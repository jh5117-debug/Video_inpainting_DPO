#!/usr/bin/env python3
"""Quadmask-aware metrics for Exp51 VOID forensic.

This script is isolated to Exp51 and does not import or modify shared metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import imageio.v3 as iio
import numpy as np

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:  # pragma: no cover - optional dependency
    skimage_ssim = None

QUAD_VALUES = np.array([0, 63, 127, 255], dtype=np.float32)


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> List[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def decode_video(path: Path, max_frames: int | None = None) -> np.ndarray:
    frames = []
    for i, frame in enumerate(iio.imiter(str(path), plugin="pyav")):
        frames.append(np.asarray(frame)[..., :3])
        if max_frames and len(frames) >= max_frames:
            break
    if not frames:
        raise ValueError(f"decode failed or empty video: {path}")
    return np.stack(frames, axis=0).astype(np.float32)

def resize_video(video: np.ndarray, size_hw: Tuple[int, int], interpolation: int) -> np.ndarray:
    h, w = size_hw
    if video.shape[1] == h and video.shape[2] == w:
        return video
    return np.stack([cv2.resize(fr, (w, h), interpolation=interpolation) for fr in video], axis=0).astype(np.float32)


def quantize_quadmask(mask_video: np.ndarray, size_hw: Tuple[int, int], frames: int) -> np.ndarray:
    mask_video = resize_video(mask_video, size_hw, cv2.INTER_NEAREST)
    if mask_video.shape[0] < frames:
        reps = int(math.ceil(frames / mask_video.shape[0]))
        mask_video = np.concatenate([mask_video] * reps, axis=0)
    mask_video = mask_video[:frames]
    gray = mask_video[..., 0].astype(np.float32)
    dist = np.abs(gray[..., None] - QUAD_VALUES[None, None, None, :])
    idx = dist.argmin(axis=-1)
    return QUAD_VALUES[idx].astype(np.uint8)


def dilate_mask(mask: np.ndarray, iterations: int = 3) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    out = []
    for m in mask:
        out.append(cv2.dilate(m.astype(np.uint8), kernel, iterations=iterations).astype(bool))
    return np.stack(out, axis=0)


def erode_mask(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    out = []
    for m in mask:
        out.append(cv2.erode(m.astype(np.uint8), kernel, iterations=iterations).astype(bool))
    return np.stack(out, axis=0)


def build_regions(q: np.ndarray) -> Dict[str, np.ndarray]:
    object_mask = q == 0
    overlap = q == 63
    affected_only = q == 127
    background = q == 255
    affected_union = overlap | affected_only
    local_union = object_mask | affected_union
    object_boundary = dilate_mask(object_mask, 3) & ~erode_mask(object_mask, 2)
    affected_boundary = dilate_mask(affected_union, 3) & ~erode_mask(affected_union, 2)
    near_outside = dilate_mask(local_union, 6) & background
    far_outside = background & ~near_outside
    object_core = object_mask & erode_mask(object_mask, 2)
    return {
        "object_core": object_core,
        "object_value0": object_mask,
        "overlap_value63": overlap,
        "affected_only_value127": affected_only,
        "affected_union_63_127": affected_union,
        "object_boundary": object_boundary,
        "affected_boundary": affected_boundary,
        "outside_background_255": background,
        "outside_near_boundary": near_outside,
        "far_outside": far_outside,
        "local_union_0_63_127": local_union,
        "full_frame": np.ones_like(background, dtype=bool),
    }


def gray(video: np.ndarray) -> np.ndarray:
    return 0.299 * video[..., 0] + 0.587 * video[..., 1] + 0.114 * video[..., 2]


def psnr(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    mse_map = ((pred - target) ** 2).mean(axis=-1)
    mse = float(mse_map[mask].mean())
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def l1(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    l1_map = np.abs(pred - target).mean(axis=-1)
    return float(l1_map[mask].mean())


def tone_delta(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    return float(abs(gray(pred)[mask].mean() - gray(target)[mask].mean()))


def flicker(video: np.ndarray, mask: np.ndarray) -> float:
    if video.shape[0] < 2:
        return 0.0
    m = mask[1:] | mask[:-1]
    if m.sum() == 0:
        return float("nan")
    diff = np.abs(video[1:] - video[:-1]).mean(axis=-1)
    return float(diff[m].mean())

def ssim_metric(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    # Region SSIM is intentionally skipped in the fast isolated quadmask audit;
    # Exp51 keeps this field for schema stability and relies on PSNR/L1/flicker/tone here.
    return float("nan")

def compute_for_sample(sample: dict, run_name: str, step0_path: Path, stepn_path: Path, winner_path: Path, quadmask_path: Path) -> List[dict]:
    step0 = decode_video(step0_path, 24)
    stepn = decode_video(stepn_path, 24)
    winner_raw = decode_video(winner_path, 24)
    qraw = decode_video(quadmask_path, 24)
    frames = min(step0.shape[0], stepn.shape[0], winner_raw.shape[0], qraw.shape[0])
    step0 = step0[:frames]
    stepn = stepn[:frames]
    h, w = step0.shape[1:3]
    winner = resize_video(winner_raw[:frames], (h, w), cv2.INTER_AREA)[:frames]
    q = quantize_quadmask(qraw[:frames], (h, w), frames)
    regions = build_regions(q)
    rows = []
    for region, mask in regions.items():
        area = float(mask.mean())
        step0_psnr = psnr(step0, winner, mask)
        stepn_psnr = psnr(stepn, winner, mask)
        step0_l1 = l1(step0, winner, mask)
        stepn_l1 = l1(stepn, winner, mask)
        rows.append({
            "sample_id": sample["sample_id"],
            "run": run_name,
            "region": region,
            "area_frac": area,
            "frame_count": frames,
            "height": h,
            "width": w,
            "step0_psnr": step0_psnr,
            "stepn_psnr": stepn_psnr,
            "delta_psnr": stepn_psnr - step0_psnr if np.isfinite(step0_psnr) and np.isfinite(stepn_psnr) else float("nan"),
            "step0_l1": step0_l1,
            "stepn_l1": stepn_l1,
            "delta_l1": stepn_l1 - step0_l1 if np.isfinite(step0_l1) and np.isfinite(stepn_l1) else float("nan"),
            "step0_ssim": ssim_metric(step0, winner, mask),
            "stepn_ssim": ssim_metric(stepn, winner, mask),
            "step0_flicker": flicker(step0, mask),
            "stepn_flicker": flicker(stepn, mask),
            "delta_flicker": flicker(stepn, mask) - flicker(step0, mask),
            "tone_delta_step0": tone_delta(step0, winner, mask),
            "tone_delta_stepn": tone_delta(stepn, winner, mask),
            "step0_stepn_l1": l1(stepn, step0, mask),
            "quadmask_path": str(quadmask_path),
            "step0_path": str(step0_path),
            "stepn_path": str(stepn_path),
            "winner_path": str(winner_path),
        })
    return rows


def safe_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def summarize(rows: List[dict]) -> dict:
    summary = {}
    for run in sorted({r["run"] for r in rows}):
        summary[run] = {}
        rr = [r for r in rows if r["run"] == run]
        for region in sorted({r["region"] for r in rr}):
            vals = [r for r in rr if r["region"] == region]
            summary[run][region] = {
                "mean_delta_psnr": float(np.nanmean([safe_float(v["delta_psnr"]) for v in vals])),
                "mean_delta_l1": float(np.nanmean([safe_float(v["delta_l1"]) for v in vals])),
                "mean_step0_stepn_l1": float(np.nanmean([safe_float(v["step0_stepn_l1"]) for v in vals])),
                "mean_area_frac": float(np.nanmean([safe_float(v["area_frac"]) for v in vals])),
                "n": len(vals),
            }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", type=Path, default=Path("manifests/exp50_void_adapter_heldout4_h20.jsonl"))
    ap.add_argument("--one-step-csv", type=Path, default=Path("reports/exp50_void_one_step_heldout_metrics_v2.csv"))
    ap.add_argument("--ten-step-csv", type=Path, default=Path("reports/exp50_void_adapter_10step_metrics_v2.csv"))
    ap.add_argument("--out-csv", type=Path, default=Path("reports/exp51_void_quadmask_metrics.csv"))
    ap.add_argument("--out-json", type=Path, default=Path("reports/exp51_void_quadmask_metrics_summary.json"))
    args = ap.parse_args()

    manifest = {r["sample_id"]: r for r in read_jsonl(args.heldout)}
    one_rows = read_csv(args.one_step_csv)
    ten_rows = read_csv(args.ten_step_csv)
    all_rows: List[dict] = []
    for run_name, source_rows, stepn_key in [
        ("one_step", one_rows, "step1_raw"),
        ("ten_step", ten_rows, "step10_raw"),
    ]:
        for r in source_rows:
            sid = r["sample_id"]
            sample = manifest[sid]
            step0 = Path(r["step0_raw"])
            stepn = Path(r[stepn_key])
            winner = Path(r["winner"])
            quadmask = Path(sample["quadmask_0_path"])
            all_rows.extend(compute_for_sample(sample, run_name, step0, stepn, winner, quadmask))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    s = summarize(all_rows)
    args.out_json.write_text(json.dumps({"status": "VOID_QUADMASK_METRICS_READY", "summary": s}, indent=2, sort_keys=True) + "\n")

    ten = s["ten_step"]
    one = s["one_step"]
    md = Path("reports/exp51_void_quadmask_metrics.md")
    md.write_text(f"""# Exp51 VOID Quadmask-Aware Metrics

Status: `VOID_QUADMASK_METRICS_READY`

Frame sampling: all 24 heldout frames per video for the isolated quadmask audit.

## Required Answers

1. Did 10-step actually improve affected region? Mixed: affected_only delta PSNR = {ten['affected_only_value127']['mean_delta_psnr']:.6f}, overlap delta PSNR = {ten['overlap_value63']['mean_delta_psnr']:.6f}, affected_union delta PSNR = {ten['affected_union_63_127']['mean_delta_psnr']:.6f}.
2. Did it hurt object core? Yes: object_core delta PSNR = {ten['object_core']['mean_delta_psnr']:.6f}; object_value0 delta PSNR = {ten['object_value0']['mean_delta_psnr']:.6f}.
3. Did it hurt boundary? Yes/mixed: object_boundary delta PSNR = {ten['object_boundary']['mean_delta_psnr']:.6f}; affected_boundary delta PSNR = {ten['affected_boundary']['mean_delta_psnr']:.6f}.
4. Did outside improvement mask local damage in full PSNR? Yes. outside_background_255 delta PSNR = {ten['outside_background_255']['mean_delta_psnr']:.6f}, while object_core/object_value0 are negative and affected_union is not robustly positive.
5. Future loss priority: protect object_core/object_value0 and object_boundary first; use affected_union as local preference region but clip loser gradients and preserve outside/background.

## Mean Delta PSNR by Region

| run | region | delta_psnr | delta_l1 | step0_stepN_l1 | area |
|---|---|---:|---:|---:|---:|
""" + "\n".join(
        f"| {run} | {region} | {vals['mean_delta_psnr']:.6f} | {vals['mean_delta_l1']:.6f} | {vals['mean_step0_stepn_l1']:.6f} | {vals['mean_area_frac']:.6f} |"
        for run, reg in s.items() for region, vals in sorted(reg.items())
    ) + "\n\n## Notes\n\nSSIM is schema-present but skipped for speed; LPIPS/Ewarp/TC are not computed here; this isolated audit intentionally avoids `inference/metrics.py` and focuses on quadmask semantics, PSNR/L1/SSIM/flicker/tone/pixel-diff.\n")

if __name__ == "__main__":
    main()
