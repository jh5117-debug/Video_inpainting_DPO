#!/usr/bin/env python3
"""Analyze Exp49 ROSE VOR-OR Gate16 outputs.

This script is intentionally isolated to Exp49. It reads VOR-Train condition/GT/mask
and official ROSE inference outputs, computes simple region metrics, and builds
visual evidence sheets for manual review.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

REPO = Path("/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp49_rose_adapter")
ROOT = Path("/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/extracted/vor_train_audit64_exact_20260623")
OUT_ENV = os.environ.get("EXP49_GATE16_OUT", "").strip()
if OUT_ENV:
    OUT_DIR = Path(OUT_ENV)
else:
    latest = Path("/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp49_pai_rose_adapter_feasibility/milestone_f_vor_or_gate16_latest.env")
    vals = {}
    for line in latest.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            vals[k] = v
    OUT_DIR = Path(vals["OUT"])

REPORTS = REPO / "reports"
MANIFESTS = REPO / "manifests"
EVIDENCE = REPORTS / "exp49_rose_vor_or_gate16_evidence"
for p in (REPORTS, MANIFESTS, EVIDENCE):
    p.mkdir(parents=True, exist_ok=True)

IDS = [
    "BLENDER_BEACH030_00003",
    "BLENDER_FOREST009_00013",
    "BLENDER_LAVA001_00001",
    "BLENDER_RIVER001_00002",
    "BLENDER_STREET008_00006",
    "BLENDER_WAREHOUSE001_00004",
    "BLENDER_DESERT004_00001",
    "BLENDER_BEDROOM009_00083",
    "REAL_ENV005_00003_003_05",
    "REAL_ENV026_00001_002_02",
    "REAL_ENV096_00002_001_05",
    "REAL_ENV105_00001_001_03",
    "REAL_ENV134_00005_006_01",
    "REAL_ENV155_00015_003_02",
    "REAL_ENV166_00002_001_02",
    "REAL_ENV180_00009_005_05",
]

MANUAL_LABELS = {
    "BLENDER_BEACH030_00003": "MEDIUM_HARD_ELIGIBLE",
    "BLENDER_FOREST009_00013": "ROSE_OUTPUT_USABLE",
    "BLENDER_LAVA001_00001": "MEDIUM_HARD_ELIGIBLE",
    "BLENDER_RIVER001_00002": "ROSE_OUTPUT_USABLE",
    "BLENDER_STREET008_00006": "ROSE_OUTPUT_USABLE",
    "BLENDER_WAREHOUSE001_00004": "ROSE_OUTPUT_USABLE",
    "BLENDER_DESERT004_00001": "ROSE_OUTPUT_USABLE",
    "BLENDER_BEDROOM009_00083": "SIDE_EFFECT_LEFT",
    "REAL_ENV005_00003_003_05": "ROSE_OUTPUT_USABLE",
    "REAL_ENV026_00001_002_02": "MEDIUM_HARD_ELIGIBLE",
    "REAL_ENV096_00002_001_05": "MEDIUM_HARD_ELIGIBLE",
    "REAL_ENV105_00001_001_03": "ROSE_OUTPUT_USABLE",
    "REAL_ENV134_00005_006_01": "SIDE_EFFECT_LEFT",
    "REAL_ENV155_00015_003_02": "ROSE_OUTPUT_USABLE",
    "REAL_ENV166_00002_001_02": "ROSE_OUTPUT_USABLE",
    "REAL_ENV180_00009_005_05": "MEDIUM_HARD_ELIGIBLE",
}


def video_path(kind: str, sample_id: str) -> Path:
    if kind == "condition":
        return ROOT / "VOR-Train/VOR-Train/FG_BG" / f"{sample_id}.mp4"
    if kind == "gt":
        return ROOT / "VOR-Train/VOR-Train/BG" / f"{sample_id}.mp4"
    if kind == "mask":
        return ROOT / "VOR-Train-MASK/MASK" / f"{sample_id}.mp4"
    raise ValueError(kind)


def read_video(path: Path, limit: int = 17) -> Tuple[np.ndarray, Dict[str, object]]:
    cap = cv2.VideoCapture(str(path))
    frames: List[np.ndarray] = []
    ok = cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS) if ok else 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else 0
    while ok and len(frames) < limit:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    arr = np.stack(frames, axis=0) if frames else np.zeros((0, 0, 0, 3), dtype=np.uint8)
    meta = {
        "path": str(path),
        "opened": bool(ok),
        "frames_read": int(arr.shape[0]),
        "frame_count_reported": total,
        "fps": float(fps),
        "height": int(arr.shape[1]) if arr.size else 0,
        "width": int(arr.shape[2]) if arr.size else 0,
    }
    return arr, meta


def resize_like(frames: np.ndarray, h: int, w: int, interp: int = cv2.INTER_AREA) -> np.ndarray:
    if frames.shape[1] == h and frames.shape[2] == w:
        return frames
    resized = [cv2.resize(f, (w, h), interpolation=interp) for f in frames]
    return np.stack(resized, axis=0).astype(frames.dtype)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def simple_ssim(a: np.ndarray, b: np.ndarray) -> float:
    # Lightweight global SSIM approximation over RGB frames; enough for gate readback.
    x = a.astype(np.float64).reshape(-1, 3)
    y = b.astype(np.float64).reshape(-1, 3)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    vals = []
    for ch in range(3):
        xx = x[:, ch]
        yy = y[:, ch]
        mux, muy = xx.mean(), yy.mean()
        vx, vy = xx.var(), yy.var()
        cov = ((xx - mux) * (yy - muy)).mean()
        vals.append(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2)))
    return float(np.mean(vals))


def region_metric(a: np.ndarray, b: np.ndarray, mask: np.ndarray, fn) -> float:
    valid = mask.astype(bool)
    if valid.ndim == 3:
        valid = valid[..., None]
    if not valid.any():
        return float("nan")
    aa = a[valid.repeat(3, axis=-1)].reshape(-1, 3)
    bb = b[valid.repeat(3, axis=-1)].reshape(-1, 3)
    return fn(aa, bb)


def binary_regions(mask_rgb: np.ndarray, condition: np.ndarray, gt: np.ndarray) -> Dict[str, np.ndarray]:
    gray = mask_rgb[..., 0] if mask_rgb.ndim == 4 else mask_rgb
    m = gray > 127
    kernel = np.ones((9, 9), np.uint8)
    boundary = []
    dilated = []
    eroded = []
    for fr in m.astype(np.uint8):
        d = cv2.dilate(fr, kernel, iterations=1).astype(bool)
        e = cv2.erode(fr, kernel, iterations=1).astype(bool)
        dilated.append(d)
        eroded.append(e)
        boundary.append(np.logical_xor(d, e))
    dilated_arr = np.stack(dilated, axis=0)
    boundary_arr = np.stack(boundary, axis=0)
    affected = np.mean(np.abs(condition.astype(np.float32) - gt.astype(np.float32)), axis=-1) > 8.0
    outside = ~dilated_arr
    return {"mask": m, "boundary": boundary_arr, "affected": affected, "outside": outside}


def flicker(frames: np.ndarray) -> float:
    if frames.shape[0] < 2:
        return float("nan")
    diff = np.abs(frames[1:].astype(np.float32) - frames[:-1].astype(np.float32))
    return float(diff.mean())


def black_ratio(frames: np.ndarray) -> float:
    if frames.size == 0:
        return 1.0
    gray = frames.mean(axis=-1)
    return float(np.mean(gray < 5.0))


def make_sheet(path: Path, sample_id: str, cond: np.ndarray, gt: np.ndarray, out: np.ndarray, mask: np.ndarray) -> None:
    n = out.shape[0]
    picks = sorted(set([0, n // 2, n - 1]))
    h, w = out.shape[1:3]
    thumb_w = 240
    thumb_h = max(1, round(h * thumb_w / w))
    rows = []
    labels = ["condition", "rose_output", "gt_bg", "mask", "abs_out_gt"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx in picks:
        cells = []
        absdiff = np.clip(np.abs(out[idx].astype(np.int16) - gt[idx].astype(np.int16)) * 3, 0, 255).astype(np.uint8)
        mask_vis = np.repeat(mask[idx, :, :, :1], 3, axis=-1)
        imgs = [cond[idx], out[idx], gt[idx], mask_vis, absdiff]
        for label, img in zip(labels, imgs):
            bgr = cv2.cvtColor(cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
            canvas = np.full((thumb_h + 26, thumb_w, 3), 255, np.uint8)
            canvas[:thumb_h] = bgr
            cv2.putText(canvas, f"{label} f{idx}", (5, thumb_h + 18), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            cells.append(canvas)
        rows.append(np.concatenate(cells, axis=1))
    title = np.full((34, rows[0].shape[1], 3), 255, np.uint8)
    cv2.putText(title, sample_id, (5, 23), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    sheet = np.concatenate([title] + rows, axis=0)
    cv2.imwrite(str(path), sheet)


def make_temporal_strip(path: Path, sample_id: str, out: np.ndarray) -> None:
    n = out.shape[0]
    idxs = np.linspace(0, n - 1, min(16, n)).round().astype(int).tolist()
    h, w = out.shape[1:3]
    thumb_w = 120
    thumb_h = max(1, round(h * thumb_w / w))
    cells = []
    for idx in idxs:
        img = cv2.cvtColor(cv2.resize(out[idx], (thumb_w, thumb_h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
        canvas = np.full((thumb_h + 20, thumb_w, 3), 255, np.uint8)
        canvas[:thumb_h] = img
        cv2.putText(canvas, str(idx), (4, thumb_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        cells.append(canvas)
    title = np.full((26, thumb_w * len(cells), 3), 255, np.uint8)
    cv2.putText(title, sample_id, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.concatenate([title, np.concatenate(cells, axis=1)], axis=0))


def auto_label(row: Dict[str, object]) -> str:
    if not row["decode_ok"]:
        return "TECHNICAL_INVALID"
    if row["black_frame_ratio"] > 0.05:
        return "TRIVIAL_BAD"
    outside_l1 = row["outside_l1"]
    mask_l1 = row["mask_l1"]
    boundary_l1 = row["boundary_l1"]
    if outside_l1 <= 12 and boundary_l1 <= 18 and mask_l1 <= 30:
        return "ROSE_OUTPUT_USABLE"
    if outside_l1 <= 25 and mask_l1 <= 55:
        return "MEDIUM_HARD_ELIGIBLE"
    if outside_l1 > 35:
        return "OUTSIDE_COLLAPSE_OR_DRIFT"
    return "SIDE_EFFECT_LEFT"


def main() -> None:
    rows: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []
    overview_cells: List[np.ndarray] = []
    for i, sample_id in enumerate(IDS, start=1):
        cond_path = video_path("condition", sample_id)
        gt_path = video_path("gt", sample_id)
        mask_path = video_path("mask", sample_id)
        out_path = OUT_DIR / f"example-{i}.mp4"
        cond, cond_meta = read_video(cond_path)
        gt, gt_meta = read_video(gt_path)
        mask, mask_meta = read_video(mask_path)
        out, out_meta = read_video(out_path)
        decode_ok = bool(out_meta["opened"] and out_meta["frames_read"] >= 17)
        if out.shape[0] > 0:
            h, w = out.shape[1:3]
            cond = resize_like(cond[: out.shape[0]], h, w)
            gt = resize_like(gt[: out.shape[0]], h, w)
            mask = resize_like(mask[: out.shape[0]], h, w, cv2.INTER_NEAREST)
        regions = binary_regions(mask, cond, gt) if decode_ok else {}
        row: Dict[str, object] = {
            "index": i,
            "sample_id": sample_id,
            "scene_group": sample_id.rsplit("_", 1)[0],
            "condition_path": str(cond_path),
            "gt_path": str(gt_path),
            "mask_path": str(mask_path),
            "rose_output_path": str(out_path),
            "decode_ok": decode_ok,
            "frames": int(out.shape[0]),
            "height": int(out.shape[1]) if out.size else 0,
            "width": int(out.shape[2]) if out.size else 0,
            "black_frame_ratio": black_ratio(out),
            "full_psnr": psnr(out, gt) if decode_ok else float("nan"),
            "full_ssim_approx": simple_ssim(out, gt) if decode_ok else float("nan"),
            "full_l1": l1(out, gt) if decode_ok else float("nan"),
            "cond_full_psnr": psnr(cond, gt) if decode_ok else float("nan"),
            "cond_full_l1": l1(cond, gt) if decode_ok else float("nan"),
            "temporal_flicker": flicker(out),
        }
        for region_name, region_mask in regions.items():
            expanded = region_mask[..., None]
            row[f"{region_name}_area_frac"] = float(np.mean(region_mask))
            row[f"{region_name}_psnr"] = region_metric(out, gt, expanded, psnr)
            row[f"{region_name}_l1"] = region_metric(out, gt, expanded, l1)
        row["metric_delta_full_psnr_vs_condition"] = row["full_psnr"] - row["cond_full_psnr"] if decode_ok else float("nan")
        row["metric_delta_full_l1_vs_condition"] = row["cond_full_l1"] - row["full_l1"] if decode_ok else float("nan")
        row["auto_label"] = auto_label(row)
        row["visual_label"] = MANUAL_LABELS.get(sample_id, row["auto_label"])
        row["visual_notes"] = "pending manual review" if sample_id not in MANUAL_LABELS else "Codex-inspected"
        sheet_path = EVIDENCE / f"{i:02d}_{sample_id}_review.jpg"
        strip_path = EVIDENCE / f"{i:02d}_{sample_id}_temporal_strip.jpg"
        if decode_ok:
            make_sheet(sheet_path, sample_id, cond, gt, out, mask)
            make_temporal_strip(strip_path, sample_id, out)
        row["review_sheet_path"] = str(sheet_path)
        row["temporal_strip_path"] = str(strip_path)
        rows.append(row)
        manifest_rows.append({
            "index": i,
            "sample_id": sample_id,
            "scene_group": row["scene_group"],
            "condition_path": str(cond_path),
            "gt_bg_path": str(gt_path),
            "mask_path": str(mask_path),
            "rose_output_path": str(out_path),
            "review_sheet_path": str(sheet_path),
            "temporal_strip_path": str(strip_path),
            "visual_label": row["visual_label"],
            "decode_ok": decode_ok,
            "source_split": "VOR-Train",
            "no_vor_eval": True,
            "hard_comp": False,
        })
        if decode_ok:
            thumb = cv2.imread(str(sheet_path))
            thumb = cv2.resize(thumb, (600, max(1, round(thumb.shape[0] * 600 / thumb.shape[1]))), interpolation=cv2.INTER_AREA)
            overview_cells.append(thumb)
    fieldnames = list(rows[0].keys())
    metrics_csv = REPORTS / "exp49_rose_vor_or_gate16_metrics.csv"
    visual_csv = REPORTS / "exp49_rose_vor_or_gate16_visual_review.csv"
    for path in (metrics_csv, visual_csv):
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    manifest_path = MANIFESTS / "exp49_rose_vor_or_gate16_manifest.jsonl"
    with manifest_path.open("w") as f:
        for mr in manifest_rows:
            f.write(json.dumps(mr, sort_keys=True) + "\n")
    valid = sum(1 for r in rows if r["decode_ok"])
    useful_labels = {"ROSE_OUTPUT_USABLE", "MEDIUM_HARD_ELIGIBLE"}
    useful = sum(1 for r in rows if r["visual_label"] in useful_labels)
    trivial = sum(1 for r in rows if r["visual_label"] == "TRIVIAL_BAD")
    outside_bad = sum(1 for r in rows if r["visual_label"] == "OUTSIDE_COLLAPSE_OR_DRIFT")
    label_counts: Dict[str, int] = {}
    for r in rows:
        label_counts[str(r["visual_label"])] = label_counts.get(str(r["visual_label"]), 0) + 1
    gate_pass = valid >= 15 and useful >= 8 and trivial <= 4 and outside_bad < 8
    status = "ROSE_VOR_OR_GATE16_PASS" if gate_pass else ("ROSE_VOR_OR_GATE16_WEAK" if valid >= 15 else "ROSE_VOR_OR_GATE16_BLOCKED")
    summary = {
        "status": status,
        "output_dir": str(OUT_DIR),
        "technical_valid": valid,
        "total": len(rows),
        "useful_or_loser_eligible": useful,
        "trivial_bad": trivial,
        "outside_collapse_or_drift": outside_bad,
        "label_counts": label_counts,
        "metrics_csv": str(metrics_csv),
        "visual_review_csv": str(visual_csv),
        "manifest": str(manifest_path),
        "evidence_dir": str(EVIDENCE),
        "vor_eval_used": False,
        "hard_comp_used": False,
        "training_run": False,
        "optimizer_step": False,
    }
    (REPORTS / "exp49_rose_vor_or_gate16_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    md = [
        "# Exp49 ROSE VOR-OR Gate16",
        "",
        "## Scope",
        "",
        "PAI-only official ROSE inference review on 16 VOR-Train rows. No H20, no training, no optimizer step, no VOR-Eval, no hard comp, and no official ROSE source modification.",
        "",
        "## Result",
        "",
        f"- Status: `{status}`",
        f"- Output dir: `{OUT_DIR}`",
        f"- Technical valid: {valid}/{len(rows)}",
        f"- Useful baseline or loser-eligible: {useful}/{len(rows)}",
        f"- Trivial bad: {trivial}/{len(rows)}",
        f"- Outside collapse/drift: {outside_bad}/{len(rows)}",
        f"- Visual labels: `{label_counts}`",
        "",
        "## Gate Rule",
        "",
        "Gate16 requires technical-valid >= 15/16, usable baseline or medium-hard/hard-plausible >= 8/16, trivial-bad <= 4/16, and no systematic outside collapse.",
        "",
        "## Files",
        "",
        f"- Metrics: `{metrics_csv}`",
        f"- Visual review: `{visual_csv}`",
        f"- Manifest: `{manifest_path}`",
        f"- Evidence: `{EVIDENCE}`",
        "",
        "## Notes",
        "",
        "This gate can support ROSE as a baseline/loser-generator candidate only. It does not make ROSE adapter-positive because the training-forward audit remains blocked by missing official training objective/loss plumbing.",
    ]
    (REPORTS / "exp49_rose_vor_or_gate16.md").write_text("\n".join(md) + "\n")
    if overview_cells:
        rows_img = []
        for start in range(0, len(overview_cells), 2):
            pair = overview_cells[start:start+2]
            if len(pair) == 1:
                pad = np.full_like(pair[0], 255)
                pair.append(pad)
            rows_img.append(np.concatenate(pair, axis=1))
        cv2.imwrite(str(EVIDENCE / "overview_all16.jpg"), np.concatenate(rows_img, axis=0))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
