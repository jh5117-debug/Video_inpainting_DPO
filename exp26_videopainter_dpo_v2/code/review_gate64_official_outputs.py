#!/usr/bin/env python3
"""Dense per-sample review for Exp26 Gate64 VideoPainter outputs.

This script does not regenerate outputs.  It reads the already materialized
Gate64 manifest and official-generation directories, computes simple objective
signals, writes per-sample evidence sheets/crops, and records a conservative
medium-hard / hard-plausible / too-close / trivial-bad / technical-invalid
classification.

The classifications are intentionally conservative and must still be paired
with human visual inspection before a training manifest is promoted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def list_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def load_rgb_dir(path: Path, limit: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for fp in list_images(path)[:limit]:
        out.append(np.asarray(Image.open(fp).convert("RGB"), dtype=np.uint8))
    return out


def load_mask_dir(path: Path, limit: int, size: tuple[int, int] | None) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for fp in list_images(path)[:limit]:
        img = Image.open(fp).convert("L")
        if size and img.size != size:
            img = img.resize(size, Image.NEAREST)
        out.append((np.asarray(img, dtype=np.uint8) > 127).astype(np.float32))
    return out


def map_run_path(path: str | Path, home_run_root: Path, nas_run_root: Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    raw = str(p)
    home = str(home_run_root)
    if raw.startswith(home):
        candidate = nas_run_root / raw[len(home) :].lstrip("/")
        if candidate.exists():
            return candidate
    return p


def resize_like(arr: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if arr.shape[:2] == ref.shape[:2]:
        return arr
    return cv2.resize(arr, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)


def psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    pred = resize_like(pred, gt).astype(np.float32)
    gt = gt.astype(np.float32)
    if mask is not None:
        if mask.shape != gt.shape[:2]:
            mask = cv2.resize(mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        w = mask[..., None].astype(np.float32)
        denom = float(w.sum() * 3.0)
        if denom < 1:
            return float("nan")
        mse = float((((pred - gt) ** 2) * w).sum() / denom)
    else:
        mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 2:
        a_gray = a.astype(np.float32)
        ref_shape = a.shape
    else:
        a_gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
        ref_shape = a.shape[:2]
    if b.ndim == 3:
        if b.shape[:2] != ref_shape:
            b = cv2.resize(b, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_CUBIC)
        b_gray = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        if b.shape[:2] != ref_shape:
            b = cv2.resize(b, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_CUBIC)
        b_gray = b.astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_a = cv2.GaussianBlur(a_gray, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b_gray, (11, 11), 1.5)
    sigma_a = cv2.GaussianBlur(a_gray * a_gray, (11, 11), 1.5) - mu_a * mu_a
    sigma_b = cv2.GaussianBlur(b_gray * b_gray, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(a_gray * b_gray, (11, 11), 1.5) - mu_a * mu_b
    val = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / ((mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2) + 1e-8)
    return float(np.mean(val))


def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = frame.astype(np.float32).copy()
    red = np.zeros_like(out)
    red[..., 0] = 255
    alpha = (mask > 0).astype(np.float32)[..., None] * 0.45
    return np.clip(out * (1 - alpha) + red * alpha, 0, 255).astype(np.uint8)


def label(arr: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, min(img.width, 420), 22], fill=(0, 0, 0))
    draw.text((5, 5), text, fill=(255, 255, 255))
    return np.asarray(img)


def error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = resize_like(pred, gt).astype(np.float32)
    err = np.abs(pred - gt.astype(np.float32)).mean(axis=2)
    return cv2.cvtColor(cv2.applyColorMap(np.clip(err * 3.0, 0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)


def crop_to_mask(arr: np.ndarray, mask: np.ndarray, pad: int = 24) -> np.ndarray:
    if mask.shape != arr.shape[:2]:
        mask = cv2.resize(mask, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return arr
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, arr.shape[1])
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, arr.shape[0])
    crop = arr[y0:y1, x0:x1]
    return cv2.resize(crop, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_CUBIC)


def temporal_stats(frames: list[np.ndarray]) -> tuple[float, float]:
    vals = []
    for idx in range(1, len(frames)):
        prev = resize_like(frames[idx - 1], frames[idx])
        vals.append(float(np.mean(np.abs(frames[idx].astype(np.float32) - prev.astype(np.float32)))))
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.max(vals))


def selected_frames(gt: list[np.ndarray], pred: list[np.ndarray], masks: list[np.ndarray]) -> list[int]:
    n = min(len(gt), len(pred), len(masks))
    picks = {0, max(0, n // 2), max(0, n - 1)}
    if n:
        areas = [float(m.sum()) for m in masks[:n]]
        picks.add(int(np.argmax(areas)))
        errs = [float(np.mean(np.abs(resize_like(pred[i], gt[i]).astype(np.float32) - gt[i].astype(np.float32)))) for i in range(n)]
        picks.add(int(np.argmax(errs)))
    return sorted(i for i in picks if 0 <= i < n)


def classify(row: dict[str, Any]) -> tuple[str, str, str, str]:
    if row["status"] != "OK":
        return "technical-invalid", "missing frames or unreadable sample", "unknown", "unknown"
    if row["black_frame_ratio"] > 0.0:
        return "technical-invalid", "black/constant frame detected", "constant_frame", "global collapse"
    if row["mask_psnr"] < 6.0 or row["full_psnr"] < 10.0:
        return "trivial-bad", "very low PSNR in task/full frame", "possible flicker/collapse", "large task-region mismatch"
    if row["mask_psnr"] > 30.0 and row["full_psnr"] > 34.0:
        return "too-close", "candidate too close to winner", "none obvious", "defect too weak"
    if row["mask_psnr"] < 12.0 or row["temporal_absdiff_max"] > 55.0:
        return "hard-plausible", "strong but finite local/temporal defect", "possible temporal artifact", "strong local mismatch"
    return "medium-hard", "finite visible defect without technical collapse", "checked_by_frame_diffs", "localized in mask/near boundary"


def make_evidence_sheet(
    sample_id: str,
    gt: list[np.ndarray],
    masks: list[np.ndarray],
    raw: list[np.ndarray],
    comp: list[np.ndarray],
    picks: list[int],
    out_path: Path,
    crop_path: Path,
) -> None:
    rows = []
    crop_rows = []
    for idx in picks:
        raw_i = resize_like(raw[idx], gt[idx])
        comp_i = resize_like(comp[idx], gt[idx]) if idx < len(comp) else raw_i
        err = error_map(raw_i, gt[idx])
        mask_overlay = overlay_mask(gt[idx], masks[idx])
        rows.append(
            np.concatenate(
                [
                    label(gt[idx], f"GT f{idx}"),
                    label(mask_overlay, "mask"),
                    label(raw_i, "raw loser"),
                    label(comp_i, "hard-comp diag"),
                    label(err, "raw abs error"),
                ],
                axis=1,
            )
        )
        crop_rows.append(
            np.concatenate(
                [
                    label(crop_to_mask(gt[idx], masks[idx]), f"GT crop f{idx}"),
                    label(crop_to_mask(mask_overlay, masks[idx]), "mask crop"),
                    label(crop_to_mask(raw_i, masks[idx]), "raw crop"),
                    label(crop_to_mask(comp_i, masks[idx]), "comp crop"),
                    label(crop_to_mask(err, masks[idx]), "error crop"),
                ],
                axis=1,
            )
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(out_path, quality=92)
    Image.fromarray(np.concatenate(crop_rows, axis=0)).save(crop_path, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--home-run-root", type=Path, default=Path("/home/hj/exp26_gate64_runs/gate64_official_43597cf_20260625_031155"))
    parser.add_argument("--num-frames", type=int, default=49)
    args = parser.parse_args()

    manifest = args.manifest or args.run_root / "gate64_mask_ready.jsonl"
    rows = read_jsonl(manifest)
    gen_root = args.run_root / "official_generation"
    review_rows: list[dict[str, Any]] = []
    for row in rows:
        sid = row["sample_id"]
        frame_dir = map_run_path(row["frame_dir"], args.home_run_root, args.run_root)
        mask_dir = map_run_path(row["mask_dir"], args.home_run_root, args.run_root)
        gt = load_rgb_dir(frame_dir, args.num_frames)
        masks = load_mask_dir(mask_dir, args.num_frames, size=gt[0].shape[1::-1] if gt else None)
        raw = load_rgb_dir(gen_root / "raw_frames" / sid, args.num_frames)
        comp = load_rgb_dir(gen_root / "comp_frames" / sid, args.num_frames)
        status = "OK" if len(gt) == args.num_frames and len(masks) == args.num_frames and len(raw) == args.num_frames else "MISSING_FRAMES"
        item: dict[str, Any] = {
            "sample_id": sid,
            "status": status,
            "model": "VideoPainter_official",
            "video_path": str(gen_root / "side_by_side" / f"{sid}.mp4"),
            "contact_sheet": str(gen_root / "contact_sheets" / f"{sid}.jpg"),
            "frame_count": len(raw),
            "mask_profile": row.get("mask_profile", ""),
            "area_bucket": row.get("area_bucket", ""),
            "motion_bucket": row.get("motion_bucket", ""),
        }
        if status == "OK":
            n = args.num_frames
            full_psnr = [psnr(raw[i], gt[i]) for i in range(n)]
            mask_psnr = [psnr(raw[i], gt[i], masks[i]) for i in range(n)]
            full_ssim = [ssim_gray(resize_like(raw[i], gt[i]), gt[i]) for i in range(n)]
            temp_mean, temp_max = temporal_stats(raw)
            black = float(sum(1 for f in raw if float(f.mean()) < 2.0 or float(f.std()) < 1.0) / len(raw))
            picks = selected_frames(gt, raw, masks)
            item.update(
                {
                    "reviewed_frames": " ".join(str(x) for x in picks),
                    "full_psnr": float(np.nanmean(full_psnr)),
                    "mask_psnr": float(np.nanmean(mask_psnr)),
                    "full_ssim": float(np.nanmean(full_ssim)),
                    "temporal_absdiff_mean": temp_mean,
                    "temporal_absdiff_max": temp_max,
                    "black_frame_ratio": black,
                }
            )
            klass, reason, temporal_artifact, spatial_artifact = classify(item)
            item.update(
                {
                    "classification": klass,
                    "reason": reason,
                    "temporal_artifact": temporal_artifact,
                    "spatial_artifact": spatial_artifact,
                    "reviewer_pass": "false",
                    "visual_review_status": "DENSE_EVIDENCE_GENERATED_HUMAN_REVIEW_REQUIRED",
                    "evidence_sheet": str(args.output_dir / "evidence_sheets" / f"{sid}.jpg"),
                    "crop_sheet": str(args.output_dir / "crop_sheets" / f"{sid}.jpg"),
                }
            )
            make_evidence_sheet(
                sid,
                gt,
                masks,
                raw,
                comp,
                picks,
                args.output_dir / "evidence_sheets" / f"{sid}.jpg",
                args.output_dir / "crop_sheets" / f"{sid}.jpg",
            )
        else:
            item.update(
                {
                    "reviewed_frames": "",
                    "full_psnr": "",
                    "mask_psnr": "",
                    "full_ssim": "",
                    "temporal_absdiff_mean": "",
                    "temporal_absdiff_max": "",
                    "black_frame_ratio": "",
                    "classification": "technical-invalid",
                    "reason": "missing formal frames or outputs",
                    "temporal_artifact": "unknown",
                    "spatial_artifact": "unknown",
                    "reviewer_pass": "false",
                    "visual_review_status": "FAILED_MISSING_FRAMES",
                    "evidence_sheet": "",
                    "crop_sheet": "",
                }
            )
        review_rows.append(item)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "model",
        "video_path",
        "contact_sheet",
        "reviewed_frames",
        "classification",
        "temporal_artifact",
        "spatial_artifact",
        "reason",
        "reviewer_pass",
        "visual_review_status",
        "status",
        "frame_count",
        "full_psnr",
        "mask_psnr",
        "full_ssim",
        "temporal_absdiff_mean",
        "temporal_absdiff_max",
        "black_frame_ratio",
        "mask_profile",
        "area_bucket",
        "motion_bucket",
        "evidence_sheet",
        "crop_sheet",
    ]
    csv_path = args.output_dir / "gate64_visual_review.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fields} for row in review_rows])

    counts: dict[str, int] = {}
    for row in review_rows:
        key = str(row["classification"])
        counts[key] = counts.get(key, 0) + 1
    summary = {
        "status": "dense_evidence_generated_human_review_required",
        "num_samples": len(review_rows),
        "classification_counts": counts,
        "csv": str(csv_path),
        "evidence_sheets": str(args.output_dir / "evidence_sheets"),
        "crop_sheets": str(args.output_dir / "crop_sheets"),
        "note": "reviewer_pass remains false until manual visual inspection is recorded.",
    }
    (args.output_dir / "gate64_visual_review_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# Exp26 Gate64 Visual Review Evidence",
        "",
        f"Status: `{summary['status']}`",
        "",
        "## Classification Counts",
        "",
    ]
    for key in sorted(counts):
        md.append(f"- `{key}`: {counts[key]}")
    md.extend(
        [
            "",
            f"- CSV: `{csv_path}`",
            f"- Evidence sheets: `{args.output_dir / 'evidence_sheets'}`",
            f"- Crop sheets: `{args.output_dir / 'crop_sheets'}`",
            "",
            "No VideoPainter DPO training is authorized by this report alone.",
        ]
    )
    (args.output_dir / "gate64_visual_review.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
