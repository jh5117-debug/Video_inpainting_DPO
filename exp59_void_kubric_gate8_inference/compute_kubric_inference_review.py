#!/usr/bin/env python3
"""Compute Exp59 Kubric official-inference metrics and review sheets.

This script is intentionally read-only with respect to model artifacts. It reads
Exp58B Kubric Gate8 inputs plus Exp59 official VOID pass1 outputs, aligns them
to the native 24-frame 128x128 diagnostic space, and writes CSV/JSON/Markdown
reports plus per-sample contact sheets for human visual review.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from skimage.metrics import structural_similarity
except Exception:  # pragma: no cover - optional dependency
    structural_similarity = None


REGION_VALUES = {
    "object": 0,
    "overlap": 63,
    "affected": 127,
    "outside": 255,
}


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_video(path: Path, rgb: bool = True) -> Tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8), fps
    return np.stack(frames, axis=0), float(fps)


def resize_frames(frames: np.ndarray, size_hw: Tuple[int, int], interp: int) -> np.ndarray:
    height, width = size_hw
    out = []
    for frame in frames:
        out.append(cv2.resize(frame, (width, height), interpolation=interp))
    return np.stack(out, axis=0)


def quantize_mask(mask_rgb: np.ndarray) -> np.ndarray:
    if mask_rgb.ndim == 4:
        gray = mask_rgb[..., 0].astype(np.float32)
    else:
        gray = mask_rgb.astype(np.float32)
    values = np.array([0.0, 63.0, 127.0, 255.0], dtype=np.float32)
    idx = np.abs(gray[..., None] - values[None, None, None, :]).argmin(axis=-1)
    return values[idx].astype(np.uint8)


def make_boundary(mask_q: np.ndarray) -> np.ndarray:
    region = (mask_q < 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    masks = []
    for frame in region:
        dilated = cv2.dilate(frame, kernel, iterations=1)
        eroded = cv2.erode(frame, kernel, iterations=1)
        masks.append((dilated != eroded).astype(bool))
    return np.stack(masks, axis=0)


def region_mask(mask_q: np.ndarray, name: str) -> np.ndarray:
    if name == "boundary":
        return make_boundary(mask_q)
    value = REGION_VALUES[name]
    return mask_q == value


def mse_region(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    if mask is not None:
        m = mask[..., None]
        count = int(m.sum()) * a.shape[-1]
        if count == 0:
            return float("nan")
        return float((diff * diff * m).sum() / count)
    return float(np.mean(diff * diff))


def l1_region(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    if mask is not None:
        m = mask[..., None]
        count = int(m.sum()) * a.shape[-1]
        if count == 0:
            return float("nan")
        return float((diff * m).sum() / count / 255.0)
    return float(np.mean(diff) / 255.0)


def psnr_from_mse(mse: float) -> float:
    if math.isnan(mse):
        return float("nan")
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def ssim_video(a: np.ndarray, b: np.ndarray) -> float:
    if structural_similarity is None:
        return float("nan")
    vals = []
    for af, bf in zip(a, b):
        vals.append(
            structural_similarity(
                af,
                bf,
                channel_axis=-1,
                data_range=255,
            )
        )
    return float(np.mean(vals)) if vals else float("nan")


def temporal_flicker(output: np.ndarray, gt: np.ndarray) -> float:
    if len(output) < 2 or len(gt) < 2:
        return float("nan")
    do = np.diff(output.astype(np.float32), axis=0)
    dg = np.diff(gt.astype(np.float32), axis=0)
    return float(np.mean(np.abs(do - dg)) / 255.0)


def outside_tone_drift(output: np.ndarray, gt: np.ndarray, outside: np.ndarray) -> float:
    if outside.sum() == 0:
        return float("nan")
    vals = []
    for c in range(3):
        o = output[..., c][outside].astype(np.float32).mean()
        g = gt[..., c][outside].astype(np.float32).mean()
        vals.append(abs(o - g) / 255.0)
    return float(np.mean(vals))


def fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def classify(row: Dict[str, str]) -> str:
    full = float(row["full_psnr"])
    outside = float(row["outside_psnr"])
    obj = float(row["object_psnr"])
    overlap = float(row["overlap_psnr"])
    affected = float(row["affected_psnr"])
    boundary = float(row["boundary_psnr"])
    if row["target_hit"] == "false":
        weak = "KUBRIC_TARGET_HIT_WEAK"
    else:
        weak = ""
    if full < 10.0 or outside < 14.0:
        return "KUBRIC_TRIVIAL_BAD" + (f";{weak}" if weak else "")
    if outside < 20.0:
        return "KUBRIC_OUTSIDE_DAMAGE" + (f";{weak}" if weak else "")
    if min(overlap, affected, boundary) < 16.0:
        return "KUBRIC_TRANSITION_DAMAGE" + (f";{weak}" if weak else "")
    if full > 35.0 and obj > 30.0:
        return "KUBRIC_TOO_CLOSE" + (f";{weak}" if weak else "")
    if 18.0 <= full <= 32.0 and outside >= 24.0:
        return "KUBRIC_MEDIUM_HARD_LOSER" + (f";{weak}" if weak else "")
    return "KUBRIC_OUTPUT_USABLE" + (f";{weak}" if weak else "")


def load_font(size: int = 26) -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(candidate):
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def add_title(img: Image.Image, title: str, font: ImageFont.ImageFont) -> Image.Image:
    pad = 8
    title_h = 42
    out = Image.new("RGB", (img.width, img.height + title_h), "white")
    out.paste(img.convert("RGB"), (0, title_h))
    draw = ImageDraw.Draw(out)
    draw.text((pad, pad), title, fill=(0, 0, 0), font=font)
    return out


def resize_keep_width(img: Image.Image, width: int) -> Image.Image:
    if img.width <= width:
        return img.convert("RGB")
    height = max(1, int(img.height * width / img.width))
    return img.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)


def make_contact_sheet(sample_id: str, evidence_dir: Path, output_path: Path) -> None:
    font = load_font(24)
    items = [
        ("temporal strip / side-by-side proxy", "temporal_strip_16f.jpg"),
        ("object crop", "object_crop_sheet.jpg"),
        ("overlap crop", "overlap_crop_sheet.jpg"),
        ("affected crop", "affected_crop_sheet.jpg"),
        ("boundary crop", "boundary_crop_sheet.jpg"),
        ("outside crop", "outside_crop_sheet.jpg"),
        ("temporal diff heatmap", "temporal_diff_heatmap.jpg"),
        ("mask value histogram", "mask_value_histogram.jpg"),
    ]
    panels = []
    for title, filename in items:
        path = evidence_dir / filename
        if path.exists():
            img = resize_keep_width(Image.open(path), 900)
        else:
            img = Image.new("RGB", (900, 180), (245, 245, 245))
            ImageDraw.Draw(img).text((20, 70), f"missing: {filename}", fill=(0, 0, 0), font=font)
        panels.append(add_title(img, title, font))

    col_w = 930
    rows = []
    for left, right in zip(panels[0::2], panels[1::2]):
        h = max(left.height, right.height)
        row = Image.new("RGB", (col_w * 2, h), "white")
        row.paste(left, (0, 0))
        row.paste(right, (col_w, 0))
        rows.append(row)
    header = Image.new("RGB", (col_w * 2, 64), "white")
    ImageDraw.Draw(header).text((10, 12), f"Exp59 Kubric official inference review: {sample_id}", fill=(0, 0, 0), font=load_font(32))
    total_h = header.height + sum(row.height for row in rows)
    sheet = Image.new("RGB", (col_w * 2, total_h), "white")
    y = 0
    sheet.paste(header, (0, y))
    y += header.height
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def compute(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.inputs_manifest))
    out_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    contact_root = Path(args.contact_root)
    metrics_rows: List[Dict[str, str]] = []
    visual_rows: List[Dict[str, str]] = []

    for src in rows:
        sid = src["sample_id"]
        evidence_dir = Path(args.evidence_root) / sid
        raw_path = evidence_dir / "raw_output.mp4"
        gt_path = Path(src["rgb_removed"])
        input_path = Path(src["input_video"])
        mask_path = Path(src["quadmask_0"])

        raw, raw_fps = read_video(raw_path)
        gt, gt_fps = read_video(gt_path)
        inp, _ = read_video(input_path)
        mask_rgb, _ = read_video(mask_path)
        if len(raw) == 0 or len(gt) == 0 or len(mask_rgb) == 0:
            metrics_rows.append(
                {
                    "sample_id": sid,
                    "decode_status": "fail",
                    "classification_auto": "KUBRIC_TECHNICAL_INVALID",
                }
            )
            continue

        n = min(24, len(raw), len(gt), len(inp), len(mask_rgb))
        gt = gt[:n]
        inp = inp[:n]
        mask_q = quantize_mask(mask_rgb[:n])
        raw_native = resize_frames(raw[:n], gt.shape[1:3], cv2.INTER_AREA)
        if inp.shape[1:3] != gt.shape[1:3]:
            inp = resize_frames(inp, gt.shape[1:3], cv2.INTER_AREA)
        if mask_q.shape[1:3] != gt.shape[1:3]:
            mask_q = resize_frames(mask_q[..., None], gt.shape[1:3], cv2.INTER_NEAREST)[..., 0]

        region_masks = {name: region_mask(mask_q, name) for name in ["object", "overlap", "affected", "outside", "boundary"]}
        union_effect = region_masks["overlap"] | region_masks["affected"] | region_masks["boundary"]

        metric: Dict[str, str] = {
            "sample_id": sid,
            "target_hit": src.get("target_hit", ""),
            "decode_status": "pass",
            "frames_compared": str(n),
            "raw_frames": str(len(raw)),
            "raw_resolution": f"{raw.shape[2]}x{raw.shape[1]}",
            "raw_fps": fmt(raw_fps),
            "gt_resolution": f"{gt.shape[2]}x{gt.shape[1]}",
            "gt_fps": fmt(gt_fps),
            "metric_space": "first24_frames_output_downscaled_to_native_128x128",
            "lpips": "NA",
            "ewarp": "NA",
            "tc": "NA",
        }
        metric["full_psnr"] = fmt(psnr_from_mse(mse_region(raw_native, gt)))
        metric["ssim"] = fmt(ssim_video(raw_native, gt))
        metric["full_l1"] = fmt(l1_region(raw_native, gt))
        for name in ["object", "overlap", "affected", "boundary", "outside"]:
            mask = region_masks[name]
            metric[f"{name}_pixels"] = str(int(mask.sum()))
            metric[f"{name}_psnr"] = fmt(psnr_from_mse(mse_region(raw_native, gt, mask)))
            metric[f"{name}_l1"] = fmt(l1_region(raw_native, gt, mask))
        metric["outside_psnr_l1"] = f'{metric["outside_psnr"]}/{metric["outside_l1"]}'
        metric["temporal_flicker"] = fmt(temporal_flicker(raw_native, gt))
        metric["object_residual"] = fmt(l1_region(raw_native, gt, region_masks["object"]))
        metric["effect_residual"] = fmt(l1_region(raw_native, gt, union_effect))
        metric["tone_drift"] = fmt(outside_tone_drift(raw_native, gt, region_masks["outside"]))
        metric["output_input_l1"] = fmt(l1_region(raw_native, inp))
        metric["output_gt_l1"] = metric["full_l1"]
        metric["output_input_psnr"] = fmt(psnr_from_mse(mse_region(raw_native, inp)))
        metric["output_gt_psnr"] = metric["full_psnr"]
        metric["mask_values"] = "|".join(str(int(v)) for v in sorted(np.unique(mask_q).tolist()))
        metric["classification_auto"] = classify(metric)
        metrics_rows.append(metric)

        contact_path = contact_root / sid / "review_contact_sheet.jpg"
        make_contact_sheet(sid, evidence_dir, contact_path)
        visual_rows.append(
            {
                "sample_id": sid,
                "frames_reviewed": str(n),
                "input_valid": "yes",
                "mask_valid": "yes",
                "target_hit": src.get("target_hit", ""),
                "output_valid": "yes",
                "object_removed": "manual_review_pending",
                "overlap_quality": "manual_review_pending",
                "affected_quality": "manual_review_pending",
                "boundary_quality": "manual_review_pending",
                "outside_quality": "manual_review_pending",
                "temporal_quality": "manual_review_pending",
                "classification": metric["classification_auto"],
                "reason": "Auto-prefilled; Codex visual review required before final promotion.",
                "contact_sheet": str(contact_path),
                "side_by_side": str(evidence_dir / "side_by_side.mp4"),
                "temporal_strip": str(evidence_dir / "temporal_strip_16f.jpg"),
                "object_crop_sheet": str(evidence_dir / "object_crop_sheet.jpg"),
                "overlap_crop_sheet": str(evidence_dir / "overlap_crop_sheet.jpg"),
                "affected_crop_sheet": str(evidence_dir / "affected_crop_sheet.jpg"),
                "boundary_crop_sheet": str(evidence_dir / "boundary_crop_sheet.jpg"),
                "outside_crop_sheet": str(evidence_dir / "outside_crop_sheet.jpg"),
                "temporal_diff_heatmap": str(evidence_dir / "temporal_diff_heatmap.jpg"),
                "mask_value_histogram": str(evidence_dir / "mask_value_histogram.jpg"),
            }
        )

    metrics_path = out_dir / "exp59_kubric_official_inference_metrics.csv"
    visual_path = out_dir / "exp59_kubric_official_inference_visual_review.csv"
    metric_fields = sorted({k for row in metrics_rows for k in row.keys()})
    visual_fields = list(visual_rows[0].keys()) if visual_rows else []
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metrics_rows)
    with visual_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=visual_fields)
        writer.writeheader()
        writer.writerows(visual_rows)

    numeric_cols = [
        "full_psnr",
        "ssim",
        "object_psnr",
        "overlap_psnr",
        "affected_psnr",
        "boundary_psnr",
        "outside_psnr",
        "outside_l1",
        "temporal_flicker",
        "object_residual",
        "effect_residual",
        "tone_drift",
        "output_input_l1",
        "output_gt_l1",
    ]
    summary: Dict[str, object] = {
        "status": "VOID_KUBRIC_INFERENCE_REVIEW_PENDING_VISUAL",
        "sample_count": len(metrics_rows),
        "decode_pass": sum(1 for r in metrics_rows if r.get("decode_status") == "pass"),
        "target_hit_false": sum(1 for r in metrics_rows if r.get("target_hit") == "false"),
        "metric_space": "first24_frames_output_downscaled_to_native_128x128",
        "lpips": "not_available",
        "ewarp": "not_available",
        "contact_root": str(contact_root),
    }
    for col in numeric_cols:
        vals = []
        for row in metrics_rows:
            value = row.get(col, "NA")
            try:
                vals.append(float(value))
            except Exception:
                pass
        if vals:
            summary[f"mean_{col}"] = float(np.mean(vals))
    class_counts: Dict[str, int] = {}
    for row in metrics_rows:
        for cls in row.get("classification_auto", "").split(";"):
            if cls:
                class_counts[cls] = class_counts.get(cls, 0) + 1
    summary["classification_auto_counts"] = class_counts
    summary_path = out_dir / "exp59_kubric_official_inference_review_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    md_path = out_dir / "exp59_kubric_official_inference_review.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Exp59 Kubric Official Inference Metrics And Review\n\n")
        handle.write("Status: `VOID_KUBRIC_INFERENCE_REVIEW_PENDING_VISUAL`\n\n")
        handle.write("Metrics are computed against `rgb_removed` in the common native diagnostic space: first 24 frames, official output downscaled to 128x128.\n\n")
        handle.write("LPIPS/Ewarp are marked `NA` because the isolated review script did not find those metric implementations in this no-training diagnostic path.\n\n")
        handle.write("## Mean Metrics\n\n")
        for col in numeric_cols:
            key = f"mean_{col}"
            if key in summary:
                handle.write(f"- {col}: `{summary[key]:.6f}`\n")
        handle.write("\n## Auto Classification Counts\n\n")
        for cls, count in sorted(class_counts.items()):
            handle.write(f"- {cls}: {count}\n")
        handle.write("\n## Visual Review\n\n")
        handle.write("Contact sheets were generated for all 8 samples. Codex visual review must replace the pending fields in the CSV before any promotion. No training, one-step, or 10-step was run.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs-manifest", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--contact-root", required=True)
    compute(parser.parse_args())


if __name__ == "__main__":
    main()
