#!/usr/bin/env python3
"""Post-process EffectErase official 81-frame inference smoke outputs.

The script keeps raw EffectErase output as the primary OR diagnostic result.
It writes diagnostic hard-comp videos and review sheets only for inspection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity


REVIEW_FRAMES = [0, 5, 10, 16, 21, 26, 32, 37, 42, 48, 53, 58, 64, 69, 74, 80]


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_video(path: Path, *, mask: bool = False) -> tuple[list[np.ndarray], dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], {"opens": False, "frames": 0, "width": 0, "height": 0, "fps": 0.0}
    frames: list[np.ndarray] = []
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mask:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append((gray > 10).astype(np.uint8))
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, {"opens": True, "frames": len(frames), "width": width, "height": height, "fps": fps}


def write_mp4(path: Path, frames: list[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise RuntimeError(f"Cannot write empty video: {path}")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps or 24.0, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter: {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    h, w = a.shape[:2]
    win = min(65, h if h % 2 else h - 1, w if w % 2 else w - 1)
    win = max(3, win)
    return float(structural_similarity(a, b, data_range=255, channel_axis=-1, win_size=win))


def masked_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    m = mask > 0
    if not np.any(m):
        return float("nan")
    diff = a.astype(np.float64) - b.astype(np.float64)
    vals = diff[m]
    mse = float(np.mean(vals * vals))
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def bbox(mask: np.ndarray, pad: int = 0) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    h, w = mask.shape[:2]
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    return x0, y0, x1, y1


def boundary_mask(mask: np.ndarray, pixels: int = 5) -> np.ndarray:
    kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), dtype=np.uint8)
    binary = (mask > 0).astype(np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(binary, kernel, iterations=1)
    return ((dilated - eroded) > 0).astype(np.uint8)


def crop_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    box = bbox(mask, pad=0)
    if box is None:
        return float("nan")
    x0, y0, x1, y1 = box
    if x1 - x0 < 3 or y1 - y0 < 3:
        return float("nan")
    return ssim(a[y0:y1, x0:x1], b[y0:y1, x0:x1])


def finite_mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def make_comp(raw: list[np.ndarray], winner: list[np.ndarray], masks: list[np.ndarray]) -> list[np.ndarray]:
    comps: list[np.ndarray] = []
    for out, gt, mask in zip(raw, winner, masks):
        m = (mask > 0)[..., None]
        comps.append(np.where(m, out, gt).astype(np.uint8))
    return comps


def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    red = np.zeros_like(frame)
    red[..., 0] = 255
    m = (mask > 0)[..., None]
    return np.where(m, (0.55 * frame + 0.45 * red).astype(np.uint8), frame)


def diff_heat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)), axis=2)
    diff = np.clip(diff * 2.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(diff, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def resize_tile(frame: np.ndarray, width: int, height: int) -> Image.Image:
    return Image.fromarray(frame.astype(np.uint8)).resize((width, height), Image.BICUBIC)


def draw_review_page(path: Path, sample_id: str, condition: list[np.ndarray], winner: list[np.ndarray], masks: list[np.ndarray], raw: list[np.ndarray], comp: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = [min(len(raw) - 1, i) for i in REVIEW_FRAMES]
    tile_w, tile_h = 144, 84
    label_w = 130
    header_h = 42
    rows = [
        ("condition", condition),
        ("winner_bg", winner),
        ("mask_overlay", [overlay_mask(condition[i], masks[i]) for i in range(len(raw))]),
        ("raw_output", raw),
        ("diag_comp", comp),
        ("absdiff_raw_bg", [diff_heat(raw[i], winner[i]) for i in range(len(raw))]),
    ]
    canvas = Image.new("RGB", (label_w + len(indices) * tile_w, header_h + len(rows) * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"{sample_id} official81 EffectErase smoke", fill=(0, 0, 0))
    for c, frame_idx in enumerate(indices):
        draw.text((label_w + c * tile_w + 4, 24), f"f{frame_idx}", fill=(0, 0, 0))
    for r, (name, frames) in enumerate(rows):
        y = header_h + r * tile_h
        draw.text((8, y + 30), name, fill=(0, 0, 0))
        for c, frame_idx in enumerate(indices):
            x = label_w + c * tile_w
            canvas.paste(resize_tile(frames[frame_idx], tile_w, tile_h), (x, y))
    canvas.save(path)


def draw_crop_page(path: Path, sample_id: str, condition: list[np.ndarray], winner: list[np.ndarray], masks: list[np.ndarray], raw: list[np.ndarray], comp: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    areas = [float((m > 0).mean()) for m in masks]
    mask_idx = int(np.argmax(areas))
    errors = [float(np.mean(np.abs(raw[i].astype(np.float32) - winner[i].astype(np.float32)))) for i in range(len(raw))]
    err_idx = int(np.argmax(errors))
    frames = [mask_idx, err_idx, len(raw) // 2]
    tile_w, tile_h = 220, 126
    crop_w, crop_h = 220, 126
    label_w = 140
    rows = [
        ("condition", condition),
        ("winner_bg", winner),
        ("mask_overlay", [overlay_mask(condition[i], masks[i]) for i in range(len(raw))]),
        ("raw_output", raw),
        ("diag_comp", comp),
        ("absdiff_raw_bg", [diff_heat(raw[i], winner[i]) for i in range(len(raw))]),
    ]
    canvas = Image.new("RGB", (label_w + len(frames) * (tile_w + crop_w), 40 + len(rows) * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"{sample_id} crops: max-mask={mask_idx}, max-error={err_idx}, mid={len(raw)//2}", fill=(0, 0, 0))
    for r, (name, frame_list) in enumerate(rows):
        y = 40 + r * tile_h
        draw.text((8, y + 46), name, fill=(0, 0, 0))
        for c, frame_idx in enumerate(frames):
            x = label_w + c * (tile_w + crop_w)
            frame = frame_list[frame_idx]
            canvas.paste(resize_tile(frame, tile_w, tile_h), (x, y))
            box = bbox(masks[frame_idx], pad=24)
            if box is None:
                crop = frame
            else:
                x0, y0, x1, y1 = box
                crop = frame[y0:y1, x0:x1]
            canvas.paste(resize_tile(crop, crop_w, crop_h), (x + tile_w, y))
    canvas.save(path)


def per_sample_metrics(winner: list[np.ndarray], condition: list[np.ndarray], masks: list[np.ndarray], raw: list[np.ndarray]) -> dict:
    whole_psnr: list[float] = []
    whole_ssim: list[float] = []
    mask_psnr: list[float] = []
    mask_ssim: list[float] = []
    boundary_psnr: list[float] = []
    boundary_ssim: list[float] = []
    outside_abs: list[float] = []
    raw_to_condition_mask_abs: list[float] = []
    condition_to_winner_mask_abs: list[float] = []
    raw_to_winner_mask_abs: list[float] = []
    for gt, cond, mask, out in zip(winner, condition, masks, raw):
        whole_psnr.append(psnr(gt, out))
        whole_ssim.append(ssim(gt, out))
        mask_psnr.append(masked_psnr(gt, out, mask))
        mask_ssim.append(crop_ssim(gt, out, mask))
        bmask = boundary_mask(mask, pixels=5)
        boundary_psnr.append(masked_psnr(gt, out, bmask))
        boundary_ssim.append(crop_ssim(gt, out, bmask))
        outside = mask <= 0
        if np.any(outside):
            outside_abs.append(float(np.mean(np.abs(out.astype(np.float32) - gt.astype(np.float32))[outside])))
        inside = mask > 0
        if np.any(inside):
            raw_to_condition_mask_abs.append(float(np.mean(np.abs(out.astype(np.float32) - cond.astype(np.float32))[inside])))
            condition_to_winner_mask_abs.append(float(np.mean(np.abs(cond.astype(np.float32) - gt.astype(np.float32))[inside])))
            raw_to_winner_mask_abs.append(float(np.mean(np.abs(out.astype(np.float32) - gt.astype(np.float32))[inside])))
    out_temporal = [float(np.mean(np.abs(raw[i].astype(np.float32) - raw[i - 1].astype(np.float32)))) for i in range(1, len(raw))]
    gt_temporal = [float(np.mean(np.abs(winner[i].astype(np.float32) - winner[i - 1].astype(np.float32)))) for i in range(1, len(winner))]
    return {
        "whole_psnr": finite_mean(whole_psnr),
        "whole_ssim": finite_mean(whole_ssim),
        "strict_mask_psnr": finite_mean(mask_psnr),
        "mask_bbox_ssim": finite_mean(mask_ssim),
        "boundary_psnr": finite_mean(boundary_psnr),
        "boundary_ssim": finite_mean(boundary_ssim),
        "outside_abs_diff_mean": finite_mean(outside_abs),
        "raw_to_condition_mask_abs": finite_mean(raw_to_condition_mask_abs),
        "condition_to_winner_mask_abs": finite_mean(condition_to_winner_mask_abs),
        "raw_to_winner_mask_abs": finite_mean(raw_to_winner_mask_abs),
        "object_effect_residual_ratio": finite_mean(raw_to_winner_mask_abs) / finite_mean(condition_to_winner_mask_abs)
        if math.isfinite(finite_mean(condition_to_winner_mask_abs)) and finite_mean(condition_to_winner_mask_abs) > 0
        else float("nan"),
        "temporal_abs_diff_output": finite_mean(out_temporal),
        "temporal_abs_diff_winner": finite_mean(gt_temporal),
        "temporal_abs_diff_delta": finite_mean(out_temporal) - finite_mean(gt_temporal),
    }


def aggregate(rows: list[dict]) -> dict:
    out: dict = {"rows": len(rows)}
    numeric = sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    for key in numeric:
        out[f"{key}_mean"] = finite_mean([float(r[key]) for r in rows if key in r])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path, required=True)
    parser.add_argument("--branch", default="research/exp29-minimax-effecterase-adapter-feasibility-20260626")
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    args.visual_dir.mkdir(parents=True, exist_ok=True)
    review_dir = args.visual_dir / "review_pages"
    crop_dir = args.visual_dir / "crop_pages"
    per_rows: list[dict] = []
    metric_manifest: list[dict] = []
    for row in rows:
        sample_id = row["sample_id"]
        raw_path = args.output_root / "outputs" / sample_id / "raw_output.mp4"
        comp_path = args.output_root / "outputs" / sample_id / "diagnostic_comp.mp4"
        condition, cond_stats = read_video(Path(row["condition_path"]))
        winner, win_stats = read_video(Path(row["winner_path"]))
        masks, mask_stats = read_video(Path(row["mask_path"]), mask=True)
        raw, raw_stats = read_video(raw_path)
        n = min(len(condition), len(winner), len(masks), len(raw))
        errors: list[str] = []
        if cond_stats["frames"] != 81:
            errors.append("condition_frames_not_81")
        if win_stats["frames"] != 81:
            errors.append("winner_frames_not_81")
        if mask_stats["frames"] != 81:
            errors.append("mask_frames_not_81")
        if raw_stats["frames"] != 81:
            errors.append("raw_output_frames_not_81")
        if not raw_path.exists():
            errors.append("raw_output_missing")
        if n != 81:
            errors.append("aligned_frame_count_not_81")
        condition, winner, masks, raw = condition[:n], winner[:n], masks[:n], raw[:n]
        comp = make_comp(raw, winner, masks) if n else []
        if comp:
            write_mp4(comp_path, comp, raw_stats.get("fps") or 24.0)
            draw_review_page(review_dir / f"{int(row.get('selection_rank', 0)):02d}_{sample_id}.jpg", sample_id, condition, winner, masks, raw, comp)
            draw_crop_page(crop_dir / f"{int(row.get('selection_rank', 0)):02d}_{sample_id}_crops.jpg", sample_id, condition, winner, masks, raw, comp)
        metrics = per_sample_metrics(winner, condition, masks, raw) if n else {}
        rec = {
            "sample_id": sample_id,
            "source_type": row.get("source_type", ""),
            "scene_group": row.get("scene_group", ""),
            "mask_bucket": row.get("mask_bucket", ""),
            "condition_path": row["condition_path"],
            "winner_path": row["winner_path"],
            "mask_path": row["mask_path"],
            "raw_output_path": str(raw_path),
            "diagnostic_comp_path": str(comp_path),
            "review_page": str(review_dir / f"{int(row.get('selection_rank', 0)):02d}_{sample_id}.jpg"),
            "crop_page": str(crop_dir / f"{int(row.get('selection_rank', 0)):02d}_{sample_id}_crops.jpg"),
            "condition_frames": cond_stats["frames"],
            "winner_frames": win_stats["frames"],
            "mask_frames": mask_stats["frames"],
            "raw_output_frames": raw_stats["frames"],
            "width": raw_stats["width"],
            "height": raw_stats["height"],
            "fps": raw_stats["fps"],
            "raw_sha256": sha256_file(raw_path) if raw_path.exists() else "",
            "technical_valid": len(errors) == 0,
            "errors": ";".join(dict.fromkeys(errors)),
        }
        rec.update(metrics)
        per_rows.append(rec)
        metric_manifest.append(
            {
                "sample_id": sample_id,
                "model_label": "EffectErase_official81_raw",
                "gt_video_path": row["winner_path"],
                "prediction_video_path": str(raw_path),
                "mask_path": row["mask_path"],
            }
        )
    aggregate_row = aggregate([row for row in per_rows if row.get("technical_valid")])
    write_csv(args.reports_dir / "exp29_effecterase_official81_inference_smoke.csv", per_rows)
    write_csv(args.reports_dir / "exp29_effecterase_official81_metric_manifest.csv", metric_manifest)
    write_csv(args.reports_dir / "exp29_effecterase_official81_aggregate_metrics.csv", [aggregate_row])
    summary = {
        "status": "EFFECTERASE_OFFICIAL81_INFERENCE_POSTPROCESSED",
        "branch": args.branch,
        "commit": args.commit,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "output_root": str(args.output_root),
        "reports_dir": str(args.reports_dir),
        "visual_dir": str(args.visual_dir),
        "rows": len(rows),
        "technical_valid": sum(1 for row in per_rows if row.get("technical_valid")),
        "aggregate_metrics": aggregate_row,
    }
    (args.reports_dir / "exp29_effecterase_official81_inference_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md = [
        "# Exp29 EffectErase Official 81F Inference Smoke",
        "",
        f"Status: `{summary['status']}`",
        f"Rows: {summary['rows']}",
        f"Technical valid: {summary['technical_valid']} / {summary['rows']}",
        f"Manifest SHA256: `{summary['manifest_sha256']}`",
        f"Output root: `{args.output_root}`",
        "",
        "Raw output remains the primary OR diagnostic. Diagnostic hard-comp videos are generated only for inspection.",
        "",
        "## Aggregate Pixel Diagnostics",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in aggregate_row.items():
        if key == "rows":
            continue
        if isinstance(value, float):
            md.append(f"| {key} | {value:.6f} |")
    (args.reports_dir / "exp29_effecterase_official81_inference_smoke.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
