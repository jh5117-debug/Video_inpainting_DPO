#!/usr/bin/env python3
"""Thin target-domain inpainting metric adapter.

This wrapper intentionally delegates PSNR/SSIM/LPIPS/Ewarp math to the
existing project metric backend in ``inference.metrics``. It only handles
manifest parsing, video/mask loading, region cropping, and report writing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import metrics as metric_backend


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate inpainting outputs with the existing inference.metrics "
            "backend. The manifest may be CSV or JSONL."
        )
    )
    parser.add_argument("--pair_manifest", required=True, help="CSV/JSONL rows pairing GT, prediction, and mask paths.")
    parser.add_argument("--output_dir", required=True, help="Directory where metric summaries are written.")
    parser.add_argument("--sample_id_col", default="sample_id")
    parser.add_argument("--model_col", default="model_label")
    parser.add_argument("--gt_col", default="gt_video_path")
    parser.add_argument("--pred_col", default="prediction_video_path")
    parser.add_argument("--mask_col", default="mask_path")
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument("--width", type=int, default=0, help="Optional resize width before metrics.")
    parser.add_argument("--height", type=int, default=0, help="Optional resize height before metrics.")
    parser.add_argument("--boundary_pixels", type=int, default=3)
    parser.add_argument("--compute_lpips", action="store_true", help="Use inference.metrics.LPIPSMetric if available.")
    parser.add_argument("--compute_ewarp", action="store_true", help="Use inference.metrics.EwarpMetric as temporal metric.")
    parser.add_argument("--device", default=None, help="Torch device for optional LPIPS/Ewarp. Defaults to cuda if available.")
    parser.add_argument("--strict_missing", action="store_true", help="Fail on missing files instead of recording row issues.")
    return parser.parse_args()


def read_manifest(path: Path) -> List[Dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve_path(value: str, manifest_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidate = manifest_dir / path
    if candidate.exists():
        return candidate
    return path


def read_image_sequence(path: Path, max_frames: Optional[int], is_mask: bool) -> List[np.ndarray]:
    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if max_frames:
        files = files[:max_frames]
    frames: List[np.ndarray] = []
    for item in files:
        flag = cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR
        arr = cv2.imread(str(item), flag)
        if arr is None:
            continue
        if is_mask:
            frames.append((arr > 127).astype(np.uint8))
        else:
            frames.append(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    return frames


def read_video_or_frames(path: Path, max_frames: int, is_mask: bool) -> List[np.ndarray]:
    if path.is_dir():
        return read_image_sequence(path, max_frames, is_mask=is_mask)
    if is_mask:
        return metric_backend.read_mask_frames(str(path), max_frames=max_frames)
    return metric_backend.read_video_frames(str(path), max_frames=max_frames)


def resize_frames(frames: Sequence[np.ndarray], width: int, height: int, is_mask: bool) -> List[np.ndarray]:
    if not width or not height:
        return [np.asarray(frame) for frame in frames]
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    resized = []
    for frame in frames:
        arr = cv2.resize(np.asarray(frame), (width, height), interpolation=interp)
        if is_mask:
            arr = (arr > 0).astype(np.uint8)
        resized.append(arr)
    return resized


def finite_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def finite_median(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.median(vals)) if vals else float("nan")


def metric_pair(gt: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    return metric_backend.compute_psnr(gt, pred), metric_backend.compute_ssim(gt, pred)


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_metric(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    box = bbox_from_mask(mask)
    if box is None:
        return float("nan"), float("nan")
    x0, y0, x1, y1 = box
    if x1 - x0 < 3 or y1 - y0 < 3:
        return float("nan"), float("nan")
    return metric_pair(gt[y0:y1, x0:x1], pred[y0:y1, x0:x1])


def boundary_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    pixels = max(1, int(pixels))
    kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), dtype=np.uint8)
    binary = (mask > 0).astype(np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(binary, kernel, iterations=1)
    return ((dilated - eroded) > 0).astype(np.uint8)


def outside_diff(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    outside = mask <= 0
    if not np.any(outside):
        return float("nan"), float("nan")
    diff = np.abs(pred.astype(np.float32) - gt.astype(np.float32))
    vals = diff[outside]
    return float(vals.mean()), float(vals.max())


def compute_lpips_frame(gt: np.ndarray, pred: np.ndarray, device: str) -> float:
    return float(metric_backend.LPIPSMetric.compute(gt, pred, device=device))


def compute_ewarp(frames: Sequence[np.ndarray], gt_frames: Sequence[np.ndarray], masks: Sequence[np.ndarray], device: str) -> float:
    metric = metric_backend.EwarpMetric(device=device, raft_model_path=None)
    masks01 = [(m > 0).astype(np.uint8) for m in masks]
    return float(metric.compute(list(frames), masks01=masks01, gt_frames_u8_rgb=list(gt_frames), only_mask_region=True))


def evaluate_row(row: Dict[str, str], args: argparse.Namespace, manifest_dir: Path) -> Dict[str, object]:
    sample_id = row.get(args.sample_id_col) or row.get("id") or row.get("sample") or ""
    model_label = row.get(args.model_col) or "model"
    gt_path = resolve_path(row.get(args.gt_col, ""), manifest_dir)
    pred_path = resolve_path(row.get(args.pred_col, ""), manifest_dir)
    mask_path = resolve_path(row.get(args.mask_col, ""), manifest_dir)

    result: Dict[str, object] = {
        "sample_id": sample_id,
        "model_label": model_label,
        "gt_video_path": str(gt_path),
        "prediction_video_path": str(pred_path),
        "mask_path": str(mask_path),
        "status": "ok",
        "issue": "",
    }

    missing = [name for name, path in (("gt", gt_path), ("prediction", pred_path), ("mask", mask_path)) if not path.exists()]
    if missing:
        message = "missing " + ",".join(missing)
        if args.strict_missing:
            raise FileNotFoundError(f"{message}: {row}")
        result.update({"status": "skipped", "issue": message})
        return result

    gt_frames = read_video_or_frames(gt_path, args.max_frames, is_mask=False)
    pred_frames = read_video_or_frames(pred_path, args.max_frames, is_mask=False)
    mask_frames = read_video_or_frames(mask_path, args.max_frames, is_mask=True)
    gt_frames = resize_frames(gt_frames, args.width, args.height, is_mask=False)
    pred_frames = resize_frames(pred_frames, args.width, args.height, is_mask=False)
    mask_frames = resize_frames(mask_frames, args.width, args.height, is_mask=True)

    n = min(len(gt_frames), len(pred_frames), len(mask_frames))
    if n == 0:
        result.update({"status": "skipped", "issue": "zero readable aligned frames"})
        return result
    gt_frames, pred_frames, mask_frames = gt_frames[:n], pred_frames[:n], mask_frames[:n]

    whole_psnr, whole_ssim = [], []
    mask_psnr, mask_ssim = [], []
    bound_psnr, bound_ssim = [], []
    out_mean, out_max = [], []
    lpips_vals = []

    for gt, pred, mask in zip(gt_frames, pred_frames, mask_frames):
        p, s = metric_pair(gt, pred)
        whole_psnr.append(p)
        whole_ssim.append(s)

        p, s = crop_metric(gt, pred, mask)
        mask_psnr.append(p)
        mask_ssim.append(s)

        bmask = boundary_mask(mask, args.boundary_pixels)
        p, s = crop_metric(gt, pred, bmask)
        bound_psnr.append(p)
        bound_ssim.append(s)

        mean_diff, max_diff = outside_diff(gt, pred, mask)
        out_mean.append(mean_diff)
        out_max.append(max_diff)

        if args.compute_lpips:
            lpips_vals.append(compute_lpips_frame(gt, pred, device=args.device))

    result.update(
        {
            "num_frames": n,
            "whole_video_psnr": finite_mean(whole_psnr),
            "whole_video_ssim": finite_mean(whole_ssim),
            "mask_region_psnr": finite_mean(mask_psnr),
            "mask_region_ssim": finite_mean(mask_ssim),
            "boundary_psnr": finite_mean(bound_psnr),
            "boundary_ssim": finite_mean(bound_ssim),
            "outside_region_diff_mean": finite_mean(out_mean),
            "outside_region_diff_max": finite_mean(out_max),
        }
    )
    if args.compute_lpips:
        result["whole_video_lpips"] = finite_mean(lpips_vals)
    if args.compute_ewarp:
        ewarp = compute_ewarp(pred_frames, gt_frames, mask_frames, device=args.device)
        result["ewarp_mask_region"] = ewarp
        result["temporal_diff_delta_vs_gt"] = ewarp
    return result


def numeric_columns(rows: Sequence[Dict[str, object]]) -> List[str]:
    cols = set()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cols.add(key)
    return sorted(cols)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in ok_rows:
        grouped[str(row.get("model_label", "model"))].append(row)

    num_cols = [c for c in numeric_columns(ok_rows) if c != "num_frames"]
    summary = []
    for model_label, group in sorted(grouped.items()):
        item: Dict[str, object] = {"model_label": model_label, "rows": len(group)}
        for col in num_cols:
            vals = [row.get(col) for row in group]
            item[f"{col}_mean"] = finite_mean(vals)
            item[f"{col}_median"] = finite_median(vals)
        summary.append(item)
    return summary


def markdown_table(rows: Sequence[Dict[str, object]], cols: Sequence[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        values = []
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                values.append("nan" if math.isnan(val) else f"{val:.6g}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_summary_md(path: Path, args: argparse.Namespace, per_sample: Sequence[Dict[str, object]], summary: Sequence[Dict[str, object]]) -> None:
    ok_count = sum(1 for row in per_sample if row.get("status") == "ok")
    skipped = [row for row in per_sample if row.get("status") != "ok"]
    cols = [
        "model_label",
        "rows",
        "mask_region_psnr_mean",
        "mask_region_ssim_mean",
        "boundary_psnr_mean",
        "boundary_ssim_mean",
        "whole_video_psnr_mean",
        "whole_video_ssim_mean",
        "outside_region_diff_mean_mean",
        "outside_region_diff_max_mean",
        "temporal_diff_delta_vs_gt_mean",
    ]
    cols = [col for col in cols if any(col in row for row in summary)]
    text = [
        "# Inpainting Metric Summary",
        "",
        f"pair_manifest: `{args.pair_manifest}`",
        f"metric_backend: `inference/metrics.py`",
        f"rows_ok: {ok_count}",
        f"rows_skipped: {len(skipped)}",
        "",
        "PSNR/SSIM/LPIPS/Ewarp are delegated to the existing project metric backend.",
        "The wrapper only handles manifest pairing, mask/boundary crops, and aggregation.",
        "",
        "## Summary",
        "",
        markdown_table(summary, cols),
    ]
    if skipped:
        text.extend(["", "## Skipped Rows", "", markdown_table(skipped[:50], ["sample_id", "model_label", "issue"])])
    path.write_text("\n".join(text), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.device is None:
        try:
            import torch

            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"

    manifest_path = Path(args.pair_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(manifest_path)
    per_sample = [evaluate_row(row, args, manifest_path.parent) for row in rows]
    summary = summarize(per_sample)

    write_csv(metrics_dir / "per_sample_metrics.csv", per_sample)
    write_csv(metrics_dir / "summary.csv", summary)
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_md(metrics_dir / "summary.md", args, per_sample, summary)

    manifest = {
        "metric_backend": "inference/metrics.py",
        "adapter": "tools/run_inpainting_metric_eval.py",
        "pair_manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "num_rows": len(rows),
        "num_ok": sum(1 for row in per_sample if row.get("status") == "ok"),
        "compute_lpips": bool(args.compute_lpips),
        "compute_ewarp": bool(args.compute_ewarp),
    }
    (output_dir / "metric_adapter_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[metric-eval] backend=inference/metrics.py")
    print(f"[metric-eval] per_sample={metrics_dir / 'per_sample_metrics.csv'}")
    print(f"[metric-eval] summary={metrics_dir / 'summary.csv'}")
    print(f"[metric-eval] report={metrics_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
