#!/usr/bin/env python3
"""Postprocess saved Pair001 raw frames for outer-ring diagnostics.

This is an Exp23-only companion to ``eval_pair001_raw_outer_diagnostics.py``.
It avoids rerunning inference when the saved raw frames already exist, and it
builds the outer rings directly from DAVIS masks with an explicit threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import metrics as metric_backend  # noqa: E402
from inference.run_BR import (  # noqa: E402
    load_gray_masks,
    load_rgb_frames,
    normalize_length,
    parse_input_size,
    resize_frames,
    resize_masks,
)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: Iterable[float]) -> float:
    parsed = [float(v) for v in values]
    vals = [v for v in parsed if math.isfinite(v)]
    if vals:
        return float(np.mean(vals))
    if any(math.isinf(v) and v > 0 for v in parsed):
        return float("inf")
    if any(math.isinf(v) and v < 0 for v in parsed):
        return float("-inf")
    return float("nan")


def list_video_names(root: Path) -> List[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def read_saved_frames(root: Path, n: int) -> List[np.ndarray]:
    paths = sorted(root.glob("*.png"))[:n]
    frames: List[np.ndarray] = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return frames


def ring_masks(mask: np.ndarray) -> Dict[str, np.ndarray]:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    hole = (arr > 127).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dil1 = cv2.dilate(hole, kernel, iterations=1)
    dil2 = cv2.dilate(hole, kernel, iterations=2)
    return {
        "mask_core": hole,
        "outer1": ((dil1 > 0) & (hole == 0)).astype(np.uint8),
        "outer2_cumulative": ((dil2 > 0) & (hole == 0)).astype(np.uint8),
        "outer2_band": ((dil2 > 0) & (dil1 == 0)).astype(np.uint8),
    }


def masked_psnr(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(bool)
    if not np.any(m):
        return float("nan")
    diff = gt.astype(np.float64) - pred.astype(np.float64)
    mse = float(np.mean(diff[m] ** 2))
    return float("inf") if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def bbox(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def region_composite(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = gt.copy()
    m = mask.astype(bool)
    out[m] = pred[m]
    return out


def region_ssim(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    box = bbox(mask)
    if box is None:
        return float("nan")
    x0, y0, x1, y1 = box
    if x1 - x0 < 3 or y1 - y0 < 3:
        return float("nan")
    comp = region_composite(gt, pred, mask)
    return float(metric_backend.compute_ssim(gt[y0:y1, x0:x1], comp[y0:y1, x0:x1]))


def region_lpips(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray, device: str) -> float:
    if not np.any(mask > 0):
        return float("nan")
    comp = region_composite(gt, pred, mask)
    return float(metric_backend.LPIPSMetric.compute(gt, comp, device=device))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--saved-root", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--mask-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--input-size", default="432x240")
    parser.add_argument("--video-length", type=int, default=24)
    parser.add_argument("--limit-videos", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_w, target_h = parse_input_size(args.input_size)
    names = list_video_names(args.saved_root)
    if args.limit_videos > 0:
        names = names[: args.limit_videos]
    metric_backend.LPIPSMetric.get_instance(args.device)

    rows: List[Dict[str, object]] = []
    for idx, name in enumerate(names, 1):
        print(f"[raw-outer-post] {args.label} [{idx}/{len(names)}] {name}", flush=True)
        raw_frames = read_saved_frames(args.saved_root / name / "raw_frames", args.video_length)
        hard_frames = read_saved_frames(args.saved_root / name / "hard_comp_frames", args.video_length)
        gt_frames = resize_frames(load_rgb_frames(args.gt_root / name, args.video_length), target_w, target_h)
        masks = resize_masks(load_gray_masks(args.mask_root / name, args.video_length), target_w, target_h)
        n = min(len(raw_frames), len(hard_frames), len(gt_frames), len(masks))
        masks = normalize_length(masks[:n], n)
        row: Dict[str, object] = {"label": args.label, "video": name, "frames": n}
        for region_name in ["mask_core", "outer1", "outer2_cumulative", "outer2_band"]:
            raw_psnr: List[float] = []
            raw_ssim: List[float] = []
            raw_lpips: List[float] = []
            hard_psnr: List[float] = []
            hard_ssim: List[float] = []
            hard_lpips: List[float] = []
            area_ratio: List[float] = []
            for gt, raw, hard, mask in zip(gt_frames[:n], raw_frames[:n], hard_frames[:n], masks):
                region = ring_masks(mask)[region_name]
                area_ratio.append(float(region.mean()))
                raw_psnr.append(masked_psnr(gt, raw, region))
                raw_ssim.append(region_ssim(gt, raw, region))
                raw_lpips.append(region_lpips(gt, raw, region, args.device))
                hard_psnr.append(masked_psnr(gt, hard, region))
                hard_ssim.append(region_ssim(gt, hard, region))
                hard_lpips.append(region_lpips(gt, hard, region, args.device))
            row[f"{region_name}_area_ratio"] = finite_mean(area_ratio)
            row[f"raw_{region_name}_psnr"] = finite_mean(raw_psnr)
            row[f"raw_{region_name}_ssim"] = finite_mean(raw_ssim)
            row[f"raw_{region_name}_lpips"] = finite_mean(raw_lpips)
            row[f"hard_{region_name}_psnr"] = finite_mean(hard_psnr)
            row[f"hard_{region_name}_ssim"] = finite_mean(hard_ssim)
            row[f"hard_{region_name}_lpips"] = finite_mean(hard_lpips)
        rows.append(row)

    write_csv(args.saved_root / "raw_outer_per_video_postprocessed.csv", rows)
    summary: Dict[str, object] = {"label": args.label, "rows": len(rows)}
    for key in rows[0]:
        if key in {"label", "video", "frames"}:
            continue
        summary[f"{key}_mean"] = finite_mean(float(row[key]) for row in rows)
    write_csv(args.saved_root / "raw_outer_summary_postprocessed.csv", [summary])
    (args.saved_root / "raw_outer_summary_postprocessed.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
