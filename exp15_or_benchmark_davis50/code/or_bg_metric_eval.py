#!/usr/bin/env python3
"""Evaluate OR outputs with background-region metrics, without compositing."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(path: Path) -> List[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def read_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError(f"failed to read image: {path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"failed to read mask: {path}")
    return (arr > 0).astype(np.uint8)


def finite_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def psnr_from_mse(mse: float) -> float:
    return float("inf") if mse <= 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def psnr_bg(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    bg = mask <= 0
    if not np.any(bg):
        return float("nan")
    diff = gt.astype(np.float64) - pred.astype(np.float64)
    return psnr_from_mse(float(np.mean(diff[bg] ** 2)))


def ssim_bg_ignore_mask(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    """SSIM with foreground ignored by setting it to black in both images.

    This is not a strict arbitrary-pixel SSIM. It is a background-preservation
    proxy and is reported as `SSIM_bg_ignore_mask`.
    """
    from skimage.metrics import structural_similarity

    gt2 = gt.copy()
    pred2 = pred.copy()
    fg = mask > 0
    gt2[fg] = 0
    pred2[fg] = 0
    min_dim = min(gt2.shape[:2])
    win_size = min(65, min_dim if min_dim % 2 == 1 else min_dim - 1)
    win_size = max(3, win_size)
    return float(structural_similarity(gt2, pred2, data_range=255, channel_axis=-1, win_size=win_size))


def temporal_bg_score(gt_frames: Sequence[np.ndarray], pred_frames: Sequence[np.ndarray], masks: Sequence[np.ndarray]) -> float:
    """Simple background temporal-difference consistency; higher is better."""
    vals = []
    for idx in range(1, min(len(gt_frames), len(pred_frames), len(masks))):
        bg = (masks[idx] <= 0) & (masks[idx - 1] <= 0)
        if not np.any(bg):
            continue
        gt_delta = gt_frames[idx].astype(np.float32) - gt_frames[idx - 1].astype(np.float32)
        pr_delta = pred_frames[idx].astype(np.float32) - pred_frames[idx - 1].astype(np.float32)
        mse = float(np.mean((gt_delta[bg] - pr_delta[bg]) ** 2))
        vals.append(psnr_from_mse(mse))
    return finite_mean(vals)


def align_to_pred(gt: np.ndarray, mask: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = pred.shape[:2]
    if gt.shape[:2] != (h, w):
        gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_AREA)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)
    return gt, mask


def evaluate_video(frame_dir: Path, mask_dir: Path, pred_dir: Path, max_frames: int) -> Dict[str, float]:
    frame_files = image_files(frame_dir)
    mask_files = image_files(mask_dir)
    pred_files = image_files(pred_dir)
    n = min(len(frame_files), len(mask_files), len(pred_files))
    if max_frames > 0:
        n = min(n, max_frames)
    if n == 0:
        raise RuntimeError(f"zero aligned frames: {frame_dir}, {mask_dir}, {pred_dir}")
    gt_frames, pred_frames, masks = [], [], []
    psnr_vals, ssim_vals = [], []
    for fpath, mpath, ppath in zip(frame_files[:n], mask_files[:n], pred_files[:n]):
        gt = read_rgb(fpath)
        mask = read_mask(mpath)
        pred = read_rgb(ppath)
        gt, mask = align_to_pred(gt, mask, pred)
        gt_frames.append(gt)
        pred_frames.append(pred)
        masks.append(mask)
        psnr_vals.append(psnr_bg(gt, pred, mask))
        ssim_vals.append(ssim_bg_ignore_mask(gt, pred, mask))
    return {
        "num_frames": n,
        "PSNR_bg": finite_mean(psnr_vals),
        "SSIM_bg": finite_mean(ssim_vals),
        "SSIM_bg_ignore_mask": finite_mean(ssim_vals),
        "TC_bg_pixel_proxy": temporal_bg_score(gt_frames, pred_frames, masks),
    }


def evaluate_task(task: Tuple[str, dict, str, int]) -> dict:
    method, row, output_root_str, max_frames = task
    output_root = Path(output_root_str)
    pred_dir = output_root / method / "raw_frames" / row["video_name"]
    out = {
        "method": method,
        "video_name": row["video_name"],
        "status": "ok",
        "issue": "",
        "num_frames": 0,
        "PSNR_bg": float("nan"),
        "SSIM_bg": float("nan"),
        "SSIM_bg_ignore_mask": float("nan"),
        "TC_bg_pixel_proxy": float("nan"),
    }
    try:
        if not pred_dir.is_dir():
            raise FileNotFoundError(f"missing prediction dir: {pred_dir}")
        metrics = evaluate_video(Path(row["frame_dir"]), Path(row["mask_dir"]), pred_dir, max_frames)
        out.update(metrics)
    except Exception as exc:  # noqa: BLE001
        out["status"] = "failed"
        out["issue"] = str(exc)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--methods", required=True, help="Comma separated method dirs under output_root.")
    parser.add_argument("--metrics_dir", required=True)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 1) // 2)))
    args = parser.parse_args()

    output_root = Path(args.output_root)
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    with Path(args.manifest).open("r", encoding="utf-8", newline="") as handle:
        videos = list(csv.DictReader(handle))

    tasks = [(method, row, str(output_root), args.max_frames) for method in methods for row in videos]
    per_rows = []
    print(f"[metrics] evaluating {len(tasks)} method/video pairs with workers={args.workers}", flush=True)
    if args.workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            per_rows.append(evaluate_task(task))
            if idx % 25 == 0 or idx == len(tasks):
                print(f"[metrics] completed {idx}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(evaluate_task, task): task for task in tasks}
            for idx, future in enumerate(as_completed(futures), start=1):
                per_rows.append(future.result())
                if idx % 25 == 0 or idx == len(tasks):
                    print(f"[metrics] completed {idx}/{len(tasks)}", flush=True)

    order = {(method, row["video_name"]): i for i, (method, row, _root, _max_frames) in enumerate(tasks)}
    per_rows.sort(key=lambda item: order[(item["method"], item["video_name"])])

    per_csv = metrics_dir / "per_video.csv"
    with per_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "video_name",
                "status",
                "issue",
                "num_frames",
                "PSNR_bg",
                "SSIM_bg",
                "SSIM_bg_ignore_mask",
                "TC_bg_pixel_proxy",
            ],
        )
        writer.writeheader()
        writer.writerows(per_rows)

    by_method: Dict[str, List[dict]] = defaultdict(list)
    for row in per_rows:
        by_method[row["method"]].append(row)
    summary_rows = []
    for method in methods:
        rows = by_method[method]
        ok = [r for r in rows if r["status"] == "ok"]
        summary_rows.append(
            {
                "method": method,
                "status": "ok" if len(ok) == len(rows) else ("partial" if ok else "failed_or_blocked"),
                "num_videos": len(rows),
                "num_success": len(ok),
                "num_failed": len(rows) - len(ok),
                "PSNR_bg": finite_mean(float(r["PSNR_bg"]) for r in ok),
                "SSIM_bg": finite_mean(float(r["SSIM_bg"]) for r in ok),
                "SSIM_bg_ignore_mask": finite_mean(float(r["SSIM_bg_ignore_mask"]) for r in ok),
                "TC_bg_pixel_proxy": finite_mean(float(r["TC_bg_pixel_proxy"]) for r in ok),
                "LPIPS_if_available": "not_available",
                "VFID_if_available": "not_available",
                "metric_protocol": "minimax_compatible_or_davis_subset_no_comp_bg_proxy",
                "notes": "" if ok else "no successful predictions",
            }
        )

    summary_csv = metrics_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "status",
                "num_videos",
                "num_success",
                "num_failed",
                "PSNR_bg",
                "SSIM_bg",
                "SSIM_bg_ignore_mask",
                "TC_bg_pixel_proxy",
                "LPIPS_if_available",
                "VFID_if_available",
                "metric_protocol",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    (metrics_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    md = [
        "# Exp15 DAVIS50 OR Quantitative Summary",
        "",
        "| Method | Status | Success | PSNR_bg | SSIM_bg | TC_bg_pixel_proxy | Notes |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        md.append(
            f"| {row['method']} | {row['status']} | {row['num_success']}/{row['num_videos']} | "
            f"{float(row['PSNR_bg']):.4f} | {float(row['SSIM_bg']):.4f} | {float(row['TC_bg_pixel_proxy']):.4f} | {row['notes']} |"
        )
    md.append("")
    md.append("Protocol: no comp; metrics are computed on raw method outputs. PSNR_bg is strict mask-outside pixels. SSIM_bg is a background-preservation proxy implemented as SSIM_bg_ignore_mask. TC_bg_pixel_proxy is not the MiniMax paper CLIP-feature TC.")
    (metrics_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[metrics] wrote {summary_csv}")


if __name__ == "__main__":
    main()
