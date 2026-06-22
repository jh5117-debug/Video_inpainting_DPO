#!/usr/bin/env python3
"""Exp23-only raw outer-ring diagnostics for Pair001 main endpoints.

The canonical DAVIS50 evaluator computes the official protocol on hard-composed
frames.  This diagnostic reruns only selected endpoints to preserve raw
DiffuEraser frames and measure whether a candidate damages the outer background
before hard composition restores it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from time import time
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.run_BR import (  # noqa: E402
    DiffuEraser,
    Propainter,
    composite_with_gt,
    ensure_same_hw,
    list_video_names,
    load_gray_masks,
    load_rgb_frames,
    normalize_length,
    parse_input_size,
    resize_frames,
    resize_masks,
    save_frames_to_dir,
    save_masks_to_dir,
)
from inference import metrics as metric_backend  # noqa: E402


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


def save_frames(frames: Sequence[np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        cv2.imwrite(str(out_dir / f"{idx:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


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


def ring_masks(mask: np.ndarray) -> Dict[str, np.ndarray]:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dil1 = cv2.dilate(binary, kernel, iterations=1)
    dil2 = cv2.dilate(binary, kernel, iterations=2)
    return {
        "mask_core": binary,
        "outer1": ((dil1 - binary) > 0).astype(np.uint8),
        "outer2_cumulative": ((dil2 - binary) > 0).astype(np.uint8),
        "outer2_band": ((dil2 - dil1) > 0).astype(np.uint8),
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
    return metric_backend.compute_ssim(gt[y0:y1, x0:x1], comp[y0:y1, x0:x1])


def region_lpips(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray, device: str) -> float:
    if not np.any(mask > 0):
        return float("nan")
    comp = region_composite(gt, pred, mask)
    return float(metric_backend.LPIPSMetric.compute(gt, comp, device=device))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--diffueraser-path", required=True)
    parser.add_argument("--save-path", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--mask-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--propainter-model-dir", required=True)
    parser.add_argument("--pcm-weights-path", required=True)
    parser.add_argument("--input-size", default="432x240")
    parser.add_argument("--video-length", type=int, default=24)
    parser.add_argument("--num-inference-steps", type=int, default=6)
    parser.add_argument("--limit-videos", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_w, target_h = parse_input_size(args.input_size)
    args.save_path.mkdir(parents=True, exist_ok=True)
    names = list_video_names(args.video_root)
    if args.limit_videos > 0:
        names = names[: args.limit_videos]

    metric_backend.LPIPSMetric.get_instance(args.device)
    propainter = Propainter(args.propainter_model_dir, args.device)
    diffueraser = DiffuEraser(
        args.device,
        args.base_model_path,
        args.vae_path,
        args.diffueraser_path,
        pcm_weights_path=args.pcm_weights_path,
        use_pcm=False,
        num_inference_steps_override=args.num_inference_steps,
    )
    rows: List[Dict[str, object]] = []
    for index, name in enumerate(names, 1):
        print(f"[raw-outer] {args.label} [{index}/{len(names)}] {name}", flush=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f"exp23_raw_outer_{name}_"))
        try:
            in_frames = resize_frames(load_rgb_frames(args.video_root / name, args.video_length), target_w, target_h)
            gt_frames = resize_frames(load_rgb_frames(args.gt_root / name, args.video_length), target_w, target_h)
            masks = resize_masks(load_gray_masks(args.mask_root / name, args.video_length), target_w, target_h)
            n = min(len(in_frames), len(gt_frames), len(masks))
            in_frames, gt_frames, masks = ensure_same_hw(in_frames[:n], gt_frames[:n], normalize_length(masks[:n], n))
            model_vdir = save_frames_to_dir(in_frames, temp_dir / "frames")
            model_mdir = save_masks_to_dir(masks, temp_dir / "masks")
            start = time()
            prior_frames = propainter.forward(
                video=str(model_vdir),
                mask=str(model_mdir),
                output_path=str(args.save_path / name / "propainter.mp4"),
                resize_ratio=1.0,
                video_length=args.video_length,
                height=-1,
                width=-1,
                mask_dilation=0,
                ref_stride=3,
                neighbor_length=25,
                subvideo_length=80,
                raft_iter=20,
                save_fps=24,
                save_frames=False,
                fp16=True,
                return_frames=True,
            )
            print(f"  propainter {time() - start:.1f}s", flush=True)
            start = time()
            pred_frames = diffueraser.forward(
                validation_image=str(model_vdir),
                validation_mask=str(model_mdir),
                priori="__unused__",
                output_path=str(args.save_path / name / "raw.mp4"),
                max_img_size=max(in_frames[0].shape[1], in_frames[0].shape[0]) + 100,
                video_length=args.video_length,
                mask_dilation_iter=0,
                nframes=22,
                seed=None,
                blended=False,
                priori_frames=prior_frames,
                return_frames=True,
            )
            print(f"  diffueraser {time() - start:.1f}s", flush=True)
            comp_frames, masks01 = composite_with_gt(pred_frames, gt_frames, masks, mask_inverse=False)
            save_frames(pred_frames, args.save_path / name / "raw_frames")
            save_frames(comp_frames, args.save_path / name / "hard_comp_frames")

            row: Dict[str, object] = {"label": args.label, "video": name, "frames": len(comp_frames)}
            for region_name in ["mask_core", "outer1", "outer2_cumulative", "outer2_band"]:
                raw_psnr, raw_ssim, raw_lpips = [], [], []
                hard_psnr, hard_ssim, hard_lpips = [], [], []
                for gt, raw, hard, mask in zip(gt_frames, pred_frames, comp_frames, masks01):
                    region = ring_masks(mask)[region_name]
                    raw_psnr.append(masked_psnr(gt, raw, region))
                    raw_ssim.append(region_ssim(gt, raw, region))
                    raw_lpips.append(region_lpips(gt, raw, region, args.device))
                    hard_psnr.append(masked_psnr(gt, hard, region))
                    hard_ssim.append(region_ssim(gt, hard, region))
                    hard_lpips.append(region_lpips(gt, hard, region, args.device))
                row[f"raw_{region_name}_psnr"] = finite_mean(raw_psnr)
                row[f"raw_{region_name}_ssim"] = finite_mean(raw_ssim)
                row[f"raw_{region_name}_lpips"] = finite_mean(raw_lpips)
                row[f"hard_{region_name}_psnr"] = finite_mean(hard_psnr)
                row[f"hard_{region_name}_ssim"] = finite_mean(hard_ssim)
                row[f"hard_{region_name}_lpips"] = finite_mean(hard_lpips)
            rows.append(row)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(args.save_path / "raw_outer_per_video.csv", rows)
    summary: Dict[str, object] = {"label": args.label, "rows": len(rows)}
    for key in rows[0]:
        if key in {"label", "video", "frames"}:
            continue
        summary[f"{key}_mean"] = finite_mean([float(row[key]) for row in rows])
    write_csv(args.save_path / "raw_outer_summary.csv", [summary])
    (args.save_path / "raw_outer_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
