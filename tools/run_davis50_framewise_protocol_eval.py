#!/usr/bin/env python3
"""Run the fixed DAVIS-50 frame-wise DiffuEraser evaluation protocol.

This tool intentionally computes metrics on in-memory hard-composited frames,
not on mp4 outputs. That is the protocol that reproduces the SFT-48000 DAVIS
raw6 score near/above 32 PSNR.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from time import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.run_BR import (  # noqa: E402
    DiffuEraser,
    Propainter,
    composite_with_gt,
    create_comparison_video_from_frames,
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
    save_mp4,
)
from inference import metrics as metric_backend  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--mask_root", required=True)
    parser.add_argument("--gt_root", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--diffueraser_path", required=True)
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--propainter_model_dir", required=True)
    parser.add_argument("--pcm_weights_path", required=True)
    parser.add_argument("--input_size", default="432x240")
    parser.add_argument("--video_length", type=int, default=24)
    parser.add_argument("--num_inference_steps", type=int, default=6)
    parser.add_argument("--use_pcm", default="false")
    parser.add_argument("--mask_dilation_iter", type=int, default=0)
    parser.add_argument("--ref_stride", type=int, default=3)
    parser.add_argument("--neighbor_length", type=int, default=25)
    parser.add_argument("--subvideo_length", type=int, default=80)
    parser.add_argument("--limit_videos", type=int, default=0)
    parser.add_argument("--save_videos", action="store_true")
    parser.add_argument("--save_comp_frames", action="store_true")
    parser.add_argument("--compute_lpips", action="store_true")
    parser.add_argument("--compute_vfid", action="store_true")
    parser.add_argument("--compute_tc", action="store_true")
    parser.add_argument("--compute_ewarp", action="store_true")
    parser.add_argument("--i3d_model_path", default=str(metric_backend.DEFAULT_I3D_MODEL))
    parser.add_argument("--tc_model_path", default="")
    parser.add_argument("--raft_model_path", default=str(metric_backend.DEFAULT_RAFT_MODEL))
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def str_to_bool(value: object) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"}


def bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_metric(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    box = bbox(mask)
    if box is None:
        return float("nan"), float("nan")
    x0, y0, x1, y1 = box
    if x1 - x0 < 3 or y1 - y0 < 3:
        return float("nan"), float("nan")
    return (
        metric_backend.compute_psnr(gt[y0:y1, x0:x1], pred[y0:y1, x0:x1]),
        metric_backend.compute_ssim(gt[y0:y1, x0:x1], pred[y0:y1, x0:x1]),
    )


def finite_values(values: Iterable[float]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            val = float(value)
        except Exception:
            continue
        if math.isfinite(val):
            out.append(val)
    return out


def finite_mean(values: Iterable[float]) -> float:
    vals = finite_values(values)
    return float(np.mean(vals)) if vals else float("nan")


def finite_median(values: Iterable[float]) -> float:
    vals = finite_values(values)
    return float(np.median(vals)) if vals else float("nan")


def init_optional_metrics(args: argparse.Namespace, device: str) -> Dict[str, object]:
    optional: Dict[str, object] = {}
    if args.compute_vfid:
        i3d_path = Path(args.i3d_model_path)
        if not i3d_path.exists():
            raise FileNotFoundError(f"--compute_vfid requires --i3d_model_path: {i3d_path}")
        optional["i3d"] = metric_backend.init_i3d_model(str(i3d_path), device)
        optional["ori_i3d_activations"] = []
        optional["comp_i3d_activations"] = []
    if args.compute_tc:
        model_path = args.tc_model_path or None
        if model_path and not Path(model_path).exists():
            raise FileNotFoundError(f"--compute_tc model path does not exist: {model_path}")
        optional["tc"] = metric_backend.TemporalConsistencyMetric(device=device, model_path=model_path)
    if args.compute_ewarp:
        raft_path = args.raft_model_path or None
        if raft_path and not Path(raft_path).exists():
            raise FileNotFoundError(f"--compute_ewarp requires a valid --raft_model_path: {raft_path}")
        optional["ewarp"] = metric_backend.EwarpMetric(device=device, raft_model_path=raft_path)
    if args.compute_lpips:
        metric_backend.LPIPSMetric.get_instance(device)
    return optional


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(label: str, rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    summary: Dict[str, object] = {"model_label": label, "rows": len(rows)}
    numeric_keys = []
    for row in rows:
        for key, value in row.items():
            if key in {"model_label", "video"}:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if key not in numeric_keys:
                    numeric_keys.append(key)
    for key in numeric_keys:
        values = [row[key] for row in rows if key in row]
        summary[f"{key}_mean"] = finite_mean(values)
        summary[f"{key}_median"] = finite_median(values)
    return summary


def save_frames(frames: Sequence[np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        cv2.imwrite(str(out_dir / f"{idx:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def main() -> int:
    args = parse_args()
    use_pcm = str_to_bool(args.use_pcm)
    target_w, target_h = parse_input_size(args.input_size)
    save_root = Path(args.save_path)
    save_root.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    video_root = Path(args.video_root)
    mask_root = Path(args.mask_root)
    gt_root = Path(args.gt_root)
    names = list_video_names(video_root)
    if args.limit_videos > 0:
        names = names[: args.limit_videos]

    print(
        f"[davis50-framewise] label={args.label} videos={len(names)} "
        f"device={device} steps={args.num_inference_steps} use_pcm={use_pcm}"
    )
    optional = init_optional_metrics(args, device)
    propainter = Propainter(args.propainter_model_dir, device)

    old_disable = logging.root.manager.disable
    logging.disable(logging.WARNING)
    diffueraser = DiffuEraser(
        device,
        args.base_model_path,
        args.vae_path,
        args.diffueraser_path,
        pcm_weights_path=args.pcm_weights_path,
        use_pcm=use_pcm,
        num_inference_steps_override=args.num_inference_steps,
    )
    logging.disable(old_disable)

    rows: List[Dict[str, object]] = []
    for index, name in enumerate(names, 1):
        print(f"[{index}/{len(names)}] {name}", flush=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f"davis50_framewise_{name}_"))
        try:
            in_frames = resize_frames(load_rgb_frames(video_root / name, args.video_length), target_w, target_h)
            gt_frames = resize_frames(load_rgb_frames(gt_root / name, args.video_length), target_w, target_h)
            masks = resize_masks(load_gray_masks(mask_root / name, args.video_length), target_w, target_h)
            n = min(len(in_frames), len(gt_frames), len(masks))
            in_frames = in_frames[:n]
            gt_frames = gt_frames[:n]
            masks = normalize_length(masks[:n], n)
            in_frames, gt_frames, masks = ensure_same_hw(in_frames, gt_frames, masks)

            model_vdir = save_frames_to_dir(in_frames, temp_dir / "frames")
            model_mdir = save_masks_to_dir(masks, temp_dir / "masks")

            start = time()
            prior_frames = propainter.forward(
                video=str(model_vdir),
                mask=str(model_mdir),
                output_path=str(save_root / name / "propainter.mp4"),
                resize_ratio=1.0,
                video_length=args.video_length,
                height=-1,
                width=-1,
                mask_dilation=args.mask_dilation_iter,
                ref_stride=args.ref_stride,
                neighbor_length=args.neighbor_length,
                subvideo_length=args.subvideo_length,
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
                output_path=str(save_root / name / "diffueraser.mp4"),
                max_img_size=max(in_frames[0].shape[1], in_frames[0].shape[0]) + 100,
                video_length=args.video_length,
                mask_dilation_iter=args.mask_dilation_iter,
                nframes=22,
                seed=None,
                blended=False,
                priori_frames=prior_frames,
                return_frames=True,
            )
            print(f"  diffueraser {time() - start:.1f}s", flush=True)

            comp_frames, masks01 = composite_with_gt(pred_frames, gt_frames, masks, mask_inverse=False)

            if args.save_videos:
                out_dir = save_root / name
                out_dir.mkdir(parents=True, exist_ok=True)
                save_mp4(comp_frames, out_dir / "diffueraser_comp.mp4")
                create_comparison_video_from_frames(
                    in_frames,
                    masks,
                    gt_frames,
                    comp_frames,
                    str(out_dir / "comparison_input_gt_current.mp4"),
                    fps=12,
                )
            if args.save_comp_frames:
                save_frames(comp_frames, save_root / name / "diffueraser_comp_frames")

            psnr_vals: List[float] = []
            ssim_vals: List[float] = []
            mask_psnr_vals: List[float] = []
            mask_ssim_vals: List[float] = []
            outside_vals: List[float] = []
            lpips_vals: List[float] = []

            for gt, comp, mask in zip(gt_frames, comp_frames, masks01):
                psnr_vals.append(metric_backend.compute_psnr(gt, comp))
                ssim_vals.append(metric_backend.compute_ssim(gt, comp))
                mask_psnr, mask_ssim = crop_metric(gt, comp, mask)
                mask_psnr_vals.append(mask_psnr)
                mask_ssim_vals.append(mask_ssim)
                outside = mask <= 0
                if outside.any():
                    diff = np.abs(comp.astype(np.float32) - gt.astype(np.float32))
                    outside_vals.append(float(diff[outside].mean()))
                if args.compute_lpips:
                    lpips_vals.append(float(metric_backend.LPIPSMetric.compute(gt, comp, device=device)))

            row: Dict[str, object] = {
                "model_label": args.label,
                "video": name,
                "frames": n,
                "whole_video_psnr": finite_mean(psnr_vals),
                "whole_video_ssim": finite_mean(ssim_vals),
                "mask_region_psnr": finite_mean(mask_psnr_vals),
                "mask_region_ssim": finite_mean(mask_ssim_vals),
                "outside_region_diff_mean": finite_mean(outside_vals),
            }
            if args.compute_lpips:
                row["whole_video_lpips"] = finite_mean(lpips_vals)
            if args.compute_tc:
                row["tc"] = float(optional["tc"].compute(comp_frames))
            if args.compute_ewarp:
                row["ewarp"] = float(optional["ewarp"].compute(comp_frames, masks01=masks01, gt_frames_u8_rgb=gt_frames))
            if args.compute_vfid:
                ori_act, comp_act = metric_backend.calculate_i3d_activations(
                    [Image.fromarray(frame.astype(np.uint8)) for frame in gt_frames],
                    [Image.fromarray(frame.astype(np.uint8)) for frame in comp_frames],
                    optional["i3d"],
                    device,
                )
                optional["ori_i3d_activations"].append(ori_act)
                optional["comp_i3d_activations"].append(comp_act)
            print(f"  metrics {row}", flush=True)
            rows.append(row)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = summarize_rows(args.label, rows)
    if args.compute_vfid:
        summary["vfid"] = float(
            metric_backend.calculate_vfid(
                np.vstack(optional["ori_i3d_activations"]),
                np.vstack(optional["comp_i3d_activations"]),
            )
        )

    metrics_dir = save_root / "metrics"
    write_csv(metrics_dir / "per_video_metrics.csv", rows)
    write_csv(metrics_dir / "summary.csv", [summary])
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# DAVIS-50 Frame-wise Protocol Summary",
        "",
        f"label: `{args.label}`",
        f"rows: {len(rows)}",
        "protocol: raw6, no PCM, mask dilation 0, no Gaussian blur, hard comp, frame-wise metric",
        f"summary: `{metrics_dir / 'summary.csv'}`",
        "",
    ]
    (metrics_dir / "summary.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[davis50-framewise] summary {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
