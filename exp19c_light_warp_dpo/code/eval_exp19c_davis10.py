#!/usr/bin/env python3
"""Evaluate Exp19c lambda sweep checkpoints on DAVIS10."""

from __future__ import annotations

import argparse
import csv
import gc
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP19_CODE = PROJECT_ROOT / "exp19_boundary_gated_flow_adapter_dpo" / "code"
R0_CODE = PROJECT_ROOT / "exp19r0_flow_adapter_calibration" / "code"
for path in (str(PROJECT_ROOT), str(EXP19_CODE), str(R0_CODE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from inference.run_BR import (  # noqa: E402
    DiffuEraser,
    Propainter,
    composite_with_gt,
    parse_input_size,
    save_mp4,
)
from infer_exp19_flow_adapter_davis import (  # noqa: E402
    PREFERRED_DAVIS10,
    label_frame,
    mask_overlay,
    prepare_video_inputs,
    select_videos,
)
from run_exp19r0_sweep import (  # noqa: E402
    CalibratedExp19Runtime,
    init_optional,
    metrics_for,
    set_all_seeds,
)
from exp19_flow_context_provider import (  # noqa: E402
    context_sequence,
    ensure_davis_flow_cache,
    flow_condition_from_cache,
)
from propainter.inference import Propainter as FlowPropainter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--mask_root", required=True)
    parser.add_argument("--gt_root", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--flow_cache_root", required=True)
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--sft_weights", required=True)
    parser.add_argument("--exp11_weights", required=True)
    parser.add_argument("--exp19b_adapter", required=True)
    parser.add_argument("--lambda000_adapter", required=True)
    parser.add_argument("--lambda005_adapter", required=True)
    parser.add_argument("--lambda010_adapter", required=True)
    parser.add_argument("--lambda020_adapter", required=True)
    parser.add_argument("--propainter_model_dir", required=True)
    parser.add_argument("--pcm_weights_path", required=True)
    parser.add_argument("--input_size", default="432x240")
    parser.add_argument("--video_length", type=int, default=24)
    parser.add_argument("--nframes", type=int, default=22)
    parser.add_argument("--num_inference_steps", type=int, default=6)
    parser.add_argument("--limit_videos", type=int, default=10)
    parser.add_argument("--selected_videos", default=",".join(PREFERRED_DAVIS10))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mask_dilation_iter", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--compute_lpips", action="store_true")
    parser.add_argument("--compute_ewarp", action="store_true")
    parser.add_argument("--raft_model_path", default="")
    parser.add_argument("--residual_scale", type=float, default=0.5)
    parser.add_argument("--confidence_exponent", type=float, default=2.0)
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    labels = []
    for row in rows:
        if row["label"] not in labels:
            labels.append(row["label"])
    out = []
    metric_keys = ["psnr", "ssim", "strict_mask_psnr", "boundary_psnr", "lpips", "ewarp"]
    for label in labels:
        subset = [row for row in rows if row["label"] == label]
        summary = {"label": label, "videos": len(subset)}
        for key in metric_keys:
            vals = [float(row[key]) for row in subset if key in row and row[key] == row[key]]
            summary[key] = float(np.mean(vals)) if vals else float("nan")
        out.append(summary)
    return out


def save_rgb_frames(frames: Sequence[np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        cv2.imwrite(str(out_dir / f"{idx:05d}.png"), cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))


def save_side_by_side(video: str, gt, masks, outputs: dict[str, Sequence[np.ndarray]], out_root: Path) -> None:
    labels = ["GT", "mask", "Exp11", "Exp19b", "lambda000", "best_warp"]
    best_key = outputs.get("__best_key__", "lambda005")
    keys = ["exp11", "exp19b", "lambda000", best_key]
    frames = []
    count = min(len(gt), len(masks), *(len(outputs[k]) for k in keys))
    for idx in range(count):
        cols = [
            label_frame(gt[idx], labels[0]),
            label_frame(mask_overlay(gt[idx], masks[idx]), labels[1]),
            label_frame(outputs["exp11"][idx], labels[2]),
            label_frame(outputs["exp19b"][idx], labels[3]),
            label_frame(outputs["lambda000"][idx], labels[4]),
            label_frame(outputs[best_key][idx], f"best {best_key}"),
        ]
        frames.append(np.concatenate(cols, axis=1))
    side = out_root / "side_by_side"
    side.mkdir(parents=True, exist_ok=True)
    save_mp4(frames, side / f"{video}.mp4")
    sheet = np.concatenate([frames[int(i)] for i in np.linspace(0, len(frames) - 1, min(4, len(frames))).astype(int)], axis=0)
    contact = out_root / "contact_sheets"
    contact.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(contact / f"{video}.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def run_standard(args, device: str, label: str, weights: str, names, prior_cache, optional):
    model = DiffuEraser(
        device,
        args.base_model_path,
        args.vae_path,
        weights,
        pcm_weights_path=args.pcm_weights_path,
        use_pcm=False,
        num_inference_steps_override=args.num_inference_steps,
    )
    target_w, target_h = parse_input_size(args.input_size)
    rows = []
    outputs = {}
    for name in names:
        tmp = Path(tempfile.mkdtemp(prefix=f"exp19c_eval_{label}_{name}_"))
        try:
            _in, gt, masks, frame_dir, mask_dir = prepare_video_inputs(args, name, target_w, target_h, tmp)
            set_all_seeds(args.seed)
            (Path(args.save_path) / label / "videos").mkdir(parents=True, exist_ok=True)
            pred = model.forward(
                validation_image=str(frame_dir),
                validation_mask=str(mask_dir),
                priori="__unused__",
                output_path=str(Path(args.save_path) / label / "videos" / f"{name}.mp4"),
                max_img_size=max(target_w, target_h) + 100,
                video_length=args.video_length,
                mask_dilation_iter=args.mask_dilation_iter,
                nframes=args.nframes,
                seed=args.seed,
                blended=False,
                priori_frames=prior_cache[name],
                return_frames=True,
            )
            comp, masks01 = composite_with_gt(pred, gt, masks, mask_inverse=False)
            outputs[name] = comp
            save_rgb_frames(comp, Path(args.save_path) / label / "comp_frames" / name)
            rows.append(metrics_for(name, label, gt, comp, masks01, optional, args, device))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, outputs


def run_adapter(args, device: str, label: str, adapter_path: str, names, prior_cache, flow_cache, optional):
    runtime = CalibratedExp19Runtime(
        device=device,
        base_model_path=args.base_model_path,
        vae_path=args.vae_path,
        exp11_weights=args.exp11_weights,
        exp19_adapter=adapter_path,
        pcm_weights_path=args.pcm_weights_path,
        num_inference_steps=args.num_inference_steps,
        residual_scale=args.residual_scale,
        confidence_exponent=args.confidence_exponent,
    )
    target_w, target_h = parse_input_size(args.input_size)
    seq = context_sequence(args.video_length, args.nframes, args.num_inference_steps)
    rows = []
    outputs = {}
    for name in names:
        tmp = Path(tempfile.mkdtemp(prefix=f"exp19c_eval_{label}_{name}_"))
        try:
            _in, gt, masks, frame_dir, mask_dir = prepare_video_inputs(args, name, target_w, target_h, tmp)
            cond, stats = flow_condition_from_cache(
                flow_cache[name].forward_flow_path,
                flow_cache[name].backward_flow_path,
                flow_cache[name].confidence_path,
                masks,
                (target_w, target_h),
                args.video_length,
            )
            (Path(args.save_path) / label / "videos").mkdir(parents=True, exist_ok=True)
            pred = runtime.forward(
                frame_dir=frame_dir,
                mask_dir=mask_dir,
                prior_frames=prior_cache[name],
                output_path=Path(args.save_path) / label / "videos" / f"{name}.mp4",
                flow_condition=cond,
                seq=seq,
                args=args,
                adapter_enabled=True,
            )
            comp, masks01 = composite_with_gt(pred, gt, masks, mask_inverse=False)
            outputs[name] = comp
            save_rgb_frames(comp, Path(args.save_path) / label / "comp_frames" / name)
            row = metrics_for(name, label, gt, comp, masks01, optional, args, device)
            row.update({f"flow_{k}": v for k, v in stats.items()})
            rows.append(row)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    del runtime
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, outputs


def main() -> int:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.save_path)
    out_root.mkdir(parents=True, exist_ok=True)
    target_w, target_h = parse_input_size(args.input_size)
    names = select_videos(Path(args.video_root), args.selected_videos, args.limit_videos)
    optional = init_optional(args, device)
    propainter = Propainter(args.propainter_model_dir, device)
    flow_propainter = FlowPropainter(args.propainter_model_dir, torch.device(device))
    prior_cache = {}
    flow_cache = {}
    for name in names:
        tmp = Path(tempfile.mkdtemp(prefix=f"exp19c_cache_{name}_"))
        try:
            in_frames, _gt, masks, frame_dir, mask_dir = prepare_video_inputs(args, name, target_w, target_h, tmp)
            (out_root / "propainter_prior").mkdir(parents=True, exist_ok=True)
            prior_cache[name] = propainter.forward(
                video=str(frame_dir),
                mask=str(mask_dir),
                output_path=str(out_root / "propainter_prior" / f"{name}.mp4"),
                resize_ratio=1.0,
                video_length=args.video_length,
                height=-1,
                width=-1,
                mask_dilation=args.mask_dilation_iter,
                ref_stride=3,
                neighbor_length=25,
                subvideo_length=80,
                raft_iter=20,
                save_fps=24,
                save_frames=False,
                fp16=True,
                return_frames=True,
            )
            flow_cache[name] = ensure_davis_flow_cache(
                name,
                in_frames,
                masks,
                args.flow_cache_root,
                flow_propainter,
                (target_w, target_h),
                args.video_length,
                raft_iter=20,
                fp16=True,
                save_visuals=True,
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    rows = []
    outputs_by_label = {}
    for label, weights in [("sft", args.sft_weights), ("exp11", args.exp11_weights)]:
        model_rows, outputs = run_standard(args, device, label, weights, names, prior_cache, optional)
        rows.extend(model_rows)
        outputs_by_label[label] = outputs
    adapters = {
        "exp19b": args.exp19b_adapter,
        "lambda000": args.lambda000_adapter,
        "lambda005": args.lambda005_adapter,
        "lambda010": args.lambda010_adapter,
        "lambda020": args.lambda020_adapter,
    }
    for label, adapter_path in adapters.items():
        model_rows, outputs = run_adapter(args, device, label, adapter_path, names, prior_cache, flow_cache, optional)
        rows.extend(model_rows)
        outputs_by_label[label] = outputs

    metrics_dir = out_root / "metrics"
    write_csv(metrics_dir / "per_video.csv", rows)
    summaries = summary_rows(rows)
    write_csv(metrics_dir / "summary.csv", summaries)
    best = min(
        [s for s in summaries if str(s["label"]).startswith("lambda") and s["label"] != "lambda000"],
        key=lambda s: float(s.get("ewarp", float("inf"))),
    )
    best_key = str(best["label"])
    for name in names:
        tmp = Path(tempfile.mkdtemp(prefix="exp19c_vis_"))
        try:
            _in, gt, masks, _frame_dir, _mask_dir = prepare_video_inputs(args, name, target_w, target_h, tmp)
            outputs = {
                "exp11": outputs_by_label["exp11"][name],
                "exp19b": outputs_by_label["exp19b"][name],
                "lambda000": outputs_by_label["lambda000"][name],
                best_key: outputs_by_label[best_key][name],
                "__best_key__": best_key,
            }
            save_side_by_side(name, gt, masks, outputs, out_root)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    lines = [
        "# Exp19c DAVIS10 Metric Summary",
        "",
        f"- best_warp_variant_by_ewarp: `{best_key}`",
        "",
        "| label | PSNR | SSIM | LPIPS | Ewarp | strict mask PSNR | boundary PSNR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['label']} | {row.get('psnr', float('nan')):.6f} | {row.get('ssim', float('nan')):.6f} | {row.get('lpips', float('nan')):.8f} | {row.get('ewarp', float('nan')):.6f} | {row.get('strict_mask_psnr', float('nan')):.6f} | {row.get('boundary_psnr', float('nan')):.6f} |"
        )
    (Path("reports") / "exp19c_davis10_metric_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(Path("reports") / "exp19c_davis10_metric_summary.csv", summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
