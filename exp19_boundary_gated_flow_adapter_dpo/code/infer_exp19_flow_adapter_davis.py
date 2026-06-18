#!/usr/bin/env python3
"""Run legal Exp19 flow-adapter DAVIS10 inference and frame-wise metrics."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from time import time
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

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
    save_mp4,
)
from inference import metrics as metric_backend  # noqa: E402
from tools.run_davis50_framewise_protocol_eval import (  # noqa: E402
    boundary_mask,
    crop_metric,
    finite_mean,
    finite_median,
    masked_pixel_psnr,
    outside_diff,
)

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from exp19_flow_context_provider import (  # noqa: E402
    context_sequence,
    ensure_davis_flow_cache,
    flow_condition_from_cache,
)
from exp19_inference_pipeline import Exp19FlowAdapterDiffuEraser  # noqa: E402
from propainter.inference import Propainter as FlowPropainter  # noqa: E402


PREFERRED_DAVIS10 = [
    "boat",
    "rhino",
    "dog-agility",
    "blackswan",
    "lucia",
    "dance-jump",
    "flamingo",
    "soccerball",
    "camel",
    "car-roundabout",
]


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
    parser.add_argument("--exp19_adapter", required=True)
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
    parser.add_argument("--ref_stride", type=int, default=3)
    parser.add_argument("--neighbor_length", type=int, default=25)
    parser.add_argument("--subvideo_length", type=int, default=80)
    parser.add_argument("--compute_lpips", action="store_true")
    parser.add_argument("--compute_vfid", action="store_true")
    parser.add_argument("--compute_tc", action="store_true")
    parser.add_argument("--compute_ewarp", action="store_true")
    parser.add_argument("--i3d_model_path", default=str(metric_backend.DEFAULT_I3D_MODEL))
    parser.add_argument("--tc_model_path", default="")
    parser.add_argument("--raft_model_path", default=str(metric_backend.DEFAULT_RAFT_MODEL))
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--skip_preflight", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
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


def summarize_rows(label: str, rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    summary: Dict[str, object] = {"model_label": label, "rows": len(rows)}
    numeric_keys = []
    for row in rows:
        for key, value in row.items():
            if key in {"model_label", "video"}:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool) and key not in numeric_keys:
                numeric_keys.append(key)
    for key in numeric_keys:
        values = [row[key] for row in rows if key in row]
        summary[f"{key}_mean"] = finite_mean(values)
        summary[f"{key}_median"] = finite_median(values)
    return summary


def select_videos(video_root: Path, selected: str, limit: int) -> list[str]:
    available = list_video_names(video_root)
    chosen = [name.strip() for name in selected.split(",") if name.strip() and name.strip() in available]
    for name in available:
        if name not in chosen:
            chosen.append(name)
        if limit > 0 and len(chosen) >= limit:
            break
    return chosen[:limit] if limit > 0 else chosen


def save_rgb_frames(frames: Sequence[np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        cv2.imwrite(str(out_dir / f"{idx:05d}.png"), cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))


def label_frame(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def mask_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    mask_bool = mask > 0
    overlay[mask_bool] = (0.45 * overlay[mask_bool] + np.array([0, 255, 0]) * 0.55).astype(np.uint8)
    return overlay


def save_side_by_side(video: str, gt: Sequence[np.ndarray], masks: Sequence[np.ndarray], outputs: dict[str, Sequence[np.ndarray]], out_root: Path) -> None:
    frames = []
    labels = ["GT", "mask", "SFT-48000", "Exp11 outer b0.75 S2", "Exp19b"]
    for i in range(min(len(gt), *(len(v) for v in outputs.values()))):
        cols = [
            label_frame(gt[i], labels[0]),
            label_frame(mask_overlay(gt[i], masks[i]), labels[1]),
            label_frame(outputs["sft"][i], labels[2]),
            label_frame(outputs["exp11"][i], labels[3]),
            label_frame(outputs["exp19b"][i], labels[4]),
        ]
        frames.append(np.concatenate(cols, axis=1))
    side_dir = out_root / "side_by_side"
    side_dir.mkdir(parents=True, exist_ok=True)
    save_mp4(frames, side_dir / f"{video}.mp4")
    sheet_dir = out_root / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    picks = np.linspace(0, len(frames) - 1, min(4, len(frames))).astype(int)
    sheet = np.concatenate([frames[int(i)] for i in picks], axis=0)
    cv2.imwrite(str(sheet_dir / f"{video}.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def init_optional(args: argparse.Namespace, device: str):
    optional: dict[str, object] = {}
    if args.compute_lpips:
        metric_backend.LPIPSMetric.get_instance(device)
    if args.compute_tc:
        optional["tc"] = metric_backend.TemporalConsistencyMetric(device=device, model_path=args.tc_model_path or None)
    if args.compute_ewarp:
        optional["ewarp"] = metric_backend.EwarpMetric(device=device, raft_model_path=args.raft_model_path)
    if args.compute_vfid:
        optional["i3d"] = metric_backend.init_i3d_model(args.i3d_model_path, device)
        optional["acts"] = {}
    return optional


def metric_row(label: str, video: str, gt_frames, comp_frames, masks01, optional, args, device: str) -> Dict[str, object]:
    psnr_vals: list[float] = []
    ssim_vals: list[float] = []
    mask_bbox_psnr_vals: list[float] = []
    mask_bbox_ssim_vals: list[float] = []
    strict_mask_pixel_psnr_vals: list[float] = []
    boundary_pixel_psnr_vals: list[float] = []
    outside_mean_vals: list[float] = []
    outside_max_vals: list[float] = []
    lpips_vals: list[float] = []
    for gt, comp, mask in zip(gt_frames, comp_frames, masks01):
        psnr_vals.append(metric_backend.compute_psnr(gt, comp))
        ssim_vals.append(metric_backend.compute_ssim(gt, comp))
        p, s = crop_metric(gt, comp, mask)
        mask_bbox_psnr_vals.append(p)
        mask_bbox_ssim_vals.append(s)
        strict_mask_pixel_psnr_vals.append(masked_pixel_psnr(gt, comp, mask))
        boundary_pixel_psnr_vals.append(masked_pixel_psnr(gt, comp, boundary_mask(mask)))
        outside_mean, outside_max = outside_diff(gt, comp, mask)
        outside_mean_vals.append(outside_mean)
        outside_max_vals.append(outside_max)
        if args.compute_lpips:
            lpips_vals.append(float(metric_backend.LPIPSMetric.compute(gt, comp, device=device)))
    row: Dict[str, object] = {
        "model_label": label,
        "video": video,
        "frames": len(comp_frames),
        "whole_video_psnr": finite_mean(psnr_vals),
        "whole_video_ssim": finite_mean(ssim_vals),
        "mask_region_psnr": finite_mean(mask_bbox_psnr_vals),
        "mask_region_ssim": finite_mean(mask_bbox_ssim_vals),
        "strict_mask_pixel_psnr": finite_mean(strict_mask_pixel_psnr_vals),
        "boundary_pixel_psnr": finite_mean(boundary_pixel_psnr_vals),
        "outside_diff_mean": finite_mean(outside_mean_vals),
        "outside_diff_max": finite_mean(outside_max_vals),
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
        optional["acts"].setdefault(label, {"ori": [], "comp": []})
        optional["acts"][label]["ori"].append(ori_act)
        optional["acts"][label]["comp"].append(comp_act)
    return row


def prepare_video_inputs(args, name: str, target_w: int, target_h: int, temp_dir: Path):
    in_frames = resize_frames(load_rgb_frames(Path(args.video_root) / name, args.video_length), target_w, target_h)
    gt_frames = resize_frames(load_rgb_frames(Path(args.gt_root) / name, args.video_length), target_w, target_h)
    masks = resize_masks(load_gray_masks(Path(args.mask_root) / name, args.video_length), target_w, target_h)
    n = min(len(in_frames), len(gt_frames), len(masks), args.video_length)
    in_frames = in_frames[:n]
    gt_frames = gt_frames[:n]
    masks = normalize_length(masks[:n], n)
    in_frames, gt_frames, masks = ensure_same_hw(in_frames, gt_frames, masks)
    frame_dir = save_frames_to_dir(in_frames, temp_dir / "frames")
    mask_dir = save_masks_to_dir(masks, temp_dir / "masks")
    return in_frames, gt_frames, masks, frame_dir, mask_dir


def run_standard_model(args, device: str, label: str, weights: str, names: Sequence[str], prior_cache: dict[str, list[np.ndarray]], optional) -> tuple[list[dict[str, object]], dict[str, list[np.ndarray]]]:
    model = DiffuEraser(
        device,
        args.base_model_path,
        args.vae_path,
        weights,
        pcm_weights_path=args.pcm_weights_path,
        use_pcm=False,
        num_inference_steps_override=args.num_inference_steps,
    )
    rows = []
    outputs: dict[str, list[np.ndarray]] = {}
    target_w, target_h = parse_input_size(args.input_size)
    safe = "sft" if "SFT" in label else "exp11"
    for idx, name in enumerate(names, 1):
        print(f"[{label}] {idx}/{len(names)} {name}", flush=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f"exp19_eval_{safe}_{name}_"))
        try:
            _in, gt, masks, frame_dir, mask_dir = prepare_video_inputs(args, name, target_w, target_h, temp_dir)
            (Path(args.save_path) / safe / "videos").mkdir(parents=True, exist_ok=True)
            pred = model.forward(
                validation_image=str(frame_dir),
                validation_mask=str(mask_dir),
                priori="__unused__",
                output_path=str(Path(args.save_path) / safe / "videos" / f"{name}.mp4"),
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
            save_rgb_frames(pred, Path(args.save_path) / safe / "raw_frames" / name)
            save_rgb_frames(comp, Path(args.save_path) / safe / "comp_frames" / name)
            rows.append(metric_row(label, name, gt, comp, masks01, optional, args, device))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, outputs


def run_exp19_model(args, device: str, names: Sequence[str], prior_cache: dict[str, list[np.ndarray]], flow_cache: dict[str, object], optional, adapter_enabled: bool = True, suffix: str = "exp19b"):
    model = Exp19FlowAdapterDiffuEraser(
        device=device,
        base_model_path=args.base_model_path,
        vae_path=args.vae_path,
        diffueraser_path=args.exp11_weights,
        adapter_checkpoint=args.exp19_adapter,
        pcm_weights_path=args.pcm_weights_path,
        use_pcm=False,
        num_inference_steps_override=args.num_inference_steps,
    )
    model.write_checkpoint_audit("reports/exp19_inference_checkpoint_loading_audit.md")
    rows = []
    outputs: dict[str, list[np.ndarray]] = {}
    target_w, target_h = parse_input_size(args.input_size)
    seq = context_sequence(args.video_length, args.nframes, args.num_inference_steps)
    for idx, name in enumerate(names, 1):
        print(f"[Exp19b] {idx}/{len(names)} {name}", flush=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f"exp19_eval_{suffix}_{name}_"))
        try:
            _in, gt, masks, frame_dir, mask_dir = prepare_video_inputs(args, name, target_w, target_h, temp_dir)
            flow_result = flow_cache[name]
            cond, flow_stats = flow_condition_from_cache(
                flow_result.forward_flow_path,
                flow_result.backward_flow_path,
                flow_result.confidence_path,
                masks,
                (target_w, target_h),
                args.video_length,
            )
            (Path(args.save_path) / suffix / "videos").mkdir(parents=True, exist_ok=True)
            pred = model.forward(
                validation_image=str(frame_dir),
                validation_mask=str(mask_dir),
                priori_frames=prior_cache[name],
                output_path=str(Path(args.save_path) / suffix / "videos" / f"{name}.mp4"),
                flow_condition=cond,
                context_sequence=seq,
                video_length=args.video_length,
                nframes=args.nframes,
                seed=args.seed,
                adapter_enabled=adapter_enabled,
                max_img_size=max(target_w, target_h) + 100,
                mask_dilation_iter=args.mask_dilation_iter,
            )
            comp, masks01 = composite_with_gt(pred, gt, masks, mask_inverse=False)
            outputs[name] = comp
            save_rgb_frames(pred, Path(args.save_path) / suffix / "raw_frames" / name)
            save_rgb_frames(comp, Path(args.save_path) / suffix / "comp_frames" / name)
            row = metric_row("Exp19b_stage2_500", name, gt, comp, masks01, optional, args, device)
            row.update(flow_stats)
            rows.append(row)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows, outputs


def mean_abs_frames(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    vals = [np.abs(a[i].astype(np.float32) - b[i].astype(np.float32)).mean() for i in range(n)]
    return float(np.mean(vals))


def run_preflight(args, device: str, names, prior_cache, flow_cache) -> None:
    name = names[0]
    optional = {}
    print(f"[preflight] video={name}", flush=True)
    exp11_rows, exp11_out = run_standard_model(args, device, "Exp11_boundary_outer_b075_S2", args.exp11_weights, [name], prior_cache, optional)
    disabled_rows, disabled_out = run_exp19_model(args, device, [name], prior_cache, flow_cache, optional, adapter_enabled=False, suffix="preflight_disabled")
    enabled_rows, enabled_out = run_exp19_model(args, device, [name], prior_cache, flow_cache, optional, adapter_enabled=True, suffix="preflight_enabled")
    # Shuffled-flow ablation: roll the cached condition by using a different video's flow when available.
    shuffled_name = names[1] if len(names) > 1 else name
    shuffled_flow_cache = dict(flow_cache)
    shuffled_flow_cache[name] = flow_cache[shuffled_name]
    _shuffle_rows, shuffle_out = run_exp19_model(args, device, [name], prior_cache, shuffled_flow_cache, optional, adapter_enabled=True, suffix="preflight_shuffled_flow")
    disabled_mae = mean_abs_frames(exp11_out[name], disabled_out[name])
    enabled_mae = mean_abs_frames(disabled_out[name], enabled_out[name])
    shuffled_mae = mean_abs_frames(enabled_out[name], shuffle_out[name])
    status = "PASS"
    reasons = []
    if disabled_mae > 1.0:
        status = "FAILED"
        reasons.append(f"disabled wrapper differs from Exp11 too much: mae={disabled_mae}")
    if enabled_mae <= 0:
        status = "FAILED"
        reasons.append("enabled adapter output is identical to disabled output")
    if shuffled_mae <= 0:
        status = "FAILED"
        reasons.append("shuffled flow output is identical to real-flow output")
    lines = [
        "# Exp19 Inference Preflight",
        "",
        f"status: `{status}`",
        f"video: `{name}`",
        f"disabled_vs_exp11_mae: `{disabled_mae}`",
        f"enabled_vs_disabled_mae: `{enabled_mae}`",
        f"real_vs_shuffled_flow_mae: `{shuffled_mae}`",
        "",
    ]
    if reasons:
        lines.extend(["## Reasons", ""] + [f"- {reason}" for reason in reasons])
    Path("reports/exp19_inference_preflight.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if status != "PASS":
        raise RuntimeError("; ".join(reasons))


def main() -> int:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.save_path)
    out_root.mkdir(parents=True, exist_ok=True)
    target_w, target_h = parse_input_size(args.input_size)
    names = select_videos(Path(args.video_root), args.selected_videos, args.limit_videos)
    if not names:
        raise RuntimeError("No DAVIS videos selected")
    print(f"[exp19-eval] videos={names}", flush=True)

    # Build/reuse ProPainter prior frames once per video.
    propainter = Propainter(args.propainter_model_dir, device)
    flow_propainter = FlowPropainter(args.propainter_model_dir, torch.device(device))
    prior_cache: dict[str, list[np.ndarray]] = {}
    flow_cache = {}
    flow_rows = []
    for idx, name in enumerate(names, 1):
        print(f"[cache] {idx}/{len(names)} {name}", flush=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f"exp19_cache_{name}_"))
        try:
            in_frames, _gt, masks, frame_dir, mask_dir = prepare_video_inputs(args, name, target_w, target_h, temp_dir)
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
                ref_stride=args.ref_stride,
                neighbor_length=args.neighbor_length,
                subvideo_length=args.subvideo_length,
                raft_iter=20,
                save_fps=24,
                save_frames=False,
                fp16=True,
                return_frames=True,
            )
            result = ensure_davis_flow_cache(
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
            flow_cache[name] = result
            flow_rows.append(result.__dict__ | {
                "sample_root": str(result.sample_root),
                "forward_flow_path": str(result.forward_flow_path),
                "backward_flow_path": str(result.backward_flow_path),
                "confidence_path": str(result.confidence_path),
                "fb_error_path": str(result.fb_error_path),
            })
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(Path("reports/exp19_davis10_flow_cache_quality.csv"), flow_rows)
    Path("reports/exp19_davis10_flow_cache_quality.md").write_text(
        "\n".join(
            [
                "# Exp19 DAVIS10 Flow Cache Quality",
                "",
                f"- videos: `{len(flow_rows)}`",
                f"- mean flow_conf_mean: `{finite_mean([r['flow_conf_mean'] for r in flow_rows])}`",
                f"- mean valid_flow_ratio: `{finite_mean([r['valid_flow_ratio'] for r in flow_rows])}`",
                f"- mean forward_backward_error: `{finite_mean([r['forward_backward_error'] for r in flow_rows])}`",
                f"- cache_root: `{args.flow_cache_root}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    if not args.skip_preflight:
        run_preflight(args, device, names, prior_cache, flow_cache)
    if args.preflight_only:
        return 0

    optional = init_optional(args, device)
    all_rows = []
    outputs_by_method: dict[str, dict[str, list[np.ndarray]]] = {}
    rows, outputs = run_standard_model(args, device, "SFT48000_baseline", args.sft_weights, names, prior_cache, optional)
    all_rows.extend(rows)
    outputs_by_method["sft"] = outputs
    rows, outputs = run_standard_model(args, device, "Exp11_boundary_outer_b075_S2", args.exp11_weights, names, prior_cache, optional)
    all_rows.extend(rows)
    outputs_by_method["exp11"] = outputs
    rows, outputs = run_exp19_model(args, device, names, prior_cache, flow_cache, optional, adapter_enabled=True, suffix="exp19b")
    all_rows.extend(rows)
    outputs_by_method["exp19b"] = outputs

    # Add VFID to summaries if requested.
    summaries = []
    for label in ["SFT48000_baseline", "Exp11_boundary_outer_b075_S2", "Exp19b_stage2_500"]:
        label_rows = [row for row in all_rows if row["model_label"] == label]
        summary = summarize_rows(label, label_rows)
        if args.compute_vfid and label in optional["acts"]:
            summary["vfid"] = float(
                metric_backend.calculate_vfid(
                    np.vstack(optional["acts"][label]["ori"]),
                    np.vstack(optional["acts"][label]["comp"]),
                )
            )
        summaries.append(summary)

    metrics_dir = out_root / "metrics"
    write_csv(metrics_dir / "per_video_metrics.csv", all_rows)
    write_csv(metrics_dir / "summary.csv", summaries)
    (metrics_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    for name in names:
        temp_dir = Path(tempfile.mkdtemp(prefix=f"exp19_vis_{name}_"))
        try:
            _in, gt, masks, _frame_dir, _mask_dir = prepare_video_inputs(args, name, target_w, target_h, temp_dir)
            save_side_by_side(
                name,
                gt,
                masks,
                {key: outputs_by_method[key][name] for key in ("sft", "exp11", "exp19b")},
                out_root,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    lines = [
        "# Exp19b DAVIS10 Metric Summary",
        "",
        "| method | PSNR | SSIM | LPIPS | VFID | TC | Ewarp | strict mask PSNR | boundary PSNR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {model_label} | {psnr} | {ssim} | {lpips} | {vfid} | {tc} | {ewarp} | {strict} | {boundary} |".format(
                model_label=row.get("model_label"),
                psnr=row.get("whole_video_psnr_mean", ""),
                ssim=row.get("whole_video_ssim_mean", ""),
                lpips=row.get("whole_video_lpips_mean", ""),
                vfid=row.get("vfid", ""),
                tc=row.get("tc_mean", ""),
                ewarp=row.get("ewarp_mean", ""),
                strict=row.get("strict_mask_pixel_psnr_mean", ""),
                boundary=row.get("boundary_pixel_psnr_mean", ""),
            )
        )
    (Path("reports/exp19b_davis10_metric_summary.md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(metrics_dir / "summary.csv", "reports/exp19b_davis10_metric_summary.csv")
    (out_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
