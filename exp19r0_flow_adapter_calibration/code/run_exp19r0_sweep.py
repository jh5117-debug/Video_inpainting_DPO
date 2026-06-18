#!/usr/bin/env python3
"""Zero-training Exp19-R0 residual scale / confidence sweep and causality audit."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXP19_CODE = PROJECT_ROOT / "exp19_boundary_gated_flow_adapter_dpo" / "code"
if str(EXP19_CODE) not in sys.path:
    sys.path.insert(0, str(EXP19_CODE))

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
from tools.run_davis50_framewise_protocol_eval import (  # noqa: E402
    boundary_mask,
    crop_metric,
    finite_mean,
    masked_pixel_psnr,
)
from exp19_flow_context_provider import (  # noqa: E402
    context_sequence,
    ensure_davis_flow_cache,
    flow_condition_from_cache,
)
from exp19_inference_pipeline import load_flow_adapter_checkpoint  # noqa: E402
from infer_exp19_flow_adapter_davis import select_videos  # noqa: E402
from unet_motion_flow_adapter_wrapper import DEFAULT_TARGET_MODULES, UNetMotionFlowAdapterWrapper  # noqa: E402
from propainter.inference import Propainter as FlowPropainter  # noqa: E402


PREFERRED = [
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
    parser.add_argument("--exp11_weights", required=True)
    parser.add_argument("--exp19_adapter", required=True)
    parser.add_argument("--propainter_model_dir", required=True)
    parser.add_argument("--pcm_weights_path", required=True)
    parser.add_argument("--input_size", default="432x240")
    parser.add_argument("--video_length", type=int, default=24)
    parser.add_argument("--nframes", type=int, default=22)
    parser.add_argument("--num_inference_steps", type=int, default=6)
    parser.add_argument("--limit_videos", type=int, default=10)
    parser.add_argument("--selected_videos", default=",".join(PREFERRED))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mask_dilation_iter", type=int, default=0)
    parser.add_argument("--scales", default="0,0.5,1,2,4")
    parser.add_argument("--confidence_exponents", default="0.5,1,2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--compute_lpips", action="store_true")
    parser.add_argument("--compute_ewarp", action="store_true")
    parser.add_argument("--raft_model_path", default=str(metric_backend.DEFAULT_RAFT_MODEL))
    return parser.parse_args()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CalibratedUNetMotionFlowAdapterWrapper(UNetMotionFlowAdapterWrapper):
    def __init__(
        self,
        unet,
        *,
        residual_scale: float = 1.0,
        confidence_exponent: float = 1.0,
        target_module_names: tuple[str, ...] = DEFAULT_TARGET_MODULES,
    ):
        super().__init__(unet, target_module_names=target_module_names, gate_mode="boundary")
        self.residual_scale = float(residual_scale)
        self.confidence_exponent = float(confidence_exponent)

    def _gate_from_flat_flow(self, flat: torch.Tensor) -> torch.Tensor:
        conf = flat[:, 4:5].clamp(0.0, 1.0).pow(self.confidence_exponent)
        hole = flat[:, 5:6].clamp(0.0, 1.0)
        boundary = flat[:, 6:7].clamp(0.0, 1.0)
        return (conf * torch.clamp(hole + 0.75 * boundary, 0.0, 1.0)).clamp(0.0, 1.0)

    def _build_residual(self, name: str, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual, gate = super()._build_residual(name, hidden)
        return residual * self.residual_scale, gate


class CalibratedExp19Runtime:
    def __init__(
        self,
        *,
        device: str,
        base_model_path: str,
        vae_path: str,
        exp11_weights: str,
        exp19_adapter: str,
        pcm_weights_path: str,
        num_inference_steps: int,
        residual_scale: float,
        confidence_exponent: float,
    ) -> None:
        self.device = device
        self.runtime = DiffuEraser(
            device,
            base_model_path,
            vae_path,
            exp11_weights,
            pcm_weights_path=pcm_weights_path,
            use_pcm=False,
            num_inference_steps_override=num_inference_steps,
        )
        self.wrapper = CalibratedUNetMotionFlowAdapterWrapper(
            self.runtime.pipeline.unet,
            residual_scale=residual_scale,
            confidence_exponent=confidence_exponent,
        )
        self.checkpoint_audit = load_flow_adapter_checkpoint(self.wrapper, exp19_adapter)
        self.wrapper.to(device=device, dtype=self.runtime.pipeline.unet.dtype)
        self.wrapper.eval()
        self.runtime.pipeline.unet = self.wrapper
        self.runtime.unet_main = self.wrapper

    def forward(self, *, frame_dir: Path, mask_dir: Path, prior_frames, output_path: Path, flow_condition: torch.Tensor, seq, args, adapter_enabled: bool = True):
        self.wrapper.set_auto_flow_context(
            flow_condition.to(device=self.device, dtype=self.runtime.pipeline.unet.dtype),
            seq,
            enabled=adapter_enabled,
        )
        try:
            set_all_seeds(args.seed)
            frames = self.runtime.forward(
                validation_image=str(frame_dir),
                validation_mask=str(mask_dir),
                priori="__unused__",
                output_path=str(output_path),
                max_img_size=max(parse_input_size(args.input_size)) + 100,
                video_length=args.video_length,
                mask_dilation_iter=args.mask_dilation_iter,
                nframes=args.nframes,
                seed=args.seed,
                blended=False,
                priori_frames=prior_frames,
                return_frames=True,
                apply_composite=True,
            )
            consumed, total = self.wrapper.auto_context_consumed()
            if consumed != total:
                raise RuntimeError(f"context consumed {consumed}/{total}")
            return frames
        finally:
            self.wrapper.clear_auto_flow_context()


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


def init_optional(args, device: str):
    optional = {}
    if args.compute_lpips:
        metric_backend.LPIPSMetric.get_instance(device)
    if args.compute_ewarp:
        optional["ewarp"] = metric_backend.EwarpMetric(device=device, raft_model_path=args.raft_model_path)
    return optional


def metrics_for(video: str, label: str, gt_frames, comp_frames, masks01, optional, args, device: str) -> dict[str, object]:
    psnr_vals = []
    ssim_vals = []
    strict_vals = []
    boundary_vals = []
    bbox_psnr_vals = []
    lpips_vals = []
    for gt, comp, mask in zip(gt_frames, comp_frames, masks01):
        psnr_vals.append(metric_backend.compute_psnr(gt, comp))
        ssim_vals.append(metric_backend.compute_ssim(gt, comp))
        p, _s = crop_metric(gt, comp, mask)
        bbox_psnr_vals.append(p)
        strict_vals.append(masked_pixel_psnr(gt, comp, mask))
        boundary_vals.append(masked_pixel_psnr(gt, comp, boundary_mask(mask)))
        if args.compute_lpips:
            lpips_vals.append(float(metric_backend.LPIPSMetric.compute(gt, comp, device=device)))
    row: dict[str, object] = {
        "video": video,
        "label": label,
        "psnr": finite_mean(psnr_vals),
        "ssim": finite_mean(ssim_vals),
        "mask_bbox_psnr": finite_mean(bbox_psnr_vals),
        "strict_mask_psnr": finite_mean(strict_vals),
        "boundary_psnr": finite_mean(boundary_vals),
    }
    if args.compute_lpips:
        row["lpips"] = finite_mean(lpips_vals)
    if args.compute_ewarp:
        row["ewarp"] = float(optional["ewarp"].compute(comp_frames, masks01=masks01, gt_frames_u8_rgb=gt_frames))
    return row


def summarize(rows: Sequence[dict[str, object]], keys: Sequence[str]) -> dict[str, float]:
    return {f"{key}_mean": finite_mean([row[key] for row in rows if key in row]) for key in keys}


def condition_variant(cond: torch.Tensor, mode: str, shuffled: torch.Tensor | None = None) -> torch.Tensor:
    out = cond.clone()
    if mode == "real":
        return out
    if mode == "zero":
        out[:, :, 0:4] = 0
        return out
    if mode == "reversed":
        return out.flip(1)
    if mode == "shuffled":
        if shuffled is None:
            raise ValueError("shuffled condition required")
        return shuffled.clone()
    if mode == "disabled":
        return out
    raise ValueError(mode)


def main() -> int:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.save_path)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root = Path("reports")
    report_root.mkdir(exist_ok=True)
    target_w, target_h = parse_input_size(args.input_size)
    all_names = select_videos(Path(args.video_root), args.selected_videos, args.limit_videos)
    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    exponents = [float(x) for x in args.confidence_exponents.split(",") if x.strip()]
    optional = init_optional(args, device)

    propainter = Propainter(args.propainter_model_dir, device)
    flow_propainter = FlowPropainter(args.propainter_model_dir, torch.device(device))

    samples: dict[str, dict[str, object]] = {}
    motion_rows = []
    for idx, name in enumerate(all_names, 1):
        print(f"[R0 cache/motion] {idx}/{len(all_names)} {name}", flush=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f"exp19r0_sweep_cache_{name}_"))
        try:
            in_frames, gt, masks, frame_dir, mask_dir = prepare_video_inputs(args, name, target_w, target_h, temp_dir)
            prior_path = out_root / "propainter_prior" / f"{name}.mp4"
            prior_path.parent.mkdir(parents=True, exist_ok=True)
            prior_frames = propainter.forward(
                video=str(frame_dir),
                mask=str(mask_dir),
                output_path=str(prior_path),
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
            flow_result = ensure_davis_flow_cache(
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
            cond, stats = flow_condition_from_cache(
                flow_result.forward_flow_path,
                flow_result.backward_flow_path,
                flow_result.confidence_path,
                masks,
                (target_w, target_h),
                args.video_length,
            )
            motion_score = stats["mean_flow_magnitude"] * stats["valid_flow_ratio"] * max(stats["nonzero_gate_ratio"], 1e-6)
            samples[name] = {
                "gt": gt,
                "masks": masks,
                "prior": prior_frames,
                "cond": cond,
                "flow_stats": stats,
            }
            motion_rows.append({
                "video": name,
                "motion_score": motion_score,
                **stats,
            })
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    sorted_motion = sorted(motion_rows, key=lambda r: float(r["motion_score"]), reverse=True)
    high = [r["video"] for r in sorted_motion[:3]]
    mid_pool = sorted_motion[3:]
    mid = [r["video"] for r in mid_pool[:2]] if len(mid_pool) >= 2 else [r["video"] for r in sorted_motion[3:5]]
    sweep_names = list(dict.fromkeys(high + mid))
    for row in motion_rows:
        row["motion_bin"] = "high" if row["video"] in high else ("medium" if row["video"] in mid else "low")
    write_csv(report_root / "exp19_motion_bins.csv", motion_rows)
    (report_root / "exp19_motion_score_audit.md").write_text(
        "# Exp19 Motion Score Audit\n\n"
        + f"- selected_for_sweep: `{sweep_names}`\n"
        + "\n| video | motion score | mean flow | valid ratio | nonzero gate | bin |\n"
        + "|---|---:|---:|---:|---:|---|\n"
        + "\n".join(
            f"| {r['video']} | {r['motion_score']:.6f} | {r['mean_flow_magnitude']:.4f} | {r['valid_flow_ratio']:.4f} | {r['nonzero_gate_ratio']:.4f} | {r['motion_bin']} |"
            for r in sorted_motion
        )
        + "\n",
        encoding="utf-8",
    )

    # Baseline Exp11 for selected sweep videos.
    standard = DiffuEraser(
        device,
        args.base_model_path,
        args.vae_path,
        args.exp11_weights,
        pcm_weights_path=args.pcm_weights_path,
        use_pcm=False,
        num_inference_steps_override=args.num_inference_steps,
    )
    exp11_rows = []
    exp11_outputs = {}
    seq = context_sequence(args.video_length, args.nframes, args.num_inference_steps)
    for name in sweep_names:
        temp_dir = Path(tempfile.mkdtemp(prefix=f"exp19r0_exp11_{name}_"))
        try:
            _inp, gt, masks, frame_dir, mask_dir = prepare_video_inputs(args, name, target_w, target_h, temp_dir)
            set_all_seeds(args.seed)
            pred = standard.forward(
                validation_image=str(frame_dir),
                validation_mask=str(mask_dir),
                priori="__unused__",
                output_path=str(out_root / "exp11" / f"{name}.mp4"),
                max_img_size=max(target_w, target_h) + 100,
                video_length=args.video_length,
                mask_dilation_iter=args.mask_dilation_iter,
                nframes=args.nframes,
                seed=args.seed,
                blended=False,
                priori_frames=samples[name]["prior"],
                return_frames=True,
            )
            comp, masks01 = composite_with_gt(pred, gt, masks, mask_inverse=False)
            exp11_outputs[name] = comp
            row = metrics_for(name, "Exp11", gt, comp, masks01, optional, args, device)
            exp11_rows.append(row)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    del standard
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    baseline_by_video = {row["video"]: row for row in exp11_rows}
    sweep_rows = []
    metric_keys = ["psnr", "ssim", "strict_mask_psnr", "boundary_psnr", "lpips", "ewarp"]
    for scale in scales:
        for exponent in exponents:
            print(f"[R0 sweep] scale={scale} conf_exp={exponent}", flush=True)
            runtime = CalibratedExp19Runtime(
                device=device,
                base_model_path=args.base_model_path,
                vae_path=args.vae_path,
                exp11_weights=args.exp11_weights,
                exp19_adapter=args.exp19_adapter,
                pcm_weights_path=args.pcm_weights_path,
                num_inference_steps=args.num_inference_steps,
                residual_scale=scale,
                confidence_exponent=exponent,
            )
            combo_rows = []
            for name in sweep_names:
                temp_dir = Path(tempfile.mkdtemp(prefix=f"exp19r0_sweep_{name}_"))
                try:
                    _inp, gt, masks, frame_dir, mask_dir = prepare_video_inputs(args, name, target_w, target_h, temp_dir)
                    cond = samples[name]["cond"]
                    pred = runtime.forward(
                        frame_dir=frame_dir,
                        mask_dir=mask_dir,
                        prior_frames=samples[name]["prior"],
                        output_path=out_root / "sweep_videos" / f"s{scale}_p{exponent}_{name}.mp4",
                        flow_condition=cond,
                        seq=seq,
                        args=args,
                        adapter_enabled=True,
                    )
                    comp, masks01 = composite_with_gt(pred, gt, masks, mask_inverse=False)
                    row = metrics_for(name, "Exp19R0", gt, comp, masks01, optional, args, device)
                    base = baseline_by_video[name]
                    row.update({
                        "residual_scale": scale,
                        "confidence_exponent": exponent,
                        "psnr_delta_vs_exp11": row["psnr"] - base["psnr"],
                        "ewarp_delta_vs_exp11": row.get("ewarp", np.nan) - base.get("ewarp", np.nan),
                        "lpips_delta_vs_exp11": row.get("lpips", np.nan) - base.get("lpips", np.nan),
                        "strict_mask_psnr_delta_vs_exp11": row["strict_mask_psnr"] - base["strict_mask_psnr"],
                        "boundary_psnr_delta_vs_exp11": row["boundary_psnr"] - base["boundary_psnr"],
                    })
                    row.update({f"flow_{k}": v for k, v in samples[name]["flow_stats"].items()})
                    combo_rows.append(row)
                    sweep_rows.append(row)
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            del runtime
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(report_root / "exp19_residual_scale_confidence_sweep.csv", sweep_rows)
    grouped: dict[tuple[float, float], list[dict[str, object]]] = defaultdict(list)
    for row in sweep_rows:
        grouped[(float(row["residual_scale"]), float(row["confidence_exponent"]))].append(row)
    combo_summaries = []
    for (scale, exponent), rows in grouped.items():
        summary = {"residual_scale": scale, "confidence_exponent": exponent, "videos": len(rows)}
        for key in metric_keys + [
            "psnr_delta_vs_exp11",
            "ewarp_delta_vs_exp11",
            "lpips_delta_vs_exp11",
            "strict_mask_psnr_delta_vs_exp11",
            "boundary_psnr_delta_vs_exp11",
        ]:
            vals = [row[key] for row in rows if key in row and not np.isnan(float(row[key]))]
            summary[f"{key}_mean"] = finite_mean(vals)
        combo_summaries.append(summary)
    combo_summaries.sort(key=lambda r: (r.get("ewarp_delta_vs_exp11_mean", 999.0), -r.get("psnr_delta_vs_exp11_mean", -999.0)))
    write_csv(report_root / "exp19_residual_scale_confidence_sweep_summary.csv", combo_summaries)
    valid = [r for r in combo_summaries if r.get("psnr_delta_vs_exp11_mean", -999.0) >= -0.05]
    top3 = valid[:3] if len(valid) >= 3 else combo_summaries[:3]
    lines = [
        "# Exp19 Residual Scale / Confidence Sweep",
        "",
        f"- sweep_videos: `{sweep_names}`",
        f"- combinations: `{len(combo_summaries)}`",
        "",
        "| scale | conf exp | PSNR delta | LPIPS delta | Ewarp delta | strict mask delta | boundary delta |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combo_summaries:
        lines.append(
            "| {residual_scale} | {confidence_exponent} | {psnr_delta_vs_exp11_mean:.6f} | {lpips_delta_vs_exp11_mean:.8f} | {ewarp_delta_vs_exp11_mean:.6f} | {strict_mask_psnr_delta_vs_exp11_mean:.6f} | {boundary_psnr_delta_vs_exp11_mean:.6f} |".format(**row)
        )
    lines += ["", "## Top 3", ""]
    for row in top3:
        lines.append(f"- scale={row['residual_scale']} confidence_exponent={row['confidence_exponent']}")
    (report_root / "exp19_residual_scale_confidence_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Causality audit on best combo over sweep videos.
    best = top3[0]
    best_scale = float(best["residual_scale"])
    best_exp = float(best["confidence_exponent"])
    print(f"[R0 causality] best scale={best_scale} exp={best_exp}", flush=True)
    causality_rows = []
    runtime = CalibratedExp19Runtime(
        device=device,
        base_model_path=args.base_model_path,
        vae_path=args.vae_path,
        exp11_weights=args.exp11_weights,
        exp19_adapter=args.exp19_adapter,
        pcm_weights_path=args.pcm_weights_path,
        num_inference_steps=args.num_inference_steps,
        residual_scale=best_scale,
        confidence_exponent=best_exp,
    )
    shuffled_map = {}
    for i, name in enumerate(sweep_names):
        shuffled_map[name] = samples[sweep_names[(i + 1) % len(sweep_names)]]["cond"]
    for mode in ["real", "zero", "shuffled", "reversed", "disabled"]:
        for name in sweep_names:
            temp_dir = Path(tempfile.mkdtemp(prefix=f"exp19r0_causal_{mode}_{name}_"))
            try:
                _inp, gt, masks, frame_dir, mask_dir = prepare_video_inputs(args, name, target_w, target_h, temp_dir)
                cond = condition_variant(samples[name]["cond"], mode, shuffled=shuffled_map.get(name))
                pred = runtime.forward(
                    frame_dir=frame_dir,
                    mask_dir=mask_dir,
                    prior_frames=samples[name]["prior"],
                    output_path=out_root / "causality_videos" / f"{mode}_{name}.mp4",
                    flow_condition=cond,
                    seq=seq,
                    args=args,
                    adapter_enabled=(mode != "disabled"),
                )
                comp, masks01 = composite_with_gt(pred, gt, masks, mask_inverse=False)
                row = metrics_for(name, mode, gt, comp, masks01, optional, args, device)
                base = baseline_by_video[name]
                row.update({
                    "mode": mode,
                    "residual_scale": best_scale,
                    "confidence_exponent": best_exp,
                    "psnr_delta_vs_exp11": row["psnr"] - base["psnr"],
                    "ewarp_delta_vs_exp11": row.get("ewarp", np.nan) - base.get("ewarp", np.nan),
                    "lpips_delta_vs_exp11": row.get("lpips", np.nan) - base.get("lpips", np.nan),
                })
                causality_rows.append(row)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    del runtime
    write_csv(report_root / "exp19r0_flow_causality_metrics.csv", causality_rows)
    by_mode = defaultdict(list)
    for row in causality_rows:
        by_mode[row["mode"]].append(row)
    causality_summary = []
    for mode, rows in by_mode.items():
        summary = {"mode": mode, "videos": len(rows)}
        for key in metric_keys + ["psnr_delta_vs_exp11", "ewarp_delta_vs_exp11", "lpips_delta_vs_exp11"]:
            vals = [row[key] for row in rows if key in row and not np.isnan(float(row[key]))]
            summary[f"{key}_mean"] = finite_mean(vals)
        causality_summary.append(summary)
    write_csv(report_root / "exp19r0_flow_causality_summary.csv", causality_summary)
    real = next(r for r in causality_summary if r["mode"] == "real")
    competitors = [r for r in causality_summary if r["mode"] in {"zero", "shuffled", "reversed"}]
    real_better_ewarp = all(real.get("ewarp_mean", np.inf) < c.get("ewarp_mean", -np.inf) for c in competitors if "ewarp_mean" in c)
    real_better_psnr = all(real.get("psnr_mean", -np.inf) >= c.get("psnr_mean", np.inf) - 1e-6 for c in competitors if "psnr_mean" in c)
    causality_pass = bool(real_better_ewarp and real_better_psnr)
    md = [
        "# Exp19-R0 Flow Causality Audit",
        "",
        f"- best_residual_scale: `{best_scale}`",
        f"- best_confidence_exponent: `{best_exp}`",
        f"- causality_pass: `{causality_pass}`",
        "",
        "| mode | PSNR | LPIPS | Ewarp | PSNR delta vs Exp11 | Ewarp delta vs Exp11 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(causality_summary, key=lambda r: str(r["mode"])):
        md.append(
            "| {mode} | {psnr_mean:.6f} | {lpips_mean:.8f} | {ewarp_mean:.6f} | {psnr_delta_vs_exp11_mean:.6f} | {ewarp_delta_vs_exp11_mean:.6f} |".format(**row)
        )
    if not causality_pass:
        md += [
            "",
            "## Decision",
            "",
            "Real flow does not reliably beat zero/shuffled/reversed flow under the selected R0 setting.",
            "Do not launch Exp19c warp-loss training.",
        ]
    (report_root / "exp19r0_flow_causality_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0 if causality_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
