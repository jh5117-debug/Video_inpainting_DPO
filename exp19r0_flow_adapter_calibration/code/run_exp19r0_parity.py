#!/usr/bin/env python3
"""Exp19-R0 inference parity calibration.

This script checks whether an Exp19 wrapper with the adapter disabled can
reproduce the original Exp11 evaluator under controlled random seeds. It does
not train, modify checkpoints, or use Exp11 fallback for Exp19 outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

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
from exp19_flow_context_provider import (  # noqa: E402
    context_sequence,
    ensure_davis_flow_cache,
    flow_condition_from_cache,
)
from exp19_inference_pipeline import Exp19FlowAdapterDiffuEraser  # noqa: E402
from infer_exp19_flow_adapter_davis import mean_abs_frames, select_videos  # noqa: E402
from propainter.inference import Propainter as FlowPropainter  # noqa: E402


PREFERRED = [
    "boat",
    "rhino",
    "dog-agility",
    "blackswan",
    "lucia",
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
    parser.add_argument("--limit_videos", type=int, default=1)
    parser.add_argument("--selected_videos", default=",".join(PREFERRED))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mask_dilation_iter", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--bf16_tolerance", type=float, default=5e-4)
    return parser.parse_args()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def frame_error(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> dict[str, float]:
    n = min(len(a), len(b))
    vals = []
    max_vals = []
    psnr_vals = []
    for idx in range(n):
        da = a[idx].astype(np.float32)
        db = b[idx].astype(np.float32)
        diff = np.abs(da - db)
        vals.append(float(diff.mean()))
        max_vals.append(float(diff.max()))
        mse = float(np.mean((da - db) ** 2))
        psnr_vals.append(float("inf") if mse <= 1e-12 else float(20.0 * np.log10(255.0 / np.sqrt(mse))))
    return {
        "frames": float(n),
        "mae": float(np.mean(vals)) if vals else float("nan"),
        "max_error": float(np.max(max_vals)) if max_vals else float("nan"),
        "psnr_between_outputs": float(np.mean(psnr_vals)) if psnr_vals else float("nan"),
    }


def run_standard(args, model: DiffuEraser, name: str, prior_frames, seed: int, target_w: int, target_h: int, save_root: Path):
    temp_dir = Path(tempfile.mkdtemp(prefix=f"exp19r0_std_{name}_"))
    try:
        _inp, gt, masks, frame_dir, mask_dir = prepare_video_inputs(args, name, target_w, target_h, temp_dir)
        set_all_seeds(seed)
        pred = model.forward(
            validation_image=str(frame_dir),
            validation_mask=str(mask_dir),
            priori="__unused__",
            output_path=str(save_root / f"{name}_standard.mp4"),
            max_img_size=max(target_w, target_h) + 100,
            video_length=args.video_length,
            mask_dilation_iter=args.mask_dilation_iter,
            nframes=args.nframes,
            seed=seed,
            blended=False,
            priori_frames=prior_frames,
            return_frames=True,
        )
        comp, _ = composite_with_gt(pred, gt, masks, mask_inverse=False)
        return comp
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_exp19_disabled(
    args,
    model: Exp19FlowAdapterDiffuEraser,
    name: str,
    prior_frames,
    flow_result,
    seed: int,
    target_w: int,
    target_h: int,
    save_root: Path,
):
    temp_dir = Path(tempfile.mkdtemp(prefix=f"exp19r0_disabled_{name}_"))
    try:
        _inp, gt, masks, frame_dir, mask_dir = prepare_video_inputs(args, name, target_w, target_h, temp_dir)
        cond, _stats = flow_condition_from_cache(
            flow_result.forward_flow_path,
            flow_result.backward_flow_path,
            flow_result.confidence_path,
            masks,
            (target_w, target_h),
            args.video_length,
        )
        seq = context_sequence(args.video_length, args.nframes, args.num_inference_steps)
        set_all_seeds(seed)
        pred = model.forward(
            validation_image=str(frame_dir),
            validation_mask=str(mask_dir),
            priori_frames=prior_frames,
            output_path=str(save_root / f"{name}_disabled.mp4"),
            flow_condition=cond,
            context_sequence=seq,
            video_length=args.video_length,
            nframes=args.nframes,
            seed=seed,
            adapter_enabled=False,
            max_img_size=max(target_w, target_h) + 100,
            mask_dilation_iter=args.mask_dilation_iter,
        )
        comp, _ = composite_with_gt(pred, gt, masks, mask_inverse=False)
        return comp
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def main() -> int:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.save_path)
    out_root.mkdir(parents=True, exist_ok=True)
    report_root = Path("reports")
    report_root.mkdir(parents=True, exist_ok=True)
    target_w, target_h = parse_input_size(args.input_size)
    names = select_videos(Path(args.video_root), args.selected_videos, args.limit_videos)
    if not names:
        raise RuntimeError("no selected videos")

    propainter = Propainter(args.propainter_model_dir, device)
    flow_propainter = FlowPropainter(args.propainter_model_dir, torch.device(device))
    standard = DiffuEraser(
        device,
        args.base_model_path,
        args.vae_path,
        args.exp11_weights,
        pcm_weights_path=args.pcm_weights_path,
        use_pcm=False,
        num_inference_steps_override=args.num_inference_steps,
    )
    exp19_disabled = Exp19FlowAdapterDiffuEraser(
        device=device,
        base_model_path=args.base_model_path,
        vae_path=args.vae_path,
        diffueraser_path=args.exp11_weights,
        adapter_checkpoint=args.exp19_adapter,
        pcm_weights_path=args.pcm_weights_path,
        use_pcm=False,
        num_inference_steps_override=args.num_inference_steps,
    )

    rows: list[dict[str, object]] = []
    debug: dict[str, object] = {
        "seed_policy": "torch/random/numpy seeds reset immediately before each DiffuEraser.forward",
        "videos": names,
        "target_tolerance": args.bf16_tolerance,
    }
    for name in names:
        print(f"[R0 parity] {name}", flush=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f"exp19r0_cache_{name}_"))
        try:
            in_frames, _gt, masks, frame_dir, mask_dir = prepare_video_inputs(args, name, target_w, target_h, temp_dir)
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
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        std_a = run_standard(args, standard, name, prior_frames, args.seed, target_w, target_h, out_root / "standard_a")
        std_b = run_standard(args, standard, name, prior_frames, args.seed, target_w, target_h, out_root / "standard_b")
        disabled = run_exp19_disabled(
            args,
            exp19_disabled,
            name,
            prior_frames,
            flow_result,
            args.seed,
            target_w,
            target_h,
            out_root / "disabled",
        )
        det = frame_error(std_a, std_b)
        parity = frame_error(std_a, disabled)
        row = {
            "video": name,
            "standard_repeat_mae": det["mae"],
            "standard_repeat_max_error": det["max_error"],
            "standard_repeat_psnr": det["psnr_between_outputs"],
            "disabled_vs_exp11_mae": parity["mae"],
            "disabled_vs_exp11_max_error": parity["max_error"],
            "disabled_vs_exp11_psnr": parity["psnr_between_outputs"],
            "flow_conf_mean": flow_result.flow_conf_mean,
            "valid_flow_ratio": flow_result.valid_flow_ratio,
        }
        row["parity_pass"] = bool(row["disabled_vs_exp11_mae"] <= args.bf16_tolerance)
        rows.append(row)

    write_csv(report_root / "exp19_inference_parity_repair.csv", rows)
    max_mae = max(float(row["disabled_vs_exp11_mae"]) for row in rows)
    status = "PASS" if max_mae <= args.bf16_tolerance else "BLOCKED_BY_EVAL_PARITY"
    debug["max_disabled_vs_exp11_mae"] = max_mae
    debug["status"] = status
    (report_root / "exp19_inference_parity_debug.json").write_text(json.dumps(debug, indent=2), encoding="utf-8")
    lines = [
        "# Exp19 Inference Parity Repair",
        "",
        f"- status: `{status}`",
        f"- videos: `{len(rows)}`",
        f"- tolerance_bf16: `{args.bf16_tolerance}`",
        f"- max_disabled_vs_exp11_mae: `{max_mae}`",
        "",
        "| video | standard repeat MAE | disabled vs Exp11 MAE | disabled vs Exp11 PSNR | pass |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {video} | {standard_repeat_mae:.8f} | {disabled_vs_exp11_mae:.8f} | {disabled_vs_exp11_psnr:.4f} | {parity_pass} |".format(
                **row
            )
        )
    if status != "PASS":
        lines += [
            "",
            "## Decision",
            "",
            "Exp19-R0 parity did not reach the required threshold. Do not run residual sweeps, causality,",
            "Exp19c warp-loss training, or Exp19d motion-aware continuation until this is fixed.",
        ]
    (report_root / "exp19_inference_parity_repair.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
