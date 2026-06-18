#!/usr/bin/env python3
"""Exp19c light latent-warp continuation trainer.

All variants continue from the same Exp19b Stage2-500 flow adapter checkpoint.
The DiffuEraser base remains frozen. The Exp11 DPO loss is unchanged; this
script only adds a confidence-gated latent warp consistency term.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from types import MethodType

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parents[1]
EXP19_CODE = PROJECT_ROOT / "exp19_boundary_gated_flow_adapter_dpo" / "code"
for path in (str(CODE_DIR), str(EXP19_CODE), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from train_exp19_stage2_adapter import (  # noqa: E402
    collate_exp19,
    compute_step_loss,
    load_models,
    move_batch,
    resolve_dtype,
    save_adapter,
)
from exp19_dataset import Exp19FlowManifestDataset  # noqa: E402
from exp19_diag import grad_norm  # noqa: E402
from exp19c_diag import append_exp19c_diag_csv  # noqa: E402
from latent_warp_loss import confidence_gated_latent_warp_loss, predict_x0_latent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", required=True)
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--exp11_stage2_weights", required=True)
    parser.add_argument("--start_adapter", required=True)
    parser.add_argument("--flow_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--diag_csv", required=True)
    parser.add_argument("--report_path", required=True)
    parser.add_argument("--variant_name", required=True)
    parser.add_argument("--lambda_warp", type=float, required=True)
    parser.add_argument("--max_train_steps", type=int, default=500)
    parser.add_argument("--checkpointing_steps", type=int, default=250)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--nframes", type=int, default=16)
    parser.add_argument("--train_height", type=int, default=240)
    parser.add_argument("--train_width", type=int, default=432)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpo_diag_interval", type=int, default=10)
    parser.add_argument("--beta_dpo", type=float, default=10.0)
    parser.add_argument("--lose_gap_weight", type=float, default=0.25)
    parser.add_argument("--lose_gap_clip_tau", type=float, default=1.0)
    parser.add_argument("--winner_abs_reg_weight", type=float, default=0.05)
    parser.add_argument("--winner_gap_reg_weight", type=float, default=1.0)
    parser.add_argument("--winner_gap_reg_margin", type=float, default=0.0)
    parser.add_argument("--mask_region_weight", type=float, default=1.0)
    parser.add_argument("--boundary_region_weight", type=float, default=0.75)
    parser.add_argument("--outside_region_weight", type=float, default=0.05)
    parser.add_argument("--target_modules", default="mid_block.motion_modules.0,up_blocks.0.motion_modules.0,up_blocks.1.motion_modules.0")
    parser.add_argument("--residual_scale", type=float, default=0.5)
    parser.add_argument("--confidence_exponent", type=float, default=2.0)
    return parser.parse_args()


def load_adapter_checkpoint(unet, checkpoint_path: str | Path, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing start adapter checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("adapter_state_dict", ckpt)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"invalid adapter checkpoint: {checkpoint_path}")
    unet.load_adapter_state_dict(state)
    unet.to(device=device, dtype=dtype)
    total_norm = 0.0
    nonzero = 0
    params = 0
    for value in state.values():
        if not torch.is_tensor(value):
            continue
        params += int(value.numel())
        norm = float(value.detach().float().norm().cpu())
        total_norm += norm * norm
        if bool((value.detach().float().abs() > 0).any().item()):
            nonzero += 1
    return {
        "checkpoint": str(checkpoint_path),
        "tensors": len(state),
        "nonzero_tensors": nonzero,
        "parameter_count": params,
        "parameter_l2_norm": total_norm ** 0.5,
    }


def apply_r0_calibration(unet, residual_scale: float, confidence_exponent: float) -> None:
    """Match the R0-selected residual scale and confidence exponent."""
    original_build_residual = unet._build_residual

    def _gate_from_flat_flow(self, flat):
        conf = flat[:, 4:5].clamp(0.0, 1.0).pow(float(confidence_exponent))
        hole = flat[:, 5:6].clamp(0.0, 1.0)
        boundary = flat[:, 6:7].clamp(0.0, 1.0)
        return (conf * torch.clamp(hole + 0.75 * boundary, 0.0, 1.0)).clamp(0.0, 1.0)

    def _build_residual(self, name, hidden):
        residual, gate = original_build_residual(name, hidden)
        return residual * float(residual_scale), gate

    unet._gate_from_flat_flow = MethodType(_gate_from_flat_flow, unet)
    unet._build_residual = MethodType(_build_residual, unet)


def build_dataset(args: argparse.Namespace, tokenizer) -> Exp19FlowManifestDataset:
    dataset_args = argparse.Namespace(
        preference_manifest=args.flow_manifest,
        nframes=args.nframes,
        train_height=args.train_height,
        train_width=args.train_width,
        resolution=args.train_height,
        train_mask_mode="partial",
        mask_from_manifest=True,
        videodpo_full_mask_value=0.0,
        max_resample_attempts=64,
        proportion_empty_prompts=0.0,
    )
    return Exp19FlowManifestDataset(dataset_args, tokenizer)


def motion_bin(score: float) -> str:
    if score >= 0.15:
        return "high"
    if score >= 0.05:
        return "medium"
    return "low"


def row_from(step, args, diagnostics, warp_loss, warp_stats, tensors, unet, adapter_grad, base_grad):
    residual_norm = sum(unet.last_residual_norms.values()) if unet.last_residual_norms else 0.0
    flow_stats = tensors["flow_stats"]
    motion_score = float(flow_stats.get("mean_flow_magnitude", 0.0)) * float(flow_stats.get("valid_flow_ratio", 0.0)) * max(float(flow_stats.get("nonzero_gate_ratio", 0.0)), 1e-6)
    return {
        "step": step,
        "variant_name": args.variant_name,
        "total_loss": diagnostics.get("total_loss"),
        "dpo_loss": diagnostics.get("dpo_loss"),
        "warp_loss": float(warp_loss.detach().float().cpu()),
        "warp_loss_forward": warp_stats.get("warp_loss_forward"),
        "warp_loss_backward": warp_stats.get("warp_loss_backward"),
        "lambda_warp": args.lambda_warp,
        "m_w": diagnostics.get("mse_w"),
        "m_l": diagnostics.get("mse_l"),
        "m_w_ref": diagnostics.get("ref_mse_w"),
        "m_l_ref": diagnostics.get("ref_mse_l"),
        "norm_win_gap": diagnostics.get("norm_win_gap"),
        "norm_lose_gap": diagnostics.get("norm_lose_gap"),
        "norm_lose_gap_clipped": diagnostics.get("norm_lose_gap_clipped"),
        "winner_abs_reg": diagnostics.get("winner_abs_reg"),
        "winner_gap_reg": diagnostics.get("winner_gap_reg"),
        "loser_dominant_ratio": diagnostics.get("loser_degrade_ratio"),
        "grad_norm": adapter_grad,
        "adapter_grad_norm": adapter_grad,
        "base_grad_norm": base_grad,
        "adapter_residual_norm": residual_norm,
        "adapter_to_base_ratio": None,
        "residual_scale": args.residual_scale,
        "confidence_exponent": args.confidence_exponent,
        "gate_mean": unet.last_gate_stats.get("gate_mean", flow_stats.get("gate_mean")),
        "flow_conf_mean": flow_stats.get("flow_conf_mean"),
        "valid_flow_ratio": flow_stats.get("valid_flow_ratio"),
        "warp_valid_ratio": warp_stats.get("warp_valid_ratio"),
        "mean_flow_magnitude": flow_stats.get("mean_flow_magnitude"),
        "motion_score": motion_score,
        "motion_bin": motion_bin(motion_score),
        "lr": args.learning_rate,
    }


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = resolve_dtype(args.mixed_precision)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.diag_csv).parent.mkdir(parents=True, exist_ok=True)

    models = load_models(args, device, weight_dtype)
    apply_r0_calibration(models["unet"], args.residual_scale, args.confidence_exponent)
    ckpt_audit = load_adapter_checkpoint(models["unet"], args.start_adapter, device, weight_dtype)
    adapter_params = list(models["unet"].adapter_parameters())
    if not adapter_params:
        raise RuntimeError("No Exp19c adapter parameters after loading start checkpoint")

    dataset = build_dataset(args, models["tokenizer"])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_exp19,
    )
    first_batch = move_batch(next(iter(dataloader)), device)
    loss_base, diagnostics, tensors, model_pred, _ref_pred = compute_step_loss(first_batch, models, args, device, weight_dtype, train=True)
    policy_pos = model_pred[: tensors["noise"].shape[0]]
    z_hat0 = predict_x0_latent(tensors["noisy_pos"], policy_pos, tensors["timesteps_expanded"], models["noise_scheduler"])
    warp_loss, warp_stats = confidence_gated_latent_warp_loss(
        z_hat0_flat=z_hat0,
        batch=first_batch,
        nframes=args.nframes,
    )
    total = loss_base + args.lambda_warp * warp_loss
    total.backward()
    adapter_gn = grad_norm(adapter_params)
    base_gn = grad_norm(models["unet_base"].parameters())
    preflight_ok = (
        math.isfinite(float(total.detach().float().cpu()))
        and math.isfinite(float(warp_loss.detach().float().cpu()))
        and adapter_gn > 0.0
        and base_gn == 0.0
    )
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).write_text(
        "\n".join(
            [
                "# Exp19c Light Warp Preflight",
                "",
                f"status: `{'PASS' if preflight_ok else 'FAILED'}`",
                f"variant: `{args.variant_name}`",
                f"lambda_warp: `{args.lambda_warp}`",
                f"start_adapter: `{args.start_adapter}`",
                f"checkpoint_audit: `{json.dumps(ckpt_audit, ensure_ascii=False)}`",
                f"base_loss: `{float(loss_base.detach().float().cpu())}`",
                f"warp_loss: `{float(warp_loss.detach().float().cpu())}`",
                f"adapter_grad_norm: `{adapter_gn}`",
                f"base_grad_norm: `{base_gn}`",
                f"warp_stats: `{json.dumps(warp_stats, ensure_ascii=False)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for p in adapter_params:
        p.grad = None
    if not preflight_ok:
        return 3

    optimizer = torch.optim.AdamW(adapter_params, lr=args.learning_rate, weight_decay=0.0)
    global_step = 0
    pbar = tqdm(total=args.max_train_steps, desc=args.variant_name)
    while global_step < args.max_train_steps:
        for batch in dataloader:
            batch = move_batch(batch, device)
            loss_base, diagnostics, tensors, model_pred, _ref_pred = compute_step_loss(batch, models, args, device, weight_dtype, train=True)
            policy_pos = model_pred[: tensors["noise"].shape[0]]
            z_hat0 = predict_x0_latent(tensors["noisy_pos"], policy_pos, tensors["timesteps_expanded"], models["noise_scheduler"])
            warp_loss, warp_stats = confidence_gated_latent_warp_loss(
                z_hat0_flat=z_hat0,
                batch=batch,
                nframes=args.nframes,
            )
            total = loss_base + args.lambda_warp * warp_loss
            diagnostics["total_loss"] = float(total.detach().float().cpu())
            total.backward()
            adapter_gn = grad_norm(adapter_params)
            base_gn = grad_norm(models["unet_base"].parameters())
            torch.nn.utils.clip_grad_norm_(adapter_params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            pbar.update(1)
            if global_step == 1 or global_step % args.dpo_diag_interval == 0:
                row = row_from(global_step, args, diagnostics, warp_loss, warp_stats, tensors, models["unet"], adapter_gn, base_gn)
                append_exp19c_diag_csv(args.diag_csv, row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
            if global_step % args.checkpointing_steps == 0:
                save_adapter(output_dir, models["unet"], global_step)
            del batch, loss_base, total, warp_loss, diagnostics, tensors, model_pred
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if global_step >= args.max_train_steps:
                break
    pbar.close()
    save_adapter(output_dir, models["unet"], "last_weights")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
