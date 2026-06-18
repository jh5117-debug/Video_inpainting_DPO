#!/usr/bin/env python3
"""Exp19b Stage2 flow-adapter-only DPO trainer.

This is intentionally isolated from shared training code. It imports model and
loss helpers, but all hook-based flow adapter logic lives under Exp19.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from tqdm.auto import tqdm

CODE_DIR = Path(__file__).resolve().parent


def find_project_root() -> Path:
    candidates: list[Path] = []
    for env_name in ("EXP19_CODE_ROOT", "PROJECT_ROOT", "ROOT"):
        env_root = os.environ.get(env_name)
        if env_root:
            candidates.append(Path(env_root))
    candidates.append(Path.cwd())
    candidates.extend(Path(__file__).resolve().parents)
    candidates.extend(
        [
            Path("/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp09_10_11_pai_sync"),
            Path("/mnt/nas/hj/H20_Video_inpainting_DPO_exp09_10_11_pai_sync"),
            Path("/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate"),
            Path("/mnt/nas/hj/H20_Video_inpainting_DPO_exp14_videopainter_gate"),
        ]
    )
    for candidate in candidates:
        if (candidate / "exp11_region_boundary_ablation" / "code" / "train_stage1.py").exists():
            return candidate
    raise RuntimeError(
        "Could not locate repo root containing exp11_region_boundary_ablation/code/train_stage1.py. "
        f"Checked: {[str(c) for c in candidates[:8]]}"
    )


PROJECT_ROOT = find_project_root()
EXP11_CODE = PROJECT_ROOT / "exp11_region_boundary_ablation" / "code"
for path in (str(CODE_DIR), str(EXP11_CODE), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from diffusers import AutoencoderKL, DDPMScheduler  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from libs.brushnet_CA import BrushNetModel  # noqa: E402
from libs.unet_motion_model import UNetMotionModel  # noqa: E402
from train_stage2 import import_model_class_from_model_name_or_path  # noqa: E402
from train_stage1 import (  # noqa: E402
    build_region_loss_weight_map,
    compute_dpo_loss,
)
from exp19_dataset import Exp19FlowManifestDataset, build_flow_condition  # noqa: E402
from exp19_diag import append_exp19_diag_csv, grad_norm  # noqa: E402
from unet_motion_flow_adapter_wrapper import UNetMotionFlowAdapterWrapper  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--base_model_name_or_path", required=True)
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--exp11_stage2_weights", required=True)
    parser.add_argument("--flow_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--report_path", default="reports/exp19_isolated_wrapper_preflight.md")
    parser.add_argument("--diag_csv", default="")
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
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def collate_exp19(examples: list[dict[str, Any]]) -> dict[str, Any]:
    keys_tensor = [
        "pixel_values_pos",
        "pixel_values_neg",
        "conditioning_pixel_values",
        "masks",
        "input_ids",
        "flow_forward",
        "flow_backward",
        "flow_confidence",
        "hole_mask",
        "outer_boundary",
        "flow_gate_mask",
    ]
    batch = {}
    for key in keys_tensor:
        vals = [e[key] for e in examples]
        batch[key] = torch.stack(vals).float() if key != "input_ids" else torch.stack(vals)
    batch["sample_id"] = [e.get("sample_id") for e in examples]
    return batch


def forward_pair_member(
    brushnet,
    unet_wrapper: UNetMotionFlowAdapterWrapper,
    noisy_latents,
    timesteps_2d,
    timesteps_motion,
    encoder_hidden_states_2d,
    encoder_hidden_states_motion,
    brushnet_cond,
    weight_dtype,
    nframes,
    flow_condition,
    adapter_enabled: bool,
):
    down_samples, mid_sample, up_samples = brushnet(
        noisy_latents,
        timesteps_2d,
        encoder_hidden_states=encoder_hidden_states_2d,
        brushnet_cond=brushnet_cond,
        return_dict=False,
    )
    out = unet_wrapper(
        noisy_latents,
        timesteps_motion,
        encoder_hidden_states=encoder_hidden_states_motion,
        down_block_add_samples=[s.to(dtype=weight_dtype) for s in down_samples],
        mid_block_add_sample=mid_sample.to(dtype=weight_dtype),
        up_block_add_samples=[s.to(dtype=weight_dtype) for s in up_samples],
        return_dict=True,
        num_frames=nframes,
        flow_condition=flow_condition,
        adapter_enabled=adapter_enabled,
    ).sample
    return out


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, val in batch.items():
        out[key] = val.to(device) if torch.is_tensor(val) else val
    return out


def prepare_step_tensors(batch, models, args, device, weight_dtype):
    vae, text_encoder, noise_scheduler = models["vae"], models["text_encoder"], models["noise_scheduler"]
    vae_dtype = torch.float32
    pos_latents = vae.encode(rearrange(batch["pixel_values_pos"], "b f c h w -> (b f) c h w").to(dtype=vae_dtype)).latent_dist.sample() * vae.config.scaling_factor
    neg_latents = vae.encode(rearrange(batch["pixel_values_neg"], "b f c h w -> (b f) c h w").to(dtype=vae_dtype)).latent_dist.sample() * vae.config.scaling_factor
    n_batch = batch["conditioning_pixel_values"].shape[0]
    cond_latents = vae.encode(rearrange(batch["conditioning_pixel_values"], "b f c h w -> (b f) c h w").to(dtype=vae_dtype)).latent_dist.sample() * vae.config.scaling_factor
    pos_latents = pos_latents.to(dtype=weight_dtype)
    neg_latents = neg_latents.to(dtype=weight_dtype)
    cond_latents = rearrange(cond_latents.to(dtype=weight_dtype), "(b f) c h w -> b f c h w", b=n_batch)
    masks = F.interpolate(batch["masks"].to(dtype=weight_dtype), size=(1, pos_latents.shape[-2], pos_latents.shape[-1]))
    loss_weight_map, region_stats = build_region_loss_weight_map(
        masks,
        mask_region_weight=args.mask_region_weight,
        boundary_region_weight=args.boundary_region_weight,
        outside_region_weight=args.outside_region_weight,
    )
    brushnet_cond = rearrange(torch.cat([cond_latents, masks], dim=2), "b f c h w -> (b f) c h w")
    noise = torch.randn_like(pos_latents)
    bsz = pos_latents.shape[0] // args.nframes
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
    timesteps_expanded = timesteps.repeat_interleave(args.nframes, dim=0)
    noisy_pos = noise_scheduler.add_noise(pos_latents, noise, timesteps_expanded)
    noisy_neg = noise_scheduler.add_noise(neg_latents, noise, timesteps_expanded)
    encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]
    encoder_hidden_states_expanded = rearrange(repeat(encoder_hidden_states, "b c d -> b t c d", t=args.nframes), "b t c d -> (b t) c d")
    flow_condition, flow_stats = build_flow_condition(batch, "exp19b")
    return {
        "noisy_pos": noisy_pos,
        "noisy_neg": noisy_neg,
        "timesteps_expanded": timesteps_expanded,
        "timesteps": timesteps,
        "encoder_hidden_states_2d": encoder_hidden_states_expanded.to(dtype=weight_dtype),
        "encoder_hidden_states_motion": encoder_hidden_states.to(dtype=weight_dtype),
        "brushnet_cond": brushnet_cond,
        "noise": noise,
        "loss_weight_map": loss_weight_map,
        "region_stats": region_stats,
        "flow_condition": flow_condition.to(device=device, dtype=weight_dtype),
        "flow_stats": flow_stats,
    }


def compute_step_loss(batch, models, args, device, weight_dtype, train: bool):
    unet, brushnet = models["unet"], models["brushnet"]
    tensors = prepare_step_tensors(batch, models, args, device, weight_dtype)
    with torch.no_grad():
        ref_pos = forward_pair_member(
            brushnet,
            unet,
            tensors["noisy_pos"],
            tensors["timesteps_expanded"],
            tensors["timesteps"],
            tensors["encoder_hidden_states_2d"],
            tensors["encoder_hidden_states_motion"],
            tensors["brushnet_cond"],
            weight_dtype,
            args.nframes,
            tensors["flow_condition"],
            adapter_enabled=False,
        )
        ref_neg = forward_pair_member(
            brushnet,
            unet,
            tensors["noisy_neg"],
            tensors["timesteps_expanded"],
            tensors["timesteps"],
            tensors["encoder_hidden_states_2d"],
            tensors["encoder_hidden_states_motion"],
            tensors["brushnet_cond"],
            weight_dtype,
            args.nframes,
            tensors["flow_condition"],
            adapter_enabled=False,
        )
        ref_pred = torch.cat([ref_pos, ref_neg], dim=0)
    policy_pos = forward_pair_member(
        brushnet,
        unet,
        tensors["noisy_pos"],
        tensors["timesteps_expanded"],
        tensors["timesteps"],
        tensors["encoder_hidden_states_2d"],
        tensors["encoder_hidden_states_motion"],
        tensors["brushnet_cond"],
        weight_dtype,
        args.nframes,
        tensors["flow_condition"],
        adapter_enabled=True,
    )
    policy_neg = forward_pair_member(
        brushnet,
        unet,
        tensors["noisy_neg"],
        tensors["timesteps_expanded"],
        tensors["timesteps"],
        tensors["encoder_hidden_states_2d"],
        tensors["encoder_hidden_states_motion"],
        tensors["brushnet_cond"],
        weight_dtype,
        args.nframes,
        tensors["flow_condition"],
        adapter_enabled=True,
    )
    model_pred = torch.cat([policy_pos, policy_neg], dim=0)
    loss, diagnostics = compute_dpo_loss(
        model_pred,
        ref_pred,
        tensors["noise"],
        loss_weight_map=tensors["loss_weight_map"],
        loss_region_mode="region",
        region_stats=tensors["region_stats"],
        gap_normalization="log_ratio",
        gap_eps=1e-6,
        lose_gap_clip_tau=args.lose_gap_clip_tau,
        beta_dpo=args.beta_dpo,
        sft_reg_weight=0.0,
        lose_gap_weight=args.lose_gap_weight,
        winner_abs_reg_weight=args.winner_abs_reg_weight,
        winner_gap_reg_weight=args.winner_gap_reg_weight,
        winner_gap_reg_margin=args.winner_gap_reg_margin,
        winner_gap_reg_mode="relu",
        nframes=args.nframes,
    )
    return loss, diagnostics, tensors, model_pred, ref_pred


def load_models(args, device, weight_dtype):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path, subfolder="tokenizer", use_fast=False)
    text_encoder_cls = import_model_class_from_model_name_or_path(args.base_model_name_or_path, None)
    text_encoder = text_encoder_cls.from_pretrained(args.base_model_name_or_path, subfolder="text_encoder").to(device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder="vae").to(device, dtype=torch.float32)
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model_name_or_path, subfolder="scheduler")
    unet_base = UNetMotionModel.from_pretrained(args.exp11_stage2_weights, subfolder="unet_main").to(device, dtype=weight_dtype)
    brushnet = BrushNetModel.from_pretrained(args.exp11_stage2_weights, subfolder="brushnet").to(device, dtype=weight_dtype)
    for module in (text_encoder, vae, unet_base, brushnet):
        module.requires_grad_(False)
        module.eval()
    target_modules = tuple(x.strip() for x in args.target_modules.split(",") if x.strip())
    unet_wrapper = UNetMotionFlowAdapterWrapper(unet_base, target_module_names=target_modules, gate_mode="boundary").to(device)
    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "noise_scheduler": noise_scheduler,
        "unet": unet_wrapper,
        "unet_base": unet_base,
        "brushnet": brushnet,
    }


def save_adapter(output_dir: Path, unet_wrapper: UNetMotionFlowAdapterWrapper, step: int | str) -> None:
    save_dir = output_dir / (f"checkpoint-{step}" if isinstance(step, int) else str(step))
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_state_dict": unet_wrapper.adapter_state_dict(),
            "target_module_names": unet_wrapper.target_module_names,
            "hook_shapes": [shape.__dict__ for shape in unet_wrapper.hook_shapes],
        },
        save_dir / "flow_adapter.pt",
    )


def write_preflight_report(path: Path, status: str, details: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Exp19 Isolated Wrapper Preflight", "", f"status: `{status}`", ""]
    for key, value in details.items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def diag_row(step, diagnostics, flow_stats, unet, adapter_grad, base_grad, flow_feat_norm, lr):
    residual_norm = sum(unet.last_residual_norms.values()) if unet.last_residual_norms else 0.0
    row = {
        "step": step,
        "variant_name": "exp19b_boundary_gated_flow_adapter",
        "total_loss": diagnostics.get("total_loss"),
        "dpo_loss": diagnostics.get("dpo_loss"),
        "warp_loss": None,
        "m_w": diagnostics.get("mse_w"),
        "m_l": diagnostics.get("mse_l"),
        "m_w_ref": diagnostics.get("ref_mse_w"),
        "m_l_ref": diagnostics.get("ref_mse_l"),
        "raw_win_gap": diagnostics.get("raw_win_gap"),
        "raw_lose_gap": diagnostics.get("raw_lose_gap"),
        "norm_win_gap": diagnostics.get("norm_win_gap"),
        "norm_lose_gap": diagnostics.get("norm_lose_gap"),
        "norm_lose_gap_clipped": diagnostics.get("norm_lose_gap_clipped"),
        "winner_abs_reg": diagnostics.get("winner_abs_reg"),
        "winner_gap_reg": diagnostics.get("winner_gap_reg"),
        "loser_dominant_ratio": diagnostics.get("loser_degrade_ratio"),
        "grad_norm": adapter_grad,
        "adapter_grad_norm": adapter_grad,
        "base_grad_norm": base_grad,
        "flow_feat_norm": flow_feat_norm,
        "adapter_residual_norm": residual_norm,
        "adapter_to_base_ratio": None,
        "lr": lr,
    }
    alpha_values = list(unet.alpha_values().values())
    for idx in range(3):
        row[f"alpha_scale_{idx + 1}"] = alpha_values[idx] if idx < len(alpha_values) else None
    row.update(flow_stats)
    row.update(unet.last_gate_stats)
    return row


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = resolve_dtype(args.mixed_precision)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diag_csv = Path(args.diag_csv or output_dir / "dpo_diagnostics.csv")

    models = load_models(args, device, weight_dtype)
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
    dataset = Exp19FlowManifestDataset(dataset_args, models["tokenizer"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0, collate_fn=collate_exp19)
    first_batch = move_batch(next(iter(dataloader)), device)

    # Build projectors and validate zero-init equality.
    with torch.no_grad():
        loss0, diag0, tensors0, model_pred0, ref_pred0 = compute_step_loss(first_batch, models, args, device, weight_dtype, train=False)
        zero_init_diff = float((model_pred0 - ref_pred0).detach().abs().mean().float().cpu())
    adapter_params = list(models["unet"].adapter_parameters())
    if not adapter_params:
        raise RuntimeError("Exp19 wrapper did not create adapter parameters during preflight forward")

    loss, diagnostics, tensors, _model_pred, _ref_pred = compute_step_loss(first_batch, models, args, device, weight_dtype, train=True)
    loss.backward()
    adapter_gn = grad_norm(adapter_params)
    base_gn = grad_norm(models["unet_base"].parameters())
    preflight_details = {
        "zero_init_mean_abs_diff": zero_init_diff,
        "preflight_loss": float(loss.detach().float().cpu()),
        "adapter_grad_norm": adapter_gn,
        "base_grad_norm": base_gn,
        "gate_stats": json.dumps(models["unet"].last_gate_stats, ensure_ascii=False),
        "alpha_values": json.dumps(models["unet"].alpha_values(), ensure_ascii=False),
        "hook_shapes": json.dumps([shape.__dict__ for shape in models["unet"].hook_shapes], ensure_ascii=False),
    }
    preflight_ok = (
        math.isfinite(float(loss.detach().float().cpu()))
        and zero_init_diff < (5e-4 if weight_dtype != torch.float32 else 1e-5)
        and adapter_gn > 0
        and base_gn == 0
    )
    write_preflight_report(Path(args.report_path), "PASS" if preflight_ok else "FAILED", preflight_details)
    for p in adapter_params:
        p.grad = None
    if not preflight_ok:
        return 3
    save_adapter(output_dir, models["unet"], "preflight_adapter")
    if args.preflight_only:
        return 0

    optimizer = torch.optim.AdamW(adapter_params, lr=args.learning_rate, weight_decay=0.0)
    global_step = 0
    pbar = tqdm(total=args.max_train_steps, desc="Exp19b")
    while global_step < args.max_train_steps:
        for batch in dataloader:
            batch = move_batch(batch, device)
            loss, diagnostics, tensors, _model_pred, _ref_pred = compute_step_loss(batch, models, args, device, weight_dtype, train=True)
            loss.backward()
            adapter_gn = grad_norm(adapter_params)
            base_gn = grad_norm(models["unet_base"].parameters())
            torch.nn.utils.clip_grad_norm_(adapter_params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            pbar.update(1)
            if global_step == 1 or global_step % args.dpo_diag_interval == 0:
                row = diag_row(
                    global_step,
                    diagnostics,
                    tensors["flow_stats"],
                    models["unet"],
                    adapter_gn,
                    base_gn,
                    float(tensors["flow_condition"].detach().float().norm().cpu()),
                    args.learning_rate,
                )
                append_exp19_diag_csv(diag_csv, row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
            if global_step % args.checkpointing_steps == 0:
                save_adapter(output_dir, models["unet"], global_step)
            del batch, loss, diagnostics, tensors
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
