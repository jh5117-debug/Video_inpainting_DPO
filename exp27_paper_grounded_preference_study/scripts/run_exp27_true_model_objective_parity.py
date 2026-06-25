#!/usr/bin/env python3
"""True DiffuEraser policy/reference objective parity gates for Exp27.

This is a short validation runner, not a training study.  It loads real
DiffuEraser Stage1 policy/reference models, real generated-loser preference
rows, VAE latents, shared noise, and fixed timesteps.  The outputs are used to
validate SDPO and Linear-DPO objective math on real model predictions.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import torch
from einops import rearrange, repeat
from transformers import AutoTokenizer
from diffusers import AutoencoderKL, DDPMScheduler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.brushnet_CA import BrushNetModel  # noqa: E402
from libs.unet_2d_condition import UNet2DConditionModel  # noqa: E402
from libs.unet_motion_model import UNetMotionModel  # noqa: E402
from training.dpo.dataset.generated_loser_manifest_dataset import GeneratedLoserManifestDataset  # noqa: E402
from training.dpo.train_stage1 import (  # noqa: E402
    collate_fn,
    forward_stage1_pair_member,
    import_model_class_from_model_name_or_path,
)
from exp27_paper_grounded_preference_study.code.official_parity import (  # noqa: E402
    ema_update_tensor,
    exp27_sdpo_safe_lambda,
    linear_dpo_clip_ratio,
    linear_dpo_loss,
    load_official_sdpo_lambda,
)


@dataclass
class Paths:
    base_model: str
    vae: str
    sft_weights: str
    exp11_stage1: str
    manifest: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, default=Path("reports"))
    p.add_argument("--base-model", default="/mnt/nas/hj/weights/stable-diffusion-v1-5")
    p.add_argument("--vae", default="/mnt/nas/hj/weights/sd-vae-ft-mse")
    p.add_argument("--sft-weights", default="/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000")
    p.add_argument(
        "--exp11-stage1",
        default="/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/"
        "20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s1_2000_davis_pai/last_weights",
    )
    p.add_argument(
        "--manifest",
        default="/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/"
        "exp08c_youtubevos_gtwin_d3comp_lose_fixed_pai/manifests/selected_primary_comp.gtwin.pai_paths.jsonl",
    )
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--timesteps", default="50,250,500,750")
    p.add_argument("--nframes", type=int, default=16)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260625)
    p.add_argument("--mu", type=float, default=0.37)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--states", default="S0,S1", help="Comma-separated subset of S0,S1.")
    p.add_argument("--max-tiny-step-cases", type=int, default=0)
    p.add_argument("--tiny-step-lr", type=float, default=1e-7)
    p.add_argument("--tiny-step-min-per-class", type=int, default=4)
    return p.parse_args()


def sha256_file(path: Path, max_bytes: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = max_bytes
        while remaining > 0:
            chunk = f.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def inventory_dir(path: Path) -> dict:
    files = sorted(p for p in path.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    sample = []
    for pth in files[:20]:
        sample.append({"path": str(pth.relative_to(path)), "size": pth.stat().st_size})
    return {"path": str(path), "exists": path.exists(), "file_count": len(files), "total_size": total, "sample_files": sample}


def dtype_from_name(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float32


def load_unet2d(weights: str, base_model: str) -> UNet2DConditionModel:
    cfg = Path(weights) / "unet_main" / "config.json"
    is_motion = False
    if cfg.exists():
        data = json.loads(cfg.read_text(encoding="utf-8"))
        is_motion = data.get("_class_name") == "UNetMotionModel"
    if not is_motion:
        return UNet2DConditionModel.from_pretrained(weights, subfolder="unet_main")
    motion = UNetMotionModel.from_pretrained(weights, subfolder="unet_main")
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    unet.conv_in.load_state_dict(motion.conv_in.state_dict())
    unet.time_proj.load_state_dict(motion.time_proj.state_dict())
    unet.time_embedding.load_state_dict(motion.time_embedding.state_dict())
    for idx, block in enumerate(motion.down_blocks):
        unet.down_blocks[idx].resnets.load_state_dict(block.resnets.state_dict())
        if hasattr(unet.down_blocks[idx], "attentions") and hasattr(block, "attentions"):
            unet.down_blocks[idx].attentions.load_state_dict(block.attentions.state_dict())
        if unet.down_blocks[idx].downsamplers and block.downsamplers:
            unet.down_blocks[idx].downsamplers.load_state_dict(block.downsamplers.state_dict())
    for idx, block in enumerate(motion.up_blocks):
        unet.up_blocks[idx].resnets.load_state_dict(block.resnets.state_dict())
        if hasattr(unet.up_blocks[idx], "attentions") and hasattr(block, "attentions"):
            unet.up_blocks[idx].attentions.load_state_dict(block.attentions.state_dict())
        if unet.up_blocks[idx].upsamplers and block.upsamplers:
            unet.up_blocks[idx].upsamplers.load_state_dict(block.upsamplers.state_dict())
    unet.mid_block.resnets.load_state_dict(motion.mid_block.resnets.state_dict())
    unet.mid_block.attentions.load_state_dict(motion.mid_block.attentions.state_dict())
    if motion.conv_norm_out is not None:
        unet.conv_norm_out.load_state_dict(motion.conv_norm_out.state_dict())
    if hasattr(motion, "conv_act") and motion.conv_act is not None:
        unet.conv_act.load_state_dict(motion.conv_act.state_dict())
    unet.conv_out.load_state_dict(motion.conv_out.state_dict())
    del motion
    return unet


def make_dataset(args: argparse.Namespace, tokenizer) -> GeneratedLoserManifestDataset:
    ds_args = SimpleNamespace(
        preference_manifest=args.manifest,
        nframes=args.nframes,
        train_height=args.height,
        train_width=args.width,
        resolution=args.width,
        train_mask_mode="partial",
        mask_from_manifest=True,
        videodpo_full_mask_value=0.0,
        max_resample_attempts=128,
        proportion_empty_prompts=0.0,
    )
    return GeneratedLoserManifestDataset(ds_args, tokenizer)


def encode_batch(batch: dict, vae, noise_scheduler, text_encoder, nframes: int, timestep: int, dtype: torch.dtype, seed: int, device: torch.device) -> dict:
    with torch.no_grad():
        pos_latents = vae.encode(rearrange(batch["pixel_values_pos"], "b f c h w -> (b f) c h w").to(device=device, dtype=torch.float32)).latent_dist.sample()
        pos_latents = (pos_latents * vae.config.scaling_factor).to(dtype=dtype)
        neg_latents = vae.encode(rearrange(batch["pixel_values_neg"], "b f c h w -> (b f) c h w").to(device=device, dtype=torch.float32)).latent_dist.sample()
        neg_latents = (neg_latents * vae.config.scaling_factor).to(dtype=dtype)
        cond_latents = vae.encode(rearrange(batch["conditioning_pixel_values"], "b f c h w -> (b f) c h w").to(device=device, dtype=torch.float32)).latent_dist.sample()
        cond_latents = (cond_latents * vae.config.scaling_factor).to(dtype=dtype)
        cond_latents = rearrange(cond_latents, "(b f) c h w -> b f c h w", b=batch["conditioning_pixel_values"].shape[0])
        masks = torch.nn.functional.interpolate(
            batch["masks"].to(device=device, dtype=dtype),
            size=(1, pos_latents.shape[-2], pos_latents.shape[-1]),
        )
        brushnet_cond = rearrange(torch.cat([cond_latents, masks], 2), "b f c h w -> (b f) c h w")
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        noise = torch.randn(pos_latents.shape, device=device, dtype=dtype, generator=gen)
        bsz = pos_latents.shape[0] // nframes
        timesteps = torch.full((bsz,), int(timestep), device=device, dtype=torch.long)
        timesteps_expanded = timesteps.repeat_interleave(nframes, dim=0)
        noisy_pos = noise_scheduler.add_noise(pos_latents, noise, timesteps_expanded)
        noisy_neg = noise_scheduler.add_noise(neg_latents, noise, timesteps_expanded)
        encoder_hidden_states = text_encoder(batch["input_ids"].to(device), return_dict=False)[0]
        encoder_hidden_states = rearrange(repeat(encoder_hidden_states, "b c d -> b t c d", t=nframes), "b t c d -> (b t) c d")
    return {
        "noisy_pos": noisy_pos,
        "noisy_neg": noisy_neg,
        "noise": noise,
        "timesteps_expanded": timesteps_expanded,
        "encoder_hidden_states": encoder_hidden_states,
        "brushnet_cond": brushnet_cond,
    }


def forward_pair(policy, ref, tensors: dict, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    unet, brushnet = policy
    unet_ref, brushnet_ref = ref
    noisy_pos = tensors["noisy_pos"]
    noisy_neg = tensors["noisy_neg"]
    ts = tensors["timesteps_expanded"]
    enc = tensors["encoder_hidden_states"].to(dtype=dtype)
    cond = tensors["brushnet_cond"].to(dtype=dtype)
    with torch.no_grad():
        ref_pos = forward_stage1_pair_member(brushnet_ref, unet_ref, noisy_pos, ts, enc, cond, dtype)
        ref_neg = forward_stage1_pair_member(brushnet_ref, unet_ref, noisy_neg, ts, enc, cond, dtype)
        ref_pred = torch.cat([ref_pos, ref_neg], dim=0)
    model_pos = forward_stage1_pair_member(brushnet, unet, noisy_pos, ts, enc, cond, dtype)
    model_neg = forward_stage1_pair_member(brushnet, unet, noisy_neg, ts, enc, cond, dtype)
    return torch.cat([model_pos, model_neg], dim=0), ref_pred


def mse_losses(pred: torch.Tensor, noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target = noise.repeat(2, 1, 1, 1)
    losses = (pred.float() - target.float()).pow(2).mean(dim=(1, 2, 3))
    return losses.chunk(2)


def sdpo_loss_with_lambda(pred: torch.Tensor, noise: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    win, lose = mse_losses(pred, noise)
    return (win - lam.float() * lose).mean()


def margin_from_losses(win: torch.Tensor, lose: torch.Tensor, ref_win: torch.Tensor, ref_lose: torch.Tensor) -> torch.Tensor:
    return (lose - win) - (ref_lose - ref_win)


def grad_from_loss(pred: torch.Tensor, noise: torch.Tensor, official_lambda, mu: float, mode: str) -> tuple[torch.Tensor, float, torch.Tensor]:
    clone = pred.detach().clone().requires_grad_(True)
    target = noise.repeat(2, 1, 1, 1).detach()
    lam = official_lambda(clone, target, mu=mu, eps=1e-8, max_lambda=1.0) if mode == "official" else exp27_sdpo_safe_lambda(clone, target, mu=mu, eps=1e-8, max_lambda=1.0)
    loss = sdpo_loss_with_lambda(clone, noise.detach(), lam)
    grad = torch.autograd.grad(loss, clone)[0]
    return grad.detach(), float(loss.detach().cpu()), lam.detach()


def representative_param_grads(model_pred: torch.Tensor, loss: torch.Tensor, policy: tuple, names: Iterable[str]) -> dict:
    params = []
    name_map = {}
    for prefix, module in (("unet", policy[0]), ("brushnet", policy[1])):
        for name, param in module.named_parameters():
            full = f"{prefix}.{name}"
            if any(token in full for token in names) and param.requires_grad:
                params.append(param)
                name_map[id(param)] = full
                if len(params) >= 8:
                    break
        if len(params) >= 8:
            break
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    out = {}
    for p, g in zip(params, grads):
        out[name_map[id(p)]] = None if g is None else float(g.detach().float().norm().cpu())
    return out


def policy_param_delta_step(policy: tuple, lr: float) -> tuple[float, float, int]:
    """Apply a tiny SGD-like step and report aggregate delta norms.

    This is intentionally not a training optimizer.  It exists only to check
    whether the real SDPO output gradient produces the predicted direction on
    an actual DiffuEraser policy parameterization for selected cases.
    """

    delta_sq = 0.0
    grad_sq = 0.0
    updated = 0
    with torch.no_grad():
        for module in policy:
            for param in module.parameters():
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                step = (-float(lr) * grad).to(dtype=param.dtype)
                if not torch.isfinite(step).all():
                    raise FloatingPointError("non-finite tiny-step update")
                param.add_(step)
                delta_sq += float(step.float().pow(2).sum().cpu())
                grad_sq += float(grad.float().pow(2).sum().cpu())
                updated += 1
    return math.sqrt(delta_sq), math.sqrt(grad_sq), updated


def load_state(state: str, paths: Paths, device: torch.device, dtype: torch.dtype):
    policy_path = paths.sft_weights if state == "S0" else paths.exp11_stage1
    ref_path = paths.sft_weights
    unet = load_unet2d(policy_path, paths.base_model).to(device=device, dtype=dtype)
    brushnet = BrushNetModel.from_pretrained(policy_path, subfolder="brushnet").to(device=device, dtype=dtype)
    unet_ref = load_unet2d(ref_path, paths.base_model).to(device=device, dtype=dtype)
    brushnet_ref = BrushNetModel.from_pretrained(ref_path, subfolder="brushnet").to(device=device, dtype=dtype)
    unet.train()
    brushnet.train()
    unet_ref.eval().requires_grad_(False)
    brushnet_ref.eval().requires_grad_(False)
    return (unet, brushnet), (unet_ref, brushnet_ref), {"policy_path": policy_path, "ref_path": ref_path}


def clear_models(*models) -> None:
    for obj in models:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def select_tiny_cases(rows: list[dict], max_cases: int, min_per_class: int) -> list[dict]:
    if max_cases <= 0:
        return []
    s1_rows = [r for r in rows if r.get("state") == "S1" and r.get("finite")]
    lt1 = sorted((r for r in s1_rows if r.get("lambda_lt_1")), key=lambda r: float(r["lambda_adapter"]))
    eq1 = sorted((r for r in s1_rows if not r.get("lambda_lt_1")), key=lambda r: (int(r["row_index"]), int(r["timestep"])))
    per_class = min(min_per_class, max_cases // 2)
    selected = lt1[:per_class] + eq1[:per_class]
    if len(selected) < max_cases:
        seen = {(r["row_index"], r["timestep"]) for r in selected}
        for row in s1_rows:
            key = (row["row_index"], row["timestep"])
            if key not in seen:
                selected.append(row)
                seen.add(key)
            if len(selected) >= max_cases:
                break
    return selected[:max_cases]


def run_tiny_step_cases(
    args: argparse.Namespace,
    selected: list[dict],
    dataset,
    vae,
    noise_scheduler,
    text_encoder,
    paths: Paths,
    official_lambda,
    device,
    dtype,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    if not selected:
        return rows, {"status": "not_requested", "records": 0}
    for case_idx, case in enumerate(selected):
        policy, ref, identity = load_state("S1", paths, device, dtype)
        row_idx = int(case["row_index"])
        timestep = int(case["timestep"])
        seed = int(args.seed + row_idx * 1009 + [50, 250, 500, 750, timestep].index(timestep) * 9176 + 31) if timestep in [50, 250, 500, 750] else int(args.seed + row_idx * 1009 + timestep * 17 + 31)
        sample = dataset[row_idx]
        batch = collate_fn([sample])
        tensors = encode_batch(batch, vae, noise_scheduler, text_encoder, args.nframes, timestep, dtype, seed, device)
        model_pred, ref_pred = forward_pair(policy, ref, tensors, dtype)
        noise = tensors["noise"]
        target = noise.repeat(2, 1, 1, 1)
        lam = official_lambda(model_pred, target, mu=args.mu, eps=1e-8, max_lambda=1.0)
        win_pre, lose_pre = mse_losses(model_pred, noise)
        ref_win_pre, ref_lose_pre = mse_losses(ref_pred, noise)
        margin_pre = margin_from_losses(win_pre, lose_pre, ref_win_pre, ref_lose_pre)
        loss = sdpo_loss_with_lambda(model_pred, noise, lam)
        loss.backward()
        ref_grad_norm = 0.0
        for module in ref:
            for param in module.parameters():
                if param.grad is not None:
                    ref_grad_norm += float(param.grad.detach().float().pow(2).sum().cpu())
        param_delta_norm, param_grad_norm, updated_params = policy_param_delta_step(policy, args.tiny_step_lr)
        for module in policy + ref:
            module.zero_grad(set_to_none=True)
        with torch.no_grad():
            post_pred, post_ref_pred = forward_pair(policy, ref, tensors, dtype)
            win_post, lose_post = mse_losses(post_pred, noise)
            ref_win_post, ref_lose_post = mse_losses(post_ref_pred, noise)
            margin_post = margin_from_losses(win_post, lose_post, ref_win_post, ref_lose_post)
        rows.append(
            {
                "case_index": case_idx,
                "sample_id": sample.get("sample_id"),
                "row_index": row_idx,
                "timestep": timestep,
                "lambda_adapter_scan": float(case["lambda_adapter"]),
                "lambda_official_tiny": float(lam.float().mean().detach().cpu()),
                "lambda_lt_1": bool(case["lambda_lt_1"]),
                "tiny_step_lr": float(args.tiny_step_lr),
                "loss_pre": float(loss.detach().cpu()),
                "winner_loss_pre": float(win_pre.mean().detach().cpu()),
                "winner_loss_post": float(win_post.mean().detach().cpu()),
                "winner_loss_change": float((win_post - win_pre).mean().detach().cpu()),
                "loser_loss_pre": float(lose_pre.mean().detach().cpu()),
                "loser_loss_post": float(lose_post.mean().detach().cpu()),
                "loser_loss_change": float((lose_post - lose_pre).mean().detach().cpu()),
                "margin_pre": float(margin_pre.mean().detach().cpu()),
                "margin_post": float(margin_post.mean().detach().cpu()),
                "margin_change": float((margin_post - margin_pre).mean().detach().cpu()),
                "param_delta_norm": param_delta_norm,
                "param_grad_norm": param_grad_norm,
                "updated_param_tensors": updated_params,
                "reference_grad_norm": math.sqrt(ref_grad_norm),
                "reference_delta_norm": 0.0,
                "policy_path": identity["policy_path"],
                "ref_path": identity["ref_path"],
                "finite": bool(torch.isfinite(post_pred).all().item() and torch.isfinite(post_ref_pred).all().item()),
            }
        )
        del tensors, model_pred, ref_pred, post_pred, post_ref_pred, noise, target, lam, loss
        clear_models(policy, ref)
    lt1_count = sum(1 for r in rows if r["lambda_lt_1"])
    eq1_count = sum(1 for r in rows if not r["lambda_lt_1"])
    finite = all(r["finite"] for r in rows)
    summary = {
        "status": "passed" if finite and lt1_count >= min(args.tiny_step_min_per_class, args.max_tiny_step_cases // 2) and eq1_count >= min(args.tiny_step_min_per_class, args.max_tiny_step_cases // 2) else "partial_or_failed",
        "records": len(rows),
        "lambda_lt_1_cases": lt1_count,
        "lambda_eq_1_cases": eq1_count,
        "winner_loss_change_mean": float(sum(r["winner_loss_change"] for r in rows) / len(rows)) if rows else None,
        "margin_change_mean": float(sum(r["margin_change"] for r in rows) / len(rows)) if rows else None,
        "max_reference_grad_norm": max((r["reference_grad_norm"] for r in rows), default=None),
        "max_param_delta_norm": max((r["param_delta_norm"] for r in rows), default=None),
    }
    return rows, summary


def run_state(args: argparse.Namespace, state: str, dataset, vae, noise_scheduler, text_encoder, paths: Paths, official_lambda, device, dtype, timesteps: list[int]) -> tuple[list[dict], list[dict], dict]:
    policy, ref, identity = load_state(state, paths, device, dtype)
    rows_out: list[dict] = []
    tiny_candidates: list[dict] = []
    for row_idx in range(min(args.rows, len(dataset))):
        sample = dataset[row_idx]
        batch = collate_fn([sample])
        for t_idx, timestep in enumerate(timesteps):
            for module in policy + ref:
                module.zero_grad(set_to_none=True)
            seed = int(args.seed + row_idx * 1009 + t_idx * 9176 + (0 if state == "S0" else 31))
            tensors = encode_batch(batch, vae, noise_scheduler, text_encoder, args.nframes, timestep, dtype, seed, device)
            model_pred, ref_pred = forward_pair(policy, ref, tensors, dtype)
            noise = tensors["noise"]
            target = noise.repeat(2, 1, 1, 1)
            lam_adapter = exp27_sdpo_safe_lambda(model_pred, target, mu=args.mu, eps=1e-8, max_lambda=1.0)
            lam_official = official_lambda(model_pred, target, mu=args.mu, eps=1e-8, max_lambda=1.0)
            grad_a, loss_a, _ = grad_from_loss(model_pred, noise, official_lambda, args.mu, "adapter")
            grad_o, loss_o, _ = grad_from_loss(model_pred, noise, official_lambda, args.mu, "official")
            grad_cos = torch.nn.functional.cosine_similarity(grad_a.flatten().float(), grad_o.flatten().float(), dim=0)
            win_m, lose_m = mse_losses(model_pred, noise)
            win_r, lose_r = mse_losses(ref_pred, noise)
            sdpo_loss = sdpo_loss_with_lambda(model_pred, noise, lam_adapter)
            param_grads = representative_param_grads(model_pred, sdpo_loss, policy, ("conv_in", "down_blocks.0", "mid_block", "conv_out"))
            lambda_value = float(lam_adapter.float().mean().cpu())
            row = {
                "state": state,
                "sample_id": sample.get("sample_id"),
                "row_index": row_idx,
                "timestep": int(timestep),
                "lambda_adapter": lambda_value,
                "lambda_official": float(lam_official.float().mean().cpu()),
                "lambda_abs_diff": float((lam_adapter.float() - lam_official.float()).abs().max().cpu()),
                "lambda_lt_1": lambda_value < 0.999999,
                "adapter_loss": loss_a,
                "official_loss": loss_o,
                "loss_abs_diff": abs(loss_a - loss_o),
                "output_grad_cosine": float(grad_cos.detach().cpu()),
                "winner_policy_mse": float(win_m.mean().detach().cpu()),
                "loser_policy_mse": float(lose_m.mean().detach().cpu()),
                "winner_ref_mse": float(win_r.mean().detach().cpu()),
                "loser_ref_mse": float(lose_r.mean().detach().cpu()),
                "win_gap": float((win_m - win_r).mean().detach().cpu()),
                "lose_gap": float((lose_m - lose_r).mean().detach().cpu()),
                "param_grad_json": json.dumps(param_grads, sort_keys=True),
                "finite": bool(torch.isfinite(model_pred).all().item() and torch.isfinite(ref_pred).all().item()),
            }
            rows_out.append(row)
            if state == "S1":
                tiny_candidates.append(row)
            del tensors, model_pred, ref_pred, noise, target, lam_adapter, lam_official, grad_a, grad_o, sdpo_loss
            gc.collect()
            torch.cuda.empty_cache()
    summary = {"state": state, **identity}
    clear_models(policy, ref)
    return rows_out, tiny_candidates, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def linear_true_model_probe(scan_rows: list[dict]) -> tuple[list[dict], dict]:
    rows = []
    for idx, row in enumerate(scan_rows):
        mw = torch.tensor([row["winner_policy_mse"]], dtype=torch.float32, requires_grad=True)
        ml = torch.tensor([row["loser_policy_mse"]], dtype=torch.float32, requires_grad=True)
        rw = torch.tensor([row["winner_ref_mse"]], dtype=torch.float32)
        rl = torch.tensor([row["loser_ref_mse"]], dtype=torch.float32)
        utility = linear_dpo_clip_ratio(mw.detach(), ml.detach(), beta_dpo=5000.0, eta_dpo=0.01)
        loss = linear_dpo_loss(mw, ml, rw, rl, beta_dpo=5000.0, eta_dpo=0.01)
        loss.backward()
        ema = torch.tensor([row["winner_policy_mse"], row["loser_policy_mse"]], dtype=torch.float32)
        model = ema + torch.tensor([0.001, -0.001], dtype=torch.float32)
        ema_updated = ema_update_tensor(ema.clone(), model, decay=0.9999)
        rows.append({
            "source_state": row["state"],
            "sample_id": row["sample_id"],
            "timestep": row["timestep"],
            "utility": float(utility.item()),
            "loss": float(loss.detach().item()),
            "grad_winner_loss": float(mw.grad.detach().item()),
            "grad_loser_loss": float(ml.grad.detach().item()),
            "ema_drift": float((ema_updated - ema).abs().max().item()),
        })
        if idx >= 31:
            break
    summary = {
        "status": "passed" if rows and all(math.isfinite(float(r["loss"])) for r in rows) else "failed",
        "records": len(rows),
        "utility_min": min((r["utility"] for r in rows), default=None),
        "utility_max": max((r["utility"] for r in rows), default=None),
        "ema_max_drift": max((r["ema_drift"] for r in rows), default=None),
    }
    return rows, summary


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for true-model Exp27 parity")
    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)
    paths = Paths(args.base_model, args.vae, args.sft_weights, args.exp11_stage1, args.manifest)
    timesteps = [int(x) for x in args.timesteps.split(",") if x.strip()]
    text_cls = import_model_class_from_model_name_or_path(args.base_model, None)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, subfolder="tokenizer", use_fast=False)
    text_encoder = text_cls.from_pretrained(args.base_model, subfolder="text_encoder").to(device=device, dtype=dtype).eval().requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(args.vae, subfolder="vae").to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    dataset = make_dataset(args, tokenizer)
    official_lambda = load_official_sdpo_lambda()
    inventories = {
        "paths": asdict(paths),
        "base_model": inventory_dir(Path(args.base_model)),
        "vae": inventory_dir(Path(args.vae)),
        "sft_weights": inventory_dir(Path(args.sft_weights)),
        "exp11_stage1": inventory_dir(Path(args.exp11_stage1)),
        "manifest_exists": Path(args.manifest).exists(),
        "manifest_size": Path(args.manifest).stat().st_size if Path(args.manifest).exists() else None,
        "rows_requested": args.rows,
        "timesteps": timesteps,
        "dtype": args.dtype,
        "device": str(device),
    }
    (args.output_dir / "true_model_identity.json").write_text(json.dumps(inventories, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    all_rows: list[dict] = []
    state_summaries = []
    for state in [s.strip() for s in args.states.split(",") if s.strip()]:
        state_rows, _tiny, state_summary = run_state(args, state, dataset, vae, noise_scheduler, text_encoder, paths, official_lambda, device, dtype, timesteps)
        all_rows.extend(state_rows)
        state_summaries.append(state_summary)

    write_csv(args.output_dir / "sdpo_true_model_distribution_scan.csv", all_rows)
    tiny_rows: list[dict] = []
    tiny_summary = {"status": "not_requested", "records": 0}
    if args.max_tiny_step_cases > 0 and "S1" in [s.strip() for s in args.states.split(",") if s.strip()]:
        selected_tiny = select_tiny_cases(all_rows, args.max_tiny_step_cases, args.tiny_step_min_per_class)
        tiny_rows, tiny_summary = run_tiny_step_cases(
            args,
            selected_tiny,
            dataset,
            vae,
            noise_scheduler,
            text_encoder,
            paths,
            official_lambda,
            device,
            dtype,
        )
    write_csv(args.output_dir / "sdpo_true_model_tiny_step_cases.csv", tiny_rows)
    linear_rows, linear_summary = linear_true_model_probe(all_rows)
    write_csv(args.output_dir / "linear_true_model_probe.csv", linear_rows)

    ok = [r for r in all_rows if r.get("finite")]
    lambda_diffs = [float(r["lambda_abs_diff"]) for r in ok]
    loss_diffs = [float(r["loss_abs_diff"]) for r in ok]
    grad_cos = [float(r["output_grad_cosine"]) for r in ok]
    s1 = [r for r in ok if r["state"] == "S1"]
    lt1 = [r for r in s1 if r["lambda_lt_1"]]
    eq1 = [r for r in s1 if not r["lambda_lt_1"]]
    base_status = ok and max(lambda_diffs, default=1.0) <= 1e-6 and max(loss_diffs, default=1.0) <= 1e-6 and min(grad_cos, default=0.0) >= 0.999999
    tiny_required = args.max_tiny_step_cases > 0
    tiny_ok = (not tiny_required) or tiny_summary.get("status") == "passed"
    summary = {
        "status": "TRUE_MODEL_PARITY" if base_status and tiny_ok else "FAILED",
        "records": len(ok),
        "states": state_summaries,
        "lambda_max_abs_diff": max(lambda_diffs, default=None),
        "loss_max_abs_diff": max(loss_diffs, default=None),
        "output_gradient_cosine_min": min(grad_cos, default=None),
        "s1_records": len(s1),
        "s1_lambda_lt_1_count": len(lt1),
        "s1_lambda_lt_1_ratio": float(len(lt1) / len(s1)) if s1 else 0.0,
        "s1_lambda_eq_1_count": len(eq1),
        "tiny_step": tiny_summary,
        "linear_probe": linear_summary,
        "note": "This gate uses real DiffuEraser Stage1 policy/reference forwards. Tiny parameter-step cases are included only when max_tiny_step_cases > 0.",
    }
    (args.output_dir / "sdpo_true_model_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md = [
        "# Exp27 True DiffuEraser Policy/Reference Objective Parity",
        "",
        f"- status: `{summary['status']}`",
        f"- records: `{summary['records']}`",
        f"- lambda max abs diff: `{summary['lambda_max_abs_diff']}`",
        f"- loss max abs diff: `{summary['loss_max_abs_diff']}`",
        f"- output-gradient cosine min: `{summary['output_gradient_cosine_min']}`",
        f"- S1 lambda<1 ratio: `{summary['s1_lambda_lt_1_ratio']}`",
        f"- tiny-step status: `{tiny_summary['status']}`",
        f"- linear probe status: `{linear_summary['status']}`",
        "",
        "This report is a true-model forward gate. It does not start RC-FPO or any long training.",
    ]
    (args.report_dir / "exp27_sdpo_true_model_forward_parity.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    write_csv(args.report_dir / "exp27_sdpo_true_model_distribution_scan.csv", all_rows)
    write_csv(args.report_dir / "exp27_sdpo_true_model_tiny_step_cases.csv", tiny_rows)
    (args.report_dir / "exp27_sdpo_true_model_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.report_dir / "exp27_linear_true_model_parity.md").write_text(
        "# Exp27 Linear-DPO True Model Probe\n\n"
        f"- status: `{linear_summary['status']}`\n"
        f"- records: `{linear_summary['records']}`\n"
        f"- utility range: `{linear_summary['utility_min']}` to `{linear_summary['utility_max']}`\n"
        f"- EMA max drift: `{linear_summary['ema_max_drift']}`\n",
        encoding="utf-8",
    )
    write_csv(args.report_dir / "exp27_linear_true_model_parity.csv", linear_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "TRUE_MODEL_PARITY" and linear_summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
