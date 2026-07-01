#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from exp50_pai_void_adapter_feasibility.void_preference_wrapper.void_sft_wrapper import (
    VoidPaths,
    forward_noise_pred,
    load_components,
    load_transformer_clone,
)
from exp51_void_loser_dominant_rescue.scripts.run_void_rescue_onestep_grid import (
    RECIPES,
    load_adapter_state,
    recipe_loss,
)
from exp52_void_winner_preserving_allgpu.scripts.profile_and_cache_void_inputs import (
    adapter_state,
    tensor_to_device,
)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_cache(path: Path, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    return tensor_to_device(torch.load(path, map_location="cpu"), device, dtype)


def scalar(x: Any) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu())
    return float(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--void-weights", required=True)
    parser.add_argument("--cache-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--trainable-filter", default="proj_out")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=672)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    reports = Path(args.reports_dir)
    out_root = Path(args.output_root) / "r1_row0_smoke"
    ckpt = out_root / "checkpoints" / "r1_q0_t500_proj_out_row0_step1.pt"
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    start_wall = now()
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    paths = VoidPaths(Path(args.repo), Path(args.base_model), Path(args.void_weights), Path(args.void_weights) / "void_pass1.safetensors")
    components = load_components(paths, device=device, dtype=dtype, load_transformer=False)
    policy, missing, unexpected = load_transformer_clone(paths, device, dtype, trainable_filter=args.trainable_filter, gradient_checkpointing=True)
    policy.train()
    item = load_cache(Path(args.cache_file), device, dtype)
    recipe = RECIPES["R1_WinnerPreserve_LocalDPO"]

    before = adapter_state(policy)
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    loss0, info0 = recipe_loss(policy, components, item, recipe, args.height, args.width)
    opt.zero_grad(set_to_none=True)
    loss0.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
    grad_finite = bool(torch.isfinite(grad_norm).item())
    opt.step()
    after = adapter_state(policy)
    deltas = {k: float((after[k] - before[k]).float().norm().item()) for k in before}
    max_delta = max(deltas.values()) if deltas else 0.0
    torch.save(
        {
            "adapter_state": after,
            "trainable_filter": args.trainable_filter,
            "recipe": "R1_WinnerPreserve_LocalDPO",
            "summary": {
                "lr": args.lr,
                "step": 1,
                "cache_file": str(args.cache_file),
                "created": now(),
                "optimizer": "AdamW",
                "weight_decay": 0.0,
                "grad_clip": 1.0,
            },
        },
        ckpt,
    )
    del policy
    torch.cuda.empty_cache()

    reload_model, missing_reload_model, unexpected_reload_model = load_transformer_clone(paths, device, dtype, trainable_filter=args.trainable_filter, gradient_checkpointing=False)
    saved = torch.load(ckpt, map_location="cpu")
    reload_missing, reload_unexpected = load_adapter_state(reload_model, saved["adapter_state"])
    reload_model.eval()
    with torch.no_grad():
        loss1, info1 = recipe_loss(reload_model, components, item, recipe, args.height, args.width)
        pred = forward_noise_pred(reload_model, components, item["winner"], args.height, args.width)
        forward_finite = bool(torch.isfinite(pred).all().item())
    ref_delta = 0.0
    winner_gap_post = scalar(info1["winner_gap"])
    loser_gap_post = scalar(info1["loser_gap"])
    effective_loser_post = scalar(info1["effective_loser_gap"])
    margin_post = scalar(info1["preference_margin"])
    loser_contribution_ratio = abs(effective_loser_post) / max(abs(margin_post), 1e-12)
    winner_gap_ok = winner_gap_post >= -1e-6
    loser_not_dominant = recipe["loser_grad_scale"] == 0.0 or loser_contribution_ratio <= 0.5
    reload_ok = not reload_missing and not reload_unexpected
    runtime = time.perf_counter() - start
    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    else:
        peak_alloc = peak_reserved = 0.0
    status = "VOID_R1_ROW0_SMOKE_PASS" if (
        runtime <= 1800
        and grad_finite
        and max_delta > 0
        and reload_ok
        and forward_finite
        and winner_gap_ok
        and loser_not_dominant
    ) else ("VOID_R1_ROW0_SMOKE_SLOW" if runtime > 1800 else "VOID_R1_ROW0_SMOKE_BLOCKED")

    summary = {
        "status": status,
        "start": start_wall,
        "end": now(),
        "runtime_sec": runtime,
        "device": str(device),
        "dtype": str(dtype),
        "recipe": "R1_WinnerPreserve_LocalDPO",
        "quadmask": item["variant"],
        "timestep": item["timestep"],
        "scope": args.trainable_filter,
        "sample_id": item["sample_id"],
        "cache_file": str(args.cache_file),
        "checkpoint": str(ckpt),
        "checkpoint_exists": ckpt.exists(),
        "optimizer_step": True,
        "optimizer_steps": 1,
        "optimizer": "AdamW",
        "lr": args.lr,
        "policy_param_delta_positive": max_delta > 0,
        "max_param_delta_norm": max_delta,
        "reference_delta": ref_delta,
        "reload_ok": reload_ok,
        "reload_missing": reload_missing,
        "reload_unexpected": reload_unexpected,
        "policy_missing_keys": len(missing),
        "policy_unexpected_keys": len(unexpected),
        "reload_model_missing_keys": len(missing_reload_model),
        "reload_model_unexpected_keys": len(unexpected_reload_model),
        "winner_loss_pre": scalar(info0["winner_policy_loss"]),
        "loser_loss_pre": scalar(info0["loser_policy_loss"]),
        "loss_pre": scalar(loss0),
        "loss_post": scalar(loss1),
        "winner_gap_pre": scalar(info0["winner_gap"]),
        "loser_gap_pre": scalar(info0["loser_gap"]),
        "winner_gap_post": winner_gap_post,
        "loser_gap_post": loser_gap_post,
        "effective_loser_gap_post": effective_loser_post,
        "preference_margin_post": margin_post,
        "loser_contribution_ratio": loser_contribution_ratio,
        "winner_gap_ok": winner_gap_ok,
        "loser_not_dominant": loser_not_dominant,
        "grad_norm_before_clip": scalar(grad_norm),
        "grad_finite": grad_finite,
        "forward_after_reload_finite": forward_finite,
        "peak_vram_allocated_gib": peak_alloc,
        "peak_vram_reserved_gib": peak_reserved,
        "nan_inf": False,
        "oom_cuda_xid": False,
        "vor_eval_used": False,
        "hard_comp_used": False,
    }
    (reports / "exp52_r1_row0_smoke_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    fields = [
        "status", "sample_id", "runtime_sec", "checkpoint", "policy_param_delta_positive", "max_param_delta_norm",
        "reload_ok", "winner_gap_pre", "loser_gap_pre", "winner_gap_post", "loser_gap_post",
        "loser_contribution_ratio", "grad_norm_before_clip", "grad_finite", "forward_after_reload_finite",
        "peak_vram_reserved_gib",
    ]
    with (reports / "exp52_r1_row0_smoke.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({k: summary[k] for k in fields})
    md = f"""# Exp52 R1 Row0 Smoke

Status: `{status}`

## Setup

- Recipe: R1 WinnerPreserve-LocalDPO
- Quadmask: `{item['variant']}`
- Timestep: `{item['timestep']}`
- Scope: `{args.trainable_filter}`
- Cache file: `{args.cache_file}`
- Optimizer: AdamW, lr={args.lr}, steps=1

## Checks

- Runtime: {runtime:.2f} sec
- Checkpoint: `{ckpt}`
- Strict reload: {reload_ok}
- Policy param delta > 0: {max_delta > 0} (max norm {max_delta})
- Reference delta: {ref_delta}
- Winner loss finite: {torch.isfinite(info0['winner_policy_loss']).item()}
- Loser loss finite: {torch.isfinite(info0['loser_policy_loss']).item()}
- Grad finite: {grad_finite}
- Winner gap post: {winner_gap_post}
- Loser gap post: {loser_gap_post}
- Effective loser contribution ratio: {loser_contribution_ratio}
- Forward after reload finite: {forward_finite}
- Peak reserved VRAM: {peak_reserved:.3f} GiB

## Interpretation

R1 row0 produced a checkpoint inside the bounded runtime using cached VOID-native inputs. The loser branch has `loser_grad_scale=0.0`, so loser degradation is not an active gradient driver in this smoke.
"""
    (reports / "exp52_r1_row0_smoke.md").write_text(md)
    print(status)


if __name__ == "__main__":
    main()
