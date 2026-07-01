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


def scalar(x: Any) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu())
    return float(x)


def load_item(path: Path, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    return tensor_to_device(torch.load(path, map_location="cpu"), device, dtype)


def cache_paths(cache_root: Path, variant: str, split: str) -> list[Path]:
    return sorted((cache_root / variant / split).glob("*.pt"))


def summarize_diag(diags: list[dict[str, Any]]) -> dict[str, float]:
    out = {}
    for key in ["loss", "winner_gap", "loser_gap", "effective_loser_gap", "preference_margin", "winner_policy_loss", "loser_policy_loss"]:
        vals = [float(d[key]) for d in diags if key in d]
        out[f"mean_{key}"] = sum(vals) / max(len(vals), 1)
    margins = [abs(float(d["preference_margin"])) for d in diags if abs(float(d["preference_margin"])) > 1e-12]
    ratios = [abs(float(d.get("effective_loser_gap", 0.0))) / max(abs(float(d["preference_margin"])), 1e-12) for d in diags]
    out["mean_loser_contribution_ratio"] = sum(ratios) / max(len(ratios), 1)
    out["max_loser_contribution_ratio"] = max(ratios) if ratios else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--void-weights", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--timestep", type=int, default=500)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--trainable-filter", default="proj_out")
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=672)
    args = ap.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    recipe = RECIPES[args.recipe]
    out_root = Path(args.output_root) / "wave1_forward" / args.cell
    ckpt = out_root / "checkpoints" / f"{args.cell}_adapter_step1.pt"
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    start_wall = now()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    paths = VoidPaths(Path(args.repo), Path(args.base_model), Path(args.void_weights), Path(args.void_weights) / "void_pass1.safetensors")
    components = load_components(paths, device=device, dtype=dtype, load_transformer=False)
    policy, missing, unexpected = load_transformer_clone(paths, device, dtype, trainable_filter=args.trainable_filter, gradient_checkpointing=True)
    policy.train()
    train_files = cache_paths(Path(args.cache_root), args.variant, "train4")
    heldout_files = cache_paths(Path(args.cache_root), args.variant, "heldout4")
    if len(train_files) != 4 or len(heldout_files) != 4:
        raise RuntimeError(f"expected train4/heldout4 cache files, got {len(train_files)} / {len(heldout_files)}")
    train_items = [load_item(p, device, dtype) for p in train_files]
    before = adapter_state(policy)
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    opt.zero_grad(set_to_none=True)
    train_diag = []
    total_loss = 0.0
    for item in train_items:
        loss, info = recipe_loss(policy, components, item, recipe, args.height, args.width)
        (loss / len(train_items)).backward()
        total_loss += scalar(loss)
        rec = {
            "cell": args.cell,
            "split": "train4",
            "sample_id": item["sample_id"],
            "loss": scalar(loss),
            "winner_policy_loss": scalar(info["winner_policy_loss"]),
            "winner_reference_loss": scalar(info["winner_reference_loss"]),
            "loser_policy_loss": scalar(info["loser_policy_loss"]),
            "loser_reference_loss": scalar(info["loser_reference_loss"]),
            "winner_gap": scalar(info["winner_gap"]),
            "loser_gap": scalar(info["loser_gap"]),
            "effective_loser_gap": scalar(info["effective_loser_gap"]),
            "preference_margin": scalar(info["preference_margin"]),
            "same_noise": info["same_noise"],
            "same_timestep": info["same_timestep"],
            "prediction_type": info["prediction_type"],
        }
        train_diag.append(rec)
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
            "recipe": args.recipe,
            "cell": args.cell,
            "summary": {
                "created": now(),
                "lr": args.lr,
                "optimizer": "AdamW",
                "step": 1,
                "variant": args.variant,
                "timestep": args.timestep,
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
    heldout_diag = []
    heldout_finite = True
    with torch.no_grad():
        for path in heldout_files:
            item = load_item(path, device, dtype)
            loss, info = recipe_loss(reload_model, components, item, recipe, args.height, args.width)
            pred = forward_noise_pred(reload_model, components, item["winner"], args.height, args.width)
            finite = bool(torch.isfinite(pred).all().item()) and bool(torch.isfinite(loss).item())
            heldout_finite = heldout_finite and finite
            heldout_diag.append({
                "cell": args.cell,
                "split": "heldout4",
                "sample_id": item["sample_id"],
                "loss": scalar(loss),
                "winner_policy_loss": scalar(info["winner_policy_loss"]),
                "winner_reference_loss": scalar(info["winner_reference_loss"]),
                "loser_policy_loss": scalar(info["loser_policy_loss"]),
                "loser_reference_loss": scalar(info["loser_reference_loss"]),
                "winner_gap": scalar(info["winner_gap"]),
                "loser_gap": scalar(info["loser_gap"]),
                "effective_loser_gap": scalar(info["effective_loser_gap"]),
                "preference_margin": scalar(info["preference_margin"]),
                "finite": finite,
                "same_noise": info["same_noise"],
                "same_timestep": info["same_timestep"],
                "prediction_type": info["prediction_type"],
            })

    train_summary = summarize_diag(train_diag)
    heldout_summary = summarize_diag(heldout_diag)
    winner_ok = heldout_summary["mean_winner_gap"] >= -1e-6 or train_summary["mean_winner_gap"] >= -1e-6
    loser_ratio = heldout_summary["mean_loser_contribution_ratio"]
    loser_controlled = loser_ratio <= 0.5 or recipe.get("loser_grad_scale", 1.0) == 0.0
    reload_ok = not reload_missing and not reload_unexpected
    runtime = time.perf_counter() - start
    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    else:
        peak_alloc = peak_reserved = 0.0
    status = "FORWARD_READY" if (
        grad_finite and max_delta > 0 and reload_ok and heldout_finite and winner_ok and loser_controlled
    ) else "FORWARD_MIXED"

    summary = {
        "status": status,
        "cell": args.cell,
        "recipe": args.recipe,
        "variant": args.variant,
        "timestep": args.timestep,
        "scope": args.trainable_filter,
        "gpu": args.gpu,
        "start": start_wall,
        "end": now(),
        "runtime_sec": runtime,
        "checkpoint": str(ckpt),
        "checkpoint_exists": ckpt.exists(),
        "max_param_delta_norm": max_delta,
        "grad_norm_before_clip": scalar(grad_norm),
        "grad_finite": grad_finite,
        "reload_ok": reload_ok,
        "reload_missing": reload_missing,
        "reload_unexpected": reload_unexpected,
        "policy_missing_keys": len(missing),
        "policy_unexpected_keys": len(unexpected),
        "reload_model_missing_keys": len(missing_reload_model),
        "reload_model_unexpected_keys": len(unexpected_reload_model),
        "heldout_forward_finite": heldout_finite,
        "winner_ok": winner_ok,
        "loser_controlled": loser_controlled,
        "train": train_summary,
        "heldout": heldout_summary,
        "peak_vram_allocated_gib": peak_alloc,
        "peak_vram_reserved_gib": peak_reserved,
        "optimizer_step": True,
        "optimizer_steps": 1,
        "vor_eval_used": False,
        "hard_comp_used": False,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (out_root / "diagnostics.csv").open("w", newline="") as f:
        rows = train_diag + heldout_diag
        fields = sorted({k for r in rows for k in r.keys()})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"cell": args.cell, "status": status, "checkpoint": str(ckpt), "runtime_sec": runtime}, sort_keys=True))


if __name__ == "__main__":
    main()
