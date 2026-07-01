#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from exp50_pai_void_adapter_feasibility.void_preference_wrapper.void_sft_wrapper import (
    VoidPaths,
    forward_noise_pred,
    load_components,
    load_transformer_clone,
    weighted_mse,
)
from exp51_void_loser_dominant_rescue.scripts.run_void_rescue_onestep_grid import (
    load_adapter_state,
)
from exp52_void_winner_preserving_allgpu.scripts.profile_and_cache_void_inputs import (
    adapter_state,
    tensor_to_device,
)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def scalar(x: Any) -> float:
    if torch.is_tensor(x):
        return float(x.detach().float().cpu())
    return float(x)


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def load_item(path: Path, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    return tensor_to_device(torch.load(path, map_location="cpu"), device, dtype)


def cache_paths(cache_root: Path, variant: str, split: str) -> list[Path]:
    return sorted((cache_root / variant / split).glob("*.pt"))


def read_quad(path: str | Path, frames: int, size: tuple[int, int], device: torch.device) -> torch.Tensor:
    width, height = size
    cap = cv2.VideoCapture(str(path))
    got: list[np.ndarray] = []
    while len(got) < frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
        got.append(frame[..., 0])
    cap.release()
    if not got:
        raise RuntimeError(f"cannot decode quadmask {path}")
    while len(got) < frames:
        got.append(got[-1].copy())
    return torch.from_numpy(np.stack(got[:frames]).astype(np.float32)).to(device=device)


def resize_latent_weight(weight: torch.Tensor, latent_shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
    weight = weight.unsqueeze(0).unsqueeze(0)
    _, latent_f, _, latent_h, latent_w = tuple(latent_shape)
    weight = F.interpolate(weight, size=(latent_f, latent_h, latent_w), mode="trilinear", align_corners=False)
    weight = rearrange(weight, "b c f h w -> b f c h w")
    return weight.to(dtype=dtype)


def region_weights(item: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    row = item["row"]
    frames = int(item["frames"])
    width = int(item["width"])
    height = int(item["height"])
    q = read_quad(row["quadmask_0_path"], frames, (width, height), device)
    obj = (q <= 31).float()
    overlap = ((q > 31) & (q <= 95)).float()
    affected = ((q > 95) & (q <= 191)).float()
    outside = (q > 191).float()
    local = ((q <= 191)).float()
    kernel = np.ones((9, 9), np.uint8)
    boundary_frames: list[np.ndarray] = []
    for frame in local.detach().cpu().numpy().astype(np.uint8):
        dil = cv2.dilate(frame, kernel, iterations=1)
        ero = cv2.erode(frame, kernel, iterations=1)
        boundary_frames.append(np.clip(dil - ero, 0, 1))
    boundary = torch.from_numpy(np.stack(boundary_frames).astype(np.float32)).to(device=device)
    latent_shape = item["winner"]["target"].shape
    raw = {
        "object": obj,
        "overlap": overlap,
        "affected": affected,
        "boundary": boundary,
        "outside": outside,
    }
    return {k: resize_latent_weight(v.clamp_min(0.0), latent_shape, dtype) for k, v in raw.items()}


def mse_or_zero(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if float(weight.float().sum().detach().cpu()) <= 1e-6:
        return pred.float().sum() * 0.0
    return weighted_mse(pred, target, weight)


def r5_loss(policy, components, item: dict[str, Any], height: int, width: int, recipe: dict[str, float]):
    weights = item["r5_weights"]
    wp = forward_noise_pred(policy, components, item["winner"], height, width)
    lp = forward_noise_pred(policy, components, item["loser"], height, width)
    wr = item["winner_reference_pred"]
    lr = item["loser_reference_pred"]

    object_w = weights["object"]
    winner_policy_loss = mse_or_zero(wp, item["winner"]["target"], object_w)
    winner_reference_loss = mse_or_zero(wr, item["winner"]["target"], object_w).detach()
    loser_policy_loss = mse_or_zero(lp, item["loser"]["target"], object_w)
    loser_reference_loss = mse_or_zero(lr, item["loser"]["target"], object_w).detach()
    winner_gap = winner_reference_loss - winner_policy_loss
    loser_gap = loser_reference_loss - loser_policy_loss
    effective_loser_gap = loser_gap.detach() * float(recipe["loser_grad_scale"])
    preference_margin = winner_gap - effective_loser_gap
    dpo_loss = -F.logsigmoid(float(recipe["beta"]) * preference_margin)

    object_anchor = mse_or_zero(wp, item["winner"]["target"], weights["object"])
    overlap_anchor = mse_or_zero(wp, item["winner"]["target"], weights["overlap"])
    affected_anchor = mse_or_zero(wp, item["winner"]["target"], weights["affected"])
    boundary_anchor = mse_or_zero(wp, item["winner"]["target"], weights["boundary"])
    outside_anchor = mse_or_zero(wp, item["winner"]["target"], weights["outside"])
    anchor_loss = (
        float(recipe["winner_anchor"]) * object_anchor
        + float(recipe["overlap_preservation"]) * overlap_anchor
        + float(recipe["affected_preservation"]) * affected_anchor
        + float(recipe["boundary_preservation"]) * boundary_anchor
        + float(recipe["outside_preservation"]) * outside_anchor
    )
    loss = dpo_loss + anchor_loss
    return loss, {
        "loss": loss.detach(),
        "dpo_loss": dpo_loss.detach(),
        "anchor_loss": anchor_loss.detach(),
        "winner_policy_loss": winner_policy_loss.detach(),
        "winner_reference_loss": winner_reference_loss.detach(),
        "loser_policy_loss": loser_policy_loss.detach(),
        "loser_reference_loss": loser_reference_loss.detach(),
        "winner_gap": winner_gap.detach(),
        "loser_gap": loser_gap.detach(),
        "effective_loser_gap": effective_loser_gap.detach(),
        "preference_margin": preference_margin.detach(),
        "object_anchor_loss": object_anchor.detach(),
        "overlap_anchor_loss": overlap_anchor.detach(),
        "affected_anchor_loss": affected_anchor.detach(),
        "boundary_anchor_loss": boundary_anchor.detach(),
        "outside_anchor_loss": outside_anchor.detach(),
        "same_noise": item["same_noise"],
        "same_timestep": item["same_timestep"],
        "prediction_type": item["prediction_type"],
    }


def add_weights(item: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    item["r5_weights"] = region_weights(item, device, dtype)
    return item


def summarize(diags: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in [
        "loss",
        "dpo_loss",
        "anchor_loss",
        "winner_gap",
        "loser_gap",
        "effective_loser_gap",
        "preference_margin",
        "winner_policy_loss",
        "winner_reference_loss",
        "loser_policy_loss",
        "loser_reference_loss",
    ]:
        vals = [float(d[key]) for d in diags if key in d]
        out[f"mean_{key}"] = sum(vals) / max(len(vals), 1)
    ratios = [
        abs(float(d.get("effective_loser_gap", 0.0))) / max(abs(float(d.get("preference_margin", 0.0))), 1e-12)
        for d in diags
    ]
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
    ap.add_argument("--cell", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--variant", default="q2_strict_affected")
    ap.add_argument("--trainable-filter", default="proj_out")
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=672)
    args = ap.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    recipe = {
        "beta": 0.1,
        "winner_anchor": 0.10,
        "loser_grad_scale": 0.0,
        "outside_preservation": 0.10,
        "boundary_preservation": 0.15,
        "affected_preservation": 0.10,
        "overlap_preservation": 0.15,
    }
    out_root = Path(args.output_root) / "r5_forward" / args.cell
    ckpt = out_root / "checkpoints" / f"{args.cell}_adapter_step1.pt"
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    paths = VoidPaths(
        Path(args.repo),
        Path(args.base_model),
        Path(args.void_weights),
        Path(args.void_weights) / "void_pass1.safetensors",
    )
    log(f"{args.cell}: load components on {device}")
    components = load_components(paths, device=device, dtype=dtype, load_transformer=False)
    policy, missing, unexpected = load_transformer_clone(
        paths, device, dtype, trainable_filter=args.trainable_filter, gradient_checkpointing=True
    )
    policy.train()

    train_files = cache_paths(Path(args.cache_root), args.variant, "train4")
    heldout_files = cache_paths(Path(args.cache_root), args.variant, "heldout4")
    if len(train_files) != 4 or len(heldout_files) != 4:
        raise RuntimeError(f"expected train4/heldout4 cache files, got {len(train_files)} / {len(heldout_files)}")

    train_items = [add_weights(load_item(p, device, dtype), device, dtype) for p in train_files]
    before = adapter_state(policy)
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    opt.zero_grad(set_to_none=True)
    train_diag: list[dict[str, Any]] = []
    log(f"{args.cell}: train4 one-step forward/backward")
    for item in train_items:
        loss, info = r5_loss(policy, components, item, args.height, args.width, recipe)
        (loss / len(train_items)).backward()
        row = {
            "cell": args.cell,
            "split": "train4",
            "sample_id": item["sample_id"],
            "loser_contribution_ratio": abs(scalar(info["effective_loser_gap"])) / max(abs(scalar(info["preference_margin"])), 1e-12),
            **{k: (scalar(v) if torch.is_tensor(v) else v) for k, v in info.items()},
        }
        train_diag.append(row)
        log(f"{args.cell}: train sample {item['sample_id']} loss={row['loss']:.8f}")

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
            "recipe": "R5_ObjectOnly_RegionPreserve",
            "cell": args.cell,
            "summary": {
                "created": now(),
                "lr": args.lr,
                "optimizer": "AdamW",
                "step": 1,
                "variant": args.variant,
                "timestep": 500,
                "recipe": recipe,
                "half_step": args.lr < 1e-5,
            },
        },
        ckpt,
    )
    log(f"{args.cell}: checkpoint saved {ckpt}")
    del policy
    torch.cuda.empty_cache()

    reload_model, _, _ = load_transformer_clone(
        paths, device, dtype, trainable_filter=args.trainable_filter, gradient_checkpointing=False
    )
    saved = torch.load(ckpt, map_location="cpu")
    reload_missing, reload_unexpected = load_adapter_state(reload_model, saved["adapter_state"])
    reload_model.eval()
    heldout_diag: list[dict[str, Any]] = []
    heldout_finite = True
    with torch.no_grad():
        for path in heldout_files:
            item = add_weights(load_item(path, device, dtype), device, dtype)
            loss, info = r5_loss(reload_model, components, item, args.height, args.width, recipe)
            pred = forward_noise_pred(reload_model, components, item["winner"], args.height, args.width)
            finite = bool(torch.isfinite(pred).all().item()) and bool(torch.isfinite(loss).item())
            heldout_finite = heldout_finite and finite
            row = {
                "cell": args.cell,
                "split": "heldout4",
                "sample_id": item["sample_id"],
                "finite": finite,
                "loser_contribution_ratio": abs(scalar(info["effective_loser_gap"])) / max(abs(scalar(info["preference_margin"])), 1e-12),
                **{k: (scalar(v) if torch.is_tensor(v) else v) for k, v in info.items()},
            }
            heldout_diag.append(row)
            log(f"{args.cell}: heldout sample {item['sample_id']} finite={finite}")

    train_summary = summarize(train_diag)
    heldout_summary = summarize(heldout_diag)
    reload_ok = not reload_missing and not reload_unexpected
    runtime = time.perf_counter() - start
    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    else:
        peak_alloc = peak_reserved = 0.0
    status = "FORWARD_READY" if grad_finite and max_delta > 0 and reload_ok and heldout_finite else "FORWARD_BLOCKED"
    summary = {
        "status": status,
        "cell": args.cell,
        "recipe": "R5_ObjectOnly_RegionPreserve",
        "variant": args.variant,
        "timestep": 500,
        "scope": args.trainable_filter,
        "gpu": args.gpu,
        "lr": args.lr,
        "checkpoint": str(ckpt),
        "runtime_sec": runtime,
        "peak_vram_allocated_gib": peak_alloc,
        "peak_vram_reserved_gib": peak_reserved,
        "grad_norm_before_clip": scalar(grad_norm),
        "grad_finite": grad_finite,
        "max_param_delta_norm": max_delta,
        "reload_missing": reload_missing,
        "reload_unexpected": reload_unexpected,
        "strict_reload": reload_ok,
        "heldout_forward_finite": heldout_finite,
        "policy_missing_keys": len(missing),
        "policy_unexpected_keys": len(unexpected),
        "train_summary": train_summary,
        "heldout_summary": heldout_summary,
        "training_run": "one_optimizer_step_only",
        "optimizer_step": True,
        "ten_step_run": False,
        "vor_eval_used": False,
        "hard_comp_used": False,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    fields = sorted({k for row in train_diag + heldout_diag for k in row.keys()})
    with (out_root / "diagnostics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(train_diag + heldout_diag)
    print(status)
    if status != "FORWARD_READY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
