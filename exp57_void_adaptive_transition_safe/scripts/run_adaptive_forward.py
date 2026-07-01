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

from exp50_pai_void_adapter_feasibility.void_preference_wrapper.void_sft_wrapper import (
    VoidPaths,
    forward_noise_pred,
    load_components,
    load_transformer_clone,
)
from exp51_void_loser_dominant_rescue.scripts.run_void_rescue_onestep_grid import load_adapter_state
from exp52_void_winner_preserving_allgpu.scripts.profile_and_cache_void_inputs import (
    adapter_state,
    tensor_to_device,
)
from exp57_void_adaptive_transition_safe.adaptive_transition_loss import (
    REGIONS,
    AdaptiveLossConfig,
    build_adaptive_loss,
    config_for_cell,
    flatten_grads,
    grad_stats,
    scalar,
    select_backtracking_scale,
    transition_risk_weights,
    weighted_mse,
)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def load_item(path: Path, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    return tensor_to_device(torch.load(path, map_location="cpu"), device, dtype)


def cache_paths(cache_root: Path, variant: str, split: str) -> list[Path]:
    return sorted((cache_root / variant / split).glob("*.pt"))


def read_quad(path: str | Path, frames: int, width: int, height: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    got: list[np.ndarray] = []
    while len(got) < frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame[..., 0], (width, height), interpolation=cv2.INTER_NEAREST)
        got.append(frame)
    cap.release()
    if not got:
        raise RuntimeError(f"cannot decode quadmask {path}")
    while len(got) < frames:
        got.append(got[-1].copy())
    return np.stack(got[:frames], axis=0)


def resize_latent_weight(weight: torch.Tensor, latent_shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    weight = weight.unsqueeze(0).unsqueeze(0)
    _, latent_f, _, latent_h, latent_w = tuple(latent_shape)
    weight = F.interpolate(weight, size=(latent_f, latent_h, latent_w), mode="trilinear", align_corners=False)
    weight = weight.permute(0, 2, 1, 3, 4).contiguous()
    return weight.to(dtype=dtype)


def region_weights(item: dict[str, Any], latent_shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    row = item["row"]
    frames = int(item["frames"])
    width = int(item["width"])
    height = int(item["height"])
    q_np = read_quad(row["quadmask_0_path"], frames, width, height)
    q = torch.from_numpy(q_np.astype(np.float32)).to(device=device)
    masks = {
        "object": (q <= 31).float(),
        "overlap": ((q > 31) & (q <= 95)).float(),
        "affected": ((q > 95) & (q <= 191)).float(),
        "outside": (q > 191).float(),
    }
    local = ((q <= 191).detach().cpu().numpy().astype(np.uint8))
    kernel = np.ones((9, 9), np.uint8)
    boundary_frames: list[np.ndarray] = []
    for frame in local:
        dil = cv2.dilate(frame, kernel, iterations=1).astype(bool)
        ero = cv2.erode(frame, kernel, iterations=1).astype(bool)
        boundary_frames.append((dil ^ ero).astype(np.float32))
    masks["boundary"] = torch.from_numpy(np.stack(boundary_frames)).to(device=device)
    return {name: resize_latent_weight(mask, latent_shape, dtype).clamp_min(0.0) for name, mask in masks.items()}


def region_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {region: weighted_mse(pred, target, weights[region]) for region in REGIONS}


def compute_losses(policy: Any, components: Any, item: dict[str, Any], weights: dict[str, torch.Tensor], height: int, width: int, config: AdaptiveLossConfig, safe: dict[str, float] | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    winner_pred = forward_noise_pred(policy, components, item["winner"], height, width)
    loser_pred = forward_noise_pred(policy, components, item["loser"], height, width)
    winner_policy = region_losses(winner_pred, item["winner"]["target"], weights)
    loser_policy = region_losses(loser_pred, item["loser"]["target"], weights)
    winner_reference = region_losses(item["winner_reference_pred"], item["winner"]["target"], weights)
    loser_reference = region_losses(item["loser_reference_pred"], item["loser"]["target"], weights)
    loss, info = build_adaptive_loss(winner_policy, winner_reference, loser_policy, loser_reference, config, safe)
    info.update(
        {
            "winner_policy_loss": sum(winner_policy.values()).detach() / len(winner_policy),
            "winner_reference_loss": sum(winner_reference.values()).detach() / len(winner_reference),
            "loser_policy_loss": sum(loser_policy.values()).detach() / len(loser_policy),
            "loser_reference_loss": sum(loser_reference.values()).detach() / len(loser_reference),
            "same_noise": item["same_noise"],
            "same_timestep": item["same_timestep"],
            "prediction_type": item["prediction_type"],
        }
    )
    return loss, info


def trainable_params(model: Any) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def compute_safe_lambda(policy: Any, components: Any, item: dict[str, Any], weights: dict[str, torch.Tensor], height: int, width: int, config: AdaptiveLossConfig) -> dict[str, float]:
    params = trainable_params(policy)
    winner_pred = forward_noise_pred(policy, components, item["winner"], height, width)
    loser_pred = forward_noise_pred(policy, components, item["loser"], height, width)
    winner_losses = region_losses(winner_pred, item["winner"]["target"], weights)
    loser_losses = region_losses(loser_pred, item["loser"]["target"], weights)
    winner_obj = winner_losses["object"] + 0.25 * winner_losses["affected"]
    loser_obj = loser_losses["object"] + 0.25 * loser_losses["affected"]
    wg = torch.autograd.grad(winner_obj, params, retain_graph=False, allow_unused=True)
    lg = torch.autograd.grad(loser_obj, params, retain_graph=False, allow_unused=True)
    return grad_stats(flatten_grads(wg), flatten_grads(lg), config.lambda_max)


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "loss",
        "preference_loss",
        "winner_anchor_loss",
        "winner_gap",
        "loser_gap",
        "effective_loser_gap",
        "preference_margin",
        "winner_policy_loss",
        "winner_reference_loss",
        "loser_policy_loss",
        "loser_reference_loss",
    ]
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row]
        out[f"mean_{key}"] = sum(vals) / max(len(vals), 1)
    ratios = [
        abs(float(row.get("effective_loser_gap", 0.0))) / max(abs(float(row.get("preference_margin", 0.0))), 1.0e-12)
        for row in rows
    ]
    out["mean_loser_contribution_ratio"] = sum(ratios) / max(len(ratios), 1)
    return out


def evaluate_region_means(policy: Any, components: Any, items: list[dict[str, Any]], weights: list[dict[str, torch.Tensor]], height: int, width: int) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for item, region_w in zip(items, weights):
            pred = forward_noise_pred(policy, components, item["winner"], height, width)
            losses = region_losses(pred, item["winner"]["target"], region_w)
            losses["winner"] = sum(losses.values()) / len(losses)
            rows.append({name: scalar(value) for name, value in losses.items()})
    return {name: sum(row[name] for row in rows) / len(rows) for name in ("winner", *REGIONS)}


def restore_adapter(model: Any, state: dict[str, torch.Tensor]) -> None:
    own = dict(model.named_parameters())
    for name, tensor in state.items():
        own[name].data.copy_(tensor.to(device=own[name].device, dtype=own[name].dtype))


def save_adapter(path: Path, model: Any, config: AdaptiveLossConfig, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_state": adapter_state(model),
            "trainable_filter": summary["trainable_filter"],
            "cell": summary["cell"],
            "loss_mode": "void_adaptive_transition_safe_dpo_v0",
            "config": config.__dict__,
            "summary": summary,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zero_gap", "one_step"], required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--void-weights", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--cell", default="ATS0_Q2_T500_S0")
    parser.add_argument("--variant", default="q2_strict_affected")
    parser.add_argument("--timestep", type=int, default=500)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--trainable-filter", default="proj_out")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    start = time.perf_counter()
    start_wall = now()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    config = config_for_cell(args.cell)
    out_root = Path(args.output_root) / ("adaptive_zero_gap" if args.mode == "zero_gap" else "adaptive_forward") / args.cell
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt = out_root / "checkpoints" / f"{args.cell}_adapter_step1.pt"

    paths = VoidPaths(
        Path(args.repo),
        Path(args.base_model),
        Path(args.void_weights),
        Path(args.void_weights) / "void_pass1.safetensors",
    )
    log(f"{args.cell}: load components")
    components = load_components(paths, device=device, dtype=dtype, load_transformer=False)
    log(f"{args.cell}: load policy")
    policy, missing, unexpected = load_transformer_clone(
        paths,
        device,
        dtype,
        trainable_filter=args.trainable_filter,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    policy.train()

    train_files = cache_paths(Path(args.cache_root), args.variant, "train4")
    heldout_files = cache_paths(Path(args.cache_root), args.variant, "heldout4")
    if len(train_files) != 4 or len(heldout_files) != 4:
        raise RuntimeError(f"expected train4/heldout4 cache files, got {len(train_files)} / {len(heldout_files)}")
    train_items = [load_item(p, device, dtype) for p in train_files]
    heldout_items = [load_item(p, device, dtype) for p in heldout_files]
    train_weights = [region_weights(item, tuple(item["winner"]["target"].shape), device, dtype) for item in train_items]
    heldout_weights = [region_weights(item, tuple(item["winner"]["target"].shape), device, dtype) for item in heldout_items]

    log(f"{args.cell}: compute safe lambda")
    safe = compute_safe_lambda(policy, components, train_items[0], train_weights[0], args.height, args.width, config)
    safe.update({"lambda_loser_global": min(safe["lambda_loser_global"], config.lambda_max)})

    rows: list[dict[str, Any]] = []
    policy.zero_grad(set_to_none=True)
    for item, region_w in zip(train_items, train_weights):
        loss, info = compute_losses(policy, components, item, region_w, args.height, args.width, config, safe)
        if args.mode == "one_step":
            (loss / len(train_items)).backward()
        rec = {"cell": args.cell, "split": "train4", "sample_id": item["sample_id"], **{k: scalar(v) if torch.is_tensor(v) else v for k, v in info.items()}}
        rows.append(rec)

    params = trainable_params(policy)
    grad_norm = torch.tensor(0.0, device=device)
    grad_finite = True
    before_state = adapter_state(policy)
    before_regions = evaluate_region_means(policy, components, train_items, train_weights, args.height, args.width)
    update_decision: dict[str, Any] = {
        "finite_diff_pass_loser": True,
        "transition_safe_pass": True,
        "finite_diff_attempts": 0,
        "finite_diff_selected_scale": 0.0,
        "update_rejected": args.mode != "one_step",
        "global_update_scale": 0.0,
    }
    max_delta = 0.0
    reload_ok = False

    if args.mode == "one_step":
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        grad_finite = bool(torch.isfinite(grad_norm).item())
        candidates: list[dict[str, float]] = []
        accepted = False
        selected_scale = 0.0
        for scale in config.backtracking_scales:
            restore_adapter(policy, before_state)
            opt = torch.optim.AdamW(params, lr=args.lr * config.lr_scale * scale, betas=(0.9, 0.999), weight_decay=0.0)
            opt.step()
            after_regions = evaluate_region_means(policy, components, train_items, train_weights, args.height, args.width)
            deltas = {"scale": scale, **{name: after_regions[name] - before_regions[name] for name in ("winner", *REGIONS)}}
            candidates.append(deltas)
            decision = select_backtracking_scale(candidates, config)
            if not decision["update_rejected"] and decision["finite_diff_selected_scale"] == scale:
                update_decision = decision
                accepted = True
                selected_scale = scale
                break
        if not accepted:
            restore_adapter(policy, before_state)
            update_decision = select_backtracking_scale(candidates, config)
            selected_scale = 0.0
        after_state = adapter_state(policy)
        deltas = {k: float((after_state[k] - before_state[k]).float().norm().item()) for k in before_state}
        max_delta = max(deltas.values()) if deltas else 0.0
        summary_stub = {
            "cell": args.cell,
            "trainable_filter": args.trainable_filter,
            "selected_scale": selected_scale,
        }
        save_adapter(ckpt, policy, config, summary_stub)
        log(f"{args.cell}: checkpoint saved {ckpt}")

        del policy
        torch.cuda.empty_cache()
        reload_model, _, _ = load_transformer_clone(paths, device, dtype, trainable_filter=args.trainable_filter, gradient_checkpointing=False)
        saved = torch.load(ckpt, map_location="cpu")
        missing_reload, unexpected_reload = load_adapter_state(reload_model, saved["adapter_state"])
        reload_ok = not missing_reload and not unexpected_reload
        policy = reload_model
        policy.eval()

    heldout_finite = True
    with torch.no_grad():
        for item, region_w in zip(heldout_items, heldout_weights):
            loss, info = compute_losses(policy, components, item, region_w, args.height, args.width, config, safe)
            finite = bool(torch.isfinite(loss).item())
            heldout_finite = heldout_finite and finite
            rec = {"cell": args.cell, "split": "heldout4", "sample_id": item["sample_id"], "finite": finite, **{k: scalar(v) if torch.is_tensor(v) else v for k, v in info.items()}}
            rows.append(rec)

    train_rows = [row for row in rows if row["split"] == "train4"]
    heldout_rows = [row for row in rows if row["split"] == "heldout4"]
    train_summary = summarize(train_rows)
    heldout_summary = summarize(heldout_rows)
    risk = transition_risk_weights(
        {
            "overlap": float(update_decision.get("overlap_delta_pred", 0.0)),
            "affected": float(update_decision.get("affected_delta_pred", 0.0)),
            "boundary": float(update_decision.get("boundary_delta_pred", 0.0)),
            "outside": float(update_decision.get("outside_delta_pred", 0.0)),
        },
        config,
    )
    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    else:
        peak_alloc = peak_reserved = 0.0
    status = "EXP57_ADAPTIVE_ZERO_GAP_PASS" if args.mode == "zero_gap" else "FORWARD_READY"
    if args.mode == "zero_gap":
        zero_gap_ok = abs(heldout_summary.get("mean_preference_margin", 0.0)) <= 1e-4 and heldout_finite
        status = "EXP57_ADAPTIVE_ZERO_GAP_PASS" if zero_gap_ok else "EXP57_ADAPTIVE_ZERO_GAP_FAIL"
    else:
        status = "FORWARD_READY" if grad_finite and heldout_finite and (max_delta > 0.0 or update_decision["update_rejected"]) and reload_ok else "FORWARD_MIXED"

    summary = {
        "status": status,
        "mode": args.mode,
        "cell": args.cell,
        "loss_mode": "void_adaptive_transition_safe_dpo_v0",
        "variant": args.variant,
        "timestep": args.timestep,
        "gpu": args.gpu,
        "start": start_wall,
        "end": now(),
        "runtime_sec": time.perf_counter() - start,
        "checkpoint": str(ckpt) if args.mode == "one_step" else "",
        "checkpoint_exists": ckpt.exists() if args.mode == "one_step" else False,
        "max_param_delta_norm": max_delta,
        "grad_norm_before_clip": scalar(grad_norm),
        "grad_finite": grad_finite,
        "reload_ok": reload_ok if args.mode == "one_step" else "not_run",
        "heldout_forward_finite": heldout_finite,
        "safe_lambda": safe,
        "transition_risk": risk,
        "backtracking": update_decision,
        "train": train_summary,
        "heldout": heldout_summary,
        "peak_vram_allocated_gib": peak_alloc,
        "peak_vram_reserved_gib": peak_reserved,
        "optimizer_step": args.mode == "one_step" and not update_decision.get("update_rejected", False),
        "optimizer_steps": 1 if args.mode == "one_step" and not update_decision.get("update_rejected", False) else 0,
        "vor_eval_used": False,
        "hard_comp_used": False,
        "ten_step_run": False,
        "policy_missing_keys": len(missing),
        "policy_unexpected_keys": len(unexpected),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (out_root / "diagnostics.csv").open("w", newline="") as f:
        fields = sorted({k for row in rows for k in row})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": status, "cell": args.cell, "mode": args.mode, "checkpoint": summary["checkpoint"]}, sort_keys=True))


if __name__ == "__main__":
    main()
