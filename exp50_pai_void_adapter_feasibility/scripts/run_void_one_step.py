#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from exp50_pai_void_adapter_feasibility.void_preference_wrapper.void_sft_wrapper import (
    VoidPaths,
    forward_noise_pred,
    latent_region_weights,
    load_components,
    load_micro_row,
    load_transformer_clone,
    make_target_pack,
    trainable_parameter_summary,
    weighted_mse,
)


def adapter_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def load_adapter_state(model, state):
    own = dict(model.named_parameters())
    missing = []
    unexpected = []
    for name, tensor in state.items():
        if name not in own:
            unexpected.append(name)
            continue
        own[name].data.copy_(tensor.to(device=own[name].device, dtype=own[name].dtype))
    for name, p in own.items():
        if p.requires_grad and name not in state:
            missing.append(name)
    return missing, unexpected


def pref_loss(policy, reference, components, row, frames, size, seed, timestep, beta, height, width):
    device = next(policy.parameters()).device
    dtype = next(policy.parameters()).dtype
    winner = make_target_pack(row, components, device, dtype, frames=frames, size=size, seed=seed, timestep=timestep, target_key="winner_path")
    loser = make_target_pack(row, components, device, dtype, frames=frames, size=size, seed=seed, timestep=timestep, target_key="loser_path")
    w_weight = latent_region_weights(row, frames, size, winner["target"].shape, device, dtype)
    l_weight = latent_region_weights(row, frames, size, loser["target"].shape, device, dtype)
    with torch.no_grad():
        wr = forward_noise_pred(reference, components, winner, height, width)
        lr = forward_noise_pred(reference, components, loser, height, width)
        winner_reference_loss = weighted_mse(wr, winner["target"], w_weight)
        loser_reference_loss = weighted_mse(lr, loser["target"], l_weight)
    wp = forward_noise_pred(policy, components, winner, height, width)
    lp = forward_noise_pred(policy, components, loser, height, width)
    winner_policy_loss = weighted_mse(wp, winner["target"], w_weight)
    loser_policy_loss = weighted_mse(lp, loser["target"], l_weight)
    winner_gap = winner_reference_loss.detach() - winner_policy_loss
    loser_gap = loser_reference_loss.detach() - loser_policy_loss
    margin = winner_gap - loser_gap
    loss = -F.logsigmoid(beta * margin)
    return loss, {
        "winner_policy_loss": winner_policy_loss,
        "winner_reference_loss": winner_reference_loss.detach(),
        "loser_policy_loss": loser_policy_loss,
        "loser_reference_loss": loser_reference_loss.detach(),
        "winner_gap": winner_gap,
        "loser_gap": loser_gap,
        "preference_margin": margin,
        "winner_pred": wp.detach(),
        "prediction_type": winner["prediction_type"],
        "same_noise": bool(torch.equal(winner["noise"], loser["noise"])),
        "same_timestep": bool(torch.equal(winner["timesteps"], loser["timesteps"])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--void-weights", required=True)
    ap.add_argument("--manifest", default="manifests/exp50_void_adapter_train4.jsonl")
    ap.add_argument("--heldout", default="manifests/exp50_void_adapter_heldout4.jsonl")
    ap.add_argument("--row-index", type=int, default=0)
    ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--width", type=int, default=672)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--timestep", type=int, default=500)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--trainable-filter", default="proj_out")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-visual-csv", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    paths = VoidPaths(Path(args.repo), Path(args.base_model), Path(args.void_weights), Path(args.void_weights) / "void_pass1.safetensors")
    components = load_components(paths, device=device, dtype=dtype, load_transformer=False)
    policy, _, _ = load_transformer_clone(paths, device, dtype, trainable_filter=args.trainable_filter, gradient_checkpointing=True)
    reference, _, _ = load_transformer_clone(paths, device, dtype, trainable_filter=None, gradient_checkpointing=False)
    reference.eval().requires_grad_(False)
    policy.train()
    frames = args.frames
    temporal_compression = int(getattr(components["vae"].config, "temporal_compression_ratio", 4))
    patch_size_t = getattr(policy.config, "patch_size_t", None)
    if patch_size_t is not None:
        latent_len = (frames - 1) // temporal_compression + 1
        remainder = latent_len % int(patch_size_t)
        if remainder:
            frames = max(1, frames - remainder * temporal_compression)
    train_row = load_micro_row(args.manifest, args.row_index)
    before = adapter_state(policy)
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    loss, info = pref_loss(policy, reference, components, train_row, frames, (args.width, args.height), args.seed, args.timestep, args.beta, args.height, args.width)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
    grad_finite = bool(torch.isfinite(grad_norm).item())
    opt.step()
    after = adapter_state(policy)
    deltas = {k: float((after[k] - before[k]).float().norm().item()) for k in before}
    max_delta = max(deltas.values()) if deltas else 0.0
    ckpt = Path(args.checkpoint)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter_state": after, "trainable_filter": args.trainable_filter, "summary": {"lr": args.lr, "frames": frames}}, ckpt)
    # Strict adapter reload into a fresh policy clone.
    del reference
    torch.cuda.empty_cache()
    reload_model, _, _ = load_transformer_clone(paths, device, dtype, trainable_filter=args.trainable_filter, gradient_checkpointing=False)
    saved = torch.load(ckpt, map_location="cpu")
    missing, unexpected = load_adapter_state(reload_model, saved["adapter_state"])
    reload_model.eval()
    heldout_row = load_micro_row(args.heldout, 0)
    with torch.no_grad():
        held_pack = make_target_pack(heldout_row, components, device, dtype, frames=frames, size=(args.width, args.height), seed=args.seed, timestep=args.timestep, target_key="winner_path")
        pred_after = forward_noise_pred(reload_model, components, held_pack, args.height, args.width)
        heldout_output_finite = bool(torch.isfinite(pred_after).all().item())
        train_pack = make_target_pack(train_row, components, device, dtype, frames=frames, size=(args.width, args.height), seed=args.seed, timestep=args.timestep, target_key="winner_path")
        step1_pred = forward_noise_pred(reload_model, components, train_pack, args.height, args.width)
        step1_vs_step0_l1 = float((step1_pred.float() - info["winner_pred"].float()).abs().mean().item())
    param_delta_positive = max_delta > 0.0
    reload_ok = (not missing) and (not unexpected)
    technical_pass = param_delta_positive and reload_ok and heldout_output_finite and step1_vs_step0_l1 > 0 and grad_finite
    status = "VOID_ONE_STEP_PARETO_MIXED" if technical_pass else "VOID_ONE_STEP_BLOCKED"
    summary = {
        "status": status,
        "sample_id": train_row.get("sample_id"),
        "heldout_sample_id": heldout_row.get("sample_id"),
        "device": str(device),
        "dtype": str(dtype),
        "frames": frames,
        "prediction_type": info["prediction_type"],
        "same_noise": info["same_noise"],
        "same_timestep": info["same_timestep"],
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "loss_before_step": float(loss.detach().cpu()),
        "grad_norm_before_clip": float(grad_norm.detach().cpu()),
        "grad_finite": grad_finite,
        "param_delta_positive": param_delta_positive,
        "max_param_delta_norm": max_delta,
        "adapter_checkpoint": str(ckpt),
        "reload_missing_keys": missing,
        "reload_unexpected_keys": unexpected,
        "reload_ok": reload_ok,
        "heldout_forward_finite": heldout_output_finite,
        "step1_vs_step0_l1": step1_vs_step0_l1,
        "video_inference_generated": False,
        "visual_status": "NOT_GENERATED_FORWARD_ONLY",
        "reason_not_pass": "Technical one-step checks passed, but no video-level heldout inference/visual metric was generated; gate is conservative PARETO_MIXED, not PASS." if technical_pass else "One or more technical checks failed.",
        "training_run": "one_optimizer_step_only",
        "optimizer_step": True,
        "vor_eval_used": False,
        "hard_comp_used": False,
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    fields = ["sample_id", "heldout_sample_id", "status", "loss_before_step", "grad_norm_before_clip", "param_delta_positive", "max_param_delta_norm", "reload_ok", "heldout_forward_finite", "step1_vs_step0_l1", "video_inference_generated"]
    with Path(args.output_csv).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerow({k: summary[k] for k in fields})
    with Path(args.output_visual_csv).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "visual_status", "output_path", "notes"])
        w.writeheader(); w.writerow({"sample_id": heldout_row.get("sample_id"), "visual_status": "NOT_GENERATED_FORWARD_ONLY", "output_path": "", "notes": "One-step generated finite model noise prediction only; no video decode/inference visual artifact was produced."})
    md = f"""# Exp50 VOID One-Step Gate V2

Status: `{status}`

## Optimizer

- Optimizer: AdamW
- LR: {args.lr}
- Weight decay: 0
- Grad clip: 1.0
- Trainable filter: `{args.trainable_filter}`
- Optimizer steps: 1

## Checks

- Loss before step: {summary['loss_before_step']}
- Grad finite: {grad_finite}
- Max adapter param delta norm: {max_delta}
- Adapter checkpoint: `{ckpt}`
- Reload ok: {reload_ok}; missing={missing}; unexpected={unexpected}
- Heldout forward finite: {heldout_output_finite}
- Step1 vs Step0 prediction L1: {step1_vs_step0_l1}
- Video inference generated: no

## Interpretation

Technical one-step checks passed, but this is conservatively marked `VOID_ONE_STEP_PARETO_MIXED` rather than `VOID_ONE_STEP_PASS` because no video-level heldout inference/visual evidence was generated in this gate. Therefore H5 10-step remains locked.

## Safety

Exactly one optimizer step was run. No VOR-Eval, hard comp, deepspeed install, long training, or VOID positive claim was made.
"""
    Path(args.output_md).write_text(md)


if __name__ == "__main__":
    main()
