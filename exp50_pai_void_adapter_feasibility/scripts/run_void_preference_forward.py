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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--void-weights", required=True)
    ap.add_argument("--manifest", default="manifests/exp50_void_adapter_train4.jsonl")
    ap.add_argument("--row-index", type=int, default=0)
    ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--width", type=int, default=672)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--timestep", type=int, default=500)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--trainable-filter", default="proj_out")
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    paths = VoidPaths(Path(args.repo), Path(args.base_model), Path(args.void_weights), Path(args.void_weights) / "void_pass1.safetensors")
    components = load_components(paths, device=device, dtype=dtype, load_transformer=False)
    policy, p_missing, p_unexpected = load_transformer_clone(paths, device, dtype, trainable_filter=args.trainable_filter, gradient_checkpointing=True)
    reference, r_missing, r_unexpected = load_transformer_clone(paths, device, dtype, trainable_filter=None, gradient_checkpointing=False)
    reference.eval().requires_grad_(False)
    policy.train()
    row = load_micro_row(args.manifest, args.row_index)
    frames = args.frames
    temporal_compression = int(getattr(components["vae"].config, "temporal_compression_ratio", 4))
    patch_size_t = getattr(policy.config, "patch_size_t", None)
    if patch_size_t is not None:
        latent_len = (frames - 1) // temporal_compression + 1
        remainder = latent_len % int(patch_size_t)
        if remainder:
            frames = max(1, frames - remainder * temporal_compression)
    winner = make_target_pack(row, components, device, dtype, frames=frames, size=(args.width, args.height), seed=args.seed, timestep=args.timestep, target_key="winner_path")
    loser = make_target_pack(row, components, device, dtype, frames=frames, size=(args.width, args.height), seed=args.seed, timestep=args.timestep, target_key="loser_path")
    same_noise = bool(torch.equal(winner["noise"], loser["noise"]))
    same_timestep = bool(torch.equal(winner["timesteps"], loser["timesteps"]))
    w_weight = latent_region_weights(row, frames, (args.width, args.height), winner["target"].shape, device, dtype)
    l_weight = latent_region_weights(row, frames, (args.width, args.height), loser["target"].shape, device, dtype)
    with torch.no_grad():
        wr_pred = forward_noise_pred(reference, components, winner, args.height, args.width)
        lr_pred = forward_noise_pred(reference, components, loser, args.height, args.width)
        winner_reference_loss = weighted_mse(wr_pred, winner["target"], w_weight)
        loser_reference_loss = weighted_mse(lr_pred, loser["target"], l_weight)
    wp_pred = forward_noise_pred(policy, components, winner, args.height, args.width)
    lp_pred = forward_noise_pred(policy, components, loser, args.height, args.width)
    winner_policy_loss = weighted_mse(wp_pred, winner["target"], w_weight)
    loser_policy_loss = weighted_mse(lp_pred, loser["target"], l_weight)
    winner_gap = winner_reference_loss.detach() - winner_policy_loss
    loser_gap = loser_reference_loss.detach() - loser_policy_loss
    preference_margin = winner_gap - loser_gap
    dpo_loss = -F.logsigmoid(args.beta * preference_margin)
    dpo_loss.backward()
    grad_finite = True
    grad_norm_sq = 0.0
    grad_tensors = 0
    for p in policy.parameters():
        if p.requires_grad and p.grad is not None:
            grad_tensors += 1
            grad_finite = grad_finite and bool(torch.isfinite(p.grad).all().item())
            grad_norm_sq += float(p.grad.detach().float().norm().item() ** 2)
    reference_grad_zero = True
    for p in reference.parameters():
        if p.grad is not None and float(p.grad.detach().abs().max().item()) != 0.0:
            reference_grad_zero = False
            break
    pref = trainable_parameter_summary(policy)
    summary = {
        "status": "VOID_PREFERENCE_FORWARD_PASS" if grad_finite and reference_grad_zero and same_noise and same_timestep else "VOID_PREFERENCE_FORWARD_BLOCKED",
        "sample_id": row.get("sample_id"),
        "device": str(device),
        "dtype": str(dtype),
        "requested_frames": args.frames,
        "frames": frames,
        "width": args.width,
        "height": args.height,
        "prediction_type": winner["prediction_type"],
        "trainable_filter": args.trainable_filter,
        "parameter_summary": pref,
        "same_noise": same_noise,
        "same_timestep": same_timestep,
        "winner_policy_loss": float(winner_policy_loss.detach().cpu()),
        "winner_reference_loss": float(winner_reference_loss.detach().cpu()),
        "loser_policy_loss": float(loser_policy_loss.detach().cpu()),
        "loser_reference_loss": float(loser_reference_loss.detach().cpu()),
        "winner_gap": float(winner_gap.detach().cpu()),
        "loser_gap": float(loser_gap.detach().cpu()),
        "preference_margin": float(preference_margin.detach().cpu()),
        "dpo_loss": float(dpo_loss.detach().cpu()),
        "grad_finite": grad_finite,
        "grad_tensors": grad_tensors,
        "grad_norm": grad_norm_sq ** 0.5,
        "reference_grad_zero": reference_grad_zero,
        "policy_missing_keys": len(p_missing),
        "policy_unexpected_keys": len(p_unexpected),
        "reference_missing_keys": len(r_missing),
        "reference_unexpected_keys": len(r_unexpected),
        "optimizer_step": False,
        "training_run": False,
        "vor_eval_used": False,
        "hard_comp_used": False,
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    fields = ["sample_id", "status", "prediction_type", "same_noise", "same_timestep", "winner_policy_loss", "winner_reference_loss", "loser_policy_loss", "loser_reference_loss", "winner_gap", "loser_gap", "preference_margin", "dpo_loss", "grad_finite", "reference_grad_zero", "grad_norm", "trainable_filter"]
    with Path(args.output_csv).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({k: summary[k] for k in fields})
    md = f"""# Exp50 VOID Preference Forward

Status: `{summary['status']}`

## Setup

- Sample: `{summary['sample_id']}`
- Policy/reference: identical VOID pass1 transformer clones.
- Reference frozen: {summary['reference_grad_zero']}
- Trainable policy filter: `{args.trainable_filter}`
- Trainable parameters: {pref['trainable_parameters']} / {pref['total_parameters']}
- Target parameterization: `{summary['prediction_type']}`
- Same noise/timestep: {same_noise} / {same_timestep}

## Losses

- winner policy/reference: {summary['winner_policy_loss']} / {summary['winner_reference_loss']}
- loser policy/reference: {summary['loser_policy_loss']} / {summary['loser_reference_loss']}
- winner gap: {summary['winner_gap']}
- loser gap: {summary['loser_gap']}
- preference margin: {summary['preference_margin']}
- DPO loss: {summary['dpo_loss']}
- policy grad finite: {summary['grad_finite']} grad_norm={summary['grad_norm']}

## Safety

No optimizer step, training loop, VOR-Eval, hard comp, deepspeed install, or VOID positive claim was performed.
"""
    Path(args.output_md).write_text(md)


if __name__ == "__main__":
    main()
