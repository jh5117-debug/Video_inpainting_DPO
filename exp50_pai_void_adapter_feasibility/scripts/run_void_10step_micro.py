#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timedelta, timezone
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
    weighted_mse,
)


SH_TZ = timezone(timedelta(hours=8))


def now() -> str:
    return datetime.now(SH_TZ).replace(microsecond=0).isoformat()


def manifest_len(path: str | Path) -> int:
    return sum(1 for line in Path(path).read_text().splitlines() if line.strip())


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
    for name, param in own.items():
        if param.requires_grad and name not in state:
            missing.append(name)
    return missing, unexpected


def save_adapter(path: Path, model, step: int, args, frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_state": adapter_state(model),
            "trainable_filter": args.trainable_filter,
            "summary": {
                "step": step,
                "lr": args.lr,
                "frames": frames,
                "seed": args.seed,
                "timestep": args.timestep,
                "beta": args.beta,
                "created": now(),
                "reference_mode": "cached_base_policy_outputs",
            },
        },
        path,
    )


def build_cache(policy, components, rows_n: int, args, frames: int, device, dtype):
    cache = []
    policy.eval()
    with torch.no_grad():
        for row_index in range(rows_n):
            row = load_micro_row(args.manifest, row_index)
            seed = args.seed + row_index
            winner = make_target_pack(
                row,
                components,
                device,
                dtype,
                frames=frames,
                size=(args.width, args.height),
                seed=seed,
                timestep=args.timestep,
                target_key="winner_path",
            )
            loser = make_target_pack(
                row,
                components,
                device,
                dtype,
                frames=frames,
                size=(args.width, args.height),
                seed=seed,
                timestep=args.timestep,
                target_key="loser_path",
            )
            winner_weight = latent_region_weights(row, frames, (args.width, args.height), winner["target"].shape, device, dtype)
            loser_weight = latent_region_weights(row, frames, (args.width, args.height), loser["target"].shape, device, dtype)
            winner_ref = forward_noise_pred(policy, components, winner, args.height, args.width)
            loser_ref = forward_noise_pred(policy, components, loser, args.height, args.width)
            winner_reference_loss = weighted_mse(winner_ref, winner["target"], winner_weight).detach()
            loser_reference_loss = weighted_mse(loser_ref, loser["target"], loser_weight).detach()
            cache.append(
                {
                    "row_index": row_index,
                    "sample_id": row.get("sample_id"),
                    "seed": seed,
                    "winner": winner,
                    "loser": loser,
                    "winner_weight": winner_weight,
                    "loser_weight": loser_weight,
                    "winner_reference_loss": winner_reference_loss,
                    "loser_reference_loss": loser_reference_loss,
                    "prediction_type": winner["prediction_type"],
                    "same_noise": bool(torch.equal(winner["noise"], loser["noise"])),
                    "same_timestep": bool(torch.equal(winner["timesteps"], loser["timesteps"])),
                }
            )
            del winner_ref, loser_ref
            torch.cuda.empty_cache()
    policy.train()
    return cache


def cached_pref_loss(policy, components, item, beta: float, height: int, width: int):
    winner_pred = forward_noise_pred(policy, components, item["winner"], height, width)
    loser_pred = forward_noise_pred(policy, components, item["loser"], height, width)
    winner_policy_loss = weighted_mse(winner_pred, item["winner"]["target"], item["winner_weight"])
    loser_policy_loss = weighted_mse(loser_pred, item["loser"]["target"], item["loser_weight"])
    winner_gap = item["winner_reference_loss"] - winner_policy_loss
    loser_gap = item["loser_reference_loss"] - loser_policy_loss
    margin = winner_gap - loser_gap
    loss = -F.logsigmoid(beta * margin)
    return loss, {
        "winner_policy_loss": winner_policy_loss.detach(),
        "winner_reference_loss": item["winner_reference_loss"],
        "loser_policy_loss": loser_policy_loss.detach(),
        "loser_reference_loss": item["loser_reference_loss"],
        "winner_gap": winner_gap.detach(),
        "loser_gap": loser_gap.detach(),
        "preference_margin": margin.detach(),
        "prediction_type": item["prediction_type"],
        "same_noise": item["same_noise"],
        "same_timestep": item["same_timestep"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--void-weights", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-visual-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timestep", type=int, default=500)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--trainable-filter", default="proj_out")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    ckpt_root = output_root / "checkpoints"
    output_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    start = now()
    t0 = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    paths = VoidPaths(
        Path(args.repo),
        Path(args.base_model),
        Path(args.void_weights),
        Path(args.void_weights) / "void_pass1.safetensors",
    )
    components = load_components(paths, device=device, dtype=dtype, load_transformer=False)
    policy, missing_policy, unexpected_policy = load_transformer_clone(
        paths,
        device,
        dtype,
        trainable_filter=args.trainable_filter,
        gradient_checkpointing=True,
    )

    frames = args.frames
    temporal_compression = int(getattr(components["vae"].config, "temporal_compression_ratio", 4))
    patch_size_t = getattr(policy.config, "patch_size_t", None)
    if patch_size_t is not None:
        latent_len = (frames - 1) // temporal_compression + 1
        remainder = latent_len % int(patch_size_t)
        if remainder:
            frames = max(1, frames - remainder * temporal_compression)

    rows_n = manifest_len(args.manifest)
    cache = build_cache(policy, components, rows_n, args, frames, device, dtype)
    before = adapter_state(policy)
    save_adapter(ckpt_root / "adapter_proj_out_step0.pt", policy, 0, args, frames)
    opt = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )
    diagnostics = []
    for step in range(1, args.steps + 1):
        item = cache[(step - 1) % rows_n]
        opt.zero_grad(set_to_none=True)
        loss, info = cached_pref_loss(policy, components, item, args.beta, args.height, args.width)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
        grad_finite = bool(torch.isfinite(grad_norm).item())
        opt.step()
        if step in {1, 5, args.steps}:
            save_adapter(ckpt_root / f"adapter_proj_out_step{step}.pt", policy, step, args, frames)
        diagnostics.append(
            {
                "step": step,
                "row_index": item["row_index"],
                "sample_id": item["sample_id"],
                "seed": item["seed"],
                "loss": float(loss.detach().cpu()),
                "grad_norm_before_clip": float(grad_norm.detach().cpu()),
                "grad_finite": grad_finite,
                "winner_policy_loss": float(info["winner_policy_loss"].cpu()),
                "winner_reference_loss": float(info["winner_reference_loss"].cpu()),
                "loser_policy_loss": float(info["loser_policy_loss"].cpu()),
                "loser_reference_loss": float(info["loser_reference_loss"].cpu()),
                "winner_gap": float(info["winner_gap"].cpu()),
                "loser_gap": float(info["loser_gap"].cpu()),
                "preference_margin": float(info["preference_margin"].cpu()),
                "same_noise": info["same_noise"],
                "same_timestep": info["same_timestep"],
                "prediction_type": info["prediction_type"],
            }
        )

    after = adapter_state(policy)
    deltas = {key: float((after[key] - before[key]).float().norm().item()) for key in before}
    max_delta = max(deltas.values()) if deltas else 0.0

    torch.cuda.empty_cache()
    reload_model, missing_reload_model, unexpected_reload_model = load_transformer_clone(
        paths,
        device,
        dtype,
        trainable_filter=args.trainable_filter,
        gradient_checkpointing=False,
    )
    step10_ckpt = ckpt_root / f"adapter_proj_out_step{args.steps}.pt"
    saved = torch.load(step10_ckpt, map_location="cpu")
    reload_missing, reload_unexpected = load_adapter_state(reload_model, saved["adapter_state"])
    reload_model.eval()
    heldout_ok = True
    heldout_checks = []
    with torch.no_grad():
        for idx in range(manifest_len(args.heldout)):
            row = load_micro_row(args.heldout, idx)
            pack = make_target_pack(
                row,
                components,
                device,
                dtype,
                frames=frames,
                size=(args.width, args.height),
                seed=args.seed + idx,
                timestep=args.timestep,
                target_key="winner_path",
            )
            pred = forward_noise_pred(reload_model, components, pack, args.height, args.width)
            finite = bool(torch.isfinite(pred).all().item())
            heldout_ok = heldout_ok and finite
            heldout_checks.append({"sample_id": row.get("sample_id"), "finite": finite})

    technical_ok = (
        all(item["grad_finite"] for item in diagnostics)
        and max_delta > 0.0
        and not reload_missing
        and not reload_unexpected
        and heldout_ok
    )
    status = "VOID_ADAPTER_10STEP_FORWARD_READY" if technical_ok else "VOID_ADAPTER_10STEP_BLOCKED"
    summary = {
        "status": status,
        "start": start,
        "end": now(),
        "runtime_sec": time.time() - t0,
        "device": str(device),
        "dtype": str(dtype),
        "frames": frames,
        "steps": args.steps,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "trainable_filter": args.trainable_filter,
        "reference_mode": "cached_base_policy_outputs",
        "missing_policy": list(missing_policy),
        "unexpected_policy": list(unexpected_policy),
        "missing_reload_model": list(missing_reload_model),
        "unexpected_reload_model": list(unexpected_reload_model),
        "reload_missing_keys": reload_missing,
        "reload_unexpected_keys": reload_unexpected,
        "reload_ok": (not reload_missing and not reload_unexpected),
        "max_param_delta_norm": max_delta,
        "param_delta_positive": max_delta > 0.0,
        "heldout_forward_finite": heldout_ok,
        "heldout_checks": heldout_checks,
        "checkpoints": {
            "step0": str(ckpt_root / "adapter_proj_out_step0.pt"),
            "step1": str(ckpt_root / "adapter_proj_out_step1.pt"),
            "step5": str(ckpt_root / "adapter_proj_out_step5.pt"),
            "step10": str(step10_ckpt),
        },
        "diagnostics": diagnostics,
        "training_run": "ten_optimizer_steps_only",
        "optimizer_step": True,
        "vor_eval_used": False,
        "hard_comp_used": False,
        "long_training": False,
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    fields = list(diagnostics[0].keys())
    with Path(args.output_csv).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(diagnostics)
    with Path(args.output_visual_csv).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["status", "notes"])
        writer.writeheader()
        writer.writerow(
            {
                "status": "PENDING_HELDOUT_VIDEO_INFERENCE",
                "notes": "Forward-only 10-step training finished; video evaluation must run through official inference before promotion.",
            }
        )
    md = [
        "# Exp50 VOID Adapter 10-Step V2",
        "",
        f"Status: `{status}`",
        "",
        "## Protocol",
        "",
        f"- Optimizer steps: {args.steps}",
        "- Optimizer: AdamW",
        f"- LR: {args.lr}",
        f"- Trainable filter: `{args.trainable_filter}`",
        "- Reference mode: cached frozen base policy outputs",
        "- Deepspeed: not used",
        "- VOR-Eval: not used",
        "- Hard comp: not used",
        "",
        "## Checks",
        "",
        f"- Max param delta norm: {max_delta}",
        f"- Reload ok: {not reload_missing and not reload_unexpected}; missing={reload_missing}; unexpected={reload_unexpected}",
        f"- Heldout forward finite: {heldout_ok}",
        f"- Step10 checkpoint: `{step10_ckpt}`",
        "",
        "Heldout video metrics are pending official inference.",
    ]
    Path(args.output_md).write_text("\n".join(md) + "\n")
    print(status)
    if status != "VOID_ADAPTER_10STEP_FORWARD_READY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
