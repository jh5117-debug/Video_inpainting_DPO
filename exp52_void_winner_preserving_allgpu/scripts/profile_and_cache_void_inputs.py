#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from exp50_pai_void_adapter_feasibility.void_preference_wrapper.void_sft_wrapper import (
    VoidPaths,
    forward_noise_pred,
    load_components,
    load_transformer_clone,
    make_target_pack,
    weighted_mse,
)
from exp51_void_loser_dominant_rescue.scripts.run_void_rescue_onestep_grid import (
    RECIPES,
    recipe_loss,
    recipe_weights,
)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def shell_line(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNAVAILABLE {exc!r}"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Timer:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def run(self, name: str, fn):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        self.rows.append({"stage": name, "seconds": elapsed})
        return out


def tensor_to_cpu(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: tensor_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [tensor_to_cpu(v) for v in obj]
    return obj


def tensor_to_device(obj: Any, device: torch.device, dtype: torch.dtype | None = None) -> Any:
    if torch.is_tensor(obj):
        if dtype is not None and obj.is_floating_point() and obj.ndim > 0:
            return obj.to(device=device, dtype=dtype)
        return obj.to(device=device)
    if isinstance(obj, dict):
        return {k: tensor_to_device(v, device, dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tensor_to_device(v, device, dtype) for v in obj]
    return obj


def adapter_state(model: Any) -> dict[str, torch.Tensor]:
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def profile_row0(args, paths: VoidPaths, components, policy, frames: int, device: torch.device, dtype: torch.dtype):
    timer = Timer()
    train_rows = read_jsonl(args.train_manifest)
    row0 = train_rows[0]
    recipe = RECIPES["R1_WinnerPreserve_LocalDPO"]

    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    policy.train()
    seed = args.seed

    def make_seeded_winner():
        set_all_seeds(seed * 10 + 1)
        return make_target_pack(
            row0,
            components,
            device,
            dtype,
            frames=frames,
            size=(args.width, args.height),
            seed=seed,
            timestep=args.timestep,
            target_key="winner_path",
        )

    def make_seeded_loser():
        set_all_seeds(seed * 10 + 2)
        return make_target_pack(
            row0,
            components,
            device,
            dtype,
            frames=frames,
            size=(args.width, args.height),
            seed=seed,
            timestep=args.timestep,
            target_key="loser_path",
        )

    winner = timer.run(
        "make_winner_pack_decode_vae_text_noise",
        make_seeded_winner,
    )
    loser = timer.run(
        "make_loser_pack_decode_vae_text_noise",
        make_seeded_loser,
    )
    pref_w, anchor_w = timer.run(
        "quadmask_region_weight_resize",
        lambda: recipe_weights(row0, recipe, frames, (args.width, args.height), winner["target"].shape, device, dtype),
    )
    loser_w, _ = timer.run(
        "loser_region_weight_resize",
        lambda: recipe_weights(row0, recipe, frames, (args.width, args.height), loser["target"].shape, device, dtype),
    )
    policy.eval()
    with torch.no_grad():
        wr = timer.run("reference_winner_forward", lambda: forward_noise_pred(policy, components, winner, args.height, args.width))
        lr = timer.run("reference_loser_forward", lambda: forward_noise_pred(policy, components, loser, args.height, args.width))
        winner_ref = weighted_mse(wr, winner["target"], pref_w).detach()
        loser_ref = weighted_mse(lr, loser["target"], loser_w).detach()
    item = {
        "row_index": 0,
        "sample_id": row0.get("sample_id"),
        "seed": seed,
        "winner": winner,
        "loser": loser,
        "pref_w": pref_w,
        "anchor_w": anchor_w,
        "loser_w": loser_w,
        "winner_reference_loss": winner_ref,
        "loser_reference_loss": loser_ref,
        "prediction_type": winner["prediction_type"],
        "same_noise": bool(torch.equal(winner["noise"], loser["noise"])),
        "same_timestep": bool(torch.equal(winner["timesteps"], loser["timesteps"])),
    }
    policy.train()
    loss, info = timer.run("policy_forward_and_loss", lambda: recipe_loss(policy, components, item, recipe, args.height, args.width))
    policy.zero_grad(set_to_none=True)
    timer.run("backward_no_optimizer_step", lambda: loss.backward())
    grad_norm = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
    ckpt_path = Path(args.output_root) / "cache" / "profile_row0_adapter_state_no_step.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    timer.run("checkpoint_save_no_optimizer_step", lambda: torch.save({"adapter_state": adapter_state(policy), "note": "profile only; no optimizer step"}, ckpt_path))

    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    else:
        peak_alloc = 0.0
        peak_reserved = 0.0
    summary = {
        "sample_id": row0.get("sample_id"),
        "frames": frames,
        "seed": seed,
        "timestep": args.timestep,
        "prediction_type": winner["prediction_type"],
        "same_noise": item["same_noise"],
        "same_timestep": item["same_timestep"],
        "winner_reference_loss": float(winner_ref.detach().cpu()),
        "loser_reference_loss": float(loser_ref.detach().cpu()),
        "profile_loss": float(loss.detach().cpu()),
        "winner_gap": float(info["winner_gap"].cpu()),
        "loser_gap": float(info["loser_gap"].cpu()),
        "preference_margin": float(info["preference_margin"].cpu()),
        "grad_norm_before_clip": float(grad_norm.detach().cpu()),
        "grad_finite": bool(torch.isfinite(grad_norm).item()),
        "checkpoint_save_path": str(ckpt_path),
        "optimizer_step": False,
        "peak_vram_allocated_gib": peak_alloc,
        "peak_vram_reserved_gib": peak_reserved,
    }
    return timer.rows, summary, item


def select_variant_rows(train_rows: list[dict[str, Any]], heldout_rows: list[dict[str, Any]], variant_manifest: Path) -> dict[str, list[dict[str, Any]]]:
    rows = read_jsonl(variant_manifest)
    by_id = {r.get("sample_id"): r for r in rows}
    train_ids = [r.get("sample_id") for r in train_rows]
    heldout_ids = [r.get("sample_id") for r in heldout_rows]
    return {
        "train4": [by_id[sid] for sid in train_ids if sid in by_id],
        "heldout4": [by_id[sid] for sid in heldout_ids if sid in by_id],
    }


def cache_rows(args, paths: VoidPaths, components, policy, frames: int, device: torch.device, dtype: torch.dtype):
    cache_root = Path(args.output_root) / "cache" / "tensor_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(args.train_manifest)
    heldout_rows = read_jsonl(args.heldout_manifest)
    variant_paths = {
        "q0_current": Path("manifests/exp51_void_quadmask_ablation_q0_current.jsonl"),
        "q1_object_only": Path("manifests/exp51_void_quadmask_ablation_q1_object_only.jsonl"),
        "q2_strict_affected": Path("manifests/exp51_void_quadmask_ablation_q2_strict_affected.jsonl"),
        "q3_broad_affected": Path("manifests/exp51_void_quadmask_ablation_q3_broad_affected.jsonl"),
    }
    recipe = RECIPES["R1_WinnerPreserve_LocalDPO"]
    manifest_rows: list[dict[str, Any]] = []
    policy.eval()
    for variant, manifest in variant_paths.items():
        if not manifest.exists():
            continue
        groups = select_variant_rows(train_rows, heldout_rows, manifest)
        for split, rows in groups.items():
            for idx, row in enumerate(rows):
                seed = args.seed + idx
                t0 = time.perf_counter()
                set_all_seeds(seed * 10 + 1)
                winner = make_target_pack(row, components, device, dtype, frames=frames, size=(args.width, args.height), seed=seed, timestep=args.timestep, target_key="winner_path")
                set_all_seeds(seed * 10 + 2)
                loser = make_target_pack(row, components, device, dtype, frames=frames, size=(args.width, args.height), seed=seed, timestep=args.timestep, target_key="loser_path")
                pref_w, anchor_w = recipe_weights(row, recipe, frames, (args.width, args.height), winner["target"].shape, device, dtype)
                loser_w, _ = recipe_weights(row, recipe, frames, (args.width, args.height), loser["target"].shape, device, dtype)
                with torch.no_grad():
                    wr = forward_noise_pred(policy, components, winner, args.height, args.width)
                    lr = forward_noise_pred(policy, components, loser, args.height, args.width)
                    winner_ref = weighted_mse(wr, winner["target"], pref_w).detach()
                    loser_ref = weighted_mse(lr, loser["target"], loser_w).detach()
                cache_file = cache_root / variant / split / f"{idx:02d}_{row.get('sample_id')}.pt"
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "variant": variant,
                    "split": split,
                    "row_index": idx,
                    "sample_id": row.get("sample_id"),
                    "seed": seed,
                    "timestep": args.timestep,
                    "frames": frames,
                    "width": args.width,
                    "height": args.height,
                    "prediction_type": winner["prediction_type"],
                    "same_noise": bool(torch.equal(winner["noise"], loser["noise"])),
                    "same_timestep": bool(torch.equal(winner["timesteps"], loser["timesteps"])),
                    "row": row,
                    "winner": tensor_to_cpu(winner),
                    "loser": tensor_to_cpu(loser),
                    "pref_w": tensor_to_cpu(pref_w),
                    "anchor_w": tensor_to_cpu(anchor_w),
                    "loser_w": tensor_to_cpu(loser_w),
                    "winner_reference_loss": tensor_to_cpu(winner_ref),
                    "loser_reference_loss": tensor_to_cpu(loser_ref),
                    "winner_reference_pred": tensor_to_cpu(wr),
                    "loser_reference_pred": tensor_to_cpu(lr),
                }
                torch.save(payload, cache_file)
                elapsed = time.perf_counter() - t0
                manifest_rows.append({
                    "variant": variant,
                    "split": split,
                    "row_index": idx,
                    "sample_id": row.get("sample_id"),
                    "cache_file": str(cache_file),
                    "sha256_cache": sha256_file(cache_file),
                    "condition_sha256": sha256_file(row["condition_path"]),
                    "winner_sha256": sha256_file(row["winner_path"]),
                    "loser_sha256": sha256_file(row["loser_path"]),
                    "quadmask_sha256": sha256_file(row["quadmask_0_path"]),
                    "seconds": elapsed,
                    "prediction_type": winner["prediction_type"],
                    "same_noise": bool(torch.equal(winner["noise"], loser["noise"])),
                    "same_timestep": bool(torch.equal(winner["timesteps"], loser["timesteps"])),
                })
                del winner, loser, pref_w, anchor_w, loser_w, wr, lr, winner_ref, loser_ref
                torch.cuda.empty_cache()
    return manifest_rows


def parity_check(args, components, policy, cache_rows: list[dict[str, Any]], device: torch.device, dtype: torch.dtype):
    recipe = RECIPES["R1_WinnerPreserve_LocalDPO"]
    row = next(r for r in cache_rows if r["variant"] == "q0_current" and r["split"] == "train4" and r["row_index"] == 0)
    cached = torch.load(row["cache_file"], map_location="cpu")
    cached_dev = tensor_to_device(cached, device, dtype)
    loss_cached, info_cached = recipe_loss(policy, components, cached_dev, recipe, args.height, args.width)
    uncached_row = cached["row"]
    set_all_seeds(cached["seed"] * 10 + 1)
    winner = make_target_pack(uncached_row, components, device, dtype, frames=cached["frames"], size=(args.width, args.height), seed=cached["seed"], timestep=args.timestep, target_key="winner_path")
    set_all_seeds(cached["seed"] * 10 + 2)
    loser = make_target_pack(uncached_row, components, device, dtype, frames=cached["frames"], size=(args.width, args.height), seed=cached["seed"], timestep=args.timestep, target_key="loser_path")
    pref_w, anchor_w = recipe_weights(uncached_row, recipe, cached["frames"], (args.width, args.height), winner["target"].shape, device, dtype)
    loser_w, _ = recipe_weights(uncached_row, recipe, cached["frames"], (args.width, args.height), loser["target"].shape, device, dtype)
    with torch.no_grad():
        wr = forward_noise_pred(policy, components, winner, args.height, args.width)
        lr = forward_noise_pred(policy, components, loser, args.height, args.width)
    uncached_item = {
        "winner": winner,
        "loser": loser,
        "pref_w": pref_w,
        "anchor_w": anchor_w,
        "loser_w": loser_w,
        "winner_reference_loss": weighted_mse(wr, winner["target"], pref_w).detach(),
        "loser_reference_loss": weighted_mse(lr, loser["target"], loser_w).detach(),
        "prediction_type": winner["prediction_type"],
        "same_noise": bool(torch.equal(winner["noise"], loser["noise"])),
        "same_timestep": bool(torch.equal(winner["timesteps"], loser["timesteps"])),
    }
    loss_uncached, info_uncached = recipe_loss(policy, components, uncached_item, recipe, args.height, args.width)
    records = []
    for key in ["winner_policy_loss", "winner_reference_loss", "loser_policy_loss", "loser_reference_loss", "winner_gap", "loser_gap", "preference_margin", "loss"]:
        cv = loss_cached if key == "loss" else info_cached[key]
        uv = loss_uncached if key == "loss" else info_uncached[key]
        diff = abs(float(cv.detach().cpu()) - float(uv.detach().cpu()))
        records.append({"metric": key, "cached": float(cv.detach().cpu()), "uncached": float(uv.detach().cpu()), "abs_diff": diff})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--void-weights", required=True)
    parser.add_argument("--train-manifest", default="manifests/exp50_void_adapter_train4_h20.jsonl")
    parser.add_argument("--heldout-manifest", default="manifests/exp50_void_adapter_heldout4_h20.jsonl")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timestep", type=int, default=500)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--trainable-filter", default="proj_out")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    timer = Timer()
    paths = VoidPaths(Path(args.repo), Path(args.base_model), Path(args.void_weights), Path(args.void_weights) / "void_pass1.safetensors")
    components = timer.run("component_load_scheduler_tokenizer_text_vae", lambda: load_components(paths, device=device, dtype=dtype, load_transformer=False))
    policy, missing, unexpected = timer.run(
        "transformer_load_pass1_proj_out_trainable",
        lambda: load_transformer_clone(paths, device, dtype, trainable_filter=args.trainable_filter, gradient_checkpointing=True),
    )
    frames = args.frames
    temporal_compression = int(getattr(components["vae"].config, "temporal_compression_ratio", 4))
    patch_size_t = getattr(policy.config, "patch_size_t", None)
    if patch_size_t is not None:
        latent_len = (frames - 1) // temporal_compression + 1
        remainder = latent_len % int(patch_size_t)
        if remainder:
            frames = max(1, frames - remainder * temporal_compression)

    profile_rows, profile_summary, _ = profile_row0(args, paths, components, policy, frames, device, dtype)
    profile_rows = timer.rows + profile_rows
    cache_manifest = cache_rows(args, paths, components, policy, frames, device, dtype)
    parity = parity_check(args, components, policy, cache_manifest, device, dtype)
    max_diff = max((r["abs_diff"] for r in parity), default=999.0)
    status = "VOID_CACHE_READY" if max_diff <= 1e-6 else ("VOID_CACHE_PARITY_EXPLAINED" if max_diff <= 1e-5 else "VOID_CACHE_BLOCKED")
    total_sec = time.perf_counter() - start

    with (reports / "exp52_slow_forward_profile.csv").open("w", newline="") as f:
        fields = ["stage", "seconds"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(profile_rows)

    with (reports / "exp52_cache_parity.csv").open("w", newline="") as f:
        fields = ["metric", "cached", "uncached", "abs_diff"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(parity)

    cache_manifest_path = reports / "exp52_cache_manifest.json"
    cache_manifest_path.write_text(json.dumps({
        "cache_root": str(Path(args.output_root) / "cache" / "tensor_cache"),
        "rows": cache_manifest,
    }, indent=2, sort_keys=True) + "\n")

    summary = {
        "status": status,
        "created": now(),
        "runtime_sec": total_sec,
        "device": str(device),
        "dtype": str(dtype),
        "frames": frames,
        "profile": profile_summary,
        "cache_rows": len(cache_manifest),
        "cache_variants": sorted({r["variant"] for r in cache_manifest}),
        "cache_splits": sorted({r["split"] for r in cache_manifest}),
        "parity_max_abs_diff": max_diff,
        "policy_missing_keys": len(missing),
        "policy_unexpected_keys": len(unexpected),
        "gpu_query_final": shell_line(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv"]),
        "optimizer_step": False,
        "vor_eval_used": False,
        "hard_comp_used": False,
    }
    (reports / "exp52_cache_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    md = f"""# Exp52 Slow-Forward Forensic

Status: `{status}`

## Profile

- Device: `{device}`
- Runtime: {total_sec:.2f} sec
- Frames after official patch-size adjustment: {frames}
- Peak allocated VRAM: {profile_summary['peak_vram_allocated_gib']:.3f} GiB
- Peak reserved VRAM: {profile_summary['peak_vram_reserved_gib']:.3f} GiB
- Row0 loss: {profile_summary['profile_loss']}
- Grad finite: {profile_summary['grad_finite']}
- Optimizer step: no

Slow stages are recorded in `reports/exp52_slow_forward_profile.csv`.

## Interpretation

The Exp51 blocker came from doing VAE/text/quadmask encoding and base-policy reference forwards inside each recipe process. Exp52 fixes that by materializing CPU tensor caches for train4/heldout4 under Q0/Q1/Q2/Q3, including fixed noise, timestep, text embeddings, VAE latents, inpaint latents, region weights, scheduler target, and reference predictions/losses.
"""
    (reports / "exp52_slow_forward_forensic.md").write_text(md)

    plan = f"""# Exp52 Cache Plan

Status: `{status}`

Cache root:

`{Path(args.output_root) / 'cache' / 'tensor_cache'}`

Cached artifacts:

- rgb_full-derived condition latents
- winner VAE latents and v-prediction targets
- loser VAE latents and v-prediction targets
- quadmask/inpaint latents
- prompt tokens/text embeddings
- fixed timestep `{args.timestep}`
- fixed noise seeds beginning at `{args.seed}`
- object/affected/boundary/outside recipe weights
- reference predictions/losses for R1 row-compatible parity
- source path SHA256 metadata

No VOR-Eval, hard comp, optimizer step, or official VOID source edit was used.
"""
    (reports / "exp52_cache_plan.md").write_text(plan)
    print(status)


if __name__ == "__main__":
    main()
