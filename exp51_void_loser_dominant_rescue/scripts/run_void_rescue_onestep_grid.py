#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from exp50_pai_void_adapter_feasibility.void_preference_wrapper.void_sft_wrapper import (
    VoidPaths,
    forward_noise_pred,
    load_components,
    load_micro_row,
    load_transformer_clone,
    make_target_pack,
    weighted_mse,
)

SH_TZ = timezone(timedelta(hours=8))


RECIPES = {
    "R1_WinnerPreserve_LocalDPO": {
        "beta": 0.1,
        "winner_anchor": 0.05,
        "loser_grad_scale": 0.0,
        "loser_gap_clip_tau": None,
        "linear": False,
        "object": 1.0,
        "affected": 0.75,
        "boundary": 0.0,
        "outside": 0.0,
        "anchor_boundary": 0.05,
        "anchor_outside": 0.10,
    },
    "R2_WinnerPreserve_LoserClip": {
        "beta": 0.1,
        "winner_anchor": 0.05,
        "loser_grad_scale": 0.1,
        "loser_gap_clip_tau": 0.0005,
        "linear": False,
        "object": 1.0,
        "affected": 0.75,
        "boundary": 0.75,
        "outside": 0.0,
        "anchor_boundary": 0.05,
        "anchor_outside": 0.10,
    },
    "R3_SDPO_Safe": {
        "beta": 0.1,
        "winner_anchor": 0.05,
        "loser_grad_scale": 0.1,
        "loser_gap_clip_tau": None,
        "linear": False,
        "object": 1.0,
        "affected": 0.75,
        "boundary": 0.75,
        "outside": 0.0,
        "anchor_boundary": 0.05,
        "anchor_outside": 0.10,
    },
    "R4_LinearDPO_vPrediction": {
        "beta": 0.1,
        "winner_anchor": 0.02,
        "loser_grad_scale": 0.1,
        "loser_gap_clip_tau": None,
        "linear": True,
        "object": 1.0,
        "affected": 0.75,
        "boundary": 0.75,
        "outside": 0.0,
        "anchor_boundary": 0.05,
        "anchor_outside": 0.10,
    },
}


def now() -> str:
    return datetime.now(SH_TZ).replace(microsecond=0).isoformat()


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def adapter_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def load_adapter_state(model, state):
    own = dict(model.named_parameters())
    missing, unexpected = [], []
    for name, tensor in state.items():
        if name not in own:
            unexpected.append(name)
            continue
        own[name].data.copy_(tensor.to(device=own[name].device, dtype=own[name].dtype))
    for name, param in own.items():
        if param.requires_grad and name not in state:
            missing.append(name)
    return missing, unexpected


def save_adapter(path: Path, model, recipe: str, args, frames: int, diagnostics: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_state": adapter_state(model),
            "trainable_filter": args.trainable_filter,
            "recipe": recipe,
            "summary": {
                "step": 1,
                "lr": args.lr,
                "frames": frames,
                "seed": args.seed,
                "timestep": args.timestep,
                "created": now(),
                "diagnostics": diagnostics,
            },
        },
        path,
    )


def _read_quad(path: str | Path, frames: int, size: tuple[int, int]) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    got = []
    width, height = size
    while len(got) < frames:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, (width, height), interpolation=cv2.INTER_NEAREST)
        got.append(fr[..., 0])
    cap.release()
    if not got:
        raise RuntimeError(f"cannot decode quadmask {path}")
    while len(got) < frames:
        got.append(got[-1].copy())
    return np.stack(got[:frames], axis=0)


def _resize_latent_weight(weight: torch.Tensor, latent_shape, dtype) -> torch.Tensor:
    weight = weight.unsqueeze(0).unsqueeze(0)
    _, latent_f, _, latent_h, latent_w = tuple(latent_shape)
    weight = F.interpolate(weight, size=(latent_f, latent_h, latent_w), mode="trilinear", align_corners=False)
    weight = rearrange(weight, "b c f h w -> b f c h w")
    return weight.to(dtype=dtype)


def recipe_weights(row: dict, recipe: dict, frames: int, size: tuple[int, int], latent_shape, device, dtype):
    q = torch.from_numpy(_read_quad(row["quadmask_0_path"], frames, size).astype(np.float32)).to(device=device)
    object_mask = (q <= 95).float()
    affected = ((q > 31) & (q <= 191)).float()
    outside = (q > 191).float()
    boundary_frames = []
    kernel = np.ones((9, 9), np.uint8)
    for fr in object_mask.detach().cpu().numpy().astype(np.uint8):
        dil = cv2.dilate(fr, kernel, iterations=1)
        ero = cv2.erode(fr, kernel, iterations=1)
        boundary_frames.append(np.clip(dil - ero, 0, 1))
    boundary = torch.from_numpy(np.stack(boundary_frames).astype(np.float32)).to(device=device)
    pref = object_mask * recipe["object"] + affected * recipe["affected"] + boundary * recipe["boundary"] + outside * recipe["outside"]
    pref = pref.clamp_min(0.001)
    anchor = pref + boundary * recipe["anchor_boundary"] + outside * recipe["anchor_outside"]
    anchor = anchor.clamp_min(0.001)
    return _resize_latent_weight(pref, latent_shape, dtype), _resize_latent_weight(anchor, latent_shape, dtype)


def build_cache(base_model, components, rows: list[dict], recipe: dict, args, frames: int, device, dtype):
    cache = []
    base_model.eval()
    with torch.no_grad():
        for row_index, row in enumerate(rows):
            seed = args.seed + row_index
            winner = make_target_pack(row, components, device, dtype, frames=frames, size=(args.width, args.height), seed=seed, timestep=args.timestep, target_key="winner_path")
            loser = make_target_pack(row, components, device, dtype, frames=frames, size=(args.width, args.height), seed=seed, timestep=args.timestep, target_key="loser_path")
            pref_w, anchor_w = recipe_weights(row, recipe, frames, (args.width, args.height), winner["target"].shape, device, dtype)
            loser_w, _ = recipe_weights(row, recipe, frames, (args.width, args.height), loser["target"].shape, device, dtype)
            wr = forward_noise_pred(base_model, components, winner, args.height, args.width)
            lr = forward_noise_pred(base_model, components, loser, args.height, args.width)
            cache.append({
                "row_index": row_index,
                "sample_id": row.get("sample_id"),
                "seed": seed,
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
            })
            del wr, lr
            torch.cuda.empty_cache()
    base_model.train()
    return cache


def recipe_loss(policy, components, item, recipe: dict, height: int, width: int):
    wp = forward_noise_pred(policy, components, item["winner"], height, width)
    lp = forward_noise_pred(policy, components, item["loser"], height, width)
    winner_policy_loss = weighted_mse(wp, item["winner"]["target"], item["pref_w"])
    loser_policy_loss = weighted_mse(lp, item["loser"]["target"], item["loser_w"])
    winner_anchor_loss = weighted_mse(wp, item["winner"]["target"], item["anchor_w"])
    winner_gap = item["winner_reference_loss"] - winner_policy_loss
    loser_gap = item["loser_reference_loss"] - loser_policy_loss
    scale = recipe["loser_grad_scale"]
    if recipe["loser_gap_clip_tau"] is not None:
        clipped = torch.clamp(loser_gap, min=-recipe["loser_gap_clip_tau"], max=recipe["loser_gap_clip_tau"])
        effective_loser = loser_gap.detach() + scale * (clipped - loser_gap.detach())
    else:
        effective_loser = loser_gap.detach() + scale * (loser_gap - loser_gap.detach())
    margin = winner_gap - effective_loser
    if recipe["linear"]:
        pref = -recipe["beta"] * margin
    else:
        pref = -F.logsigmoid(recipe["beta"] * margin)
    loss = pref + recipe["winner_anchor"] * winner_anchor_loss
    return loss, {
        "winner_policy_loss": winner_policy_loss.detach(),
        "winner_reference_loss": item["winner_reference_loss"],
        "loser_policy_loss": loser_policy_loss.detach(),
        "loser_reference_loss": item["loser_reference_loss"],
        "winner_gap": winner_gap.detach(),
        "loser_gap": loser_gap.detach(),
        "effective_loser_gap": effective_loser.detach(),
        "preference_margin": margin.detach(),
        "winner_anchor_loss": winner_anchor_loss.detach(),
        "loss": loss.detach(),
        "prediction_type": item["prediction_type"],
        "same_noise": item["same_noise"],
        "same_timestep": item["same_timestep"],
    }


def run_recipe(name: str, recipe: dict, paths: VoidPaths, components, rows, heldout_rows, args, frames, device, dtype) -> dict:
    policy, missing, unexpected = load_transformer_clone(paths, device, dtype, trainable_filter=args.trainable_filter, gradient_checkpointing=True)
    policy.train()
    cache = build_cache(policy, components, rows, recipe, args, frames, device, dtype)
    before = adapter_state(policy)
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    opt.zero_grad(set_to_none=True)
    diagnostics = []
    grad_ok = True
    total_loss = 0.0
    for item in cache:
        loss, info = recipe_loss(policy, components, item, recipe, args.height, args.width)
        (loss / len(cache)).backward()
        total_loss += float(loss.detach().cpu())
        diagnostics.append({
            "recipe": name,
            "row_index": item["row_index"],
            "sample_id": item["sample_id"],
            "seed": item["seed"],
            **{k: (float(v.cpu()) if torch.is_tensor(v) else v) for k, v in info.items()},
        })
    grad_norm = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
    grad_ok = grad_ok and bool(torch.isfinite(grad_norm).item())
    opt.step()
    after = adapter_state(policy)
    deltas = {k: float((after[k] - before[k]).float().norm().item()) for k in before}
    max_delta = max(deltas.values()) if deltas else 0.0
    ckpt = Path(args.output_root) / name / "checkpoints" / "adapter_proj_out_step1.pt"
    save_adapter(ckpt, policy, name, args, frames, diagnostics)
    torch.cuda.empty_cache()
    reload_model, _, _ = load_transformer_clone(paths, device, dtype, trainable_filter=args.trainable_filter, gradient_checkpointing=False)
    saved = torch.load(ckpt, map_location="cpu")
    missing_reload, unexpected_reload = load_adapter_state(reload_model, saved["adapter_state"])
    reload_model.eval()
    heldout_ok = True
    with torch.no_grad():
        for idx, row in enumerate(heldout_rows):
            pack = make_target_pack(row, components, device, dtype, frames=frames, size=(args.width, args.height), seed=args.seed + idx, timestep=args.timestep, target_key="winner_path")
            pred = forward_noise_pred(reload_model, components, pack, args.height, args.width)
            heldout_ok = heldout_ok and bool(torch.isfinite(pred).all().item())
            del pred, pack
    del reload_model, policy
    torch.cuda.empty_cache()
    return {
        "recipe": name,
        "status": "FORWARD_READY" if grad_ok and max_delta > 0 and not missing_reload and not unexpected_reload and heldout_ok else "FORWARD_BLOCKED",
        "checkpoint": str(ckpt),
        "frames": frames,
        "train_rows": len(rows),
        "heldout_rows": len(heldout_rows),
        "total_loss_mean": total_loss / max(len(cache), 1),
        "grad_norm_before_clip": float(grad_norm.detach().cpu()),
        "grad_finite": grad_ok,
        "max_param_delta_norm": max_delta,
        "reload_missing": missing_reload,
        "reload_unexpected": unexpected_reload,
        "heldout_forward_finite": heldout_ok,
        "policy_missing": len(missing),
        "policy_unexpected": len(unexpected),
        "diagnostics": diagnostics,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--void-weights", required=True)
    ap.add_argument("--manifest", default="manifests/exp50_void_adapter_train4_h20.jsonl")
    ap.add_argument("--heldout", default="manifests/exp50_void_adapter_heldout4_h20.jsonl")
    ap.add_argument("--output-root", default="/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp51_void_loser_dominant_rescue/rescue_onestep")
    ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--width", type=int, default=672)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--timestep", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--trainable-filter", default="proj_out")
    ap.add_argument("--recipes", default="R1_WinnerPreserve_LocalDPO,R2_WinnerPreserve_LoserClip,R3_SDPO_Safe,R4_LinearDPO_vPrediction")
    args = ap.parse_args()
    t0 = time.time()
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    paths = VoidPaths(Path(args.repo), Path(args.base_model), Path(args.void_weights), Path(args.void_weights) / "void_pass1.safetensors")
    components = load_components(paths, device=device, dtype=dtype, load_transformer=False)
    # Load a temporary transformer config to apply official frame truncation.
    tmp_model, _, _ = load_transformer_clone(paths, device, dtype, trainable_filter=args.trainable_filter, gradient_checkpointing=False)
    frames = args.frames
    temporal_compression = int(getattr(components["vae"].config, "temporal_compression_ratio", 4))
    patch_size_t = getattr(tmp_model.config, "patch_size_t", None)
    if patch_size_t is not None:
        latent_len = (frames - 1) // temporal_compression + 1
        remainder = latent_len % int(patch_size_t)
        if remainder:
            frames = max(1, frames - remainder * temporal_compression)
    del tmp_model
    torch.cuda.empty_cache()
    rows = read_jsonl(args.manifest)
    heldout_rows = read_jsonl(args.heldout)
    selected = [r.strip() for r in args.recipes.split(",") if r.strip()]
    results = []
    all_diag = []
    for name in selected:
        result = run_recipe(name, RECIPES[name], paths, components, rows, heldout_rows, args, frames, device, dtype)
        results.append({k: v for k, v in result.items() if k != "diagnostics"})
        all_diag.extend(result["diagnostics"])
    status = "VOID_RESCUE_ONESTEP_FORWARD_READY" if all(r["status"] == "FORWARD_READY" for r in results) else "VOID_RESCUE_ONESTEP_BLOCKED"
    summary = {
        "status": status,
        "created": now(),
        "runtime_sec": time.time() - t0,
        "device": str(device),
        "dtype": str(dtype),
        "frames": frames,
        "recipes": results,
        "training_run": "one_optimizer_step_per_recipe_only",
        "optimizer_step": True,
        "vor_eval_used": False,
        "hard_comp_used": False,
        "long_training": False,
    }
    Path("reports/exp51_void_rescue_onestep_forward_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with Path("reports/exp51_void_rescue_onestep_diagnostics.csv").open("w", newline="") as f:
        fields = list(all_diag[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_diag)
    with Path("reports/exp51_void_rescue_onestep_forward.csv").open("w", newline="") as f:
        fields = ["recipe", "status", "checkpoint", "train_rows", "heldout_rows", "total_loss_mean", "grad_norm_before_clip", "grad_finite", "max_param_delta_norm", "heldout_forward_finite"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fields})
    print(status)
    if status != "VOID_RESCUE_ONESTEP_FORWARD_READY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
