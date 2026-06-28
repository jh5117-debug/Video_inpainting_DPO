#!/usr/bin/env python3
"""Run the Exp40 MiniMax PSNR-safe winner-SFT warmup grid.

This is still a small gate, not long training.  It trains only winner
reconstruction on the locked LocalDPO-v3 VOR-Train pool and evaluates raw
MiniMax outputs against the already materialized Step0 baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SftRecipe:
    name: str
    mask_weight: float
    boundary_weight: float
    affected_weight: float
    outside_weight: float
    far_outside_weight: float
    use_hard_noise_schedule: bool = False


RECIPES: dict[str, SftRecipe] = {
    "SFT-A": SftRecipe("SFT-A", 1.00, 1.25, 0.75, 0.10, 0.02),
    "SFT-B": SftRecipe("SFT-B", 1.00, 1.50, 0.75, 0.15, 0.02),
    "SFT-C": SftRecipe("SFT-C", 0.75, 1.50, 0.75, 0.20, 0.03),
    "SFT-D": SftRecipe("SFT-D", 1.00, 1.50, 0.75, 0.15, 0.02, use_hard_noise_schedule=True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--search-manifest", required=True)
    parser.add_argument("--baseline-metrics-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--reports-root", required=True)
    parser.add_argument("--worker-name", default="worker0")
    parser.add_argument("--recipes", default="SFT-A,SFT-B,SFT-C,SFT-D")
    parser.add_argument("--lrs", default="3e-5,1e-4,3e-4")
    parser.add_argument("--scope", choices=["S0", "S1"], default="S0")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--train-rows", type=int, default=64)
    parser.add_argument("--search-rows", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--num-inference-steps", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--eval-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--heartbeat", default="")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def heartbeat(path: Path | None, message: str) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{time.time():.0f}\t{message}\n", encoding="utf-8")


def parse_recipes(value: str) -> list[SftRecipe]:
    out: list[SftRecipe] = []
    for name in (part.strip() for part in value.split(",")):
        if not name:
            continue
        if name not in RECIPES:
            raise ValueError(f"unknown SFT recipe {name!r}; allowed={sorted(RECIPES)}")
        out.append(RECIPES[name])
    if not out:
        raise ValueError("at least one recipe is required")
    return out


def parse_lrs(value: str) -> list[float]:
    lrs = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not lrs:
        raise ValueError("at least one lr is required")
    return lrs


def read_baseline_index(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[(str(row["split"]), str(row["sample_id"]))] = row
    return rows


def float_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"})


def read_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError(f"failed to read image: {path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def read_gray(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"failed to read image: {path}")
    return arr


def save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def sample_indices(n: int, count: int = 16) -> list[int]:
    if n <= count:
        return list(range(n))
    return sorted({int(round(i * (n - 1) / (count - 1))) for i in range(count)})


def labeled_tile(frame: np.ndarray, label: str, tile_w: int = 768) -> np.ndarray:
    h, w = frame.shape[:2]
    tile_h = max(1, int(round(h * tile_w / w)))
    tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    cv2.putText(tile, label, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return tile


def save_comparison_strip(step0_dir: Path, stepn_dir: Path, mask_dir: Path, out_path: Path) -> dict[str, float]:
    step0_files = image_files(step0_dir)
    stepn_files = image_files(stepn_dir)
    mask_files = image_files(mask_dir)
    n = min(len(step0_files), len(stepn_files), len(mask_files))
    if n == 0:
        raise RuntimeError(f"empty comparison inputs: {step0_dir} {stepn_dir} {mask_dir}")
    tiles: list[np.ndarray] = []
    diffs: list[np.ndarray] = []
    mask_diffs: list[float] = []
    outside_diffs: list[float] = []
    for idx in sample_indices(n, 16):
        base = read_rgb(step0_files[idx])
        cur = read_rgb(stepn_files[idx])
        mask = read_gray(mask_files[idx])
        if mask.shape[:2] != base.shape[:2]:
            mask = cv2.resize(mask, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)
        diff = np.abs(cur.astype(np.int16) - base.astype(np.int16)).clip(0, 255).astype(np.uint8)
        m = mask > 20
        kernel = np.ones((9, 9), np.uint8)
        outside = ~cv2.dilate(m.astype(np.uint8), kernel, iterations=1).astype(bool)
        if np.any(m):
            mask_diffs.append(float(np.mean(diff[m])))
        if np.any(outside):
            outside_diffs.append(float(np.mean(diff[outside])))
        diffs.append(diff)
        tiles.append(np.concatenate([base, cur, diff], axis=1))
    rows = [labeled_tile(tile, f"f{idx:02d} step0|stepN|diff") for tile, idx in zip(tiles, sample_indices(n, 16))]
    save_rgb(out_path, np.concatenate(rows, axis=0))
    all_diff = np.concatenate([d.reshape(-1, 3) for d in diffs], axis=0)
    return {
        "step0_stepn_full_mae": float(np.mean(all_diff)),
        "step0_stepn_mask_mae": float(np.mean(mask_diffs)) if mask_diffs else float("nan"),
        "step0_stepn_outside_mae": float(np.mean(outside_diffs)) if outside_diffs else float("nan"),
    }


def latent_region_weights(record: dict[str, object], z0: torch.Tensor, recipe: SftRecipe, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, dict[str, float]]:
    mask_frames = np.stack(record["mask_frames_uint8"], axis=0)
    cond_frames = np.stack(record["condition_frames_uint8"], axis=0).astype(np.float32)
    winner_frames = np.stack(record["winner_frames_uint8"], axis=0).astype(np.float32)
    mask_np = (mask_frames > 20).astype(np.float32)
    affected_np = (np.mean(np.abs(cond_frames - winner_frames), axis=-1) > 6.0).astype(np.float32)
    mask = torch.from_numpy(mask_np)[None, None].to(device=device, dtype=dtype)
    affected = torch.from_numpy(affected_np)[None, None].to(device=device, dtype=dtype)
    mask = F.interpolate(mask, size=tuple(z0.shape[2:]), mode="nearest").clamp(0, 1)
    affected = F.interpolate(affected, size=tuple(z0.shape[2:]), mode="nearest").clamp(0, 1)
    dil = F.max_pool3d(mask, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
    ero = 1.0 - F.max_pool3d(1.0 - mask, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
    boundary = (dil - ero).clamp(0, 1)
    near_outside = (dil - mask).clamp(0, 1)
    affected_outside = (affected * (1.0 - mask)).clamp(0, 1)
    weights = torch.full_like(mask, recipe.far_outside_weight)
    weights = torch.where(near_outside > 0.5, torch.full_like(weights, recipe.outside_weight), weights)
    weights = torch.where(affected_outside > 0.5, torch.full_like(weights, recipe.affected_weight), weights)
    weights = torch.where(boundary > 0.5, torch.full_like(weights, recipe.boundary_weight), weights)
    weights = torch.where(mask > 0.5, torch.full_like(weights, recipe.mask_weight), weights)
    ratios = {
        "mask_latent_ratio": float(mask.float().mean().detach().cpu()),
        "boundary_latent_ratio": float(boundary.float().mean().detach().cpu()),
        "affected_outside_latent_ratio": float(affected_outside.float().mean().detach().cpu()),
        "near_outside_latent_ratio": float(near_outside.float().mean().detach().cpu()),
    }
    return weights.expand_as(z0), ratios


def t_schedule(step: int, recipe: SftRecipe, seed: int) -> tuple[int, float]:
    if recipe.use_hard_noise_schedule:
        # Fixed before training: alternate higher-noise and mid-noise states.
        vals = (0.23, 0.37, 0.51, 0.67)
        return seed + 7919 * step, vals[(step - 1) % len(vals)]
    vals = (0.19, 0.29, 0.39, 0.49, 0.59)
    return seed + step, vals[(step - 1) % len(vals)]


def winner_sft_loss(model: torch.nn.Module, cache, row: dict[str, object], recipe: SftRecipe, seed: int, tval: float) -> tuple[torch.Tensor, dict[str, float]]:
    record = cache.row(row)
    z0 = record["winner"]
    assert isinstance(z0, torch.Tensor)
    gen = torch.Generator(device=cache.device).manual_seed(seed)
    eps = torch.randn(z0.shape, generator=gen, device=cache.device, dtype=cache.dtype)
    t = torch.tensor([tval], device=cache.device, dtype=cache.dtype)
    zt = t.view(1, 1, 1, 1, 1) * eps + (1 - t.view(1, 1, 1, 1, 1)) * z0
    target = (eps - z0).detach()
    hidden = torch.cat([zt, record["cond"], record["mask_latents"]], dim=1)
    pred = model(hidden_states=hidden, timestep=(t * 1000.0).float())[0]
    weights, ratios = latent_region_weights(record, z0, recipe, cache.device, cache.dtype)
    sq = (pred.float() - target.float()).pow(2)
    loss = (sq * weights.float()).sum() / weights.float().sum().clamp_min(1e-6)
    with torch.no_grad():
        out = {
            "loss": float(loss.detach().cpu()),
            "t": float(tval),
            "weight_mean": float(weights.float().mean().detach().cpu()),
            "target_norm": float(target.float().norm().detach().cpu()),
            "pred_norm": float(pred.float().norm().detach().cpu()),
        }
        out.update(ratios)
    return loss, out


def mean(rows: list[dict[str, object]], key: str) -> float:
    vals = [float_or_nan(row.get(key)) for row in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def gate30_status(rows: list[dict[str, object]]) -> tuple[str, dict[str, float]]:
    stats = {
        "mean_delta_full_psnr": mean(rows, "delta_full_psnr"),
        "mean_delta_mask_psnr": mean(rows, "delta_mask_psnr"),
        "mean_delta_boundary_psnr": mean(rows, "delta_boundary_psnr"),
        "mean_delta_outside_psnr": mean(rows, "delta_outside_psnr"),
        "mean_delta_temporal_diff_mae": mean(rows, "delta_temporal_diff_mae"),
        "mean_step0_stepn_full_mae": mean(rows, "step0_stepn_full_mae"),
        "mean_step0_stepn_mask_mae": mean(rows, "step0_stepn_mask_mae"),
        "mean_step0_stepn_outside_mae": mean(rows, "step0_stepn_outside_mae"),
    }
    passed = (
        stats["mean_delta_full_psnr"] >= 0.08
        and stats["mean_delta_mask_psnr"] >= 0.05
        and stats["mean_delta_boundary_psnr"] >= -0.02
        and stats["mean_delta_outside_psnr"] >= -0.02
        and stats["mean_delta_temporal_diff_mae"] <= 0.05
        and stats["mean_step0_stepn_full_mae"] > 0.01
    )
    if passed:
        return "NUMERIC_GATE30_PASS_REQUIRES_CODEX_VISUAL_REVIEW", stats
    if stats["mean_delta_full_psnr"] > 0 and (stats["mean_delta_boundary_psnr"] < -0.02 or stats["mean_delta_outside_psnr"] < -0.02):
        return "PSNR_GAIN_WITH_BOUNDARY_OR_OUTSIDE_COST", stats
    if stats["mean_step0_stepn_full_mae"] <= 0.01:
        return "NO_MEANINGFUL_OUTPUT_CHANGE", stats
    return "NUMERIC_GATE30_FAIL", stats


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.repo_dir).resolve()))
    sys.path.insert(0, str(Path(args.project_root).resolve()))

    from diffusers.models import AutoencoderKLWan  # noqa: WPS433
    from diffusers.schedulers import UniPCMultistepScheduler  # noqa: WPS433
    from transformer_minimax_remover import Transformer3DModel  # noqa: WPS433
    from exp30_vor_or_multimodel_minimax.scripts.run_minimax_gate64_adapter_gate_v3 import BatchCache, run_pipeline  # noqa: WPS433
    from exp36_minimax_objective_rescue.scripts.run_minimax_winner_sft_positive_control import (  # noqa: WPS433
        configure_trainable,
        delta_probe,
        grad_stats,
        load_transformer_for_scope,
        save_checkpoint,
        torch_dtype,
    )

    if args.steps > 30:
        raise ValueError("this Exp40 grid runner is capped at the 30-step gate")
    recipes = parse_recipes(args.recipes)
    lrs = parse_lrs(args.lrs)
    output_root = Path(args.output_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    reports_root = Path(args.reports_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    hb = Path(args.heartbeat) if args.heartbeat else output_root / f"{args.worker_name}.heartbeat"

    train_rows = read_jsonl(Path(args.train_manifest))[: args.train_rows]
    search_rows = read_jsonl(Path(args.search_manifest))[: args.search_rows]
    baseline_index = read_baseline_index(Path(args.baseline_metrics_csv))
    model_dir = Path(args.model_dir).resolve()
    for child in ("vae", "transformer", "scheduler"):
        if not (model_dir / child).exists():
            raise FileNotFoundError(f"missing MiniMax model component: {model_dir / child}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dtype = torch_dtype(args.dtype)
    eval_dtype = torch_dtype(args.eval_dtype)
    torch.manual_seed(args.seed)
    start = time.time()

    heartbeat(hb, "load_train_vae")
    train_vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=train_dtype).to(device).eval()
    for param in train_vae.parameters():
        param.requires_grad_(False)
    latents_mean = torch.tensor(train_vae.config.latents_mean).view(1, train_vae.config.z_dim, 1, 1, 1).to(device, train_dtype)
    latents_std = (1.0 / torch.tensor(train_vae.config.latents_std).view(1, train_vae.config.z_dim, 1, 1, 1)).to(device, train_dtype)
    train_cache = BatchCache(train_vae, latents_mean, latents_std, device, train_dtype)

    all_diag: list[dict[str, object]] = []
    all_metrics: list[dict[str, object]] = []
    all_visual: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "status": "MINIMAX_SFT_PSNRSAFE_30STEP_RUNNING",
        "worker_name": args.worker_name,
        "scope": args.scope,
        "steps": args.steps,
        "train_rows": len(train_rows),
        "search_rows": len(search_rows),
        "recipes": {},
        "raw_output_primary": True,
        "diagnostic_comp_used": False,
        "dpo_used": False,
        "vor_eval_used": False,
        "metric_note": "This runner reports existing MiniMax raw PSNR/mask/boundary/outside/temporal proxy metrics; LPIPS/Ewarp are not produced here.",
    }

    for recipe in recipes:
        for lr in lrs:
            name = f"{recipe.name}_{args.scope}_lr{lr:g}".replace("-", "m")
            heartbeat(hb, f"{name}:load")
            recipe_out = output_root / args.worker_name / name
            recipe_ckpt = checkpoint_root / args.worker_name / name
            if recipe_out.exists():
                shutil.rmtree(recipe_out)
            if recipe_ckpt.exists():
                shutil.rmtree(recipe_ckpt)
            recipe_out.mkdir(parents=True, exist_ok=True)
            recipe_ckpt.mkdir(parents=True, exist_ok=True)

            policy = load_transformer_for_scope(Transformer3DModel, model_dir, train_dtype, device, args.scope, args.seed).train()
            base_for_delta = load_transformer_for_scope(Transformer3DModel, model_dir, train_dtype, device, args.scope, args.seed).eval()
            for param in base_for_delta.parameters():
                param.requires_grad_(False)
            trainable_info = configure_trainable(policy, args.scope)
            save_checkpoint(policy, recipe_ckpt / "checkpoint-0", args.scope, {"recipe": name, "weights": recipe.__dict__, "trainable_info": trainable_info})
            optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-4)

            step_rows: list[dict[str, object]] = []
            nan_detected = False
            for step in range(1, args.steps + 1):
                row = train_rows[(step - 1) % len(train_rows)]
                seed, tval = t_schedule(step, recipe, args.seed)
                heartbeat(hb, f"{name}:train_step={step}/{args.steps}:{row['sample_id']}")
                optimizer.zero_grad(set_to_none=True)
                loss, diag = winner_sft_loss(policy, train_cache, row, recipe, seed, tval)
                loss.backward()
                stats = grad_stats(policy)
                finite = math.isfinite(diag["loss"]) and math.isfinite(float(stats["grad_norm"]))
                diag.update(stats)
                diag.update(
                    {
                        "worker": args.worker_name,
                        "recipe": name,
                        "recipe_family": recipe.name,
                        "step": step,
                        "lr": lr,
                        "scope": args.scope,
                        "sample_id": row["sample_id"],
                        "finite": finite,
                        "mask_weight": recipe.mask_weight,
                        "boundary_weight": recipe.boundary_weight,
                        "affected_weight": recipe.affected_weight,
                        "outside_weight": recipe.outside_weight,
                        "far_outside_weight": recipe.far_outside_weight,
                        "hard_noise_schedule": recipe.use_hard_noise_schedule,
                    }
                )
                step_rows.append(diag)
                if not finite:
                    nan_detected = True
                    break
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                optimizer.step()
                if step in {1, 10, 20, 30}:
                    save_checkpoint(policy, recipe_ckpt / f"checkpoint-{step}", args.scope, {"recipe": name, "step": step, "weights": recipe.__dict__, "trainable_info": trainable_info})

            stepn_path = recipe_ckpt / f"checkpoint-{args.steps}"
            strict_reload_ok = False
            param_delta = float("nan")
            if stepn_path.exists() and not nan_detected:
                reloaded = load_transformer_for_scope(Transformer3DModel, model_dir, train_dtype, device, args.scope, args.seed, stepn_path).eval()
                strict_reload_ok = True
                param_delta = delta_probe(reloaded, base_for_delta)
                del reloaded

            metric_rows: list[dict[str, object]] = []
            visual_rows: list[dict[str, object]] = []
            if strict_reload_ok and not nan_detected:
                heartbeat(hb, f"{name}:load_eval")
                eval_vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=eval_dtype).to(device).eval()
                eval_model = load_transformer_for_scope(Transformer3DModel, model_dir, eval_dtype, device, args.scope, args.seed, stepn_path).eval()
                eval_latents_mean = torch.tensor(eval_vae.config.latents_mean).view(1, eval_vae.config.z_dim, 1, 1, 1).to(device, eval_dtype)
                eval_latents_std = (1.0 / torch.tensor(eval_vae.config.latents_std).view(1, eval_vae.config.z_dim, 1, 1, 1)).to(device, eval_dtype)
                eval_cache = BatchCache(eval_vae, eval_latents_mean, eval_latents_std, device, eval_dtype)
                for idx, row in enumerate(search_rows, 1):
                    sample_id = str(row["sample_id"])
                    heartbeat(hb, f"{name}:search_eval={idx}/{len(search_rows)}:{sample_id}")
                    base_row = baseline_index.get(("search", sample_id))
                    if base_row is None:
                        raise KeyError(f"missing Step0 baseline row for search/{sample_id}")
                    stepn_metrics = run_pipeline(
                        eval_model,
                        eval_vae,
                        UniPCMultistepScheduler,
                        model_dir,
                        eval_cache,
                        row,
                        recipe_out / "search_outputs" / sample_id / f"step{args.steps}",
                        args.seed,
                        args.num_inference_steps,
                        args.iterations,
                    )
                    strip_path = recipe_out / "search_outputs" / sample_id / f"step0_vs_step{args.steps}_strip_16.jpg"
                    diff_stats = save_comparison_strip(Path(base_row["frames_dir"]), Path(stepn_metrics["frames_dir"]), Path(str(row["mask_path"])), strip_path)
                    out: dict[str, object] = {
                        "worker": args.worker_name,
                        "recipe": name,
                        "recipe_family": recipe.name,
                        "lr": lr,
                        "scope": args.scope,
                        "step": args.steps,
                        "split": "search",
                        "sample_id": sample_id,
                        "scene_group": row.get("scene_group", ""),
                        "source_type": row.get("source_type", ""),
                        "profile": row.get("profile", ""),
                        "condition_path": row.get("condition_path", ""),
                        "winner_path": row.get("winner_path", ""),
                        "loser_path": row.get("loser_path", ""),
                        "mask_path": row.get("mask_path", ""),
                        "raw_step0": base_row.get("raw_step0", ""),
                        "raw_stepN": stepn_metrics["raw_output_mp4"],
                        "step0_frames": base_row.get("frames_dir", ""),
                        "stepN_frames": stepn_metrics["frames_dir"],
                        "stepN_temporal_strip_16": stepn_metrics["temporal_strip_16"],
                        "stepN_review_sheet": stepn_metrics["review_sheet"],
                        "comparison_strip": str(strip_path),
                        "raw_output_primary": True,
                        "diagnostic_comp_used": False,
                    }
                    for key in ("full_psnr", "mask_psnr", "boundary_psnr", "outside_psnr", "outside_mae", "temporal_diff_mae"):
                        base_val = float_or_nan(base_row.get(key))
                        cur_val = float_or_nan(stepn_metrics.get(key))
                        out[f"step0_{key}"] = base_val
                        out[f"stepN_{key}"] = cur_val
                        out[f"delta_{key}"] = cur_val - base_val if math.isfinite(base_val) and math.isfinite(cur_val) else float("nan")
                    out.update(diff_stats)
                    metric_rows.append(out)
                    classification = "PENDING_CODEX_REVIEW"
                    if float_or_nan(out["delta_full_psnr"]) > 0 and float_or_nan(out["delta_boundary_psnr"]) < -0.02:
                        classification = "NUMERIC_GAIN_WITH_BOUNDARY_COST"
                    elif float_or_nan(out["delta_full_psnr"]) > 0 and float_or_nan(out["delta_outside_psnr"]) < -0.02:
                        classification = "NUMERIC_GAIN_WITH_OUTSIDE_COST"
                    elif float_or_nan(out["delta_full_psnr"]) >= 0.08 and float_or_nan(out["delta_mask_psnr"]) >= 0.05:
                        classification = "NUMERIC_LOCAL_IMPROVEMENT_REQUIRES_VISUAL"
                    elif float_or_nan(out["step0_stepn_full_mae"]) <= 0.01:
                        classification = "NO_VISIBLE_NUMERIC_CHANGE"
                    visual_rows.append(
                        {
                            "sample_id": sample_id,
                            "split": "search",
                            "recipe": name,
                            "checkpoint": f"step{args.steps}",
                            "condition_path": row.get("condition_path", ""),
                            "winner_path": row.get("winner_path", ""),
                            "loser_path": row.get("loser_path", ""),
                            "mask_path": row.get("mask_path", ""),
                            "raw_step0": base_row.get("raw_step0", ""),
                            "raw_stepN": stepn_metrics["raw_output_mp4"],
                            "comp_step0": "",
                            "comp_stepN": "",
                            "frames_reviewed": "0,mid,last,16-strip",
                            "full_psnr_delta": out["delta_full_psnr"],
                            "mask_psnr_delta": out["delta_mask_psnr"],
                            "boundary_psnr_delta": out["delta_boundary_psnr"],
                            "outside_psnr_delta": out["delta_outside_psnr"],
                            "lpips_delta": "NOT_PRODUCED_BY_THIS_MINIMAX_RUNNER",
                            "ewarp_delta": "NOT_PRODUCED_BY_THIS_MINIMAX_RUNNER",
                            "temporal_artifact": "PENDING_CODEX_REVIEW",
                            "spatial_artifact": "PENDING_CODEX_REVIEW",
                            "fogging": "PENDING_CODEX_REVIEW",
                            "over_erasure": "PENDING_CODEX_REVIEW",
                            "boundary_damage": "PENDING_CODEX_REVIEW",
                            "outside_damage": "PENDING_CODEX_REVIEW",
                            "classification": classification,
                            "reason": "automated metric prelabel; Codex visual review required before promotion",
                            "comparison_strip": str(strip_path),
                        }
                    )
                del eval_vae, eval_model, eval_cache
                torch.cuda.empty_cache()

            recipe_status, recipe_stats = gate30_status(metric_rows)
            all_diag.extend(step_rows)
            all_metrics.extend(metric_rows)
            all_visual.extend(visual_rows)
            summary["recipes"][name] = {
                "recipe_family": recipe.name,
                "scope": args.scope,
                "lr": lr,
                "weights": recipe.__dict__,
                "nan_detected": nan_detected,
                "strict_reload_ok": strict_reload_ok,
                "param_delta_probe": param_delta,
                "loss_start": step_rows[0]["loss"] if step_rows else float("nan"),
                "loss_end": step_rows[-1]["loss"] if step_rows else float("nan"),
                "loss_decrease": (float(step_rows[0]["loss"]) - float(step_rows[-1]["loss"])) if step_rows else float("nan"),
                "gate30_status": recipe_status,
                "gate30_stats": recipe_stats,
                "checkpoint_root": str(recipe_ckpt),
                "output_root": str(recipe_out),
            }
            del policy, base_for_delta
            torch.cuda.empty_cache()

    best_name = ""
    best_score = -1e9
    for name, record in summary["recipes"].items():
        stats = record["gate30_stats"]
        score = (
            float(stats["mean_delta_full_psnr"])
            + 0.5 * float(stats["mean_delta_mask_psnr"])
            + 0.4 * min(float(stats["mean_delta_boundary_psnr"]), 0.25)
            + 0.4 * min(float(stats["mean_delta_outside_psnr"]), 0.25)
            - max(float(stats["mean_delta_temporal_diff_mae"]), 0.0)
        )
        if math.isfinite(score) and score > best_score:
            best_score = score
            best_name = name
    statuses = [record["gate30_status"] for record in summary["recipes"].values()]
    if any(status == "NUMERIC_GATE30_PASS_REQUIRES_CODEX_VISUAL_REVIEW" for status in statuses):
        summary["status"] = "MINIMAX_SFT_PSNRSAFE_30STEP_NUMERIC_GATE_PASS_REQUIRES_VISUAL_REVIEW"
    elif any(status == "PSNR_GAIN_WITH_BOUNDARY_OR_OUTSIDE_COST" for status in statuses):
        summary["status"] = "MINIMAX_SFT_PSNRSAFE_PARETO_MIXED_AT_30STEP"
    elif any(status == "NO_MEANINGFUL_OUTPUT_CHANGE" for status in statuses):
        summary["status"] = "MINIMAX_SFT_PSNRSAFE_NO_OUTPUT_CHANGE_AT_30STEP"
    else:
        summary["status"] = "MINIMAX_SFT_PSNRSAFE_NEGATIVE_AT_30STEP"
    summary["best_recipe_by_numeric_score"] = best_name
    summary["runtime_seconds"] = time.time() - start

    prefix = f"exp40_minimax_sft_psnr_safe_grid_{args.worker_name}"
    write_csv(reports_root / f"{prefix}_diagnostics.csv", all_diag)
    write_csv(reports_root / f"{prefix}_metrics.csv", all_metrics)
    write_csv(reports_root / f"{prefix}_visual_review.csv", all_visual)
    write_json(reports_root / f"{prefix}_summary.json", summary)
    md = [
        "# Exp40 MiniMax PSNR-Safe SFT Warmup Grid Worker",
        "",
        f"Status: `{summary['status']}`",
        f"Worker: `{args.worker_name}`",
        f"Scope: `{args.scope}`",
        f"Steps: `{args.steps}`",
        f"Train rows: `{len(train_rows)}`",
        f"Search rows: `{len(search_rows)}`",
        "",
        "This worker uses raw output as the primary evaluation output. Diagnostic comp is not used.",
        "",
        "Metric note: LPIPS/Ewarp are not produced by this existing MiniMax runner; no substitute values are invented.",
        "",
        "Recipe summary:",
    ]
    for name, record in summary["recipes"].items():
        stats = record["gate30_stats"]
        md.append(
            f"- `{name}`: `{record['gate30_status']}`, "
            f"full `{stats['mean_delta_full_psnr']:.6f}`, "
            f"mask `{stats['mean_delta_mask_psnr']:.6f}`, "
            f"boundary `{stats['mean_delta_boundary_psnr']:.6f}`, "
            f"outside `{stats['mean_delta_outside_psnr']:.6f}`, "
            f"temporal `{stats['mean_delta_temporal_diff_mae']:.6f}`"
        )
    (reports_root / f"{prefix}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
