#!/usr/bin/env python3
"""Run preregistered Exp37 MiniMax LocalDPO-badnoise 10-step recipes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


ACTIVE_RECIPES = {
    "R1": "LocalDPO-Linear-HardNoise",
    "R2": "LocalDPO-Linear-SDPO",
    "R3": "LocalDPO-SFTWarmup-Linear",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", required=True)
    p.add_argument("--project-root", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--train-states", required=True)
    p.add_argument("--heldout-states", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--checkpoint-root", required=True)
    p.add_argument("--reports-root", required=True)
    p.add_argument("--recipes", default="R1,R2,R3")
    p.add_argument("--seed", type=int, default=20260627)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--utility-scale", type=float, default=18.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lambda-winner-sft", type=float, default=0.05)
    p.add_argument("--lambda-outside", type=float, default=0.02)
    p.add_argument("--sdpo-min-lambda", type=float, default=0.0)
    p.add_argument("--sdpo-max-lambda", type=float, default=1.0)
    p.add_argument("--num-inference-steps", type=int, default=12)
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--heartbeat", default="")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


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
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def heartbeat(path: Path | None, text: str) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{time.time():.0f}\t{text}\n", encoding="utf-8")


def torch_dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "float16" else torch.bfloat16


def latent_region_masks(record: dict[str, object], z0: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    mask_frames = np.stack(record["mask_frames_uint8"], axis=0)
    mask = torch.from_numpy((mask_frames > 20).astype(np.float32))[None, None].to(device=device)
    mask = F.interpolate(mask, size=tuple(z0.shape[2:]), mode="nearest").clamp(0, 1)
    dil = F.max_pool3d(mask, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
    ero = 1.0 - F.max_pool3d(1.0 - mask, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
    boundary = (dil - ero).clamp(0, 1)
    outside = (1.0 - dil).clamp(0, 1)
    weights = torch.full_like(mask, 0.05)
    weights = torch.where(boundary > 0.5, torch.full_like(weights, 0.75), weights)
    weights = torch.where(mask > 0.5, torch.ones_like(weights), weights)
    return {
        "mask": mask.bool().expand_as(z0),
        "boundary": boundary.bool().expand_as(z0),
        "outside": outside.bool().expand_as(z0),
        "weights": weights.expand_as(z0),
    }


def region_mean(sq: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    if int(region.sum().detach().cpu()) <= 0:
        return sq.mean()
    return sq[region].mean()


def flow_components(model: torch.nn.Module, cache, row: dict, which: str, seed: int, tval: float) -> dict[str, torch.Tensor]:
    record = cache.row(row)
    z0 = record[which]
    assert isinstance(z0, torch.Tensor)
    gen = torch.Generator(device=cache.device).manual_seed(seed)
    eps = torch.randn(z0.shape, generator=gen, device=cache.device, dtype=cache.dtype)
    t = torch.tensor([tval], device=cache.device, dtype=cache.dtype)
    t_view = t.view(1, 1, 1, 1, 1)
    zt = t_view * eps + (1 - t_view) * z0
    target = (eps - z0).detach()
    hidden = torch.cat([zt, record["cond"], record["mask_latents"]], dim=1)
    pred = model(hidden_states=hidden, timestep=(t * 1000.0).float())[0]
    sq = (pred.float() - target.float()).pow(2)
    regions = latent_region_masks(record, z0, cache.device)
    weights = regions["weights"].float()
    return {
        "weighted": (sq * weights).sum() / weights.sum().clamp_min(1e-6),
        "full": sq.mean(),
        "mask": region_mean(sq, regions["mask"]),
        "boundary": region_mean(sq, regions["boundary"]),
        "outside": region_mean(sq, regions["outside"]),
    }


def recipe_loss(
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    cache,
    row: dict,
    recipe: str,
    scale: float,
    lambda_winner: float,
    lambda_outside: float,
    sdpo_lambda: float = 1.0,
    force_winner_sft: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    state = row["hard_state_A"]
    seed = int(state["noise_seed"])
    tval = float(state["t"])
    winner_policy = flow_components(policy, cache, row, "winner", seed, tval)
    loser_policy = flow_components(policy, cache, row, "loser", seed, tval)
    with torch.no_grad():
        winner_ref = flow_components(reference, cache, row, "winner", seed, tval)
        loser_ref = flow_components(reference, cache, row, "loser", seed, tval)
    win_gap = winner_ref["weighted"] - winner_policy["weighted"]
    lose_gap = loser_ref["weighted"] - loser_policy["weighted"]
    margin = win_gap - (float(sdpo_lambda) * lose_gap)
    utility = torch.clamp(0.2 * scale * margin + 0.5, 1e-6, 1.0)
    loss = -torch.log(utility)
    anchor = winner_policy["weighted"] * lambda_winner
    outside = 0.5 * (winner_policy["outside"] + loser_policy["outside"]) * lambda_outside
    if force_winner_sft:
        loss = anchor + winner_policy["outside"] * lambda_outside
        utility = torch.ones_like(utility)
        margin = torch.zeros_like(margin)
    else:
        loss = loss + anchor + outside
    diag = {
        "winner_policy_loss": float(winner_policy["weighted"].detach().cpu()),
        "loser_policy_loss": float(loser_policy["weighted"].detach().cpu()),
        "winner_reference_loss": float(winner_ref["weighted"].detach().cpu()),
        "loser_reference_loss": float(loser_ref["weighted"].detach().cpu()),
        "winner_policy_mask": float(winner_policy["mask"].detach().cpu()),
        "loser_policy_mask": float(loser_policy["mask"].detach().cpu()),
        "winner_policy_outside": float(winner_policy["outside"].detach().cpu()),
        "loser_policy_outside": float(loser_policy["outside"].detach().cpu()),
        "win_gap": float(win_gap.detach().cpu()),
        "lose_gap": float(lose_gap.detach().cpu()),
        "preference_margin": float(margin.detach().cpu()),
        "linear_utility": float(utility.detach().cpu()),
        "winner_anchor_loss": float(anchor.detach().cpu()),
        "outside_loss": float(outside.detach().cpu()),
        "loss": float(loss.detach().cpu()),
        "sdpo_lambda": float(sdpo_lambda),
        "force_winner_sft": bool(force_winner_sft),
        "noise_seed": seed,
        "t": tval,
    }
    return loss, diag


def grad_stats(model: torch.nn.Module) -> dict[str, float | int]:
    grad_sq = 0.0
    grad_max = 0.0
    grad_tensors = 0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().float()
            grad_sq += float((g * g).sum().cpu())
            grad_max = max(grad_max, float(g.abs().max().cpu()))
            grad_tensors += 1
    return {
        "grad_norm": math.sqrt(grad_sq) if math.isfinite(grad_sq) else float("nan"),
        "grad_max_abs": grad_max,
        "grad_tensors": grad_tensors,
    }


def safe_lambda_from_states(rows: list[dict], min_value: float, max_value: float) -> tuple[float, dict[str, float | int]]:
    """Compute a fixed loser-branch scale from the frozen bad-noise scan.

    This is an auditable Exp37 proxy for SDPO-style winner-safety: when the
    selected hard state has winner local residual close to or above loser local
    residual, the loser push is reduced. The value is fixed before training.
    """
    lambdas: list[float] = []
    violations = 0
    for row in rows:
        state = row.get("hard_state_A", {})
        winner = float(state.get("winner_local_score", state.get("winner_local_residual", 0.0)))
        loser = float(state.get("loser_local_score", state.get("loser_local_residual", 0.0)))
        if not (math.isfinite(winner) and math.isfinite(loser)) or loser <= 0:
            continue
        raw = max(0.0, min(1.0, winner / loser))
        if winner >= loser:
            violations += 1
        lambdas.append(max(min_value, min(max_value, raw)))
    if not lambdas:
        return float("nan"), {"valid_rows": 0, "winner_risk_rows": violations}
    return float(np.mean(lambdas)), {
        "valid_rows": len(lambdas),
        "winner_risk_rows": violations,
        "lambda_min": float(np.min(lambdas)),
        "lambda_max": float(np.max(lambdas)),
        "lambda_mean": float(np.mean(lambdas)),
        "lambda_median": float(np.median(lambdas)),
    }


def delta_probe(model_a: torch.nn.Module, model_b: torch.nn.Module, limit: int = 64) -> float:
    total = 0.0
    checked = 0
    with torch.no_grad():
        for (_, p), (_, q) in zip(model_a.named_parameters(), model_b.named_parameters()):
            total += float((p.detach().float() - q.detach().float()).abs().mean().cpu())
            checked += 1
            if checked >= limit:
                break
    return total / max(1, checked)


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"})


def read_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError(path)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


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


def comparison_strip(step0_dir: Path, step10_dir: Path, out_path: Path) -> dict[str, float]:
    step0_files = image_files(step0_dir)
    step10_files = image_files(step10_dir)
    n = min(len(step0_files), len(step10_files))
    if n == 0:
        raise RuntimeError(f"missing comparison frames: {step0_dir} {step10_dir}")
    tiles = []
    full_diffs = []
    for idx in sample_indices(n, 16):
        a = read_rgb(step0_files[idx])
        b = read_rgb(step10_files[idx])
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).clip(0, 255).astype(np.uint8)
        full_diffs.append(float(np.mean(diff)))
        tiles.append(labeled_tile(np.concatenate([a, b, diff], axis=1), f"f{idx:02d} step0|step10|diff"))
    save_rgb(out_path, np.concatenate(tiles, axis=0))
    return {"pixel_diff_mean_review_frames": float(np.mean(full_diffs)), "review_frame_count": len(tiles)}


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)


def mean_metric(rows: list[dict[str, object]], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and isinstance(r[key], (int, float)) and math.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    args = parse_args()
    recipes = [r.strip() for r in args.recipes.split(",") if r.strip()]
    if not recipes or any(r not in ACTIVE_RECIPES for r in recipes):
        raise ValueError(f"recipes must be subset of {sorted(ACTIVE_RECIPES)}")
    if args.steps > 10:
        raise ValueError("Exp37 rescue recipe gate is capped at 10 Linear-DPO steps")
    if args.warmup_steps > 5:
        raise ValueError("Exp37 R3 warmup is capped at 5 winner-SFT steps")

    sys.path.insert(0, str(Path(args.repo_dir).resolve()))
    sys.path.insert(0, str(Path(args.project_root).resolve()))
    from diffusers.models import AutoencoderKLWan  # noqa: WPS433
    from diffusers.schedulers import UniPCMultistepScheduler  # noqa: WPS433
    from transformer_minimax_remover import Transformer3DModel  # noqa: WPS433
    from exp30_vor_or_multimodel_minimax.scripts.run_minimax_gate64_adapter_gate_v3 import (  # noqa: WPS433
        BatchCache,
        run_pipeline,
    )

    output_root = Path(args.output_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    reports_root = Path(args.reports_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    hb = Path(args.heartbeat) if args.heartbeat else output_root / "rescue_10step.heartbeat"

    train_rows = read_jsonl(Path(args.train_states))
    heldout_rows = read_jsonl(Path(args.heldout_states))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch_dtype(args.dtype)
    model_dir = Path(args.model_dir).resolve()
    torch.manual_seed(args.seed)

    heartbeat(hb, "loading_vae")
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=dtype).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    latents_std = (1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)).to(device, dtype)
    cache = BatchCache(vae, latents_mean, latents_std, device, dtype)

    all_diag_rows: list[dict[str, object]] = []
    all_metric_rows: list[dict[str, object]] = []
    all_visual_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "status": "MINIMAX_LOCALDPO_BADNOISE_10STEP_REQUIRES_CODEX_VISUAL_REVIEW",
        "training_launched": True,
        "recipes": {},
        "seed": args.seed,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "lr": args.lr,
        "utility_scale": args.utility_scale,
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
        "target": "epsilon_minus_z0_flow_velocity",
    }

    start = time.time()
    for recipe in recipes:
        heartbeat(hb, f"{recipe}:load")
        recipe_name = ACTIVE_RECIPES[recipe]
        recipe_out = output_root / recipe
        recipe_ckpt = checkpoint_root / recipe
        if recipe_out.exists():
            shutil.rmtree(recipe_out)
        if recipe_ckpt.exists():
            shutil.rmtree(recipe_ckpt)
        recipe_out.mkdir(parents=True, exist_ok=True)
        recipe_ckpt.mkdir(parents=True, exist_ok=True)

        policy = Transformer3DModel.from_pretrained(model_dir / "transformer", torch_dtype=dtype).to(device).train()
        reference = Transformer3DModel.from_pretrained(model_dir / "transformer", torch_dtype=dtype).to(device).eval()
        for p in reference.parameters():
            p.requires_grad_(False)
        for p in policy.parameters():
            p.requires_grad_(True)
        sdpo_lambda = 1.0
        sdpo_preflight: dict[str, float | int | str] = {"status": "not_applicable"}
        if recipe == "R2":
            sdpo_lambda, sdpo_stats = safe_lambda_from_states(train_rows, args.sdpo_min_lambda, args.sdpo_max_lambda)
            sdpo_preflight = {"status": "passed" if math.isfinite(sdpo_lambda) else "failed", **sdpo_stats}
            if not math.isfinite(sdpo_lambda):
                summary["recipes"][recipe] = {
                    "name": recipe_name,
                    "status": "SDPO_GEOMETRY_UNSTABLE_SKIP",
                    "sdpo_preflight": sdpo_preflight,
                    "heldout_rows": 0,
                    "requires_visual_review": False,
                }
                del policy, reference
                torch.cuda.empty_cache()
                continue
        save_checkpoint(policy, recipe_ckpt / "checkpoint-0")
        zero_loss, zero_diag = recipe_loss(
            policy,
            reference,
            cache,
            train_rows[0],
            recipe,
            args.utility_scale,
            args.lambda_winner_sft,
            args.lambda_outside,
            sdpo_lambda,
        )
        zero_ok = (
            abs(zero_diag["win_gap"]) < 1e-6
            and abs(zero_diag["lose_gap"]) < 1e-6
            and abs(zero_diag["linear_utility"] - 0.5) < 1e-6
        )
        optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-4)
        nan_detected = False

        if recipe == "R3":
            for warmup_step in range(1, args.warmup_steps + 1):
                heartbeat(hb, f"{recipe}:warmup={warmup_step}/{args.warmup_steps}")
                row = train_rows[(warmup_step - 1) % len(train_rows)]
                optimizer.zero_grad(set_to_none=True)
                loss, diag = recipe_loss(
                    policy,
                    reference,
                    cache,
                    row,
                    recipe,
                    args.utility_scale,
                    args.lambda_winner_sft,
                    args.lambda_outside,
                    sdpo_lambda,
                    force_winner_sft=True,
                )
                loss.backward()
                stats = grad_stats(policy)
                finite = math.isfinite(diag["loss"]) and math.isfinite(float(stats["grad_norm"]))
                diag.update(stats)
                diag.update({
                    "recipe": recipe,
                    "recipe_name": recipe_name,
                    "phase": "winner_sft_warmup",
                    "step": warmup_step,
                    "sample_id": row["sample_id"],
                    "source_group": row.get("source_group", ""),
                    "finite": finite,
                })
                all_diag_rows.append(diag)
                if not finite:
                    nan_detected = True
                    break
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                optimizer.step()
            if not nan_detected:
                save_checkpoint(policy, recipe_ckpt / f"checkpoint-warmup-{args.warmup_steps}")

        for step in range(1, args.steps + 1):
            if nan_detected:
                break
            heartbeat(hb, f"{recipe}:step={step}")
            row = train_rows[(step - 1) % len(train_rows)]
            optimizer.zero_grad(set_to_none=True)
            loss, diag = recipe_loss(
                policy,
                reference,
                cache,
                row,
                recipe,
                args.utility_scale,
                args.lambda_winner_sft,
                args.lambda_outside,
                sdpo_lambda,
            )
            loss.backward()
            stats = grad_stats(policy)
            finite = math.isfinite(diag["loss"]) and math.isfinite(float(stats["grad_norm"]))
            diag.update(stats)
            diag.update({
                "recipe": recipe,
                "recipe_name": recipe_name,
                "phase": "linear_dpo",
                "step": step,
                "sample_id": row["sample_id"],
                "source_group": row.get("source_group", ""),
                "finite": finite,
            })
            all_diag_rows.append(diag)
            if not finite:
                nan_detected = True
                break
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()
            if step in {1, 5, 10}:
                save_checkpoint(policy, recipe_ckpt / f"checkpoint-{step}")

        base_reload = Transformer3DModel.from_pretrained(model_dir / "transformer", torch_dtype=dtype).to(device).eval()
        step1_reload = Transformer3DModel.from_pretrained(recipe_ckpt / "checkpoint-1", torch_dtype=dtype).to(device).eval()
        step10_path = recipe_ckpt / "checkpoint-10"
        step10_reload = Transformer3DModel.from_pretrained(step10_path, torch_dtype=dtype).to(device).eval() if step10_path.exists() else None
        step1_delta = delta_probe(step1_reload, base_reload)
        step10_delta = delta_probe(step10_reload, base_reload) if step10_reload is not None else float("nan")
        reference_delta = 0.0

        heldout_metric_rows = []
        if step10_reload is not None and not nan_detected:
            infer_vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float16).to(device).eval()
            base_infer = Transformer3DModel.from_pretrained(model_dir / "transformer", torch_dtype=torch.float16).to(device).eval()
            step10_infer = Transformer3DModel.from_pretrained(step10_path, torch_dtype=torch.float16).to(device).eval()
            for idx, row in enumerate(heldout_rows):
                sample_id = str(row["sample_id"])
                heartbeat(hb, f"{recipe}:heldout={idx + 1}/{len(heldout_rows)}:{sample_id}")
                sample_root = recipe_out / "heldout_outputs" / sample_id
                step0_metrics = run_pipeline(
                    base_infer,
                    infer_vae,
                    UniPCMultistepScheduler,
                    model_dir,
                    cache,
                    row,
                    sample_root / "step0",
                    args.seed,
                    args.num_inference_steps,
                    args.iterations,
                )
                step10_metrics = run_pipeline(
                    step10_infer,
                    infer_vae,
                    UniPCMultistepScheduler,
                    model_dir,
                    cache,
                    row,
                    sample_root / "step10",
                    args.seed,
                    args.num_inference_steps,
                    args.iterations,
                )
                strip_path = sample_root / "step0_vs_step10_strip.jpg"
                diff_stats = comparison_strip(Path(step0_metrics["frames_dir"]), Path(step10_metrics["frames_dir"]), strip_path)
                metric_row: dict[str, object] = {
                    "recipe": recipe,
                    "recipe_name": recipe_name,
                    "sample_id": sample_id,
                    "source_group": row.get("source_group", ""),
                    "step0_frames": step0_metrics["frames_dir"],
                    "step10_frames": step10_metrics["frames_dir"],
                    "step0_raw_output_mp4": step0_metrics["raw_output_mp4"],
                    "step10_raw_output_mp4": step10_metrics["raw_output_mp4"],
                    "step0_temporal_strip_16": step0_metrics["temporal_strip_16"],
                    "step10_temporal_strip_16": step10_metrics["temporal_strip_16"],
                    "step0_vs_step10_strip": str(strip_path),
                    **diff_stats,
                }
                for key in ("full_psnr", "mask_psnr", "boundary_psnr", "outside_psnr", "outside_mae", "temporal_diff_mae"):
                    metric_row[f"step0_{key}"] = step0_metrics.get(key, "")
                    metric_row[f"step10_{key}"] = step10_metrics.get(key, "")
                    if isinstance(step0_metrics.get(key), float) and isinstance(step10_metrics.get(key), float):
                        metric_row[f"delta_{key}"] = float(step10_metrics[key]) - float(step0_metrics[key])
                heldout_metric_rows.append(metric_row)
                all_metric_rows.append(metric_row)
                all_visual_rows.append({
                    "recipe": recipe,
                    "sample_id": sample_id,
                    "source_group": row.get("source_group", ""),
                    "condition_path": row.get("condition_path", ""),
                    "winner_path": row.get("winner_path", ""),
                    "loser_path": row.get("loser_path", ""),
                    "mask_path": row.get("mask_path", ""),
                    "step0_output": step0_metrics["raw_output_mp4"],
                    "step10_output": step10_metrics["raw_output_mp4"],
                    "step0_vs_step10_strip": str(strip_path),
                    "frames_reviewed": "PENDING_CODEX_REVIEW",
                    "pixel_diff_mean": diff_stats["pixel_diff_mean_review_frames"],
                    "mask_diff_mean": "",
                    "affected_diff_mean": "",
                    "outside_diff_mean": metric_row.get("delta_outside_mae", ""),
                    "object_removed": "PENDING_CODEX_REVIEW",
                    "effect_removed": "PENDING_CODEX_REVIEW",
                    "mask_region_quality": "PENDING_CODEX_REVIEW",
                    "boundary_quality": "PENDING_CODEX_REVIEW",
                    "affected_region_quality": "PENDING_CODEX_REVIEW",
                    "outside_damage": "PENDING_CODEX_REVIEW",
                    "temporal_flicker": "PENDING_CODEX_REVIEW",
                    "ghosting": "PENDING_CODEX_REVIEW",
                    "color_shift": "PENDING_CODEX_REVIEW",
                    "artifact": "PENDING_CODEX_REVIEW",
                    "classification": "PENDING_CODEX_VISUAL_REVIEW",
                    "reason": "",
                })
            del infer_vae, base_infer, step10_infer
            torch.cuda.empty_cache()

        summary["recipes"][recipe] = {
            "name": recipe_name,
            "zero_gap_status": "MINIMAX_ZERO_GAP_PASSED" if zero_ok else "MINIMAX_ZERO_GAP_FAILED",
            "zero_gap": zero_diag,
            "nan_detected": nan_detected,
            "step1_delta_probe": step1_delta,
            "step10_delta_probe": step10_delta,
            "reference_delta_probe": reference_delta,
            "sdpo_lambda": sdpo_lambda,
            "sdpo_preflight": sdpo_preflight,
            "mean_delta_full_psnr": mean_metric(heldout_metric_rows, "delta_full_psnr"),
            "mean_delta_mask_psnr": mean_metric(heldout_metric_rows, "delta_mask_psnr"),
            "mean_delta_boundary_psnr": mean_metric(heldout_metric_rows, "delta_boundary_psnr"),
            "mean_delta_outside_psnr": mean_metric(heldout_metric_rows, "delta_outside_psnr"),
            "mean_pixel_diff_review_frames": mean_metric(heldout_metric_rows, "pixel_diff_mean_review_frames"),
            "checkpoint_root": str(recipe_ckpt),
            "output_root": str(recipe_out),
            "heldout_rows": len(heldout_metric_rows),
            "peak_vram_mib": torch.cuda.max_memory_allocated() / 1024 / 1024,
            "requires_visual_review": True,
        }
        del policy, reference, base_reload, step1_reload, step10_reload
        torch.cuda.empty_cache()

    summary["runtime_seconds"] = time.time() - start
    write_json(reports_root / "exp37_minimax_localdpo_badnoise_10step_summary.json", summary)
    write_csv(reports_root / "exp37_minimax_localdpo_badnoise_10step_diagnostics.csv", all_diag_rows)
    write_csv(reports_root / "exp37_minimax_localdpo_badnoise_10step_metrics.csv", all_metric_rows)
    write_csv(reports_root / "exp37_minimax_localdpo_badnoise_10step_visual_review.csv", all_visual_rows)
    md = [
        "# Exp37 MiniMax LocalDPO-BadNoise 10-Step Recipes",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"- Recipes: `{','.join(recipes)}`",
        f"- Train rows: `{len(train_rows)}`",
        f"- Heldout rows: `{len(heldout_rows)}`",
        f"- Linear-DPO steps: `{args.steps}`",
        f"- R3 winner-SFT warmup steps: `{args.warmup_steps}`",
        f"- LR: `{args.lr}`",
        f"- Utility scale: `{args.utility_scale}`",
        "- Training launched: true.",
        "- 30-step launched: false.",
        "- Codex visual review is required before any pass or positive status.",
    ]
    (reports_root / "exp37_minimax_localdpo_badnoise_10step.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
