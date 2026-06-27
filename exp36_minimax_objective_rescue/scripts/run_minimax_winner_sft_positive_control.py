#!/usr/bin/env python3
"""Run the Exp36 MiniMax winner-SFT positive-control micro gate."""

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
from torch import nn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", required=True)
    p.add_argument("--project-root", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--heldout-manifest", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--checkpoint-root", required=True)
    p.add_argument("--reports-root", required=True)
    p.add_argument("--seed", type=int, default=20260627)
    p.add_argument("--scopes", default="S0,S1")
    p.add_argument("--lrs", default="1e-5,3e-5,1e-4")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--train-rows", type=int, default=4)
    p.add_argument("--heldout-rows", type=int, default=4)
    p.add_argument("--num-inference-steps", type=int, default=12)
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--heartbeat", default="")
    return p.parse_args()


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16) -> None:
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.scale = float(alpha) / float(rank)
        nn.init.normal_(self.lora_a.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_b.weight)
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_a.to(device=device, dtype=dtype)
        self.lora_b.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale


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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def heartbeat(path: Path | None, text: str) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{time.time():.0f}\t{text}\n", encoding="utf-8")


def torch_dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "float16" else torch.bfloat16


def get_submodule(root: nn.Module, path: str) -> nn.Module:
    module: nn.Module = root
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def set_submodule(root: nn.Module, path: str, value: nn.Module) -> None:
    if "." not in path:
        setattr(root, path, value)
        return
    parent_path, name = path.rsplit(".", 1)
    parent = get_submodule(root, parent_path)
    if name.isdigit():
        parent[int(name)] = value  # type: ignore[index]
    else:
        setattr(parent, name, value)


def s1_lora_target(name: str, module: nn.Module) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if ".attn1.to_q" in name or ".attn1.to_k" in name or ".attn1.to_v" in name:
        return True
    if ".attn1.to_out.0" in name:
        return True
    return name == "proj_out"


def apply_s1_lora(model: nn.Module, seed: int, rank: int = 8, alpha: int = 16) -> list[str]:
    torch.manual_seed(seed)
    targets = [name for name, module in model.named_modules() if s1_lora_target(name, module)]
    for name in targets:
        base = get_submodule(model, name)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"S1 target is not Linear: {name}")
        set_submodule(model, name, LoRALinear(base, rank=rank, alpha=alpha))
    return targets


def configure_trainable(model: nn.Module, scope: str) -> dict[str, object]:
    if scope == "S0":
        for param in model.parameters():
            param.requires_grad_(True)
        names = [name for name, param in model.named_parameters() if param.requires_grad]
        return {
            "scope": scope,
            "trainable_tensors": len(names),
            "trainable_names_preview": names[:32],
            "lora_targets": [],
        }
    if scope == "S1":
        for name, param in model.named_parameters():
            param.requires_grad_("lora_" in name)
        names = [name for name, param in model.named_parameters() if param.requires_grad]
        return {
            "scope": scope,
            "trainable_tensors": len(names),
            "trainable_names_preview": names[:32],
            "lora_targets": sorted({name.rsplit(".", 2)[0] for name in names}),
        }
    raise ValueError(f"unsupported scope: {scope}")


def load_transformer_for_scope(transformer_cls, model_dir: Path, dtype: torch.dtype, device: torch.device, scope: str, seed: int, checkpoint_dir: Path | None = None) -> nn.Module:
    model = transformer_cls.from_pretrained(model_dir / "transformer", torch_dtype=dtype).to(device)
    if scope == "S1":
        apply_s1_lora(model, seed)
    elif scope != "S0":
        raise ValueError(f"unsupported scope: {scope}")
    if checkpoint_dir is not None:
        state_path = checkpoint_dir / "model_state.pt"
        state = torch.load(state_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"strict load failed for {checkpoint_dir}: missing={missing} unexpected={unexpected}")
    return model


def save_checkpoint(model: nn.Module, path: Path, scope: str, metadata: dict[str, object]) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model_state.pt")
    write_json(path / "metadata.json", {"scope": scope, **metadata})


def latent_region_weights(record: dict, z0: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask_frames = np.stack(record["mask_frames_uint8"], axis=0)
    mask = torch.from_numpy((mask_frames > 20).astype(np.float32))[None, None].to(device=device, dtype=dtype)
    mask = F.interpolate(mask, size=tuple(z0.shape[2:]), mode="nearest").clamp(0, 1)
    dil = F.max_pool3d(mask, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
    ero = 1.0 - F.max_pool3d(1.0 - mask, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
    boundary = (dil - ero).clamp(0, 1)
    weights = torch.full_like(mask, 0.05)
    weights = torch.where(boundary > 0.5, torch.full_like(weights, 0.75), weights)
    weights = torch.where(mask > 0.5, torch.ones_like(weights), weights)
    return weights.expand_as(z0)


def winner_sft_loss(model: torch.nn.Module, cache, row: dict, seed: int, tval: float) -> tuple[torch.Tensor, dict[str, float]]:
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
    weights = latent_region_weights(record, z0, cache.device, cache.dtype)
    sq = (pred.float() - target.float()).pow(2)
    loss = (sq * weights.float()).sum() / weights.float().sum().clamp_min(1e-6)
    with torch.no_grad():
        mask_mean = float((sq * (weights > 0.9).float()).sum().detach().cpu() / (weights > 0.9).float().sum().detach().cpu().clamp_min(1e-6))
        outside_mean = float((sq * (weights < 0.1).float()).sum().detach().cpu() / (weights < 0.1).float().sum().detach().cpu().clamp_min(1e-6))
    return loss, {
        "loss": float(loss.detach().cpu()),
        "mask_weighted_mse_proxy": mask_mean,
        "outside_mse_proxy": outside_mean,
        "t": tval,
    }


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


def read_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError(path)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"})


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


def contact_sheet(frames: list[np.ndarray], labels: list[str]) -> np.ndarray:
    tiles = [labeled_tile(f, l) for f, l in zip(frames, labels)]
    return np.concatenate(tiles, axis=0)


def comparison_strip(step0_dir: Path, step10_dir: Path, out_path: Path) -> None:
    a_files = image_files(step0_dir)
    b_files = image_files(step10_dir)
    n = min(len(a_files), len(b_files))
    frames = []
    labels = []
    for idx in sample_indices(n, 16):
        a = read_rgb(a_files[idx])
        b = read_rgb(b_files[idx])
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).clip(0, 255).astype(np.uint8)
        frames.append(np.concatenate([a, b, diff], axis=1))
        labels.append(f"f{idx:02d} step0|step10|diff")
    save_rgb(out_path, contact_sheet(frames, labels))


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.repo_dir).resolve()))
    sys.path.insert(0, str(Path(args.project_root).resolve()))
    from diffusers.models import AutoencoderKLWan  # noqa: WPS433
    from diffusers.schedulers import UniPCMultistepScheduler  # noqa: WPS433
    from transformer_minimax_remover import Transformer3DModel  # noqa: WPS433
    from exp30_vor_or_multimodel_minimax.scripts.run_minimax_gate64_adapter_gate_v3 import (  # noqa: WPS433
        BatchCache,
        aggregate_metrics,
        read_jsonl,
        run_pipeline,
    )

    if args.steps > 10:
        raise ValueError("winner-SFT positive control is capped at 10 steps")
    output_root = Path(args.output_root)
    checkpoint_root = Path(args.checkpoint_root)
    reports_root = Path(args.reports_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    hb = Path(args.heartbeat) if args.heartbeat else output_root / "winner_sft.heartbeat"
    train_rows = read_jsonl(Path(args.train_manifest))[: args.train_rows]
    heldout_rows = read_jsonl(Path(args.heldout_manifest))[: args.heldout_rows]
    lrs = [float(x) for x in args.lrs.split(",") if x.strip()]
    scopes = [x.strip() for x in args.scopes.split(",") if x.strip()]
    if any(scope not in {"S0", "S1"} for scope in scopes):
        raise ValueError("scopes must be S0 and/or S1")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch_dtype(args.dtype)
    model_dir = Path(args.model_dir)
    torch.manual_seed(args.seed)
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=dtype).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    latents_std = (1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)).to(device, dtype)
    cache = BatchCache(vae, latents_mean, latents_std, device, dtype)

    all_diag: list[dict[str, object]] = []
    all_metrics: list[dict[str, object]] = []
    all_visual: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "status": "MINIMAX_POSITIVE_CONTROL_BLOCKED",
        "recipes": {},
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
        "steps": args.steps,
        "scopes": scopes,
        "training_type": "winner_sft_positive_control_not_dpo",
    }

    for scope in scopes:
      for lr in lrs:
        recipe = f"{scope}_adamw_lr{lr:g}".replace("-", "m")
        heartbeat(hb, f"recipe={recipe}:load")
        recipe_out = output_root / recipe
        recipe_ckpt = checkpoint_root / recipe
        if recipe_out.exists():
            shutil.rmtree(recipe_out)
        if recipe_ckpt.exists():
            shutil.rmtree(recipe_ckpt)
        recipe_out.mkdir(parents=True)
        recipe_ckpt.mkdir(parents=True)

        policy = load_transformer_for_scope(Transformer3DModel, model_dir, dtype, device, scope, args.seed).train()
        base = load_transformer_for_scope(Transformer3DModel, model_dir, dtype, device, scope, args.seed).eval()
        for p in base.parameters():
            p.requires_grad_(False)
        trainable_info = configure_trainable(policy, scope)
        save_checkpoint(policy, recipe_ckpt / "checkpoint-0", scope, {"recipe": recipe, "trainable_info": trainable_info})
        optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-4)
        step_rows = []
        nan_detected = False
        for step in range(1, args.steps + 1):
            row = train_rows[(step - 1) % len(train_rows)]
            tval = 0.19 + 0.05 * (step % 11)
            heartbeat(hb, f"recipe={recipe}:step={step}")
            optimizer.zero_grad(set_to_none=True)
            loss, diag = winner_sft_loss(policy, cache, row, args.seed + step, tval)
            loss.backward()
            stats = grad_stats(policy)
            finite = math.isfinite(diag["loss"]) and math.isfinite(float(stats["grad_norm"]))
            diag.update(stats)
            diag.update({"recipe": recipe, "step": step, "sample_id": row["sample_id"], "lr": lr, "finite": finite})
            step_rows.append(diag)
            if not finite:
                nan_detected = True
                break
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            if step in {1, 5, 10}:
                save_checkpoint(policy, recipe_ckpt / f"checkpoint-{step}", scope, {"recipe": recipe, "step": step, "trainable_info": trainable_info})

        step10_path = recipe_ckpt / "checkpoint-10"
        step10_reload = load_transformer_for_scope(Transformer3DModel, model_dir, dtype, device, scope, args.seed, step10_path).eval() if step10_path.exists() else None
        step10_delta = delta_probe(step10_reload, base) if step10_reload is not None else float("nan")
        loss_start = step_rows[0]["loss"] if step_rows else float("nan")
        loss_end = step_rows[-1]["loss"] if step_rows else float("nan")
        loss_decrease = float(loss_start) - float(loss_end) if math.isfinite(float(loss_start)) and math.isfinite(float(loss_end)) else float("nan")

        heldout_metrics = []
        if step10_reload is not None and not nan_detected:
            infer_vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float16).to(device).eval()
            base_infer = load_transformer_for_scope(Transformer3DModel, model_dir, torch.float16, device, scope, args.seed).eval()
            step10_infer = load_transformer_for_scope(Transformer3DModel, model_dir, torch.float16, device, scope, args.seed, step10_path).eval()
            for idx, row in enumerate(heldout_rows):
                heartbeat(hb, f"recipe={recipe}:heldout={idx + 1}/{len(heldout_rows)}")
                sample_id = str(row["sample_id"])
                step0_metrics = run_pipeline(base_infer, infer_vae, UniPCMultistepScheduler, model_dir, cache, row, recipe_out / "heldout_outputs" / sample_id / "step0", args.seed, args.num_inference_steps, args.iterations)
                step10_metrics = run_pipeline(step10_infer, infer_vae, UniPCMultistepScheduler, model_dir, cache, row, recipe_out / "heldout_outputs" / sample_id / "step10", args.seed, args.num_inference_steps, args.iterations)
                combined = recipe_out / "heldout_outputs" / sample_id / "sft_comparison_strip_16.jpg"
                comparison_strip(Path(step0_metrics["frames_dir"]), Path(step10_metrics["frames_dir"]), combined)
                metric_row: dict[str, object] = {
                    "recipe": recipe,
                    "sample_id": sample_id,
                    "source_group": row.get("source_group", ""),
                    "step0_frames": step0_metrics["frames_dir"],
                    "step10_frames": step10_metrics["frames_dir"],
                    "comparison_strip": str(combined),
                    "step0_raw_output_mp4": step0_metrics["raw_output_mp4"],
                    "step10_raw_output_mp4": step10_metrics["raw_output_mp4"],
                }
                for key in ("full_psnr", "mask_psnr", "boundary_psnr", "outside_psnr", "outside_mae", "temporal_diff_mae"):
                    metric_row[f"step0_{key}"] = step0_metrics.get(key, "")
                    metric_row[f"step10_{key}"] = step10_metrics.get(key, "")
                    if isinstance(step0_metrics.get(key), float) and isinstance(step10_metrics.get(key), float):
                        metric_row[f"delta_{key}"] = float(step10_metrics[key]) - float(step0_metrics[key])
                heldout_metrics.append(metric_row)
                all_visual.append({
                    "recipe": recipe,
                    "sample_id": sample_id,
                    "source_group": row.get("source_group", ""),
                    "step0_output": step0_metrics["raw_output_mp4"],
                    "step10_output": step10_metrics["raw_output_mp4"],
                    "comparison_strip": str(combined),
                    "frames_reviewed": "0,mid,last,16-strip",
                    "classification": "PENDING_CODEX_VISUAL_REVIEW",
                    "reason": "",
                })
            del infer_vae, base_infer, step10_infer
            torch.cuda.empty_cache()

        all_diag.extend(step_rows)
        all_metrics.extend(heldout_metrics)
        mean_mask_delta = float(np.mean([r.get("delta_mask_psnr", 0.0) for r in heldout_metrics])) if heldout_metrics else float("nan")
        mean_boundary_delta = float(np.mean([r.get("delta_boundary_psnr", 0.0) for r in heldout_metrics])) if heldout_metrics else float("nan")
        output_changed = any(abs(float(r.get("delta_mask_psnr", 0.0))) > 1e-5 for r in heldout_metrics)
        summary["recipes"][recipe] = {
            "lr": lr,
            "scope": scope,
            "trainable_info": trainable_info,
            "nan_detected": nan_detected,
            "loss_start": loss_start,
            "loss_end": loss_end,
            "loss_decrease": loss_decrease,
            "step10_delta_probe": step10_delta,
            "heldout_mean_delta_mask_psnr": mean_mask_delta,
            "heldout_mean_delta_boundary_psnr": mean_boundary_delta,
            "heldout_output_changed": output_changed,
            "checkpoint_root": str(recipe_ckpt),
            "output_root": str(recipe_out),
        }
        del policy, base, step10_reload
        torch.cuda.empty_cache()

    candidates = [
        (name, rec) for name, rec in summary["recipes"].items()
        if not rec["nan_detected"] and rec["loss_decrease"] > 0 and rec["step10_delta_probe"] > 0 and rec["heldout_output_changed"]
    ]
    summary["status"] = "MINIMAX_POSITIVE_CONTROL_PASS" if candidates else "MINIMAX_POSITIVE_CONTROL_FAILED"
    summary["best_recipe_by_loss_decrease"] = max(summary["recipes"], key=lambda name: summary["recipes"][name]["loss_decrease"]) if summary["recipes"] else ""
    write_json(reports_root / "exp36_minimax_winner_sft_summary.json", summary)
    write_csv(reports_root / "exp36_minimax_winner_sft_positive_control.csv", all_diag)
    write_csv(reports_root / "exp36_minimax_winner_sft_metrics.csv", all_metrics)
    write_csv(reports_root / "exp36_minimax_winner_sft_visual_review.csv", all_visual)
    md = [
        "# Exp36 MiniMax Winner-SFT Positive-Control",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"- Scopes: `{', '.join(summary['scopes'])}`",
        f"- Recipes: `{', '.join(summary['recipes'])}`",
        f"- Steps per recipe: `{args.steps}`",
        "- Training type: winner reconstruction SFT positive-control, not DPO.",
        "",
        "Recipe summaries:",
    ]
    for name, rec in summary["recipes"].items():
        md.append(f"- `{name}`: loss decrease `{rec['loss_decrease']}`, delta probe `{rec['step10_delta_probe']}`, heldout mask delta `{rec['heldout_mean_delta_mask_psnr']}`")
    (reports_root / "exp36_minimax_winner_sft_positive_control.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
