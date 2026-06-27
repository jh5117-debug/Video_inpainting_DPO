#!/usr/bin/env python3
"""Run the Exp30 MiniMax Gate64 10-step adapter gate.

This is intentionally a micro gate.  It validates the MiniMax flow-matching
preference path on the locked Exp30 Gate64 train32/heldout16 manifests and
never launches long training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
USABLE_RECIPES = {"frozen", "ema"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", required=True)
    p.add_argument("--project-root", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--heldout-manifest", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--checkpoint-root", required=True)
    p.add_argument("--reports-root", default="")
    p.add_argument("--recipes", default="frozen,ema")
    p.add_argument("--seed", type=int, default=20260627)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--num-inference-steps", type=int, default=12)
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--limit-heldout", type=int, default=16)
    p.add_argument("--heartbeat", default="")
    return p.parse_args()


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def read_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError(f"failed to read RGB frame: {path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def read_gray(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"failed to read mask frame: {path}")
    return arr


def save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def heartbeat(path: Path | None, text: str) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{time.time():.0f}\t{text}\n", encoding="utf-8")


def torch_dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "float16" else torch.bfloat16


def frame_to_uint8(frame: object) -> np.ndarray:
    arr = np.array(frame)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.5:
            arr *= 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"unexpected output frame shape: {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    return arr


def write_mp4(path: Path, frames: list[np.ndarray], fps: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open mp4 writer: {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()


def sample_indices(n: int, count: int = 16) -> list[int]:
    if n <= count:
        return list(range(n))
    return sorted({int(round(i * (n - 1) / (count - 1))) for i in range(count)})


def labeled_tile(frame: np.ndarray, label: str, tile_w: int = 192) -> np.ndarray:
    h, w = frame.shape[:2]
    tile_h = max(1, int(round(h * tile_w / w)))
    tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    cv2.putText(tile, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
    return tile


def contact_sheet(frames: list[np.ndarray], labels: list[str], cols: int = 4, tile_w: int = 192) -> np.ndarray:
    tiles = [labeled_tile(frame, label, tile_w) for frame, label in zip(frames, labels)]
    rows = []
    for start in range(0, len(tiles), cols):
        row_tiles = tiles[start : start + cols]
        if len(row_tiles) < cols:
            row_tiles += [np.zeros_like(row_tiles[0]) for _ in range(cols - len(row_tiles))]
        rows.append(np.concatenate(row_tiles, axis=1))
    return np.concatenate(rows, axis=0)


def mask_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = frame.copy()
    m = mask > 20
    out[m] = (0.55 * out[m] + 0.45 * np.array([255, 40, 40])).astype(np.uint8)
    return out


def prepare_frame_tensor(frame_dir: Path, mask_dir: Path, width: int, height: int, num_frames: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    frame_paths = image_files(frame_dir)
    mask_paths = image_files(mask_dir)
    n = min(len(frame_paths), len(mask_paths), num_frames)
    if n != num_frames:
        raise RuntimeError(f"{frame_dir.name}: expected {num_frames} frames, got frames={len(frame_paths)} masks={len(mask_paths)}")
    frames: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for frame_path, mask_path in zip(frame_paths[:n], mask_paths[:n]):
        frame = read_rgb(frame_path)
        mask = read_gray(mask_path)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        frames.append(frame.astype(np.float32) / 127.5 - 1.0)
        masks.append((mask > 20).astype(np.float32)[:, :, None])
    return torch.from_numpy(np.stack(frames, 0)), torch.from_numpy(np.stack(masks, 0)), n


def prepare_video_tensor(frame_dir: Path, width: int, height: int, num_frames: int) -> torch.Tensor:
    frame_paths = image_files(frame_dir)
    if len(frame_paths) < num_frames:
        raise RuntimeError(f"{frame_dir}: expected at least {num_frames} frames, got {len(frame_paths)}")
    frames = []
    for frame_path in frame_paths[:num_frames]:
        frame = read_rgb(frame_path)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        frames.append(frame.astype(np.float32) / 127.5 - 1.0)
    return torch.from_numpy(np.stack(frames, 0))


def to_model_tensors(images: torch.Tensor, masks: torch.Tensor, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    image_tensor = rearrange(images, "f h w c -> c f h w")[None].to(device=device, dtype=dtype)
    mask_tensor = masks.permute(3, 0, 1, 2)[None].repeat(1, 3, 1, 1, 1).to(device=device, dtype=dtype)
    return image_tensor, mask_tensor.clamp(0, 1)


def image_tensor_for_vae(images: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return rearrange(images, "f h w c -> c f h w")[None].to(device=device, dtype=dtype)


class BatchCache:
    def __init__(self, vae, latents_mean: torch.Tensor, latents_std: torch.Tensor, device: torch.device, dtype: torch.dtype):
        self.vae = vae
        self.latents_mean = latents_mean
        self.latents_std = latents_std
        self.device = device
        self.dtype = dtype
        self.cache: dict[str, dict[str, torch.Tensor | int | list[np.ndarray] | dict]] = {}

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        return ((self.vae.encode(tensor).latent_dist.mode() - self.latents_mean) * self.latents_std).detach()

    def row(self, row: dict) -> dict[str, torch.Tensor | int | list[np.ndarray] | dict]:
        sample_id = str(row["sample_id"])
        if sample_id in self.cache:
            return self.cache[sample_id]
        width = int(row.get("width", 512))
        height = int(row.get("height", 512))
        num_frames = int(row.get("num_frames", 17))
        condition_dir = Path(str(row["condition_path"]))
        winner_dir = Path(str(row["winner_path"]))
        loser_dir = Path(str(row["loser_path"]))
        mask_dir = Path(str(row["mask_path"]))
        condition_np, masks_np, n = prepare_frame_tensor(condition_dir, mask_dir, width, height, num_frames)
        winner_np = prepare_video_tensor(winner_dir, width, height, num_frames)
        loser_np = prepare_video_tensor(loser_dir, width, height, num_frames)
        condition_tensor, mask_tensor = to_model_tensors(condition_np, masks_np, self.device, self.dtype)
        winner_tensor = image_tensor_for_vae(winner_np, self.device, self.dtype)
        loser_tensor = image_tensor_for_vae(loser_np, self.device, self.dtype)
        masked_condition = condition_tensor * (1 - mask_tensor)
        with torch.no_grad():
            cond_latent = self.encode(masked_condition)
            mask_latent = self.encode(2 * mask_tensor - 1.0)
            winner_latent = self.encode(winner_tensor)
            loser_latent = self.encode(loser_tensor)
        cond_frames = [((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8) for arr in condition_np.numpy()]
        win_frames = [((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8) for arr in winner_np.numpy()]
        mask_frames = [(arr[:, :, 0] * 255.0).clip(0, 255).astype(np.uint8) for arr in masks_np.numpy()]
        record = {
            "row": row,
            "cond": cond_latent,
            "mask_latents": mask_latent,
            "winner": winner_latent,
            "loser": loser_latent,
            "condition_images": condition_np,
            "condition_frames_uint8": cond_frames,
            "winner_frames_uint8": win_frames,
            "mask_frames_uint8": mask_frames,
            "original_n": n,
            "model_n": n,
            "width": width,
            "height": height,
        }
        self.cache[sample_id] = record
        return record


def flow_loss(model: torch.nn.Module, cache: BatchCache, row: dict, which: str, seed: int, tval: float) -> torch.Tensor:
    record = cache.row(row)
    z0 = record[which]
    assert isinstance(z0, torch.Tensor)
    gen = torch.Generator(device=cache.device).manual_seed(seed)
    eps = torch.randn(z0.shape, generator=gen, device=cache.device, dtype=cache.dtype)
    t = torch.tensor([tval], device=cache.device, dtype=cache.dtype)
    zt = t.view(1, 1, 1, 1, 1) * eps + (1 - t.view(1, 1, 1, 1, 1)) * z0
    target = (eps - z0).detach()
    hidden = torch.cat([zt, record["cond"], record["mask_latents"]], dim=1)
    pred = model(hidden_states=hidden, timestep=(t * 1000.0).float())[0]
    return F.mse_loss(pred.float(), target.float())


def preference_loss(
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    cache: BatchCache,
    row: dict,
    seed: int,
    tval: float,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    winner_policy = flow_loss(policy, cache, row, "winner", seed, tval)
    loser_policy = flow_loss(policy, cache, row, "loser", seed, tval)
    with torch.no_grad():
        winner_ref = flow_loss(reference, cache, row, "winner", seed, tval)
        loser_ref = flow_loss(reference, cache, row, "loser", seed, tval)
    win_gap = winner_ref - winner_policy
    lose_gap = loser_ref - loser_policy
    margin = win_gap - lose_gap
    utility = torch.clamp(0.2 * beta * margin + 0.5, 1e-6, 1.0)
    loss = -torch.log(utility)
    diag = {
        "winner_policy_loss": float(winner_policy.detach().cpu()),
        "loser_policy_loss": float(loser_policy.detach().cpu()),
        "winner_reference_loss": float(winner_ref.detach().cpu()),
        "loser_reference_loss": float(loser_ref.detach().cpu()),
        "win_gap": float(win_gap.detach().cpu()),
        "lose_gap": float(lose_gap.detach().cpu()),
        "preference_margin": float(margin.detach().cpu()),
        "linear_utility": float(utility.detach().cpu()),
        "loss": float(loss.detach().cpu()),
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


def delta_probe(model_a: torch.nn.Module, model_b: torch.nn.Module, limit: int = 64) -> float:
    delta = 0.0
    checked = 0
    with torch.no_grad():
        for (_, p), (_, q) in zip(model_a.named_parameters(), model_b.named_parameters()):
            delta += float((p.detach().float() - q.detach().float()).abs().mean().cpu())
            checked += 1
            if checked >= limit:
                break
    return delta / max(1, checked)


def update_ema(ref: torch.nn.Module, policy: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for r, p in zip(ref.parameters(), policy.parameters()):
            r.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)


def psnr_from_mse(mse: float) -> float:
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10((255.0 * 255.0) / mse)


def metric_frame(output: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    diff = output.astype(np.float32) - target.astype(np.float32)
    mask_bool = mask > 20
    kernel = np.ones((9, 9), np.uint8)
    dil = cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1).astype(bool)
    ero = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=1).astype(bool)
    boundary = np.logical_xor(dil, ero)
    outside = ~dil
    out: dict[str, float] = {}
    for name, region in [("full", np.ones(mask_bool.shape, dtype=bool)), ("mask", mask_bool), ("boundary", boundary), ("outside", outside)]:
        if not np.any(region):
            out[f"{name}_psnr"] = float("nan")
            out[f"{name}_mae"] = float("nan")
            continue
        region_diff = diff[region]
        mse = float(np.mean(region_diff * region_diff))
        out[f"{name}_psnr"] = psnr_from_mse(mse)
        out[f"{name}_mae"] = float(np.mean(np.abs(region_diff)))
    return out


def aggregate_metrics(output_frames: list[np.ndarray], target_frames: list[np.ndarray], mask_frames: list[np.ndarray]) -> dict[str, float]:
    rows = [metric_frame(o, t, m) for o, t, m in zip(output_frames, target_frames, mask_frames)]
    agg: dict[str, float] = {}
    for key in rows[0]:
        vals = [r[key] for r in rows if math.isfinite(r[key])]
        agg[key] = float(np.mean(vals)) if vals else float("nan")
    if len(output_frames) > 1:
        temporal = []
        for idx in range(1, len(output_frames)):
            pred_delta = output_frames[idx].astype(np.float32) - output_frames[idx - 1].astype(np.float32)
            tgt_delta = target_frames[idx].astype(np.float32) - target_frames[idx - 1].astype(np.float32)
            temporal.append(float(np.mean(np.abs(pred_delta - tgt_delta))))
        agg["temporal_diff_mae"] = float(np.mean(temporal))
    else:
        agg["temporal_diff_mae"] = float("nan")
    return agg


def run_pipeline(transformer, vae, scheduler_cls, model_dir: Path, cache: BatchCache, row: dict, outdir: Path, seed: int, steps: int, iterations: int) -> dict[str, object]:
    from pipeline_minimax_remover import Minimax_Remover_Pipeline  # noqa: WPS433

    if outdir.exists():
        shutil.rmtree(outdir)
    frames_dir = outdir / "frames"
    evidence_dir = outdir / "evidence"
    frames_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    record = cache.row(row)
    scheduler = scheduler_cls.from_pretrained(model_dir / "scheduler")
    pipe = Minimax_Remover_Pipeline(transformer=transformer, vae=vae, scheduler=scheduler).to(cache.device)
    gen = torch.Generator(device=cache.device).manual_seed(seed)
    with torch.inference_mode():
        result = pipe(
            images=record["condition_images"],
            masks=torch.from_numpy(np.stack(record["mask_frames_uint8"], 0).astype(np.float32)[:, :, :, None] / 255.0),
            num_frames=int(record["model_n"]),
            height=int(record["height"]),
            width=int(record["width"]),
            num_inference_steps=steps,
            generator=gen,
            iterations=iterations,
        ).frames[0]
    output_frames = [frame_to_uint8(frame) for frame in result[: int(record["original_n"])]]
    for idx, frame in enumerate(output_frames):
        save_rgb(frames_dir / f"{idx:05d}.png", frame)
    write_mp4(evidence_dir / "raw_output.mp4", output_frames)
    side = []
    for idx, frame in enumerate(output_frames):
        cond = record["condition_frames_uint8"][idx]
        win = record["winner_frames_uint8"][idx]
        mask = record["mask_frames_uint8"][idx]
        side.append(np.concatenate([cond, mask_overlay(cond, mask), win, frame], axis=1))
    write_mp4(evidence_dir / "side_by_side.mp4", side)
    idxs = sample_indices(len(output_frames), 16)
    strip_frames = []
    strip_labels = []
    for idx in idxs:
        strip_frames.append(side[idx])
        strip_labels.append(f"f{idx:02d}")
    strip = contact_sheet(strip_frames, strip_labels, cols=1, tile_w=768)
    save_rgb(evidence_dir / "temporal_strip_16.jpg", strip)
    mid = len(output_frames) // 2
    review = np.concatenate([
        record["condition_frames_uint8"][mid],
        mask_overlay(record["condition_frames_uint8"][mid], record["mask_frames_uint8"][mid]),
        record["winner_frames_uint8"][mid],
        output_frames[mid],
    ], axis=1)
    save_rgb(evidence_dir / "midframe_review_sheet.jpg", review)
    metrics = aggregate_metrics(output_frames, record["winner_frames_uint8"], record["mask_frames_uint8"])
    metrics.update({
        "frames_dir": str(frames_dir),
        "raw_output_mp4": str(evidence_dir / "raw_output.mp4"),
        "side_by_side_mp4": str(evidence_dir / "side_by_side.mp4"),
        "temporal_strip_16": str(evidence_dir / "temporal_strip_16.jpg"),
        "review_sheet": str(evidence_dir / "midframe_review_sheet.jpg"),
    })
    return metrics


def main() -> None:
    args = parse_args()
    recipes = [r.strip() for r in args.recipes.split(",") if r.strip()]
    if not recipes or any(r not in USABLE_RECIPES for r in recipes):
        raise ValueError(f"recipes must be subset of {sorted(USABLE_RECIPES)}")
    if args.steps > 10:
        raise ValueError("Exp30 prompt caps this gate at 10 steps")

    sys.path.insert(0, str(Path(args.repo_dir).resolve()))
    sys.path.insert(0, str(Path(args.project_root).resolve()))
    from diffusers.models import AutoencoderKLWan  # noqa: WPS433
    from diffusers.schedulers import UniPCMultistepScheduler  # noqa: WPS433
    from transformer_minimax_remover import Transformer3DModel  # noqa: WPS433

    train_rows = read_jsonl(Path(args.train_manifest))
    heldout_rows = read_jsonl(Path(args.heldout_manifest))[: args.limit_heldout]
    output_root = Path(args.output_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    reports_root = Path(args.reports_root).resolve() if args.reports_root else output_root / "reports"
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    hb = Path(args.heartbeat) if args.heartbeat else output_root / "minimax_gate64_adapter.heartbeat"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch_dtype(args.dtype)
    torch.manual_seed(args.seed)
    model_dir = Path(args.model_dir).resolve()
    for child in ("vae", "transformer", "scheduler"):
        if not (model_dir / child).exists():
            raise FileNotFoundError(f"missing MiniMax model component: {model_dir / child}")

    start = time.time()
    heartbeat(hb, "loading_model")
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=dtype).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    latents_std = (1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)).to(device, dtype)
    cache = BatchCache(vae, latents_mean, latents_std, device, dtype)

    loader_audit = {
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
        "train_manifest_sha256": sha256_file(Path(args.train_manifest)),
        "heldout_manifest_sha256": sha256_file(Path(args.heldout_manifest)),
        "target": "MiniMax flow velocity epsilon_minus_z0",
        "frames": 17,
        "dtype": args.dtype,
        "vor_eval_used": False,
    }
    write_json(reports_root / "exp30_minimax_gate64_dataset_loader_audit_v3.json", loader_audit)

    summary: dict[str, object] = {
        "dataset_loader_audit": loader_audit,
        "recipes": {},
        "seed": args.seed,
        "steps": args.steps,
        "lr": args.lr,
        "beta": args.beta,
        "grad_clip": args.grad_clip,
    }
    all_diag_rows: list[dict[str, object]] = []
    all_metric_rows: list[dict[str, object]] = []
    all_visual_rows: list[dict[str, object]] = []

    for recipe in recipes:
        heartbeat(hb, f"recipe={recipe}:load")
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
        save_checkpoint(policy, recipe_ckpt / "checkpoint-0")

        zero_loss, zero_diag = preference_loss(policy, reference, cache, train_rows[0], args.seed, 0.37, args.beta)
        zero_gap_ok = (
            abs(zero_diag["win_gap"]) < 1e-6
            and abs(zero_diag["lose_gap"]) < 1e-6
            and abs(zero_diag["loss"] - math.log(2.0)) < 1e-5
        )

        optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-4)
        step_records = []
        nan_detected = False
        for step in range(1, args.steps + 1):
            heartbeat(hb, f"recipe={recipe}:step={step}")
            row = train_rows[(step - 1) % len(train_rows)]
            tval = 0.19 + 0.05 * (step % 11)
            optimizer.zero_grad(set_to_none=True)
            loss, diag = preference_loss(policy, reference, cache, row, args.seed + step, tval, args.beta)
            loss.backward()
            stats = grad_stats(policy)
            finite = math.isfinite(diag["loss"]) and math.isfinite(float(stats["grad_norm"]))
            if not finite:
                nan_detected = True
                diag.update(stats)
                diag.update({"step": step, "sample_id": row["sample_id"], "recipe": recipe, "finite": False})
                step_records.append(diag)
                break
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()
            if recipe == "ema":
                update_ema(reference, policy, args.ema_decay)
            diag.update(stats)
            diag.update({
                "step": step,
                "sample_id": row["sample_id"],
                "recipe": recipe,
                "finite": True,
                "t": tval,
            })
            step_records.append(diag)
            if step in {1, 5, 10}:
                save_checkpoint(policy, recipe_ckpt / f"checkpoint-{step}")

        # Strict reload and parameter-delta probes.
        base_reload = Transformer3DModel.from_pretrained(model_dir / "transformer", torch_dtype=dtype).to(device).eval()
        step1_reload = Transformer3DModel.from_pretrained(recipe_ckpt / "checkpoint-1", torch_dtype=dtype).to(device).eval()
        step10_path = recipe_ckpt / "checkpoint-10"
        step10_reload = Transformer3DModel.from_pretrained(step10_path, torch_dtype=dtype).to(device).eval() if step10_path.exists() else None
        strict_step1 = {"missing_keys": [], "unexpected_keys": []}
        strict_step10 = {"missing_keys": [] if step10_reload else ["checkpoint-10-missing"], "unexpected_keys": []}
        step1_delta = delta_probe(step1_reload, base_reload)
        step10_delta = delta_probe(step10_reload, base_reload) if step10_reload else float("nan")
        reference_delta = 0.0 if recipe == "frozen" else delta_probe(reference, base_reload)

        heldout_metric_rows = []
        if step10_reload is not None and not nan_detected:
            # The official MiniMax pipeline internally casts masked inputs to
            # float16. Keep the training gate in the requested dtype, but use
            # fresh float16 inference copies so heldout generation follows the
            # official pipeline contract without mutating the training/cache VAE.
            infer_vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float16).to(device).eval()
            base_infer = Transformer3DModel.from_pretrained(model_dir / "transformer", torch_dtype=torch.float16).to(device).eval()
            step10_infer = Transformer3DModel.from_pretrained(step10_path, torch_dtype=torch.float16).to(device).eval()
            for idx, row in enumerate(heldout_rows):
                heartbeat(hb, f"recipe={recipe}:heldout={idx + 1}/{len(heldout_rows)}")
                sample_id = str(row["sample_id"])
                step0_metrics = run_pipeline(
                    base_infer,
                    infer_vae,
                    UniPCMultistepScheduler,
                    model_dir,
                    cache,
                    row,
                    recipe_out / "heldout_outputs" / sample_id / "step0",
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
                    recipe_out / "heldout_outputs" / sample_id / "step10",
                    args.seed,
                    args.num_inference_steps,
                    args.iterations,
                )
                metric_row = {
                    "recipe": recipe,
                    "sample_id": sample_id,
                    "source_group": row.get("source_group", ""),
                    "loser_model": row.get("model", ""),
                    "classification_final": row.get("classification_final", ""),
                    "step0_frames": step0_metrics["frames_dir"],
                    "step10_frames": step10_metrics["frames_dir"],
                    "step0_raw_output_mp4": step0_metrics["raw_output_mp4"],
                    "step10_raw_output_mp4": step10_metrics["raw_output_mp4"],
                    "step0_temporal_strip_16": step0_metrics["temporal_strip_16"],
                    "step10_temporal_strip_16": step10_metrics["temporal_strip_16"],
                    "step0_review_sheet": step0_metrics["review_sheet"],
                    "step10_review_sheet": step10_metrics["review_sheet"],
                }
                for key in ("full_psnr", "mask_psnr", "boundary_psnr", "outside_psnr", "outside_mae", "temporal_diff_mae"):
                    metric_row[f"step0_{key}"] = step0_metrics.get(key, "")
                    metric_row[f"step10_{key}"] = step10_metrics.get(key, "")
                    if isinstance(step0_metrics.get(key), float) and isinstance(step10_metrics.get(key), float):
                        metric_row[f"delta_{key}"] = float(step10_metrics[key]) - float(step0_metrics[key])
                heldout_metric_rows.append(metric_row)
                visual_row = {
                    "sample_id": sample_id,
                    "recipe": recipe,
                    "model": "MiniMax",
                    "checkpoint": "step10",
                    "condition_path": row.get("condition_path", ""),
                    "winner_path": row.get("winner_path", ""),
                    "loser_path": row.get("loser_path", ""),
                    "mask_path": row.get("mask_path", ""),
                    "step0_path": step0_metrics["raw_output_mp4"],
                    "step10_path": step10_metrics["raw_output_mp4"],
                    "frames_reviewed": "0,8,16,16-strip",
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
                }
                all_visual_rows.append(visual_row)
            del infer_vae, base_infer, step10_infer
            torch.cuda.empty_cache()

        all_diag_rows.extend(step_records)
        all_metric_rows.extend(heldout_metric_rows)
        mean_mask_delta = float(np.mean([r.get("delta_mask_psnr", 0.0) for r in heldout_metric_rows])) if heldout_metric_rows else float("nan")
        mean_boundary_delta = float(np.mean([r.get("delta_boundary_psnr", 0.0) for r in heldout_metric_rows])) if heldout_metric_rows else float("nan")
        mean_outside_delta = float(np.mean([r.get("delta_outside_psnr", 0.0) for r in heldout_metric_rows])) if heldout_metric_rows else float("nan")
        summary["recipes"][recipe] = {
            "zero_gap_status": "MINIMAX_ZERO_GAP_PASSED" if zero_gap_ok else "MINIMAX_ZERO_GAP_FAILED",
            "zero_gap": zero_diag,
            "one_step_status": "MINIMAX_ONE_STEP_STRICT_RELOAD_PASSED"
            if not strict_step1["missing_keys"] and not strict_step1["unexpected_keys"] and step1_delta > 0
            else "MINIMAX_ONE_STEP_FAILED",
            "ten_step_status": "MINIMAX_10STEP_COMPLETED_NEEDS_CODEX_REVIEW"
            if step10_reload is not None and not nan_detected and not strict_step10["missing_keys"] and not strict_step10["unexpected_keys"]
            else "MINIMAX_10STEP_FAILED",
            "strict_reload_step1": strict_step1,
            "strict_reload_step10": strict_step10,
            "step1_delta_probe": step1_delta,
            "step10_delta_probe": step10_delta,
            "reference_delta_probe": reference_delta,
            "nan_detected": nan_detected,
            "mean_delta_mask_psnr": mean_mask_delta,
            "mean_delta_boundary_psnr": mean_boundary_delta,
            "mean_delta_outside_psnr": mean_outside_delta,
            "checkpoint_root": str(recipe_ckpt),
            "output_root": str(recipe_out),
            "peak_vram_mib": torch.cuda.max_memory_allocated() / 1024 / 1024,
        }
        del policy, reference, base_reload, step1_reload, step10_reload
        torch.cuda.empty_cache()

    summary["runtime_seconds"] = time.time() - start
    summary["status"] = "MINIMAX_10STEP_REQUIRES_CODEX_VISUAL_REVIEW"
    write_json(reports_root / "exp30_minimax_gate64_adapter_summary_v3.json", summary)
    write_csv(reports_root / "exp30_minimax_gate64_adapter_diagnostics_v3.csv", all_diag_rows)
    write_csv(reports_root / "exp30_minimax_gate64_adapter_10step_metrics_v3.csv", all_metric_rows)
    write_csv(reports_root / "exp30_minimax_gate64_adapter_10step_visual_review_v3.csv", all_visual_rows)
    md = [
        "# Exp30 MiniMax Gate64 Adapter 10-Step V3",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"- Train rows: {len(train_rows)}",
        f"- Heldout rows evaluated: {len(heldout_rows)}",
        f"- Recipes: `{','.join(recipes)}`",
        f"- Target: MiniMax flow velocity `epsilon - z0`",
        f"- Steps: {args.steps}",
        f"- Long training: false",
        "",
        "Codex visual review is required before promotion.",
    ]
    (reports_root / "exp30_minimax_gate64_adapter_10step_v3.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
