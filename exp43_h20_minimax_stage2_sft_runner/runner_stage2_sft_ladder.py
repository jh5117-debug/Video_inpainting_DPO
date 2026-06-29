#!/usr/bin/env python3
"""Exp43 isolated MiniMax Stage2 SFT ladder runner and BF16 preflight."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from .precision_policy import (
        PrecisionPolicy,
        apply_runtime_policy,
        clamp_timestep,
        classify_runtime_failure,
        dtype_from_name,
    )
except ImportError:  # Allows direct script execution on H20.
    from precision_policy import (  # type: ignore
        PrecisionPolicy,
        apply_runtime_policy,
        clamp_timestep,
        classify_runtime_failure,
        dtype_from_name,
    )


PREFLIGHT_CASES = ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7")
SFT_RECIPES = {
    "SFT-A": {"mask": 1.00, "boundary": 1.25, "affected": 0.75, "outside": 0.10, "far_outside": 0.02, "hard_state": ""},
    "SFT-B": {"mask": 0.75, "boundary": 1.50, "affected": 0.75, "outside": 0.20, "far_outside": 0.03, "hard_state": ""},
    "SFT-C": {"mask": 0.75, "boundary": 1.50, "affected": 0.75, "outside": 0.20, "far_outside": 0.03, "hard_state": "H1/H3_if_available"},
}
GATE_30_TO_100 = {
    "full_psnr": 0.08,
    "mask_psnr": 0.05,
    "boundary_psnr": -0.02,
    "outside_psnr": -0.02,
    "lpips_max_worse": 0.001,
    "ewarp_max_worse": 0.05,
    "visual_worse_max_fraction": 0.25,
}
GATE_100_TO_300 = {
    "full_psnr": 0.15,
    "mask_psnr": 0.10,
    "boundary_psnr": 0.0,
    "outside_psnr": 0.0,
    "lpips_max_worse": 0.001,
    "ewarp_max_worse": 0.05,
    "visual_better_min_fraction": 0.30,
    "visual_worse_max_fraction": 0.20,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def is_finite_tensor(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor.detach()).all().cpu())


def rank_info() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )


def init_distributed_if_needed() -> tuple[int, int, int, bool]:
    rank, local_rank, world_size = rank_info()
    ddp = world_size > 1
    if ddp:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, ddp


def finish_distributed(ddp: bool) -> None:
    if ddp and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def setup_imports(repo_dir: Path, project_root: Path) -> None:
    sys.path.insert(0, str(repo_dir.resolve()))
    sys.path.insert(0, str(project_root.resolve()))


def load_modules(repo_dir: Path, project_root: Path):
    setup_imports(repo_dir, project_root)
    from diffusers.models import AutoencoderKLWan  # noqa: WPS433
    from transformer_minimax_remover import Transformer3DModel  # noqa: WPS433
    from exp30_vor_or_multimodel_minimax.scripts.run_minimax_gate64_adapter_gate_v3 import (  # noqa: WPS433
        flow_loss,
        image_tensor_for_vae,
        prepare_frame_tensor,
        prepare_video_tensor,
        run_pipeline,
        save_checkpoint,
        to_model_tensors,
    )

    return (
        AutoencoderKLWan,
        Transformer3DModel,
        flow_loss,
        image_tensor_for_vae,
        prepare_frame_tensor,
        prepare_video_tensor,
        run_pipeline,
        save_checkpoint,
        to_model_tensors,
    )


class SafeBatchCache:
    """VAE-fp32 cache with explicit latent dtype for MiniMax DiT inputs."""

    def __init__(
        self,
        vae,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        device: torch.device,
        latent_dtype: torch.dtype,
        prepare_frame_tensor,
        prepare_video_tensor,
        to_model_tensors,
        image_tensor_for_vae,
    ):
        self.vae = vae
        self.latents_mean = latents_mean
        self.latents_std = latents_std
        self.device = device
        self.dtype = latent_dtype
        self.prepare_frame_tensor = prepare_frame_tensor
        self.prepare_video_tensor = prepare_video_tensor
        self.to_model_tensors = to_model_tensors
        self.image_tensor_for_vae = image_tensor_for_vae
        self.cache: dict[str, dict[str, Any]] = {}

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        latent = (self.vae.encode(tensor.float()).latent_dist.mode() - self.latents_mean) * self.latents_std
        return latent.to(dtype=self.dtype).detach()

    def row(self, row: dict[str, Any]) -> dict[str, Any]:
        sample_id = str(row["sample_id"])
        if sample_id in self.cache:
            return self.cache[sample_id]
        width = int(row.get("width", 512))
        height = int(row.get("height", 512))
        num_frames = int(row.get("num_frames", 17))
        condition_np, masks_np, n = self.prepare_frame_tensor(
            Path(str(row["condition_path"])),
            Path(str(row["mask_path"])),
            width,
            height,
            num_frames,
        )
        winner_np = self.prepare_video_tensor(Path(str(row["winner_path"])), width, height, num_frames)
        loser_np = self.prepare_video_tensor(Path(str(row["loser_path"])), width, height, num_frames)
        condition_tensor, mask_tensor = self.to_model_tensors(condition_np, masks_np, self.device, torch.float32)
        winner_tensor = self.image_tensor_for_vae(winner_np, self.device, torch.float32)
        loser_tensor = self.image_tensor_for_vae(loser_np, self.device, torch.float32)
        masked_condition = condition_tensor * (1 - mask_tensor)
        with torch.no_grad():
            cond_latent = self.encode(masked_condition)
            mask_latent = self.encode(2 * mask_tensor - 1.0)
            winner_latent = self.encode(winner_tensor)
            loser_latent = self.encode(loser_tensor)
        cond_frames = [((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8) for arr in condition_np.numpy()]
        win_frames = [((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8) for arr in winner_np.numpy()]
        loser_frames = [((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8) for arr in loser_np.numpy()]
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
            "loser_frames_uint8": loser_frames,
            "mask_frames_uint8": mask_frames,
            "original_n": n,
            "model_n": n,
            "width": width,
            "height": height,
        }
        self.cache[sample_id] = record
        return record


def model_context(args: argparse.Namespace, dtype: torch.dtype):
    repo_dir = Path(args.repo_dir)
    project_root = Path(args.project_root)
    model_dir = Path(args.model_dir)
    manifest = Path(args.manifest)
    for path in (repo_dir, project_root, model_dir, manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    modules = load_modules(repo_dir, project_root)
    rows = read_jsonl(manifest)
    if not rows:
        raise RuntimeError(f"empty manifest: {manifest}")
    return (model_dir, rows, *modules)


def build_cache(args: argparse.Namespace, device: torch.device, latent_dtype: torch.dtype):
    (
        model_dir,
        rows,
        AutoencoderKLWan,
        Transformer3DModel,
        flow_loss,
        image_tensor_for_vae,
        prepare_frame_tensor,
        prepare_video_tensor,
        run_pipeline,
        save_checkpoint,
        to_model_tensors,
    ) = model_context(args, latent_dtype)
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float32).to(device).eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, torch.float32)
    latents_std = (1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)).to(device, torch.float32)
    cache = SafeBatchCache(
        vae,
        latents_mean,
        latents_std,
        device,
        latent_dtype,
        prepare_frame_tensor,
        prepare_video_tensor,
        to_model_tensors,
        image_tensor_for_vae,
    )
    return model_dir, rows, vae, cache, Transformer3DModel, flow_loss, image_tensor_for_vae, prepare_video_tensor, run_pipeline, save_checkpoint


def common_result(args: argparse.Namespace, rank: int, local_rank: int, world_size: int, policy: PrecisionPolicy) -> dict[str, Any]:
    return {
        "case": args.case,
        "hostname": socket.gethostname(),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "seed": args.seed,
        "precision_policy": policy.as_dict(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def p0_torch_bf16(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    a = torch.randn((2048, 2048), device=device, dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn((2048, 2048), device=device, dtype=torch.bfloat16)
    out = (a @ b).float().pow(2).mean()
    out.backward()
    grad = a.grad.detach().float()
    finite = is_finite_tensor(out) and is_finite_tensor(grad)
    result.update(
        status="PASS" if finite else "FAIL",
        device=str(device),
        dtype="bf16",
        loss=float(out.detach().cpu()),
        grad_norm=float(grad.norm().cpu()),
        finite=finite,
        max_memory_allocated=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    )
    return result


def p1_vae_fp32(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0")
    model_dir, rows, vae, _cache, _Transformer3DModel, _flow_loss, image_tensor_for_vae, prepare_video_tensor, _run_pipeline, _save_checkpoint = build_cache(args, device, torch.float32)
    row = rows[0]
    width = int(row.get("width", 512))
    height = int(row.get("height", 512))
    num_frames = int(row.get("num_frames", 17))
    winner_np = prepare_video_tensor(Path(str(row["winner_path"])), width, height, num_frames)
    winner_tensor = image_tensor_for_vae(winner_np, device, torch.float32)
    with torch.no_grad():
        latent = vae.encode(winner_tensor).latent_dist.mode()
        decoded = vae.decode(latent).sample
    finite = is_finite_tensor(latent) and is_finite_tensor(decoded)
    result.update(
        status="PASS" if finite else "FAIL",
        dtype="torch.float32",
        model_dir=str(model_dir),
        sample_id=str(row.get("sample_id", "")),
        latent_shape=list(latent.shape),
        decoded_shape=list(decoded.shape),
        latent_finite=is_finite_tensor(latent),
        decoded_finite=is_finite_tensor(decoded),
        finite=finite,
        max_memory_allocated=int(torch.cuda.max_memory_allocated(device)),
    )
    return result


def grad_norm_fp32(model: torch.nn.Module) -> tuple[float, bool]:
    grad_sq = 0.0
    finite = True
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        finite = finite and is_finite_tensor(grad)
        grad_sq += float((grad * grad).sum().cpu())
    return math.sqrt(grad_sq) if math.isfinite(grad_sq) else float("nan"), finite


def read_rgb_frames(path: Path) -> list[np.ndarray]:
    frames = []
    for frame_path in sorted((path).glob("*.png")):
        arr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if arr is None:
            raise RuntimeError(f"failed to read frame: {frame_path}")
        frames.append(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    if not frames:
        raise RuntimeError(f"no PNG frames in {path}")
    return frames


def psnr_from_mse(mse: float) -> float:
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10((255.0 * 255.0) / mse)


def region_psnr_mae(output: np.ndarray, target: np.ndarray, region: np.ndarray) -> tuple[float, float]:
    if not np.any(region):
        return float("nan"), float("nan")
    diff = output.astype(np.float32) - target.astype(np.float32)
    region_diff = diff[region]
    mse = float(np.mean(region_diff * region_diff))
    mae = float(np.mean(np.abs(region_diff)))
    return psnr_from_mse(mse), mae


def frame_regions(mask: np.ndarray) -> dict[str, np.ndarray]:
    mask_bool = mask > 20
    kernel = np.ones((9, 9), np.uint8)
    dil = cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1).astype(bool)
    ero = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=1).astype(bool)
    boundary = np.logical_xor(dil, ero)
    outside = ~dil
    return {
        "full": np.ones(mask_bool.shape, dtype=bool),
        "mask": mask_bool,
        "boundary": boundary,
        "outside": outside,
    }


def region_replaced(output: np.ndarray, target: np.ndarray, region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = target.copy()
    gt = target.copy()
    pred[region] = output[region]
    return pred, gt


def safe_nanmean(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def compute_extended_metrics(
    output_frames: list[np.ndarray],
    target_frames: list[np.ndarray],
    mask_frames: list[np.ndarray],
    *,
    device: str,
    compute_lpips: bool = True,
    compute_ewarp: bool = True,
) -> dict[str, Any]:
    setup_imports(Path.cwd(), Path.cwd())
    try:
        from inference.metrics import EwarpMetric, LPIPSMetric, calc_psnr_and_ssim  # noqa: WPS433
    except Exception as exc:  # noqa: BLE001
        EwarpMetric = None  # type: ignore
        LPIPSMetric = None  # type: ignore
        calc_psnr_and_ssim = None  # type: ignore
        import_error = f"{type(exc).__name__}: {exc}"
    else:
        import_error = ""

    rows: list[dict[str, float]] = []
    lpips_values: dict[str, list[float]] = {"full": [], "mask": [], "boundary": []}
    ssim_values: dict[str, list[float]] = {"full": [], "mask": [], "boundary": []}
    for output, target, mask in zip(output_frames, target_frames, mask_frames):
        regions = frame_regions(mask)
        row: dict[str, float] = {}
        if calc_psnr_and_ssim is not None:
            _psnr, ssim = calc_psnr_and_ssim(target, output)
            ssim_values["full"].append(float(ssim))
        for name, region in regions.items():
            psnr, mae = region_psnr_mae(output, target, region)
            row[f"{name}_psnr"] = psnr
            row[f"{name}_mae"] = mae
            if name in {"mask", "boundary"} and calc_psnr_and_ssim is not None:
                pred_region, gt_region = region_replaced(output, target, region)
                _rpsnr, rssim = calc_psnr_and_ssim(gt_region, pred_region)
                ssim_values[name].append(float(rssim))
        if compute_lpips and LPIPSMetric is not None:
            try:
                lpips_values["full"].append(float(LPIPSMetric.compute(target, output, device)))
                for name in ("mask", "boundary"):
                    pred_region, gt_region = region_replaced(output, target, regions[name])
                    lpips_values[name].append(float(LPIPSMetric.compute(gt_region, pred_region, device)))
            except Exception as exc:  # noqa: BLE001
                row["lpips_error"] = float("nan")
                import_error = f"LPIPS:{type(exc).__name__}: {exc}"
        rows.append(row)

    temporal = []
    if len(output_frames) > 1:
        for idx in range(1, len(output_frames)):
            pred_delta = output_frames[idx].astype(np.float32) - output_frames[idx - 1].astype(np.float32)
            tgt_delta = target_frames[idx].astype(np.float32) - target_frames[idx - 1].astype(np.float32)
            temporal.append(float(np.mean(np.abs(pred_delta - tgt_delta))))

    metrics: dict[str, Any] = {}
    for key in rows[0]:
        if key.endswith("_error"):
            continue
        metrics[key] = safe_nanmean([r[key] for r in rows if key in r])
    metrics["temporal_diff_mae"] = safe_nanmean(temporal)
    for name in ("full", "mask", "boundary"):
        metrics[f"{name}_ssim"] = safe_nanmean(ssim_values[name])
        metrics[f"{name}_lpips"] = safe_nanmean(lpips_values[name])
    metrics["lpips_status"] = "OK" if compute_lpips and LPIPSMetric is not None and math.isfinite(float(metrics["full_lpips"])) else "BLOCKED"
    metrics["metrics_import_error"] = import_error

    if compute_ewarp and EwarpMetric is not None:
        try:
            masks01 = [(m > 20).astype(np.uint8) for m in mask_frames]
            metrics["ewarp"] = float(EwarpMetric(device=device, raft_model_path=None).compute(output_frames, masks01=masks01, gt_frames_u8_rgb=target_frames))
            metrics["ewarp_status"] = "OK"
        except Exception as exc:  # noqa: BLE001
            metrics["ewarp"] = float("nan")
            metrics["ewarp_status"] = f"BLOCKED:{type(exc).__name__}: {exc}"
    else:
        metrics["ewarp"] = float("nan")
        metrics["ewarp_status"] = "BLOCKED"
    return metrics


def downsample_region(arr: np.ndarray, size: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(arr.astype(np.float32))[None, None].to(device)
    return F.interpolate(tensor, size=size, mode="area").clamp(0, 1)


def build_region_weight(record: dict[str, Any], latent_shape: torch.Size, weights: dict[str, float], device: torch.device) -> torch.Tensor:
    mask_frames = np.stack([(m > 20).astype(np.float32) for m in record["mask_frames_uint8"]], axis=0)
    boundary_frames = []
    outside_frames = []
    far_frames = []
    for mask in record["mask_frames_uint8"]:
        regions = frame_regions(mask)
        boundary_frames.append(regions["boundary"].astype(np.float32))
        outside_frames.append(regions["outside"].astype(np.float32))
        far = cv2.erode(regions["outside"].astype(np.uint8), np.ones((17, 17), np.uint8), iterations=1).astype(np.float32)
        far_frames.append(far)
    cond = np.stack(record["condition_frames_uint8"], axis=0).astype(np.float32)
    win = np.stack(record["winner_frames_uint8"], axis=0).astype(np.float32)
    affected = np.mean(np.abs(cond - win), axis=3) / 255.0
    if float(affected.max()) > 1e-6:
        affected = affected / float(affected.max())
    size = (int(latent_shape[2]), int(latent_shape[3]), int(latent_shape[4]))
    mask = downsample_region(mask_frames, size, device)
    boundary = downsample_region(np.stack(boundary_frames, axis=0), size, device)
    outside = downsample_region(np.stack(outside_frames, axis=0), size, device)
    far = downsample_region(np.stack(far_frames, axis=0), size, device)
    affected_t = downsample_region(affected, size, device)
    weight = torch.full_like(mask, float(weights["far_outside"]))
    weight = weight + (float(weights["outside"]) - float(weights["far_outside"])) * outside
    weight = weight + float(weights["mask"]) * mask
    weight = weight + float(weights["boundary"]) * boundary
    weight = weight + float(weights["affected"]) * affected_t
    return weight.clamp_min(1e-4)


def sft_weighted_loss(
    model: torch.nn.Module,
    cache: SafeBatchCache,
    row: dict[str, Any],
    *,
    seed: int,
    tval: float,
    weights: dict[str, float],
    policy: PrecisionPolicy,
) -> tuple[torch.Tensor, dict[str, float]]:
    record = cache.row(row)
    z0 = record["winner"]
    assert isinstance(z0, torch.Tensor)
    gen = torch.Generator(device=cache.device).manual_seed(seed)
    eps = torch.randn(z0.shape, generator=gen, device=cache.device, dtype=cache.dtype)
    t_scalar = clamp_timestep(tval, policy)
    t = torch.tensor([t_scalar], device=cache.device, dtype=cache.dtype)
    zt = t.view(1, 1, 1, 1, 1) * eps + (1 - t.view(1, 1, 1, 1, 1)) * z0
    target = (eps - z0).detach()
    hidden = torch.cat([zt, record["cond"], record["mask_latents"]], dim=1)
    pred = model(hidden_states=hidden, timestep=(t * 1000.0).float())[0]
    region_weight = build_region_weight(record, target.shape, weights, cache.device).to(dtype=torch.float32)
    sq = (pred.float() - target.float()).pow(2)
    loss = (sq * region_weight).sum() / (region_weight.sum() * sq.shape[1]).clamp_min(1.0)
    diag = {
        "loss": float(loss.detach().cpu()),
        "region_weight_mean": float(region_weight.mean().detach().cpu()),
        "region_weight_max": float(region_weight.max().detach().cpu()),
        "t": t_scalar,
    }
    return loss, diag


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    keys = list(row.keys())
    if exists:
        first = path.read_text(encoding="utf-8").splitlines()
        if first:
            keys = first[0].split(",")
            for key in row:
                if key not in keys:
                    keys.append(key)
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def lr_slug(lr: float) -> str:
    return f"{lr:.0e}".replace("-", "m").replace("+", "")


def run_id_for(recipe: str, lr: float, step: int) -> str:
    return f"{recipe}_lr{lr_slug(lr)}_step{step}"


def flow_step(
    args: argparse.Namespace,
    result: dict[str, Any],
    *,
    dtype: torch.dtype,
    backward: bool,
    train_step: bool,
    ddp: bool,
    rank: int,
    local_rank: int,
    policy: PrecisionPolicy,
) -> dict[str, Any]:
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)
    model_dir, rows, _vae, cache, Transformer3DModel, flow_loss, _image_tensor_for_vae, _prepare_video_tensor, _run_pipeline, save_checkpoint = build_cache(args, device, dtype)
    row = rows[rank % len(rows)]
    tval = clamp_timestep(args.timestep, policy)
    model = Transformer3DModel.from_pretrained(model_dir / "transformer", torch_dtype=dtype).to(device)
    model.train(mode=train_step or backward)
    for param in model.parameters():
        param.requires_grad_(train_step or backward)
    wrapped = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True) if ddp else model
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=args.lr) if train_step else None
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    loss = flow_loss(wrapped, cache, row, "winner", args.seed + rank + 17, tval)
    finite_loss = is_finite_tensor(loss)
    grad_norm = 0.0
    grad_finite = True
    if backward or train_step:
        loss.backward()
        grad_norm, grad_finite = grad_norm_fp32(wrapped)
    if optimizer is not None:
        torch.nn.utils.clip_grad_norm_(wrapped.parameters(), args.grad_clip)
        optimizer.step()
    ckpt_status = "not_requested"
    ckpt_path = ""
    if train_step and args.save_checkpoint and rank == 0:
        ckpt_path = str(Path(args.output_root) / args.case / "checkpoint-1")
        model_to_save = wrapped.module if hasattr(wrapped, "module") else wrapped
        save_checkpoint(model_to_save, Path(ckpt_path))
        reloaded = Transformer3DModel.from_pretrained(Path(ckpt_path), torch_dtype=dtype).to(device).eval()
        with torch.no_grad():
            reload_loss = flow_loss(reloaded, cache, row, "winner", args.seed + 29, clamp_timestep(0.41, policy))
        ckpt_status = "PASS" if is_finite_tensor(reload_loss) else "NONFINITE_RELOAD"
        del reloaded
    checkpoint_ok = not train_step or rank != 0 or ckpt_status == "PASS"
    status = "PASS" if finite_loss and grad_finite and checkpoint_ok else "FAIL"
    result.update(
        status=status,
        model_dir=str(model_dir),
        sample_id=str(row.get("sample_id", "")),
        dtype=str(dtype),
        backward=backward,
        train_step=train_step,
        ddp=ddp,
        timestep=tval,
        loss=float(loss.detach().cpu()),
        finite_loss=finite_loss,
        grad_norm=grad_norm,
        grad_finite=grad_finite,
        checkpoint_status=ckpt_status,
        checkpoint_path=ckpt_path,
        max_memory_allocated=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    )
    return result


def run_preflight_case(args: argparse.Namespace) -> dict[str, Any]:
    policy = apply_runtime_policy(PrecisionPolicy())
    rank, local_rank, world_size, ddp = init_distributed_if_needed()
    result = common_result(args, rank, local_rank, world_size, policy)
    try:
        if args.case == "P0":
            result = p0_torch_bf16(args, result)
        elif args.case == "P1":
            result = p1_vae_fp32(args, result)
        elif args.case == "P2":
            result = flow_step(args, result, dtype=torch.bfloat16, backward=False, train_step=False, ddp=False, rank=rank, local_rank=local_rank, policy=policy)
        elif args.case == "P3":
            result = flow_step(args, result, dtype=torch.bfloat16, backward=True, train_step=False, ddp=False, rank=rank, local_rank=local_rank, policy=policy)
        elif args.case == "P4":
            result = flow_step(args, result, dtype=torch.float32, backward=True, train_step=True, ddp=False, rank=rank, local_rank=local_rank, policy=policy)
        elif args.case == "P5":
            result = flow_step(args, result, dtype=torch.bfloat16, backward=True, train_step=True, ddp=False, rank=rank, local_rank=local_rank, policy=policy)
        elif args.case in {"P6", "P7"}:
            result = flow_step(args, result, dtype=torch.bfloat16, backward=True, train_step=True, ddp=ddp, rank=rank, local_rank=local_rank, policy=policy)
        else:
            raise ValueError(args.case)
    except BaseException as exc:  # noqa: BLE001
        result.update(
            status="FAIL",
            error_type=type(exc).__name__,
            error=str(exc),
            failure_class=classify_runtime_failure(f"{type(exc).__name__}: {exc}"),
        )
        raise
    finally:
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json(Path(args.output_root) / args.case / f"rank{rank}.json", result)
        finish_distributed(ddp)
    return result


def flatten_case_rows(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in PREFLIGHT_CASES:
        for path in sorted((output_root / case).glob("rank*.json")):
            obj = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "case": obj.get("case", case),
                    "rank": obj.get("rank", ""),
                    "world_size": obj.get("world_size", ""),
                    "status": obj.get("status", ""),
                    "dtype": obj.get("dtype", obj.get("train_dtype", "")),
                    "loss": obj.get("loss", ""),
                    "grad_norm": obj.get("grad_norm", ""),
                    "finite_loss": obj.get("finite_loss", obj.get("finite", "")),
                    "grad_finite": obj.get("grad_finite", ""),
                    "checkpoint_status": obj.get("checkpoint_status", ""),
                    "max_memory_allocated_mib": round(float(obj.get("max_memory_allocated", 0)) / 1024 / 1024, 3),
                    "error_type": obj.get("error_type", ""),
                    "failure_class": obj.get("failure_class", ""),
                    "started_at": obj.get("started_at", ""),
                    "finished_at": obj.get("finished_at", ""),
                    "json_path": str(path),
                }
            )
    return rows


def summarize_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    reports_dir = Path(args.reports_dir)
    rows = flatten_case_rows(output_root)
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case"]), []).append(row)
    cases: list[dict[str, Any]] = []
    all_pass = True
    for case in PREFLIGHT_CASES:
        case_rows = by_case.get(case, [])
        status = "PASS" if case_rows and all(r.get("status") == "PASS" for r in case_rows) else "FAIL"
        all_pass = all_pass and status == "PASS"
        cases.append(
            {
                "case": case,
                "status": status,
                "rank_count": len(case_rows),
                "world_size": max([int(r.get("world_size") or 0) for r in case_rows], default=0),
                "rank0_loss": next((r.get("loss") for r in case_rows if str(r.get("rank")) == "0"), ""),
                "rank0_grad_norm": next((r.get("grad_norm") for r in case_rows if str(r.get("rank")) == "0"), ""),
                "rank0_checkpoint": next((r.get("checkpoint_status") for r in case_rows if str(r.get("rank")) == "0"), ""),
                "peak_mib_rank0": next((r.get("max_memory_allocated_mib") for r in case_rows if str(r.get("rank")) == "0"), ""),
            }
        )
    final_status = "H20_EXP43_BF16_SAFE_READY" if all_pass else "H20_EXP43_BF16_BLOCKED"
    summary = {
        "status": final_status,
        "hostname": socket.gethostname(),
        "output_root": str(output_root),
        "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cases": cases,
    }
    write_json(reports_dir / "exp43_h20_bf16_safe_preflight_summary.json", summary)
    write_csv(reports_dir / "exp43_h20_bf16_safe_preflight.csv", rows)
    lines = [
        "# Exp43 H20 BF16 Safe Preflight",
        "",
        f"Status: `{final_status}`",
        "",
        f"- Output root: `{output_root}`",
        f"- Torch/CUDA: `{torch.__version__}` / `{torch.version.cuda}`",
        f"- BF16 supported: `{summary['bf16_supported']}`",
        "",
        "| case | status | ranks | world size | rank0 loss | rank0 grad norm | checkpoint | rank0 peak MiB |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in cases:
        lines.append(
            f"| {row['case']} | {row['status']} | {row['rank_count']} | {row['world_size']} | "
            f"{row['rank0_loss']} | {row['rank0_grad_norm']} | {row['rank0_checkpoint']} | {row['peak_mib_rank0']} |"
        )
    lines.extend(
        [
            "",
            "Policy:",
            "",
            "- VAE encode/decode fp32.",
            "- Transformer bf16 for bf16 cases.",
            "- Loss, residual, reduction, and gradient norm fp32.",
            "- Flash and memory-efficient SDPA disabled when PyTorch exposes backend toggles.",
            "- xFormers/flash-attn disabled by environment flags.",
            "- Timesteps clamped away from exact 0/1.",
            "- GradScaler disabled for bf16.",
            "- No silent fallback is accepted.",
            "",
            "This is runtime stability evidence only. It does not claim MiniMax quality improvement.",
        ]
    )
    (reports_dir / "exp43_h20_bf16_safe_preflight.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def make_policy(dtype_name: str) -> PrecisionPolicy:
    if dtype_name in {"fp32", "float32"}:
        return PrecisionPolicy(name="fp32_safe", transformer_dtype="fp32")
    if dtype_name in {"bf16", "bfloat16"}:
        return PrecisionPolicy()
    raise ValueError(f"unsupported Exp43 training dtype: {dtype_name}")


def train_sft(args: argparse.Namespace) -> dict[str, Any]:
    recipe = args.recipe
    if recipe not in SFT_RECIPES:
        raise ValueError(f"unknown recipe {recipe}; choices={sorted(SFT_RECIPES)}")
    policy = apply_runtime_policy(make_policy(args.dtype))
    rank, local_rank, world_size, ddp = init_distributed_if_needed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(policy.transformer_dtype)
    torch.manual_seed(args.seed + rank)

    setup_imports(Path(args.repo_dir), Path(args.project_root))
    (
        model_dir,
        train_rows,
        vae,
        cache,
        Transformer3DModel,
        _flow_loss,
        _image_tensor_for_vae,
        _prepare_video_tensor,
        _run_pipeline,
        save_checkpoint,
    ) = build_cache(args, device, dtype)

    run_id = args.run_id or run_id_for(recipe, args.lr, args.target_steps)
    output_root = Path(args.output_root) / "sft_ladder" / run_id
    checkpoint_root = output_root / "checkpoints"
    log_root = Path(args.log_root)
    monitor_csv = log_root / "monitor_5min.csv"
    diag_csv = output_root / "diagnostics.csv"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    source = Path(args.resume_checkpoint) if args.resume_checkpoint else model_dir / "transformer"
    model = Transformer3DModel.from_pretrained(source, torch_dtype=dtype).to(device).train()
    for param in model.parameters():
        param.requires_grad_(True)
    wrapped = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True) if ddp else model
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay)
    weights = SFT_RECIPES[recipe]
    start_step = int(args.resume_step)
    if rank == 0 and start_step == 0:
        save_checkpoint(model, checkpoint_root / "checkpoint-0")

    if ddp:
        torch.distributed.barrier()

    last_monitor = 0.0
    rows_seen = 0
    finite = True
    error_text = ""
    peak_vram = 0.0
    start_time = time.time()
    try:
        for step in range(start_step + 1, args.target_steps + 1):
            row = train_rows[((step - 1) * max(1, world_size) + rank) % len(train_rows)]
            tval = 0.11 + 0.78 * (((step * 37 + rank * 17) % 997) / 996.0)
            optimizer.zero_grad(set_to_none=True)
            loss, diag = sft_weighted_loss(
                wrapped,
                cache,
                row,
                seed=args.seed + step * 1009 + rank,
                tval=tval,
                weights=weights,
                policy=policy,
            )
            finite = finite and is_finite_tensor(loss)
            loss.backward()
            grad_norm, grad_finite = grad_norm_fp32(wrapped)
            finite = finite and grad_finite and math.isfinite(grad_norm)
            torch.nn.utils.clip_grad_norm_(wrapped.parameters(), args.grad_clip)
            optimizer.step()
            rows_seen += 1
            if device.type == "cuda":
                peak_vram = max(peak_vram, torch.cuda.max_memory_allocated(device) / 1024 / 1024)
            if rank == 0:
                diag_row = {
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "hostname": socket.gethostname(),
                    "run_id": run_id,
                    "recipe": recipe,
                    "lr": args.lr,
                    "target_steps": args.target_steps,
                    "step": step,
                    "rank": rank,
                    "world_size": world_size,
                    "sample_id": row.get("sample_id", ""),
                    "loss": diag["loss"],
                    "grad_norm": grad_norm,
                    "grad_finite": grad_finite,
                    "finite": finite,
                    "t": diag["t"],
                    "region_weight_mean": diag["region_weight_mean"],
                    "region_weight_max": diag["region_weight_max"],
                    "peak_vram_mib": peak_vram,
                }
                append_csv_row(diag_csv, diag_row)
                now = time.time()
                if now - last_monitor >= 300 or step in {1, args.target_steps}:
                    append_csv_row(
                        monitor_csv,
                        {
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "hostname": socket.gethostname(),
                            "branch": args.branch,
                            "commit": args.commit,
                            "GPU list": os.environ.get("CUDA_VISIBLE_DEVICES", "ALL"),
                            "PID": os.getpid(),
                            "PGID": os.getpgrp(),
                            "lane": "Exp43-SFT",
                            "recipe": recipe,
                            "step": step,
                            "loss": diag["loss"],
                            "grad norm": grad_norm,
                            "VRAM": peak_vram,
                            "util": "",
                            "SIGFPE": False,
                            "OOM": False,
                            "CUDA": False,
                            "Xid": "",
                            "NaN/Inf": not finite,
                            "checkpoint": str(checkpoint_root / f"checkpoint-{step}") if step == args.target_steps else "",
                            "next action": "continue" if step < args.target_steps else "evaluate",
                        },
                    )
                    last_monitor = now
            if not finite:
                error_text = "nonfinite loss or gradient"
                break
            if rank == 0 and (step == args.target_steps or step % args.checkpoint_interval == 0):
                save_checkpoint(model, checkpoint_root / f"checkpoint-{step}")
            if ddp and (step == args.target_steps or step % args.checkpoint_interval == 0):
                torch.distributed.barrier()
    except BaseException as exc:  # noqa: BLE001
        finite = False
        error_text = f"{type(exc).__name__}: {exc}"
        if rank == 0:
            append_csv_row(
                monitor_csv,
                {
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "hostname": socket.gethostname(),
                    "branch": args.branch,
                    "commit": args.commit,
                    "GPU list": os.environ.get("CUDA_VISIBLE_DEVICES", "ALL"),
                    "PID": os.getpid(),
                    "PGID": os.getpgrp(),
                    "lane": "Exp43-SFT",
                    "recipe": recipe,
                    "step": "ERROR",
                    "loss": "",
                    "grad norm": "",
                    "VRAM": peak_vram,
                    "util": "",
                    "SIGFPE": "sigfpe" in error_text.lower(),
                    "OOM": "out of memory" in error_text.lower(),
                    "CUDA": "cuda" in error_text.lower(),
                    "Xid": "",
                    "NaN/Inf": "nan" in error_text.lower() or "inf" in error_text.lower(),
                    "checkpoint": "",
                    "next action": classify_runtime_failure(error_text),
                },
            )
        raise
    finally:
        result = {
            "status": "TRAIN_DONE" if finite and not error_text else "TRAIN_FAILED",
            "failure_class": classify_runtime_failure(error_text) if error_text else "",
            "error": error_text,
            "run_id": run_id,
            "recipe": recipe,
            "lr": args.lr,
            "target_steps": args.target_steps,
            "resume_checkpoint": args.resume_checkpoint,
            "resume_step": args.resume_step,
            "world_size": world_size,
            "rank": rank,
            "dtype": policy.transformer_dtype,
            "precision_policy": policy.as_dict(),
            "rows_seen_this_rank": rows_seen,
            "runtime_seconds": time.time() - start_time,
            "peak_vram_mib": peak_vram,
            "checkpoint_root": str(checkpoint_root),
            "target_checkpoint": str(checkpoint_root / f"checkpoint-{args.target_steps}"),
            "diagnostics_csv": str(diag_csv),
            "train_manifest": args.manifest,
        }
        if rank == 0:
            write_json(output_root / "train_summary.json", result)
        finish_distributed(ddp)
    return result


def pipeline_output_ready(outdir: Path) -> bool:
    return (outdir / "evidence" / "raw_output.mp4").exists() and len(list((outdir / "frames").glob("*.png"))) > 0


def evaluate_one_pipeline(
    *,
    transformer,
    vae,
    scheduler_cls,
    model_dir: Path,
    cache: SafeBatchCache,
    row: dict[str, Any],
    outdir: Path,
    seed: int,
    inference_steps: int,
    iterations: int,
    run_pipeline,
    reuse_existing: bool,
) -> dict[str, Any]:
    if not (reuse_existing and pipeline_output_ready(outdir)):
        return run_pipeline(transformer, vae, scheduler_cls, model_dir, cache, row, outdir, seed, inference_steps, iterations)
    output_frames = read_rgb_frames(outdir / "frames")
    record = cache.row(row)
    basic = compute_extended_metrics(
        output_frames,
        record["winner_frames_uint8"],
        record["mask_frames_uint8"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_lpips=False,
        compute_ewarp=False,
    )
    basic.update(
        {
            "frames_dir": str(outdir / "frames"),
            "raw_output_mp4": str(outdir / "evidence" / "raw_output.mp4"),
            "side_by_side_mp4": str(outdir / "evidence" / "side_by_side.mp4"),
            "temporal_strip_16": str(outdir / "evidence" / "temporal_strip_16.jpg"),
            "review_sheet": str(outdir / "evidence" / "midframe_review_sheet.jpg"),
        }
    )
    return basic


def evaluate_sft(args: argparse.Namespace) -> dict[str, Any]:
    policy = apply_runtime_policy(make_policy(args.dtype))
    setup_imports(Path(args.repo_dir), Path(args.project_root))
    from diffusers.models import AutoencoderKLWan  # noqa: WPS433
    from diffusers.schedulers import UniPCMultistepScheduler  # noqa: WPS433
    from transformer_minimax_remover import Transformer3DModel  # noqa: WPS433
    from exp30_vor_or_multimodel_minimax.scripts.run_minimax_gate64_adapter_gate_v3 import run_pipeline  # noqa: WPS433

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    model_dir = Path(args.model_dir)
    eval_root = Path(args.output_root) / "sft_ladder" / args.run_id / "eval"
    reports_root = Path(args.reports_dir)
    metric_rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    target_ckpt = Path(args.checkpoint)
    if not target_ckpt.exists():
        raise FileNotFoundError(target_ckpt)
    infer_vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=dtype).to(device).eval()
    cache_vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float32).to(device).eval()
    base = Transformer3DModel.from_pretrained(model_dir / "transformer", torch_dtype=dtype).to(device).eval()
    candidate = Transformer3DModel.from_pretrained(target_ckpt, torch_dtype=dtype).to(device).eval()
    modules = load_modules(Path(args.repo_dir), Path(args.project_root))
    cache = SafeBatchCache(
        cache_vae,
        torch.tensor(cache_vae.config.latents_mean).view(1, cache_vae.config.z_dim, 1, 1, 1).to(device, torch.float32),
        (1.0 / torch.tensor(cache_vae.config.latents_std).view(1, cache_vae.config.z_dim, 1, 1, 1)).to(device, torch.float32),
        device,
        dtype,
        modules[4],
        modules[5],
        modules[8],
        modules[3],
    )
    split_specs = [("search", Path(args.search_manifest), args.limit_search), ("shadow", Path(args.shadow_manifest), args.limit_shadow)]
    for split, manifest, limit in split_specs:
        rows = read_jsonl(manifest)[:limit]
        for idx, row in enumerate(rows):
            sample_id = str(row["sample_id"])
            base_out = Path(args.output_root) / "sft_ladder" / "_step0_cache" / split / sample_id
            cand_out = eval_root / split / sample_id / f"step{args.target_steps}"
            step0_basic = evaluate_one_pipeline(
                transformer=base,
                vae=infer_vae,
                scheduler_cls=UniPCMultistepScheduler,
                model_dir=model_dir,
                cache=cache,
                row=row,
                outdir=base_out,
                seed=args.seed,
                inference_steps=args.num_inference_steps,
                iterations=args.iterations,
                run_pipeline=run_pipeline,
                reuse_existing=args.reuse_existing,
            )
            stepn_basic = evaluate_one_pipeline(
                transformer=candidate,
                vae=infer_vae,
                scheduler_cls=UniPCMultistepScheduler,
                model_dir=model_dir,
                cache=cache,
                row=row,
                outdir=cand_out,
                seed=args.seed,
                inference_steps=args.num_inference_steps,
                iterations=args.iterations,
                run_pipeline=run_pipeline,
                reuse_existing=args.reuse_existing,
            )
            record = cache.row(row)
            step0_frames = read_rgb_frames(Path(str(step0_basic["frames_dir"])))
            stepn_frames = read_rgb_frames(Path(str(stepn_basic["frames_dir"])))
            step0_ext = compute_extended_metrics(step0_frames, record["winner_frames_uint8"], record["mask_frames_uint8"], device=str(device))
            stepn_ext = compute_extended_metrics(stepn_frames, record["winner_frames_uint8"], record["mask_frames_uint8"], device=str(device))
            metric_row: dict[str, Any] = {
                "run_id": args.run_id,
                "recipe": args.recipe,
                "lr": args.lr,
                "split": split,
                "sample_id": sample_id,
                "checkpoint": f"step{args.target_steps}",
                "condition_path": row.get("condition_path", ""),
                "winner_path": row.get("winner_path", ""),
                "mask_path": row.get("mask_path", ""),
                "raw_step0": step0_basic["raw_output_mp4"],
                "raw_stepN": stepn_basic["raw_output_mp4"],
                "review_step0": step0_basic["temporal_strip_16"],
                "review_stepN": stepn_basic["temporal_strip_16"],
                "side_by_side_stepN": stepn_basic["side_by_side_mp4"],
            }
            for key, value in step0_ext.items():
                metric_row[f"step0_{key}"] = value
            for key, value in stepn_ext.items():
                metric_row[f"stepN_{key}"] = value
                base_val = step0_ext.get(key)
                if isinstance(value, (float, int)) and isinstance(base_val, (float, int)):
                    metric_row[f"delta_{key}"] = float(value) - float(base_val)
            metric_rows.append(metric_row)
            visual_rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "recipe": args.recipe,
                    "checkpoint": f"step{args.target_steps}",
                    "condition_path": row.get("condition_path", ""),
                    "winner_path": row.get("winner_path", ""),
                    "mask_path": row.get("mask_path", ""),
                    "raw_step0": step0_basic["raw_output_mp4"],
                    "raw_stepN": stepn_basic["raw_output_mp4"],
                    "full_psnr_delta": metric_row.get("delta_full_psnr", ""),
                    "mask_psnr_delta": metric_row.get("delta_mask_psnr", ""),
                    "boundary_psnr_delta": metric_row.get("delta_boundary_psnr", ""),
                    "outside_psnr_delta": metric_row.get("delta_outside_psnr", ""),
                    "lpips_delta": metric_row.get("delta_full_lpips", ""),
                    "ewarp_delta": metric_row.get("delta_ewarp", ""),
                    "fogging": "PENDING_CODEX_REVIEW",
                    "over_erasure": "PENDING_CODEX_REVIEW",
                    "boundary_damage": "PENDING_CODEX_REVIEW",
                    "outside_damage": "PENDING_CODEX_REVIEW",
                    "temporal_artifact": "PENDING_CODEX_REVIEW",
                    "classification": "PENDING_CODEX_VISUAL_REVIEW",
                    "reason": "Generated raw output and 16-frame strip; requires actual visual review before pass/promotion.",
                }
            )
            append_csv_row(Path(args.log_root) / "monitor_5min.csv", {
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "hostname": socket.gethostname(),
                "branch": args.branch,
                "commit": args.commit,
                "GPU list": os.environ.get("CUDA_VISIBLE_DEVICES", "ALL"),
                "PID": os.getpid(),
                "PGID": os.getpgrp(),
                "lane": "Exp43-SFT-eval",
                "recipe": args.recipe,
                "step": f"{split}:{idx + 1}/{len(rows)}",
                "loss": "",
                "grad norm": "",
                "VRAM": torch.cuda.max_memory_allocated(device) / 1024 / 1024 if torch.cuda.is_available() else 0,
                "util": "",
                "SIGFPE": False,
                "OOM": False,
                "CUDA": False,
                "Xid": "",
                "NaN/Inf": False,
                "checkpoint": str(target_ckpt),
                "next action": "continue_eval",
            })
    metric_csv = eval_root / "metrics.csv"
    visual_csv = eval_root / "visual_review.csv"
    write_csv(metric_csv, metric_rows)
    write_csv(visual_csv, visual_rows)
    summary = {
        "status": "EVAL_DONE",
        "run_id": args.run_id,
        "checkpoint": str(target_ckpt),
        "metric_csv": str(metric_csv),
        "visual_review_csv": str(visual_csv),
        "rows": len(metric_rows),
        "visual_review_status": "PENDING_CODEX_REVIEW",
        "precision_policy": policy.as_dict(),
    }
    write_json(eval_root / "eval_summary.json", summary)
    reports_root.mkdir(parents=True, exist_ok=True)
    return summary


def numeric(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def bootstrap_ci(values: list[float], samples: int = 2000, seed: int = 20260629) -> tuple[float, float, float]:
    vals = np.array([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(vals, size=vals.size, replace=True).mean()) for _ in range(samples)]
    return float(np.mean(vals)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_sft(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root) / "sft_ladder"
    reports_dir = Path(args.reports_dir)
    all_metric_rows: list[dict[str, Any]] = []
    all_visual_rows: list[dict[str, Any]] = []
    for metrics_path in sorted(output_root.glob("*/eval/metrics.csv")):
        with metrics_path.open("r", encoding="utf-8") as fh:
            all_metric_rows.extend(list(csv.DictReader(fh)))
    for visual_path in sorted(output_root.glob("*/eval/visual_review.csv")):
        with visual_path.open("r", encoding="utf-8") as fh:
            all_visual_rows.extend(list(csv.DictReader(fh)))

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in all_metric_rows:
        grouped.setdefault((str(row.get("run_id", "")), str(row.get("recipe", "")), str(row.get("split", ""))), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    gate_summaries: list[dict[str, Any]] = []
    for (run_id, recipe, split), rows in grouped.items():
        agg: dict[str, Any] = {
            "run_id": run_id,
            "recipe": recipe,
            "split": split,
            "n": len(rows),
        }
        for key in ("full_psnr", "mask_psnr", "boundary_psnr", "outside_psnr", "full_lpips", "ewarp", "temporal_diff_mae"):
            vals = [numeric(r, f"delta_{key}") for r in rows]
            mean, ci_lo, ci_hi = bootstrap_ci(vals)
            agg[f"mean_delta_{key}"] = mean
            agg[f"ci95_lo_delta_{key}"] = ci_lo
            agg[f"ci95_hi_delta_{key}"] = ci_hi
            if key in {"full_psnr", "mask_psnr", "boundary_psnr", "outside_psnr"}:
                agg[f"win_rate_{key}"] = float(np.mean([v > 0 for v in vals if math.isfinite(v)])) if any(math.isfinite(v) for v in vals) else float("nan")
        visual_for_group = [
            v for v in all_visual_rows
            if v.get("run_id", run_id) == run_id and v.get("recipe", recipe) == recipe and v.get("split", split) == split
        ]
        classifications = [str(v.get("classification", "")) for v in visual_for_group]
        reviewed = classifications and all("PENDING" not in c and c for c in classifications)
        worse = [c for c in classifications if c in {"WORSE", "BAD", "ARTIFACT_REGRESSION"}]
        better = [c for c in classifications if c in {"BETTER", "IMPROVED"}]
        agg["visual_review_completed"] = bool(reviewed)
        agg["visual_worse_fraction"] = float(len(worse) / len(classifications)) if classifications else float("nan")
        agg["visual_better_fraction"] = float(len(better) / len(classifications)) if classifications else float("nan")
        summary_rows.append(agg)

        target_step = int(run_id.rsplit("step", 1)[-1]) if "step" in run_id else 0
        gate = GATE_30_TO_100 if target_step == 30 else GATE_100_TO_300 if target_step == 100 else {}
        metric_pass = True
        blockers: list[str] = []
        if gate:
            for key in ("full_psnr", "mask_psnr", "boundary_psnr", "outside_psnr"):
                threshold = float(gate[key])
                actual = float(agg.get(f"mean_delta_{key}", float("nan")))
                if not math.isfinite(actual) or actual < threshold:
                    metric_pass = False
                    blockers.append(f"{key} {actual:.6g} < {threshold}")
            lpips_delta = float(agg.get("mean_delta_full_lpips", float("nan")))
            ewarp_delta = float(agg.get("mean_delta_ewarp", float("nan")))
            if not math.isfinite(lpips_delta) or lpips_delta > float(gate["lpips_max_worse"]):
                metric_pass = False
                blockers.append(f"LPIPS delta {lpips_delta:.6g} > {gate['lpips_max_worse']}")
            if not math.isfinite(ewarp_delta) or ewarp_delta > float(gate["ewarp_max_worse"]):
                metric_pass = False
                blockers.append(f"Ewarp delta {ewarp_delta:.6g} > {gate['ewarp_max_worse']}")
            if not reviewed:
                metric_pass = False
                blockers.append("visual review pending; no metric-only promotion")
        gate_summaries.append(
            {
                "run_id": run_id,
                "recipe": recipe,
                "split": split,
                "target_step": target_step,
                "gate_pass": metric_pass,
                "blockers": "; ".join(blockers),
                **agg,
            }
        )

    reports_dir.mkdir(parents=True, exist_ok=True)
    write_csv(reports_dir / "exp43_h20_stage2_sft_ladder_metrics.csv", all_metric_rows)
    write_csv(reports_dir / "exp43_h20_stage2_sft_ladder_visual_review.csv", all_visual_rows)
    write_csv(reports_dir / "exp43_h20_stage2_sft_ladder_summary_rows.csv", summary_rows)
    write_json(
        reports_dir / "exp43_h20_stage2_sft_ladder_summary.json",
        {
            "status": "H20_EXP43_SFT_PARETO_MIXED" if all_metric_rows else "H20_EXP43_SFT_BLOCKED",
            "metric_rows": len(all_metric_rows),
            "visual_rows": len(all_visual_rows),
            "gate_summaries": gate_summaries,
            "promotion_rule": "No PASS/POSITIVE without completed visual review.",
        },
    )
    lines = [
        "# Exp43 H20 MiniMax Stage2 SFT Ladder",
        "",
        f"Status: `{'H20_EXP43_SFT_PARETO_MIXED' if all_metric_rows else 'H20_EXP43_SFT_BLOCKED'}`",
        "",
        "This report intentionally blocks PASS/POSITIVE if visual review is pending.",
        "",
        "| run | split | n | dPSNR | dMaskPSNR | dBoundaryPSNR | dOutsidePSNR | dLPIPS | dEwarp | visual reviewed | gate | blockers |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in gate_summaries:
        lines.append(
            f"| {row['run_id']} | {row['split']} | {row['n']} | {row.get('mean_delta_full_psnr','')} | "
            f"{row.get('mean_delta_mask_psnr','')} | {row.get('mean_delta_boundary_psnr','')} | "
            f"{row.get('mean_delta_outside_psnr','')} | {row.get('mean_delta_full_lpips','')} | "
            f"{row.get('mean_delta_ewarp','')} | {row.get('visual_review_completed')} | "
            f"{row.get('gate_pass')} | {row.get('blockers','')} |"
        )
    lines.extend(
        [
            "",
            "Claim boundary:",
            "",
            "- Technical training/evaluation completion is not scientific positive.",
            "- MiniMax third-adapter evidence remains blocked until shadow metrics and visual review pass.",
            "- No universal adapter, final SOTA, or top-conference novelty claim is made here.",
        ]
    )
    (reports_dir / "exp43_h20_stage2_sft_ladder.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rows": len(all_metric_rows), "gate_summaries": gate_summaries}


def add_common_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-dir", default="")
    parser.add_argument("--project-root", default="")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--timestep", type=float, default=0.37)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--grad-clip", type=float, default=1.0)


def add_h20_bookkeeping_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-root", default="/home/nvme01/H20_Video_inpainting_DPO/logs/autoresearch/exp43_h20_minimax_stage2_sft_runner")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--branch", default="")
    parser.add_argument("--commit", default="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--case", required=True, choices=PREFLIGHT_CASES)
    add_common_runtime_args(preflight)
    summarize = sub.add_parser("summarize-preflight")
    summarize.add_argument("--output-root", required=True)
    summarize.add_argument("--reports-dir", required=True)
    train = sub.add_parser("train-sft")
    add_common_runtime_args(train)
    add_h20_bookkeeping_args(train)
    train.add_argument("--recipe", required=True, choices=sorted(SFT_RECIPES))
    train.add_argument("--target-steps", type=int, required=True)
    train.add_argument("--checkpoint-interval", type=int, default=10)
    train.add_argument("--resume-checkpoint", default="")
    train.add_argument("--resume-step", type=int, default=0)
    train.add_argument("--dtype", choices=["bf16", "bfloat16", "fp32", "float32"], default="bf16")
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--run-id", default="")
    evaluate = sub.add_parser("evaluate-sft")
    evaluate.add_argument("--repo-dir", required=True)
    evaluate.add_argument("--project-root", required=True)
    evaluate.add_argument("--model-dir", required=True)
    evaluate.add_argument("--output-root", required=True)
    add_h20_bookkeeping_args(evaluate)
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--recipe", required=True)
    evaluate.add_argument("--lr", type=float, required=True)
    evaluate.add_argument("--target-steps", type=int, required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--search-manifest", required=True)
    evaluate.add_argument("--shadow-manifest", required=True)
    evaluate.add_argument("--limit-search", type=int, default=24)
    evaluate.add_argument("--limit-shadow", type=int, default=24)
    evaluate.add_argument("--seed", type=int, default=20260629)
    evaluate.add_argument("--num-inference-steps", type=int, default=12)
    evaluate.add_argument("--iterations", type=int, default=6)
    evaluate.add_argument("--dtype", choices=["bf16", "bfloat16", "fp32", "float32"], default="bf16")
    evaluate.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    sft_summary = sub.add_parser("summarize-sft")
    sft_summary.add_argument("--output-root", required=True)
    sft_summary.add_argument("--reports-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        result = run_preflight_case(args)
        if result.get("rank", 0) == 0:
            print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "summarize-preflight":
        print(json.dumps(summarize_preflight(args), indent=2, sort_keys=True))
    elif args.command == "train-sft":
        result = train_sft(args)
        if result.get("rank", 0) == 0:
            print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "evaluate-sft":
        print(json.dumps(evaluate_sft(args), indent=2, sort_keys=True))
    elif args.command == "summarize-sft":
        print(json.dumps(summarize_sft(args), indent=2, sort_keys=True))
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
