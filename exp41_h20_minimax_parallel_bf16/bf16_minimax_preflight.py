#!/usr/bin/env python3
"""Exp41 MiniMax BF16/SIGFPE runtime preflight.

This helper intentionally lives under Exp41. It imports existing MiniMax and
Exp30 utilities but does not modify training source files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7"])
    parser.add_argument("--repo-dir", default="")
    parser.add_argument("--project-root", default="")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--train-dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(name)


def is_finite_tensor(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor.detach()).all().cpu())


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rank_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


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


def configure_safe_runtime() -> None:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    os.environ.setdefault("DISABLE_XFORMERS", "1")
    os.environ.setdefault("FLASH_ATTENTION_FORCE_DISABLE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)


def common_result(args: argparse.Namespace, rank: int, local_rank: int, world_size: int) -> dict[str, Any]:
    return {
        "case": args.case,
        "hostname": socket.gethostname(),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "seed": args.seed,
        "train_dtype": args.train_dtype,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


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
        save_checkpoint,
        to_model_tensors,
    )

    return AutoencoderKLWan, Transformer3DModel, flow_loss, image_tensor_for_vae, prepare_frame_tensor, prepare_video_tensor, save_checkpoint, to_model_tensors


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
        record = {
            "row": row,
            "cond": cond_latent,
            "mask_latents": mask_latent,
            "winner": winner_latent,
            "loser": loser_latent,
            "original_n": n,
            "model_n": n,
            "width": width,
            "height": height,
        }
        self.cache[sample_id] = record
        return record


def p0_torch_bf16(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    a = torch.randn((2048, 2048), device=device, dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn((2048, 2048), device=device, dtype=torch.bfloat16)
    out = (a @ b).float().pow(2).mean()
    out.backward()
    grad = a.grad.detach().float()
    result.update(
        status="PASS",
        device=str(device),
        loss=float(out.detach().cpu()),
        grad_norm=float(grad.norm().cpu()),
        finite=is_finite_tensor(out) and is_finite_tensor(grad),
        max_memory_allocated=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    )
    return result


def model_context(args: argparse.Namespace, dtype: torch.dtype):
    repo_dir = Path(args.repo_dir)
    project_root = Path(args.project_root)
    model_dir = Path(args.model_dir)
    manifest = Path(args.manifest)
    if not repo_dir.exists():
        raise FileNotFoundError(repo_dir)
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    modules = load_modules(repo_dir, project_root)
    AutoencoderKLWan, Transformer3DModel, flow_loss, image_tensor_for_vae, prepare_frame_tensor, prepare_video_tensor, save_checkpoint, to_model_tensors = modules
    rows = read_jsonl(manifest)
    if not rows:
        raise RuntimeError(f"empty manifest: {manifest}")
    return model_dir, rows, AutoencoderKLWan, Transformer3DModel, flow_loss, image_tensor_for_vae, prepare_frame_tensor, prepare_video_tensor, save_checkpoint, to_model_tensors


def build_cache(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    model_dir, rows, AutoencoderKLWan, Transformer3DModel, flow_loss, image_tensor_for_vae, prepare_frame_tensor, prepare_video_tensor, save_checkpoint, to_model_tensors = model_context(args, dtype)
    vae_dtype = torch.float32
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=vae_dtype).to(device).eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, vae_dtype)
    latents_std = (1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)).to(device, vae_dtype)
    cache = SafeBatchCache(
        vae,
        latents_mean,
        latents_std,
        device,
        dtype,
        prepare_frame_tensor,
        prepare_video_tensor,
        to_model_tensors,
        image_tensor_for_vae,
    )
    return model_dir, rows, vae, cache, Transformer3DModel, flow_loss, image_tensor_for_vae, prepare_video_tensor, save_checkpoint


def p1_vae_fp32(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0")
    dtype = torch.float32
    model_dir, rows, vae, cache, _Transformer3DModel, _flow_loss, image_tensor_for_vae, prepare_video_tensor, _save_checkpoint = build_cache(args, device, dtype)
    row = rows[0]
    width = int(row.get("width", 512))
    height = int(row.get("height", 512))
    num_frames = int(row.get("num_frames", 17))
    winner_np = prepare_video_tensor(Path(str(row["winner_path"])), width, height, num_frames)
    winner_tensor = image_tensor_for_vae(winner_np, device, dtype)
    with torch.no_grad():
        latent = vae.encode(winner_tensor).latent_dist.mode()
        decode_input = latent
        decoded = vae.decode(decode_input).sample
    result.update(
        status="PASS",
        dtype=str(dtype),
        model_dir=str(model_dir),
        sample_id=str(row.get("sample_id", "")),
        latent_shape=list(latent.shape),
        decoded_shape=list(decoded.shape),
        latent_finite=is_finite_tensor(latent),
        decoded_finite=is_finite_tensor(decoded),
        finite=is_finite_tensor(latent) and is_finite_tensor(decoded),
        max_memory_allocated=int(torch.cuda.max_memory_allocated(device)),
    )
    return result


def flow_step(args: argparse.Namespace, result: dict[str, Any], *, dtype: torch.dtype, backward: bool, train_step: bool, ddp: bool, rank: int, local_rank: int) -> dict[str, Any]:
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)
    model_dir, rows, vae, cache, Transformer3DModel, flow_loss, _image_tensor_for_vae, _prepare_video_tensor, save_checkpoint = build_cache(args, device, dtype)
    row = rows[rank % len(rows)]
    policy = Transformer3DModel.from_pretrained(model_dir / "transformer", torch_dtype=dtype).to(device)
    policy.train(mode=train_step or backward)
    for param in policy.parameters():
        param.requires_grad_(train_step or backward)
    wrapped = torch.nn.parallel.DistributedDataParallel(policy, device_ids=[local_rank], find_unused_parameters=True) if ddp else policy
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-7) if train_step else None
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    loss = flow_loss(wrapped, cache, row, "winner", args.seed + rank + 17, 0.37)
    finite_loss = is_finite_tensor(loss)
    grad_norm = 0.0
    grad_finite = True
    if backward or train_step:
        loss.backward()
        grad_sq = 0.0
        for param in wrapped.parameters():
            if param.grad is not None:
                grad = param.grad.detach().float()
                grad_finite = grad_finite and is_finite_tensor(grad)
                grad_sq += float((grad * grad).sum().cpu())
        grad_norm = math.sqrt(grad_sq) if math.isfinite(grad_sq) else float("nan")
    if optimizer is not None:
        torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0)
        optimizer.step()
    ckpt_status = "not_requested"
    ckpt_path = ""
    if train_step and args.save_checkpoint and rank == 0:
        ckpt_path = str(Path(args.output_root) / args.case / "checkpoint-1")
        model_to_save = wrapped.module if hasattr(wrapped, "module") else wrapped
        save_checkpoint(model_to_save, Path(ckpt_path))
        reloaded = Transformer3DModel.from_pretrained(Path(ckpt_path), torch_dtype=dtype).to(device).eval()
        with torch.no_grad():
            reload_loss = flow_loss(reloaded, cache, row, "winner", args.seed + 29, 0.41)
        ckpt_status = "PASS" if is_finite_tensor(reload_loss) else "NONFINITE_RELOAD"
        del reloaded
    result.update(
        status="PASS" if finite_loss and grad_finite and (not train_step or ckpt_status in {"PASS", "not_requested"}) else "FAIL",
        model_dir=str(model_dir),
        sample_id=str(row.get("sample_id", "")),
        dtype=str(dtype),
        backward=backward,
        train_step=train_step,
        ddp=ddp,
        loss=float(loss.detach().cpu()),
        finite_loss=finite_loss,
        grad_norm=grad_norm,
        grad_finite=grad_finite,
        checkpoint_status=ckpt_status,
        checkpoint_path=ckpt_path,
        max_memory_allocated=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    )
    return result


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    rank, local_rank, world_size, ddp = init_distributed_if_needed()
    result = common_result(args, rank, local_rank, world_size)
    try:
        if args.case == "P0":
            result = p0_torch_bf16(args, result)
        elif args.case == "P1":
            result = p1_vae_fp32(args, result)
        elif args.case == "P2":
            result = flow_step(args, result, dtype=torch.bfloat16, backward=False, train_step=False, ddp=False, rank=rank, local_rank=local_rank)
        elif args.case == "P3":
            result = flow_step(args, result, dtype=torch.bfloat16, backward=True, train_step=False, ddp=False, rank=rank, local_rank=local_rank)
        elif args.case == "P4":
            result = flow_step(args, result, dtype=torch.float32, backward=True, train_step=True, ddp=False, rank=rank, local_rank=local_rank)
        elif args.case == "P5":
            result = flow_step(args, result, dtype=torch.bfloat16, backward=True, train_step=True, ddp=False, rank=rank, local_rank=local_rank)
        elif args.case in {"P6", "P7"}:
            result = flow_step(args, result, dtype=torch.bfloat16, backward=True, train_step=True, ddp=ddp, rank=rank, local_rank=local_rank)
        else:
            raise ValueError(args.case)
    except BaseException as exc:  # noqa: BLE001 - report exact runtime failure.
        result.update(status="FAIL", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json(Path(args.output_root) / args.case / f"rank{rank}.json", result)
        finish_distributed(ddp)
    return result


def main() -> None:
    configure_safe_runtime()
    args = parse_args()
    result = run_case(args)
    if result.get("rank", 0) == 0:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
