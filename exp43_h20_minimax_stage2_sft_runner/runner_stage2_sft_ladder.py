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

import torch

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
    return model_dir, rows, vae, cache, Transformer3DModel, flow_loss, image_tensor_for_vae, prepare_video_tensor, save_checkpoint


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
    model_dir, rows, vae, _cache, _Transformer3DModel, _flow_loss, image_tensor_for_vae, prepare_video_tensor, _save_checkpoint = build_cache(args, device, torch.float32)
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
    model_dir, rows, _vae, cache, Transformer3DModel, flow_loss, _image_tensor_for_vae, _prepare_video_tensor, save_checkpoint = build_cache(args, device, dtype)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--case", required=True, choices=PREFLIGHT_CASES)
    add_common_runtime_args(preflight)
    summarize = sub.add_parser("summarize-preflight")
    summarize.add_argument("--output-root", required=True)
    summarize.add_argument("--reports-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        result = run_preflight_case(args)
        if result.get("rank", 0) == 0:
            print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "summarize-preflight":
        print(json.dumps(summarize_preflight(args), indent=2, sort_keys=True))
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
