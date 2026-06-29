#!/usr/bin/env python3
"""Exp46 pseudo-success BF16-safe preflight with no optimizer step."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import torch


def load_exp43_runner(project_root: Path):
    runner_dir = project_root / "exp43_h20_minimax_stage2_sft_runner"
    sys.path.insert(0, str(runner_dir.resolve()))
    import runner_stage2_sft_ladder as runner  # type: ignore
    return runner


def is_finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t.detach()).all().cpu())


def init_dist() -> tuple[int, int, int, bool]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size > 1
    if ddp:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, ddp


def finish_dist(ddp: bool) -> None:
    if ddp and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def common(args: argparse.Namespace, rank: int, local_rank: int, world_size: int, policy: Any) -> dict[str, Any]:
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
        "training_run": False,
        "optimizer_step": False,
    }


def p0(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    a = torch.randn((2048, 2048), device=device, dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn((2048, 2048), device=device, dtype=torch.bfloat16)
    loss = (a @ b).float().pow(2).mean()
    loss.backward()
    finite = is_finite(loss) and is_finite(a.grad)
    result.update(status="PASS" if finite else "FAIL", loss=float(loss.detach().cpu()), grad_norm=float(a.grad.detach().float().norm().cpu()), finite=finite, max_memory_allocated=int(torch.cuda.max_memory_allocated(device)))
    return result


def build(args: argparse.Namespace, runner: Any, device: torch.device, dtype: torch.dtype):
    return runner.build_cache(args, device, dtype)


def load_model(model_dir: Path, Transformer3DModel: Any, dtype: torch.dtype, device: torch.device) -> torch.nn.Module:
    model = Transformer3DModel.from_pretrained(model_dir / "transformer", torch_dtype=dtype).to(device)
    return model


def model_case(args: argparse.Namespace, result: dict[str, Any], runner: Any, *, backward: bool, save_reload: bool, ddp: bool, rank: int, local_rank: int) -> dict[str, Any]:
    device = torch.device(f"cuda:{local_rank}")
    policy = runner.apply_runtime_policy(runner.PrecisionPolicy())
    dtype = torch.bfloat16
    torch.manual_seed(args.seed + rank)
    model_dir, rows, _vae, cache, Transformer3DModel, _flow_loss, _image_tensor_for_vae, _prepare_video_tensor, _run_pipeline, save_checkpoint = build(args, runner, device, dtype)
    row = rows[rank % len(rows)]
    model = load_model(model_dir, Transformer3DModel, dtype, device)
    model.train(mode=backward)
    for param in model.parameters():
        param.requires_grad_(backward)
    wrapped = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True) if ddp else model
    weights = {"mask": 0.75, "boundary": 1.50, "affected": 0.75, "outside": 0.20, "far_outside": 0.03}
    if backward:
        loss, diag = runner.sft_weighted_loss(wrapped, cache, row, seed=args.seed + rank + 31, tval=args.timestep, weights=weights, policy=policy)
        loss.backward()
        grad_norm, grad_finite = runner.grad_norm_fp32(wrapped)
    else:
        with torch.no_grad():
            loss, diag = runner.sft_weighted_loss(wrapped, cache, row, seed=args.seed + rank + 31, tval=args.timestep, weights=weights, policy=policy)
        grad_norm, grad_finite = 0.0, True
    finite_loss = is_finite(loss)
    ckpt_status = "not_requested"
    ckpt_path = ""
    if save_reload and rank == 0:
        ckpt_path = str(Path(args.output_root) / args.case / "checkpoint-dry-run")
        model_to_save = wrapped.module if hasattr(wrapped, "module") else wrapped
        save_checkpoint(model_to_save, Path(ckpt_path))
        reloaded = Transformer3DModel.from_pretrained(Path(ckpt_path), torch_dtype=dtype).to(device).eval()
        with torch.no_grad():
            reload_loss, _reload_diag = runner.sft_weighted_loss(reloaded, cache, row, seed=args.seed + 97, tval=0.41, weights=weights, policy=policy)
        ckpt_status = "PASS" if is_finite(reload_loss) else "NONFINITE_RELOAD"
        del reloaded
    status = "PASS" if finite_loss and grad_finite and (ckpt_status in {"not_requested", "PASS"}) else "FAIL"
    result.update(status=status, sample_id=str(row.get("sample_id", "")), dtype=str(dtype), backward=backward, ddp=ddp, loss=float(loss.detach().cpu()), finite_loss=finite_loss, grad_norm=float(grad_norm), grad_finite=bool(grad_finite), checkpoint_status=ckpt_status, checkpoint_path=ckpt_path, max_memory_allocated=int(torch.cuda.max_memory_allocated(device)))
    for key, value in diag.items():
        result[f"diag_{key}" if key in result else key] = value
    return result


def p1(args: argparse.Namespace, result: dict[str, Any], runner: Any) -> dict[str, Any]:
    device = torch.device("cuda:0")
    model_dir, rows, vae, _cache, _Transformer3DModel, _flow_loss, image_tensor_for_vae, prepare_video_tensor, _run_pipeline, _save_checkpoint = build(args, runner, device, torch.float32)
    row = rows[0]
    winner_np = prepare_video_tensor(Path(str(row["winner_path"])), int(row.get("width", 512)), int(row.get("height", 512)), int(row.get("num_frames", 17)))
    winner_tensor = image_tensor_for_vae(winner_np, device, torch.float32)
    with torch.no_grad():
        latent = vae.encode(winner_tensor).latent_dist.mode()
        decoded = vae.decode(latent).sample
    finite = is_finite(latent) and is_finite(decoded)
    result.update(status="PASS" if finite else "FAIL", model_dir=str(model_dir), sample_id=str(row.get("sample_id", "")), dtype="torch.float32", latent_shape=list(latent.shape), decoded_shape=list(decoded.shape), latent_finite=is_finite(latent), decoded_finite=is_finite(decoded), finite=finite, max_memory_allocated=int(torch.cuda.max_memory_allocated(device)))
    return result


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    runner = load_exp43_runner(Path(args.project_root))
    policy = runner.apply_runtime_policy(runner.PrecisionPolicy())
    rank, local_rank, world_size, ddp = init_dist()
    result = common(args, rank, local_rank, world_size, policy)
    try:
        if args.case == "P0":
            result = p0(args, result)
        elif args.case == "P1":
            result = p1(args, result, runner)
        elif args.case in {"P2", "P3"}:
            result = model_case(args, result, runner, backward=False, save_reload=False, ddp=False, rank=rank, local_rank=local_rank)
        elif args.case == "P4":
            result = model_case(args, result, runner, backward=True, save_reload=False, ddp=False, rank=rank, local_rank=local_rank)
        elif args.case == "P5":
            result = model_case(args, result, runner, backward=True, save_reload=True, ddp=False, rank=rank, local_rank=local_rank)
        elif args.case in {"P6", "P7"}:
            result = model_case(args, result, runner, backward=True, save_reload=False, ddp=ddp, rank=rank, local_rank=local_rank)
        else:
            raise ValueError(args.case)
    except BaseException as exc:  # noqa: BLE001
        result.update(status="FAIL", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        output = Path(args.output_root) / args.case
        output.mkdir(parents=True, exist_ok=True)
        (output / f"rank{rank}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        finish_dist(ddp)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--timestep", type=float, default=0.41)
    args = parser.parse_args()
    result = run_case(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
