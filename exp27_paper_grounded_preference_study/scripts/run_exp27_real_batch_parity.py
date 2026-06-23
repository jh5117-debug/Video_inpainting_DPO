#!/usr/bin/env python3
"""GPU real-batch objective parity smoke for Exp27 SDPO and Linear-DPO.

This gate intentionally stays short and does not start a study/training run. It
uses real GPU tensors shaped like DiffuEraser epsilon predictions so that the
paper-objective helpers are exercised with batch/device/autograd semantics,
after the CPU exact formula parity has already passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp27_paper_grounded_preference_study.code.official_parity import (  # noqa: E402
    ema_update_tensor,
    exp27_sdpo_safe_lambda,
    linear_dpo_clip_ratio,
    linear_dpo_loss,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260623)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--mode", choices=["sdpo", "linear", "all"], default="all")
    return p.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float32


def sdpo_real_batch(seed: int, dtype: torch.dtype, device: torch.device) -> dict:
    torch.manual_seed(seed)
    pred = torch.randn(8, 4, 8, 30, 54, device=device, dtype=dtype, requires_grad=True)
    target = torch.randn_like(pred)
    lam = exp27_sdpo_safe_lambda(pred, target, mu=0.37, eps=1e-8, max_lambda=1.0)
    pred_w, pred_l = pred.chunk(2, dim=0)
    target_w, target_l = target.detach().chunk(2, dim=0)
    win_loss = ((pred_w.float() - target_w.float()) ** 2).mean(dim=(1, 2, 3, 4))
    lose_loss = ((pred_l.float() - target_l.float()) ** 2).mean(dim=(1, 2, 3, 4))
    objective = (win_loss - lam.float() * lose_loss).mean()
    objective.backward()
    grad_norm = float(pred.grad.detach().float().norm().cpu())
    finite = bool(torch.isfinite(pred.grad).all().item()) and bool(torch.isfinite(lam).all().item())
    return {
        "status": "passed" if finite and grad_norm > 0 else "failed",
        "lambda_shape": list(lam.shape),
        "lambda_min": float(lam.float().min().cpu()),
        "lambda_max": float(lam.float().max().cpu()),
        "objective": float(objective.detach().float().cpu()),
        "grad_norm": grad_norm,
        "finite": finite,
        "device": str(device),
        "dtype": str(dtype),
    }


def linear_real_batch(seed: int, dtype: torch.dtype, device: torch.device) -> dict:
    torch.manual_seed(seed + 17)
    model_w = torch.rand(6, device=device, dtype=dtype, requires_grad=True)
    model_l = torch.rand(6, device=device, dtype=dtype, requires_grad=True)
    ref_w = torch.rand(6, device=device, dtype=dtype)
    ref_l = torch.rand(6, device=device, dtype=dtype)
    ratio = linear_dpo_clip_ratio(model_w.float(), model_l.float(), beta_dpo=5000.0, eta_dpo=0.01)
    loss = linear_dpo_loss(model_w.float(), model_l.float(), ref_w.float(), ref_l.float(), beta_dpo=5000.0, eta_dpo=0.01)
    loss.backward()
    ema = torch.randn(32, device=device, dtype=dtype)
    model = torch.randn(32, device=device, dtype=dtype)
    updated = ema_update_tensor(ema.clone(), model, decay=0.9999)
    manual = ema * 0.9999 + model * 0.0001
    ema_diff = float((updated.float() - manual.float()).abs().max().cpu())
    grad_norm = float(model_w.grad.detach().float().norm().cpu() + model_l.grad.detach().float().norm().cpu())
    finite = bool(torch.isfinite(model_w.grad).all().item()) and bool(torch.isfinite(model_l.grad).all().item())
    return {
        "status": "passed" if finite and grad_norm > 0 and ema_diff <= 5e-4 else "failed",
        "loss": float(loss.detach().float().cpu()),
        "ratio_min": float(ratio.float().min().cpu()),
        "ratio_max": float(ratio.float().max().cpu()),
        "ema_max_abs_diff": ema_diff,
        "grad_norm": grad_norm,
        "finite": finite,
        "device": str(device),
        "dtype": str(dtype),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real-batch parity")
    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {"status": "passed", "seed": args.seed, "mode": args.mode}
    if args.mode in {"sdpo", "all"}:
        result["sdpo"] = sdpo_real_batch(args.seed, dtype, device)
    if args.mode in {"linear", "all"}:
        result["linear"] = linear_real_batch(args.seed, dtype, device)
    statuses = [v.get("status") for v in result.values() if isinstance(v, dict) and "status" in v]
    if not statuses or any(s != "passed" for s in statuses):
        result["status"] = "failed"
    (args.output_dir / f"real_batch_{args.mode}_parity.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
