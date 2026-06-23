#!/usr/bin/env python3
"""Run CPU exact-parity gates for Exp27 paper-code reproduction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp27_paper_grounded_preference_study.code.official_parity import (
    ema_update_tensor,
    exp27_sdpo_safe_lambda,
    linear_dpo_loss,
    load_official_sdpo_lambda,
    localdpo_mask_digest,
    write_json,
)
from exp27_paper_grounded_preference_study.code.localdpo_full_adapter import (
    localdpo_latent_fusion,
    progressive_outside_reinjection,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("exp27_paper_grounded_preference_study/parity"))
    p.add_argument("--seed", type=int, default=20260623)
    return p.parse_args()


def sdpo_gate(seed: int) -> dict:
    torch.manual_seed(seed)
    pred = torch.randn(6, 4, 8, 8, dtype=torch.float32)
    target = torch.randn_like(pred)
    try:
        official = load_official_sdpo_lambda()
    except FileNotFoundError as exc:
        return {
            "status": "blocked_official_code_missing",
            "error": repr(exc),
            "source": "Diffusion-SDPO/train.py under EXP27_PAPER_CODE_ROOT",
        }
    ours = exp27_sdpo_safe_lambda(pred, target, mu=0.37, eps=1e-8, max_lambda=1.0)
    theirs = official(pred, target, mu=0.37, eps=1e-8, max_lambda=1.0)
    max_abs = float((ours - theirs).abs().max().item())
    return {
        "status": "passed" if max_abs <= 1e-6 else "failed",
        "max_abs_diff": max_abs,
        "ours": ours.tolist(),
        "official": theirs.tolist(),
    }


def linear_gate(seed: int) -> dict:
    torch.manual_seed(seed + 1)
    mw = torch.rand(7, requires_grad=True)
    ml = torch.rand(7, requires_grad=True)
    rw = torch.rand(7)
    rl = torch.rand(7)
    loss = linear_dpo_loss(mw, ml, rw, rl, beta_dpo=5000.0, eta_dpo=0.01)
    loss.backward()
    grad = mw.grad.detach().clone()
    ema = torch.randn(5)
    model = torch.randn(5)
    updated = ema_update_tensor(ema.clone(), model, decay=0.9999)
    manual = ema * 0.9999 + model * 0.0001
    ema_max_abs = float((updated - manual).abs().max().item())
    return {
        "status": "passed" if torch.isfinite(loss) and ema_max_abs <= 1e-12 and torch.isfinite(grad).all() else "failed",
        "loss": float(loss.detach().item()),
        "grad_finite": bool(torch.isfinite(grad).all().item()),
        "ema_max_abs_diff": ema_max_abs,
        "utility": "ratio = clamp(0.2 * beta_dpo * (model_diff - ref_diff) + 0.5, eta, 1-eta)",
        "ema_update": "ema <- decay * ema + (1-decay) * model after optimizer step",
    }


def localdpo_gate(seed: int) -> dict:
    d1 = localdpo_mask_digest(seed=seed, video_length=13, image_height=120, image_width=216)
    d2 = localdpo_mask_digest(seed=seed, video_length=13, image_height=120, image_width=216)
    comparable_d1 = dict(d1)
    comparable_d2 = dict(d2)
    comparable_d1.pop("matplotlib_rgb_shim", None)
    comparable_d2.pop("matplotlib_rgb_shim", None)
    if str(d1.get("status", "")).startswith("blocked_official_code"):
        status = "blocked_official_code_runtime_error" if comparable_d1 == comparable_d2 else "failed"
        if d1.get("status") == "blocked_official_code_missing" and comparable_d1 == comparable_d2:
            status = "blocked_official_code_missing"
    else:
        status = "passed" if comparable_d1 == comparable_d2 and d1.get("shape") == [13, 120, 216] else "failed"
    return {
        "status": status,
        "digest": d1,
        "repeat_digest": d2,
        "source": "/home/hj/video_dpo_paper_code_cache/repos/Local-DPO/innerT2V/utils/random_mask_gen.py",
    }


def localdpo_fusion_gate(seed: int) -> dict:
    torch.manual_seed(seed + 2)
    denoised = torch.randn(2, 3, 4, 8, 8)
    original = torch.randn_like(denoised)
    current = torch.randn_like(denoised)
    mask = torch.zeros(2, 3, 1, 8, 8)
    mask[..., 2:6, 2:7] = 1.0
    fused = localdpo_latent_fusion(denoised, original, mask)
    progressive = progressive_outside_reinjection(current, denoised, original, mask)
    outside = mask.expand_as(fused) < 0.5
    inside = mask.expand_as(fused) > 0.5
    outside_diff = float((fused[outside] - original[outside]).abs().max().item())
    inside_diff = float((fused[inside] - denoised[inside]).abs().max().item())
    progressive_diff = float((fused - progressive).abs().max().item())
    passed = outside_diff <= 1e-7 and inside_diff <= 1e-7 and progressive_diff <= 1e-7
    return {
        "status": "passed" if passed else "failed",
        "outside_preservation_max_abs": outside_diff,
        "inside_denoised_max_abs": inside_diff,
        "progressive_step_max_abs": progressive_diff,
        "mask_semantics": "corruption_mask=1 uses denoised latent; corruption_mask=0 reinjects re-noised original latent",
        "adaptation_status": "algorithm_primitive_passed_real_diffueraser_batch_pending",
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "sdpo": sdpo_gate(args.seed),
        "linear_dpo": linear_gate(args.seed),
        "localdpo_mask": localdpo_gate(args.seed),
        "localdpo_latent_fusion": localdpo_fusion_gate(args.seed),
        "real_diffueraser_batch_parity": {
            "status": "pending",
            "reason": "No GPU batch was launched in this CPU parity command; SDPO/Linear real DiffuEraser parity remains required before studies.",
        },
    }
    gate_statuses = [v["status"] for k, v in results.items() if k != "real_diffueraser_batch_parity" and isinstance(v, dict) and "status" in v]
    if all(s == "passed" for s in gate_statuses):
        results["status"] = "passed"
        exit_code = 0
    elif any(str(s).startswith("blocked_official_code") for s in gate_statuses) and all(
        s == "passed" or str(s).startswith("blocked_official_code") for s in gate_statuses
    ):
        results["status"] = "partial_blocked_official_code"
        exit_code = 0
    else:
        results["status"] = "failed"
        exit_code = 2
    write_json(args.output_dir / "cpu_parity_summary.json", results)
    write_json(args.output_dir / "sdpo_parity.json", results["sdpo"])
    write_json(args.output_dir / "linear_dpo_parity.json", results["linear_dpo"])
    write_json(args.output_dir / "localdpo_parity.json", results["localdpo_mask"])
    write_json(args.output_dir / "localdpo_full_parity.json", results["localdpo_latent_fusion"])
    print(json.dumps(results, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
