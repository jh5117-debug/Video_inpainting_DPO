#!/usr/bin/env python3
"""Exp27 LocalDPO 6-video corruption-pair and original-loss smoke.

This is a plumbing gate, not RC-FPO and not DiffuEraser training. It verifies
that LocalDPO-style corruption/restoration masks can form six complete video
pairs and that the original region-aware loss path supports 1-step and 10-step
optimizer updates with finite gradients.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp27_paper_grounded_preference_study.code.localdpo_full_adapter import (  # noqa: E402
    LocalDpoMasks,
    localdpo_latent_fusion,
    region_aware_l1,
)
from exp27_paper_grounded_preference_study.code.official_parity import localdpo_mask_digest  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260624)
    p.add_argument("--videos", type=int, default=6)
    p.add_argument("--frames", type=int, default=13)
    p.add_argument("--height", type=int, default=48)
    p.add_argument("--width", type=int, default=80)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def make_mask(batch: int, frames: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(batch, frames, 1, height, width, device=device)
    for b in range(batch):
        y0 = 4 + (b * 5) % max(5, height // 3)
        x0 = 6 + (b * 7) % max(7, width // 3)
        h = max(8, height // 4)
        w = max(10, width // 4)
        for t in range(frames):
            yy = min(height - h, y0 + t % 5)
            xx = min(width - w, x0 + (2 * t) % 7)
            mask[b, t, :, yy : yy + h, xx : xx + w] = 1.0
    return mask


def run_steps(steps: int, args: argparse.Namespace, device: torch.device) -> dict:
    torch.manual_seed(args.seed + steps)
    model = torch.nn.Conv3d(4, 4, kernel_size=1, bias=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    corruption = make_mask(args.videos, args.frames, args.height, args.width, device)
    restoration = corruption.clone()
    task = corruption.clone()
    masks = LocalDpoMasks(task_mask=task, corruption_mask=corruption, restoration_region=restoration)
    masks.validate()
    losses: list[float] = []
    grad_norms: list[float] = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        current = torch.randn(args.videos, args.frames, 4, args.height, args.width, device=device)
        original = torch.randn_like(current)
        # Conv3d expects [B,C,T,H,W].
        pred = model(current.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        fused = localdpo_latent_fusion(pred, original, corruption)
        target = torch.zeros_like(fused)
        loss = region_aware_l1(fused, target, restoration)
        loss.backward()
        grad = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad += float(p.grad.detach().float().pow(2).sum().cpu())
        grad_norms.append(math.sqrt(grad))
        opt.step()
        losses.append(float(loss.detach().cpu()))
    finite = all(math.isfinite(v) for v in losses + grad_norms)
    return {
        "status": "passed" if finite and all(g > 0 for g in grad_norms) else "failed",
        "steps": steps,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "grad_norm_min": min(grad_norms),
        "grad_norm_max": max(grad_norms),
        "pair_shape": [args.videos, args.frames, 4, args.height, args.width],
        "corruption_mask_mean": float(corruption.mean().cpu()),
    }


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device if args.device.startswith("cuda") else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    official_digest = localdpo_mask_digest(seed=args.seed, video_length=args.frames, image_height=args.height, image_width=args.width)
    result = {
        "status": "passed",
        "note": "LocalDPO plumbing/original-loss smoke only; RC-FPO not started.",
        "official_mask_digest": official_digest,
        "one_step": run_steps(1, args, device),
        "ten_step": run_steps(10, args, device),
    }
    if result["one_step"]["status"] != "passed" or result["ten_step"]["status"] != "passed":
        result["status"] = "failed"
    (args.output_dir / "localdpo_six_video_smoke.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
