"""Faithful LocalDPO algorithm primitives for Exp27 adaptation gates.

These helpers do not claim exact official backbone output. They encode the
paper-code semantics that must hold before adapting LocalDPO-style data to
DiffuEraser:

- task masks, corruption masks, and restoration-critical masks are separate;
- corruption-mask inside uses denoised/current latent;
- outside corruption mask is re-noised original latent at every step;
- region-aware training can use the restoration-critical region without
  silently replacing it with the task mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LocalDpoMasks:
    task_mask: torch.Tensor
    corruption_mask: torch.Tensor
    restoration_region: torch.Tensor

    def validate(self) -> None:
        for name, value in (
            ("task_mask", self.task_mask),
            ("corruption_mask", self.corruption_mask),
            ("restoration_region", self.restoration_region),
        ):
            if value.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                raise TypeError(f"{name} must be floating point")
            if value.min().item() < 0 or value.max().item() > 1:
                raise ValueError(f"{name} must be in [0,1]")
        if self.task_mask.shape != self.corruption_mask.shape:
            raise ValueError("task_mask and corruption_mask must have matching shapes")
        if self.restoration_region.shape != self.task_mask.shape:
            raise ValueError("restoration_region must match task_mask shape")


def resize_mask_like(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Resize a [B,T,1,H,W] or broadcastable mask to latent shape."""
    if mask.ndim != target.ndim:
        while mask.ndim < target.ndim:
            mask = mask.unsqueeze(2)
    if mask.shape[-2:] != target.shape[-2:]:
        flat = mask.reshape(-1, 1, mask.shape[-2], mask.shape[-1]).float()
        resized = F.interpolate(flat, size=target.shape[-2:], mode="nearest")
        mask = resized.reshape(*mask.shape[:-2], *target.shape[-2:]).to(dtype=target.dtype, device=target.device)
    return mask.to(dtype=target.dtype, device=target.device)


def localdpo_latent_fusion(
    denoised_latent: torch.Tensor,
    renoised_original_latent: torch.Tensor,
    corruption_mask: torch.Tensor,
) -> torch.Tensor:
    """Fuse latents according to LocalDPO outside-region preservation.

    Mask value 1 means the local corruption/restoration region: keep denoised.
    Mask value 0 means outside: reinject the re-noised original latent.
    """

    if denoised_latent.shape != renoised_original_latent.shape:
        raise ValueError("denoised and original latents must have identical shapes")
    mask = resize_mask_like(corruption_mask, denoised_latent)
    return denoised_latent * mask + renoised_original_latent * (1.0 - mask)


def progressive_outside_reinjection(
    current_latent: torch.Tensor,
    model_step_latent: torch.Tensor,
    renoised_original_latent: torch.Tensor,
    corruption_mask: torch.Tensor,
) -> torch.Tensor:
    """One progressive denoising step with outside latent reinjection."""

    del current_latent  # Current latent is explicit in the signature for audit logs.
    return localdpo_latent_fusion(model_step_latent, renoised_original_latent, corruption_mask)


def region_aware_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    restoration_region: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Region-aware loss used for adaptation diagnostics."""

    if pred.shape != target.shape:
        raise ValueError("pred and target must have identical shapes")
    region = resize_mask_like(restoration_region, pred)
    return (torch.abs(pred - target) * region).sum() / (region.sum() * pred.shape[2] + eps)
