"""Max-pool morphology region construction for Exp23.

The public experiment parameters are pool-grid steps, not image pixels.  The
functions here intentionally work on the loss/latent mask grid and optionally
use a finer intermediate pool grid before area-pooling the partition back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


EPS = 1e-8


@dataclass(frozen=True)
class PoolMorphologyConfig:
    pool_grid_scale: int = 1
    inner_pool_steps: int = 0
    outer_pool_steps: int = 1

    def validate(self) -> None:
        if self.pool_grid_scale not in (1, 2, 4):
            raise ValueError(f"pool_grid_scale must be one of 1/2/4, got {self.pool_grid_scale}")
        if self.inner_pool_steps < 0 or self.outer_pool_steps < 0:
            raise ValueError("pool steps must be non-negative")


def _flatten_to_nchw(x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    if x.ndim < 2:
        raise ValueError(f"mask must have at least H/W dims, got shape {tuple(x.shape)}")
    original_shape = x.shape
    h, w = x.shape[-2:]
    flat = x.reshape(-1, 1, h, w).float()
    return flat, original_shape


def _restore_from_nchw(x: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    return x.reshape(*original_shape[:-2], x.shape[-2], x.shape[-1])


def max_pool_steps(mask: torch.Tensor, steps: int) -> torch.Tensor:
    """Apply 3x3 stride-1 max-pool dilation steps to an NCHW tensor."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    x = mask
    for _ in range(steps):
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    return x


def dilate_steps(mask: torch.Tensor, steps: int) -> torch.Tensor:
    flat, original_shape = _flatten_to_nchw(mask)
    out = max_pool_steps(flat, steps)
    return _restore_from_nchw(out, original_shape)


def erode_steps(mask: torch.Tensor, steps: int) -> torch.Tensor:
    flat, original_shape = _flatten_to_nchw(mask)
    out = 1.0 - max_pool_steps(1.0 - flat, steps)
    return _restore_from_nchw(out, original_shape)


def _resize_mask(mask_nchw: torch.Tensor, scale: int) -> torch.Tensor:
    if scale == 1:
        return mask_nchw
    return F.interpolate(mask_nchw, scale_factor=scale, mode="nearest")


def _downsample_region(region_nchw: torch.Tensor, scale: int, target_hw: tuple[int, int]) -> torch.Tensor:
    if scale == 1:
        return region_nchw
    return F.interpolate(region_nchw, size=target_hw, mode="area")


def build_pool_regions(mask: torch.Tensor, config: PoolMorphologyConfig) -> Dict[str, torch.Tensor]:
    """Build core/inner/outer/outside partition.

    Args:
        mask: binary hole mask with 1=hole. Supports arbitrary leading dims.
        config: pool-grid morphology config.

    Returns:
        Dict of tensors matching mask shape:
            mask_core, inner_ring, outer_ring, far_outside.
        Regions are non-negative and sum to 1 per loss cell up to numerical eps.
    """
    config.validate()
    flat, original_shape = _flatten_to_nchw(mask)
    flat = (flat > 0.5).float()
    h, w = flat.shape[-2:]
    scale = config.pool_grid_scale
    high = _resize_mask(flat, scale)

    core = 1.0 - max_pool_steps(1.0 - high, config.inner_pool_steps)
    core = core.clamp(0.0, 1.0)
    inner = (high - core).clamp(0.0, 1.0)
    outer = (max_pool_steps(high, config.outer_pool_steps) - high).clamp(0.0, 1.0)
    outside = (1.0 - core - inner - outer).clamp(0.0, 1.0)

    regions = {
        "mask_core": _downsample_region(core, scale, (h, w)),
        "inner_ring": _downsample_region(inner, scale, (h, w)),
        "outer_ring": _downsample_region(outer, scale, (h, w)),
        "far_outside": _downsample_region(outside, scale, (h, w)),
    }

    total = sum(regions.values()).clamp_min(EPS)
    regions = {k: (v / total).clamp(0.0, 1.0) for k, v in regions.items()}
    return {k: _restore_from_nchw(v, original_shape) for k, v in regions.items()}


def build_region_weight_map(
    mask: torch.Tensor,
    config: PoolMorphologyConfig,
    core_weight: float = 1.0,
    inner_weight: float = 0.0,
    outer_weight: float = 0.75,
    outside_weight: float = 0.05,
) -> torch.Tensor:
    regions = build_pool_regions(mask, config)
    return (
        core_weight * regions["mask_core"]
        + inner_weight * regions["inner_ring"]
        + outer_weight * regions["outer_ring"]
        + outside_weight * regions["far_outside"]
    )


def region_area_stats(regions: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {k: float(v.detach().float().mean().cpu()) for k, v in regions.items()}

