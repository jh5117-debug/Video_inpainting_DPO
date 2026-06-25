"""Exp28 image-space inner boundary geometry.

Exp28 keeps the Exp11/Exp23 legacy outer ring exactly: one 3x3 max-pool
dilation step on the latent/loss mask grid.  Only the inner mask boundary is
introduced from image-space pixel erosion and area-pooled back to the loss grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


EPS = 1e-8


@dataclass(frozen=True)
class InnerBoundaryConfig:
    inner_radius_px: int = 0
    mask_core_weight: float = 1.0
    inner_weight: float = 0.75
    outer_weight: float = 0.75
    outside_weight: float = 0.05

    def validate(self) -> None:
        if self.inner_radius_px < 0:
            raise ValueError("inner_radius_px must be non-negative")
        for name in ("mask_core_weight", "inner_weight", "outer_weight", "outside_weight"):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")


def _flatten_spatial(x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    if x.ndim < 2:
        raise ValueError(f"expected tensor with spatial H/W dims, got shape={tuple(x.shape)}")
    original_shape = x.shape
    h, w = x.shape[-2:]
    return x.reshape(-1, 1, h, w).float(), original_shape


def _restore_spatial(x: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    return x.reshape(*original_shape[:-2], x.shape[-2], x.shape[-1])


def _resize_spatial(
    x: torch.Tensor,
    target_hw: Tuple[int, int],
    *,
    mode: str,
) -> torch.Tensor:
    flat, original_shape = _flatten_spatial(x)
    out = F.interpolate(flat, size=target_hw, mode=mode)
    return _restore_spatial(out, original_shape)


def legacy_outer_one_ring(loss_hole_mask: torch.Tensor) -> torch.Tensor:
    """Return the exact Exp11/Exp23 outer one-ring on the loss grid."""
    flat, original_shape = _flatten_spatial((loss_hole_mask > 0.5).float())
    dilated = F.max_pool2d(flat, kernel_size=3, stride=1, padding=1)
    outer = (dilated - flat).clamp(0.0, 1.0)
    return _restore_spatial(outer, original_shape)


def erode_image_mask(mask: torch.Tensor, radius_px: int) -> torch.Tensor:
    """Erode an image-space binary hole mask by a square pixel radius."""
    if radius_px < 0:
        raise ValueError("radius_px must be non-negative")
    binary = (mask > 0.5).float()
    if radius_px == 0:
        return binary
    flat, original_shape = _flatten_spatial(binary)
    kernel = 2 * int(radius_px) + 1
    eroded = 1.0 - F.max_pool2d(1.0 - flat, kernel_size=kernel, stride=1, padding=radius_px)
    return _restore_spatial(eroded.clamp(0.0, 1.0), original_shape)


def image_inner_ring_to_loss_grid(
    image_hole_mask: torch.Tensor,
    loss_hole_mask: torch.Tensor,
    inner_radius_px: int,
) -> torch.Tensor:
    """Build a fractional loss-grid inner ring via image-space erosion + area pooling."""
    if inner_radius_px == 0:
        return torch.zeros_like(loss_hole_mask, dtype=torch.float32)
    image_hole = (image_hole_mask > 0.5).float()
    eroded = erode_image_mask(image_hole, inner_radius_px)
    ring_image = (image_hole - eroded).clamp(0.0, 1.0)
    ring_loss = _resize_spatial(ring_image, loss_hole_mask.shape[-2:], mode="area")
    return torch.minimum(ring_loss.to(loss_hole_mask.device), (loss_hole_mask > 0.5).float()).clamp(0.0, 1.0)


def build_inner_boundary_regions(
    image_hole_mask: torch.Tensor,
    loss_hole_mask: torch.Tensor,
    config: InnerBoundaryConfig,
) -> Dict[str, torch.Tensor]:
    """Return mask_core, inner_ring, outer_ring, far_outside on the loss grid.

    `image_hole_mask` is the original image-space mask with 1=hole/object.
    `loss_hole_mask` is the explicit loss-grid mask used by the fresh control.
    """
    config.validate()
    if image_hole_mask.shape[:-2] != loss_hole_mask.shape[:-2]:
        raise ValueError(
            "image_hole_mask and loss_hole_mask must share leading dims; "
            f"got {tuple(image_hole_mask.shape)} and {tuple(loss_hole_mask.shape)}"
        )
    hole_loss = (loss_hole_mask > 0.5).float()
    inner = image_inner_ring_to_loss_grid(image_hole_mask, hole_loss, config.inner_radius_px)
    core = (hole_loss - inner).clamp(0.0, 1.0)
    outer = legacy_outer_one_ring(hole_loss)
    far_outside = (1.0 - core - inner - outer).clamp(0.0, 1.0)

    return {
        "mask_core": core,
        "inner_ring": inner,
        "outer_ring": outer,
        "far_outside": far_outside,
    }


def build_inner_boundary_weight_map(
    image_hole_mask: torch.Tensor,
    loss_hole_mask: torch.Tensor,
    config: InnerBoundaryConfig,
) -> torch.Tensor:
    regions = build_inner_boundary_regions(image_hole_mask, loss_hole_mask, config)
    return (
        float(config.mask_core_weight) * regions["mask_core"]
        + float(config.inner_weight) * regions["inner_ring"]
        + float(config.outer_weight) * regions["outer_ring"]
        + float(config.outside_weight) * regions["far_outside"]
    )


def region_area_stats(regions: Dict[str, torch.Tensor]) -> Dict[str, float]:
    stats = {k: float(v.detach().float().mean().cpu()) for k, v in regions.items()}
    total = sum(regions.values())
    stats["partition_sum_max_abs_error"] = float((total - 1.0).abs().max().detach().cpu())
    stats["illegal_inner_outer_overlap"] = float((regions["inner_ring"] * regions["outer_ring"]).sum().detach().cpu())
    return stats
