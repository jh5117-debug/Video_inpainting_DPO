"""Boundary-map utilities for Exp20.

This module is intentionally standalone. It does not import or modify the
legacy Exp11 trainer. The legacy latent map path is implemented so tests can
verify parity before image-space boundary variants are enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-8


@dataclass(frozen=True)
class RadiusStats:
    radius_px: float
    area_median: float
    perimeter_median: float
    sqrt_area_median: float
    clamp_min: bool
    clamp_max: bool


def _as_bthw(mask: torch.Tensor) -> torch.Tensor:
    """Normalize mask to [B,T,H,W] float tensor with values in {0,1}."""
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[None]
    elif mask.ndim == 5 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    elif mask.ndim != 4:
        raise ValueError(f"Expected mask with ndim 2/3/4/5, got {tuple(mask.shape)}")
    return (mask.float() > 0.5).float()


def legacy_latent_outer_ring(mask_latent: torch.Tensor) -> torch.Tensor:
    """Exp11 outer ring: one latent-cell 3x3 max-pool dilation minus mask."""
    m = _as_bthw(mask_latent)
    b, t, h, w = m.shape
    flat = m.reshape(b * t, 1, h, w)
    dilated = F.max_pool2d(flat, kernel_size=3, stride=1, padding=1)
    outer = torch.clamp(dilated - flat, 0.0, 1.0)
    return outer.reshape(b, t, h, w)


def legacy_latent_inner_ring(mask_latent: torch.Tensor) -> torch.Tensor:
    """Exp11 inner ring: mask minus one latent-cell erosion."""
    m = _as_bthw(mask_latent)
    b, t, h, w = m.shape
    flat = m.reshape(b * t, 1, h, w)
    eroded = 1.0 - F.max_pool2d(1.0 - flat, kernel_size=3, stride=1, padding=1)
    inner = torch.clamp(flat - eroded, 0.0, 1.0)
    return inner.reshape(b, t, h, w)


def legacy_latent_boundary(mask_latent: torch.Tensor, mode: str) -> torch.Tensor:
    mode = mode.lower()
    if mode == "inner":
        return legacy_latent_inner_ring(mask_latent)
    if mode == "outer":
        return legacy_latent_outer_ring(mask_latent)
    if mode == "both":
        return torch.clamp(
            legacy_latent_inner_ring(mask_latent) + legacy_latent_outer_ring(mask_latent),
            0.0,
            1.0,
        )
    raise ValueError(f"Unsupported legacy boundary mode: {mode}")


def _distance_transform_outside(mask_2d: np.ndarray) -> np.ndarray:
    """Distance from every outside pixel to nearest mask pixel.

    Uses scipy/cv2 if available, with a brute-force fallback for tiny unit tests.
    The input convention is 1 = hole/mask, 0 = outside/context.
    """
    mask_bool = mask_2d.astype(bool)
    if mask_bool.sum() == 0:
        return np.full(mask_2d.shape, np.inf, dtype=np.float32)
    if mask_bool.all():
        return np.zeros(mask_2d.shape, dtype=np.float32)
    try:
        from scipy.ndimage import distance_transform_edt

        return distance_transform_edt(~mask_bool).astype(np.float32)
    except Exception:
        pass
    try:
        import cv2

        outside = (~mask_bool).astype(np.uint8)
        return cv2.distanceTransform(outside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE).astype(np.float32)
    except Exception:
        pass

    yy, xx = np.indices(mask_2d.shape)
    my, mx = np.nonzero(mask_bool)
    coords = np.stack([yy[..., None] - my, xx[..., None] - mx], axis=-1)
    return np.sqrt((coords.astype(np.float32) ** 2).sum(axis=-1)).min(axis=-1)


def image_space_outer_ring(mask_image: torch.Tensor, radius_px: float) -> torch.Tensor:
    """Compute an image-space Euclidean outer ring before pooling to latent.

    Returns [B,T,H,W], 1 outside the hole and within radius_px of the hole.
    """
    if radius_px <= 0:
        return torch.zeros_like(_as_bthw(mask_image))
    m = _as_bthw(mask_image)
    out = []
    for frame in m.detach().cpu().numpy().reshape(-1, *m.shape[-2:]):
        dist = _distance_transform_outside(frame)
        ring = ((frame < 0.5) & (dist > 0.0) & (dist <= float(radius_px))).astype(np.float32)
        out.append(ring)
    arr = np.stack(out, axis=0).reshape(m.shape)
    return torch.from_numpy(arr).to(device=m.device, dtype=m.dtype)


def area_pool_to_shape(mask_image: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Area-pool an image-space soft map to loss resolution."""
    m = _as_bthw(mask_image)
    b, t, h, w = m.shape
    flat = m.reshape(b * t, 1, h, w)
    pooled = F.interpolate(flat, size=target_hw, mode="area")
    return pooled.reshape(b, t, *target_hw)


def nearest_to_shape(mask_image: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    m = _as_bthw(mask_image)
    b, t, h, w = m.shape
    flat = m.reshape(b * t, 1, h, w)
    pooled = F.interpolate(flat, size=target_hw, mode="nearest")
    return pooled.reshape(b, t, *target_hw)


def frame_area_perimeter(mask_2d: torch.Tensor) -> Tuple[float, float]:
    """Return area and 4-neighborhood perimeter for a 2D binary mask."""
    m = (mask_2d.float() > 0.5).float()
    area = float(m.sum().item())
    if area == 0:
        return 0.0, 0.0
    padded = F.pad(m[None, None], (1, 1, 1, 1), value=0.0)[0, 0]
    center = padded[1:-1, 1:-1]
    perimeter = (
        (center != padded[:-2, 1:-1]).float()
        + (center != padded[2:, 1:-1]).float()
        + (center != padded[1:-1, :-2]).float()
        + (center != padded[1:-1, 2:]).float()
    )
    return area, float((perimeter * center).sum().item())


def adaptive_radius(
    mask_image: torch.Tensor,
    mode: str,
    k: float,
    r_min: float = 2.0,
    r_max: float = 48.0,
) -> RadiusStats:
    """Stable clip-level adaptive radius from median frame statistics."""
    m = _as_bthw(mask_image)
    areas = []
    perimeters = []
    sqrt_areas = []
    for frame in m.reshape(-1, *m.shape[-2:]):
        area, perimeter = frame_area_perimeter(frame)
        if area <= 0:
            continue
        areas.append(area)
        perimeters.append(max(perimeter, EPS))
        sqrt_areas.append(float(np.sqrt(area / np.pi)))
    if not areas:
        radius = r_min
        return RadiusStats(radius, 0.0, 0.0, 0.0, True, False)
    area_med = float(np.median(areas))
    perim_med = float(np.median(perimeters))
    sqrt_med = float(np.median(sqrt_areas))
    if mode == "adaptive_area_perimeter":
        raw = float(k) * area_med / max(perim_med, EPS)
    elif mode == "adaptive_sqrt_area":
        raw = float(k) * sqrt_med
    else:
        raise ValueError(f"Unsupported adaptive radius mode: {mode}")
    radius = float(np.clip(raw, r_min, r_max))
    return RadiusStats(radius, area_med, perim_med, sqrt_med, radius <= r_min + EPS, radius >= r_max - EPS)


def build_region_maps(
    mask_image: torch.Tensor,
    loss_hw: Tuple[int, int],
    radius_mode: str = "legacy_latent_exact",
    radius_value: float = 0.0,
    adaptive_k: float = 1.0,
    boundary_mode: str = "outer",
) -> Dict[str, torch.Tensor | RadiusStats]:
    """Build mask, boundary, and outside maps at loss resolution."""
    mask_loss_nearest = nearest_to_shape(mask_image, loss_hw)
    if radius_mode == "legacy_latent_exact":
        boundary = legacy_latent_boundary(mask_loss_nearest, boundary_mode)
        radius_stats = RadiusStats(1.0, 0.0, 0.0, 0.0, False, False)
    else:
        if radius_mode == "fixed_image_px":
            radius = float(radius_value)
            radius_stats = RadiusStats(radius, 0.0, 0.0, 0.0, False, False)
        else:
            radius_stats = adaptive_radius(mask_image, radius_mode, adaptive_k)
            radius = radius_stats.radius_px
        boundary_img = image_space_outer_ring(mask_image, radius)
        boundary = area_pool_to_shape(boundary_img, loss_hw).clamp(0.0, 1.0)
    outside = torch.clamp(1.0 - torch.maximum(mask_loss_nearest, (boundary > 0).float()), 0.0, 1.0)
    return {
        "mask": mask_loss_nearest,
        "boundary": boundary,
        "outside": outside,
        "radius_stats": radius_stats,
    }


def summarize_maps(region_maps: Dict[str, torch.Tensor | RadiusStats]) -> Dict[str, float]:
    mask = region_maps["mask"]
    boundary = region_maps["boundary"]
    outside = region_maps["outside"]
    assert isinstance(mask, torch.Tensor)
    assert isinstance(boundary, torch.Tensor)
    assert isinstance(outside, torch.Tensor)
    total = float(mask.numel())
    stats = {
        "mask_area_ratio": float(mask.mean().item()),
        "boundary_area_ratio": float(boundary.mean().item()),
        "outside_area_ratio": float(outside.mean().item()),
        "latent_boundary_coverage": float((boundary > 0).float().mean().item()),
    }
    radius = region_maps.get("radius_stats")
    if isinstance(radius, RadiusStats):
        stats.update(
            {
                "radius_px": radius.radius_px,
                "radius_area_median": radius.area_median,
                "radius_perimeter_median": radius.perimeter_median,
                "radius_sqrt_area_median": radius.sqrt_area_median,
                "radius_clamp_min": float(radius.clamp_min),
                "radius_clamp_max": float(radius.clamp_max),
            }
        )
    stats["pixel_count"] = total
    return stats
