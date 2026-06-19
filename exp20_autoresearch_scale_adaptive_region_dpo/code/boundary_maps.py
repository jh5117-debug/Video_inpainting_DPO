"""Boundary-map utilities for Exp20.

This module is intentionally standalone. It does not import or modify the
legacy Exp11 trainer. The legacy latent map path is implemented so tests can
verify parity before image-space boundary variants are enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from pathlib import Path
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


@dataclass(frozen=True)
class BatchRadiusStats:
    radius_px: torch.Tensor
    empty_clip: torch.Tensor
    area_median: torch.Tensor
    perimeter_median: torch.Tensor
    sqrt_area_median: torch.Tensor
    clamp_min: torch.Tensor
    clamp_max: torch.Tensor


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


def _radius_for_batch(radius_px: float | torch.Tensor, batch: int, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(radius_px):
        r = radius_px.to(device=device, dtype=torch.float32).flatten()
    else:
        r = torch.full((batch,), float(radius_px), device=device, dtype=torch.float32)
    if r.numel() == 1 and batch > 1:
        r = r.expand(batch)
    if r.numel() != batch:
        raise ValueError(f"radius batch={r.numel()} does not match mask batch={batch}")
    return r


def image_space_outer_ring(mask_image: torch.Tensor, radius_px: float | torch.Tensor) -> torch.Tensor:
    """Compute an image-space Euclidean outer ring before pooling to latent.

    Returns [B,T,H,W], 1 outside the hole and within radius_px of the hole.
    """
    m = _as_bthw(mask_image)
    radii = _radius_for_batch(radius_px, m.shape[0], m.device)
    out = []
    frames = m.detach().cpu().numpy()
    radii_np = radii.detach().cpu().numpy()
    for b in range(m.shape[0]):
        for frame in frames[b]:
            radius = float(radii_np[b])
            if radius <= 0:
                out.append(np.zeros(frame.shape, dtype=np.float32))
                continue
            dist = _distance_transform_outside(frame)
            ring = ((frame < 0.5) & (dist > 0.0) & (dist <= radius)).astype(np.float32)
            out.append(ring)
    arr = np.stack(out, axis=0).reshape(m.shape)
    return torch.from_numpy(arr).to(device=m.device, dtype=m.dtype)


class DistanceCache:
    """Persistent image-space outside-distance cache for Exp20 masks."""

    version = "exp20_distance_cache_v1"

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hit_count = 0
        self.miss_count = 0
        self.build_seconds = 0.0

    @staticmethod
    def mask_hash(mask_np: np.ndarray) -> str:
        return hashlib.sha256(mask_np.astype(np.uint8).tobytes()).hexdigest()

    def key(self, mask_np: np.ndarray, identity: str = "") -> str:
        payload = {
            "version": self.version,
            "identity": identity,
            "shape": list(mask_np.shape),
            "mask_hash": self.mask_hash(mask_np),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get(self, mask_np: np.ndarray, identity: str = "") -> np.ndarray:
        key = self.key(mask_np, identity)
        path = self.cache_dir / f"{key}.npz"
        if path.exists():
            self.hit_count += 1
            return np.load(path)["distance"].astype(np.float32)
        t0 = time.time()
        distance = np.stack([_distance_transform_outside(frame) for frame in mask_np], axis=0).astype(np.float32)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, distance=distance, key=key, version=self.version)
        tmp.replace(path)
        self.miss_count += 1
        self.build_seconds += time.time() - t0
        return distance

    def stats(self) -> Dict[str, float]:
        total = self.hit_count + self.miss_count
        return {
            "cache_hits": float(self.hit_count),
            "cache_misses": float(self.miss_count),
            "cache_hit_ratio": float(self.hit_count / total) if total else 0.0,
            "cache_miss_ratio": float(self.miss_count / total) if total else 0.0,
            "cache_build_seconds": float(self.build_seconds),
        }


def image_space_outer_ring_cached(
    mask_image: torch.Tensor,
    radius_px: float | torch.Tensor,
    cache: DistanceCache,
    identities: Iterable[str] | None = None,
) -> torch.Tensor:
    m = _as_bthw(mask_image)
    radii = _radius_for_batch(radius_px, m.shape[0], m.device).detach().cpu().numpy()
    masks_np = m.detach().cpu().numpy().astype(np.uint8)
    if identities is None:
        identity_list = [f"clip_{i}" for i in range(m.shape[0])]
    else:
        identity_list = list(identities)
    if len(identity_list) != m.shape[0]:
        raise ValueError(f"cache identities={len(identity_list)} does not match mask batch={m.shape[0]}")
    out = []
    for b, identity in enumerate(identity_list):
        dist = cache.get(masks_np[b], identity=identity)
        radius = float(radii[b])
        ring = ((masks_np[b] < 0.5) & (dist > 0.0) & (dist <= radius)).astype(np.float32)
        out.append(ring)
    arr = np.stack(out, axis=0)
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


def adaptive_radius_per_clip(
    mask_image: torch.Tensor,
    mode: str,
    k: float,
    r_min: float = 2.0,
    r_max: float = 48.0,
) -> BatchRadiusStats:
    """Compute one stable median radius per clip in a [B,T,H,W] batch."""
    m = _as_bthw(mask_image)
    radii = []
    empty = []
    area_meds = []
    perim_meds = []
    sqrt_meds = []
    clamp_min = []
    clamp_max = []
    for clip in m:
        areas = []
        perimeters = []
        sqrt_areas = []
        for frame in clip:
            area, perimeter = frame_area_perimeter(frame)
            if area <= 0:
                continue
            areas.append(area)
            perimeters.append(max(perimeter, EPS))
            sqrt_areas.append(float(np.sqrt(area / np.pi)))
        if not areas:
            raw = r_min
            area_med = perim_med = sqrt_med = 0.0
            is_empty = True
        else:
            area_med = float(np.median(areas))
            perim_med = float(np.median(perimeters))
            sqrt_med = float(np.median(sqrt_areas))
            if mode == "adaptive_area_perimeter":
                raw = float(k) * area_med / max(perim_med, EPS)
            elif mode == "adaptive_sqrt_area":
                raw = float(k) * sqrt_med
            else:
                raise ValueError(f"Unsupported adaptive radius mode: {mode}")
            is_empty = False
        radius = float(np.clip(raw, r_min, r_max))
        radii.append(radius)
        empty.append(float(is_empty))
        area_meds.append(area_med)
        perim_meds.append(perim_med)
        sqrt_meds.append(sqrt_med)
        clamp_min.append(float(radius <= r_min + EPS))
        clamp_max.append(float(radius >= r_max - EPS))
    device = m.device
    return BatchRadiusStats(
        radius_px=torch.tensor(radii, device=device, dtype=torch.float32),
        empty_clip=torch.tensor(empty, device=device, dtype=torch.float32),
        area_median=torch.tensor(area_meds, device=device, dtype=torch.float32),
        perimeter_median=torch.tensor(perim_meds, device=device, dtype=torch.float32),
        sqrt_area_median=torch.tensor(sqrt_meds, device=device, dtype=torch.float32),
        clamp_min=torch.tensor(clamp_min, device=device, dtype=torch.float32),
        clamp_max=torch.tensor(clamp_max, device=device, dtype=torch.float32),
    )


def build_region_maps(
    mask_image: torch.Tensor,
    loss_hw: Tuple[int, int],
    radius_mode: str = "legacy_latent_exact",
    radius_value: float = 0.0,
    adaptive_k: float = 1.0,
    boundary_mode: str = "outer",
    distance_cache: DistanceCache | None = None,
    cache_identities: Iterable[str] | None = None,
) -> Dict[str, torch.Tensor | RadiusStats]:
    """Build mask, boundary, and outside maps at loss resolution."""
    mask_loss_nearest = nearest_to_shape(mask_image, loss_hw)
    if radius_mode == "legacy_latent_exact":
        boundary = legacy_latent_boundary(mask_loss_nearest, boundary_mode)
        mask_loss = mask_loss_nearest
        outside = torch.clamp(1.0 - torch.clamp(mask_loss + boundary, 0.0, 1.0), 0.0, 1.0)
        radius_stats = RadiusStats(1.0, 0.0, 0.0, 0.0, False, False)
    else:
        if radius_mode == "fixed_image_px":
            radius = float(radius_value)
            radius_stats = RadiusStats(radius, 0.0, 0.0, 0.0, False, False)
        else:
            radius_stats = adaptive_radius_per_clip(mask_image, radius_mode, adaptive_k)
            radius = radius_stats.radius_px
        if distance_cache is None:
            boundary_img = image_space_outer_ring(mask_image, radius)
        else:
            boundary_img = image_space_outer_ring_cached(mask_image, radius, distance_cache, cache_identities)
        mask_img = _as_bthw(mask_image)
        outside_img = torch.clamp(1.0 - mask_img - boundary_img, 0.0, 1.0)
        mask_loss = area_pool_to_shape(mask_img, loss_hw).clamp_min(0.0)
        boundary = area_pool_to_shape(boundary_img, loss_hw).clamp_min(0.0)
        outside = area_pool_to_shape(outside_img, loss_hw).clamp_min(0.0)
        part_sum = (mask_loss + boundary + outside).clamp_min(EPS)
        mask_loss = mask_loss / part_sum
        boundary = boundary / part_sum
        outside = outside / part_sum
    return {
        "mask": mask_loss,
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
    elif isinstance(radius, BatchRadiusStats):
        stats.update(
            {
                "radius_px_mean": float(radius.radius_px.mean().item()),
                "radius_px_p10": float(torch.quantile(radius.radius_px, 0.10).item()),
                "radius_px_p50": float(torch.quantile(radius.radius_px, 0.50).item()),
                "radius_px_p90": float(torch.quantile(radius.radius_px, 0.90).item()),
                "radius_empty_clip_ratio": float(radius.empty_clip.mean().item()),
                "radius_clamp_min_ratio": float(radius.clamp_min.mean().item()),
                "radius_clamp_max_ratio": float(radius.clamp_max.mean().item()),
            }
        )
    stats["pixel_count"] = total
    return stats
