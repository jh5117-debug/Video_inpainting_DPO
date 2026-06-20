"""Region aggregation functions for Exp23 morphology search."""

from __future__ import annotations

from typing import Dict, Mapping

import torch


EPS = 1e-8


DEFAULT_WEIGHTS = {
    "mask_core": 1.0,
    "inner_ring": 0.0,
    "outer_ring": 0.75,
    "far_outside": 0.05,
}


def legacy_weighted_mean(
    error_map: torch.Tensor,
    regions: Mapping[str, torch.Tensor],
    weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    weight_map = torch.zeros_like(error_map, dtype=error_map.dtype)
    for name, region in regions.items():
        weight_map = weight_map + float(weights.get(name, 0.0)) * region.to(error_map)
    return (weight_map * error_map).sum(dim=(-2, -1)) / weight_map.sum(dim=(-2, -1)).clamp_min(EPS)


def region_balanced_mean(
    error_map: torch.Tensor,
    regions: Mapping[str, torch.Tensor],
    alphas: Mapping[str, float] | None = None,
) -> torch.Tensor:
    alphas = dict(DEFAULT_WEIGHTS if alphas is None else alphas)
    total = torch.zeros(error_map.shape[:-2], device=error_map.device, dtype=error_map.dtype)
    alpha_total = torch.zeros_like(total)
    for name, region in regions.items():
        alpha = float(alphas.get(name, 0.0))
        if alpha <= 0:
            continue
        r = region.to(error_map)
        denom = r.sum(dim=(-2, -1))
        valid = denom > EPS
        mean = (r * error_map).sum(dim=(-2, -1)) / denom.clamp_min(EPS)
        total = total + torch.where(valid, alpha * mean, torch.zeros_like(mean))
        alpha_total = alpha_total + torch.where(valid, torch.full_like(mean, alpha), torch.zeros_like(mean))
    return total / alpha_total.clamp_min(EPS)


def effective_alpha(
    regions: Mapping[str, torch.Tensor],
    alphas: Mapping[str, float],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, region in regions.items():
        alpha = float(alphas.get(name, 0.0))
        area = float(region.detach().float().sum().cpu())
        out[name] = alpha if alpha > 0 and area > EPS else 0.0
    return out

