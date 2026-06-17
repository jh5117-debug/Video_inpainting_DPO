#!/usr/bin/env python3
"""Loss helpers for Exp18 Multi-frame Propagation-Confidence Gated DPO.

The helpers in this file are intentionally isolated under Exp18. They do not
modify the shared training code or older experiment folders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PropagationConfidenceConfig:
    confidence_mode: str = "flow_agreement"
    tau_conf: float = 0.5
    mask_weight: float = 1.0
    boundary_weight: float = 0.75
    outside_weight: float = 0.05
    lambda_prop: float = 0.1
    lambda_gen: float = 0.05
    lambda_boundary_extra: float = 0.1
    eps: float = 1e-6


def brushnet_known_to_hole_mask(brushnet_masks: torch.Tensor) -> torch.Tensor:
    """Convert DiffuEraser BrushNet masks to a hole mask.

    Existing DiffuEraser DPO convention is `0 = hole / unknown`, `1 = known`.
    Exp18 losses use `1 = hole`.
    """

    if brushnet_masks.ndim not in {4, 5}:
        raise ValueError(f"expected [B,F,1,H,W] or [N,1,H,W], got {tuple(brushnet_masks.shape)}")
    return (1.0 - brushnet_masks.float()).clamp(0.0, 1.0)


def flatten_video_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 5:
        b, f, c, h, w = mask.shape
        if c != 1:
            raise ValueError(f"expected one mask channel, got {tuple(mask.shape)}")
        return mask.reshape(b * f, c, h, w)
    if mask.ndim == 4:
        return mask
    raise ValueError(f"expected [B,F,1,H,W] or [N,1,H,W], got {tuple(mask.shape)}")


def resize_mask_nearest(mask: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(flatten_video_mask(mask).float(), size=size, mode="nearest")


def resize_confidence(confidence: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(flatten_video_mask(confidence).float(), size=size, mode="bilinear", align_corners=False).clamp(0.0, 1.0)


def boundary_outer_from_hole(hole_mask: torch.Tensor) -> torch.Tensor:
    hole4 = flatten_video_mask(hole_mask)
    binary = (hole4 > 0.5).float()
    dilated = F.max_pool2d(binary, kernel_size=3, stride=1, padding=1)
    return (dilated - binary).clamp(0.0, 1.0)


def summarize_propagation_confidence(
    propagation_confidence: torch.Tensor,
    hole_mask: torch.Tensor,
    tau_conf: float = 0.5,
    avg_num_sources_used: float | None = None,
    propagated_region_psnr: float | None = None,
) -> Dict[str, float | str | None]:
    """Summarize confidence maps without using GT-error as confidence."""

    conf = propagation_confidence.float().clamp(0.0, 1.0)
    hole = hole_mask.float()
    if hole.ndim == 4 and conf.ndim == 5:
        b, f = conf.shape[:2]
        hole = hole.reshape(b, f, 1, hole.shape[-2], hole.shape[-1])
    if conf.shape[-2:] != hole.shape[-2:]:
        hole = F.interpolate(flatten_video_mask(hole), size=conf.shape[-2:], mode="nearest")
        if conf.ndim == 5:
            hole = hole.reshape(conf.shape[0], conf.shape[1], 1, conf.shape[-2], conf.shape[-1])
    valid = hole > 0.5
    conf_inside = conf[valid.expand_as(conf)] if valid.any() else conf.reshape(-1)
    hard = ((conf > float(tau_conf)) & valid).float()
    hole_sum = valid.float().sum().clamp(min=1.0)
    return {
        "prop_conf_mean": float(conf_inside.mean().detach().cpu()),
        "prop_conf_p10": float(torch.quantile(conf_inside.float(), 0.10).detach().cpu()),
        "prop_conf_p50": float(torch.quantile(conf_inside.float(), 0.50).detach().cpu()),
        "prop_conf_p90": float(torch.quantile(conf_inside.float(), 0.90).detach().cpu()),
        "propagation_coverage": float(hard.sum().detach().cpu() / hole_sum.detach().cpu()),
        "generate_area_ratio": float(((valid.float() * (1.0 - hard)).sum() / hole_sum).detach().cpu()),
        "avg_num_sources_used": avg_num_sources_used,
        "propagated_region_psnr": propagated_region_psnr,
    }


def weighted_l1(value: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if value.shape != target.shape:
        raise ValueError(f"value/target mismatch: {tuple(value.shape)} vs {tuple(target.shape)}")
    w = weight.to(device=value.device, dtype=value.dtype)
    if w.shape[-2:] != value.shape[-2:]:
        w = F.interpolate(w, size=value.shape[-2:], mode="bilinear", align_corners=False)
    if w.shape[1] == 1 and value.shape[1] != 1:
        w = w.expand(-1, value.shape[1], -1, -1)
    numer = ((value.float() - target.float()).abs() * w.float()).sum()
    denom = w.float().sum().clamp(min=float(eps))
    return numer / denom


def predict_x0_from_model_output(
    noisy_latents: torch.Tensor,
    model_output: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler,
) -> torch.Tensor:
    """Recover predicted clean latent x0 from a Diffusers DDPM scheduler."""

    prediction_type = str(getattr(scheduler.config, "prediction_type", "epsilon"))
    if prediction_type == "sample":
        return model_output
    if not hasattr(scheduler, "alphas_cumprod"):
        raise RuntimeError("scheduler has no alphas_cumprod; cannot reconstruct x0 safely")
    alphas_cumprod = scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    t = timesteps.to(device=noisy_latents.device).long()
    alpha_prod_t = alphas_cumprod[t].reshape(-1, 1, 1, 1)
    beta_prod_t = (1.0 - alpha_prod_t).clamp(min=0.0)
    sqrt_alpha = alpha_prod_t.sqrt()
    sqrt_beta = beta_prod_t.sqrt()
    if prediction_type == "epsilon":
        return (noisy_latents - sqrt_beta * model_output) / sqrt_alpha.clamp(min=1e-8)
    if prediction_type == "v_prediction":
        return sqrt_alpha * noisy_latents - sqrt_beta * model_output
    raise RuntimeError(f"unsupported scheduler prediction_type={prediction_type!r}; Exp18 cannot use a proxy")


def compute_propagation_gated_losses(
    z_hat_x0: torch.Tensor,
    z_prop: torch.Tensor,
    z_gt: torch.Tensor,
    brushnet_masks: torch.Tensor,
    propagation_confidence: torch.Tensor,
    cfg: PropagationConfidenceConfig,
) -> tuple[torch.Tensor, Dict[str, float | str | None]]:
    """Compute L_prop/L_gen/L_boundary on real latent targets.

    `z_prop` must come from multi-frame propagated pixels. This function does
    not know how to create propagation targets, and therefore cannot silently
    fall back to ProPainter, generated loser, or frozen-reference proxies.
    """

    latent_hw = z_hat_x0.shape[-2:]
    hole = resize_mask_nearest(brushnet_known_to_hole_mask(brushnet_masks), latent_hw)
    conf = resize_confidence(propagation_confidence, latent_hw)
    boundary = boundary_outer_from_hole(hole)
    prop_soft = (hole * conf).clamp(0.0, 1.0)
    prop_hard = (hole * (conf > float(cfg.tau_conf)).float()).clamp(0.0, 1.0)
    gen = (hole * (1.0 - (conf > float(cfg.tau_conf)).float())).clamp(0.0, 1.0)
    hole_sum = hole.float().sum().clamp(min=float(cfg.eps))

    l_prop = weighted_l1(z_hat_x0, z_prop, prop_soft, cfg.eps)
    l_gen = weighted_l1(z_hat_x0, z_gt, gen, cfg.eps)
    l_boundary = weighted_l1(z_hat_x0, z_gt, boundary, cfg.eps)
    extra = cfg.lambda_prop * l_prop + cfg.lambda_gen * l_gen + cfg.lambda_boundary_extra * l_boundary

    conf_inside = conf[hole > 0.5] if (hole > 0.5).any() else conf.reshape(-1)
    diagnostics: Dict[str, float | str | None] = {
        "L_prop": float(l_prop.detach().cpu()),
        "L_gen": float(l_gen.detach().cpu()),
        "L_boundary": float(l_boundary.detach().cpu()),
        "lambda_prop": cfg.lambda_prop,
        "lambda_gen": cfg.lambda_gen,
        "lambda_boundary_extra": cfg.lambda_boundary_extra,
        "propagation_coverage": float(prop_hard.float().sum().detach().cpu() / hole_sum.detach().cpu()),
        "generate_area_ratio": float(gen.float().sum().detach().cpu() / hole_sum.detach().cpu()),
        "prop_conf_mean": float(conf_inside.float().mean().detach().cpu()),
        "prop_conf_p10": float(torch.quantile(conf_inside.float(), 0.10).detach().cpu()),
        "prop_conf_p50": float(torch.quantile(conf_inside.float(), 0.50).detach().cpu()),
        "prop_conf_p90": float(torch.quantile(conf_inside.float(), 0.90).detach().cpu()),
        "boundary_area_ratio": float((boundary > 0.5).float().mean().detach().cpu()),
        "confidence_mode": cfg.confidence_mode,
        "tau_conf": cfg.tau_conf,
        "propagation_target_mode": "latent_x0",
    }
    return extra, diagnostics

