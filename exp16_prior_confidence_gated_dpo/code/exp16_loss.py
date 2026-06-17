#!/usr/bin/env python3
"""Loss helpers for Exp16 Prior-Confidence Gated DPO.

This module is intentionally self-contained under the Exp16 folder. It does not
modify the shared DPO training code. The helpers here only operate on tensors
that the isolated Exp16 trainer/preflight passes in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PriorConfidenceConfig:
    confidence_mode: str = "gt_error"
    confidence_alpha: float = 5.0
    mask_weight: float = 1.0
    boundary_weight: float = 0.75
    outside_weight: float = 0.05
    lambda_prior: float = 0.1
    lambda_gen: float = 0.05
    lambda_boundary_extra: float = 0.1
    eps: float = 1e-6


def brushnet_known_to_hole_mask(brushnet_masks: torch.Tensor) -> torch.Tensor:
    """Convert DiffuEraser BrushNet masks to a hole mask.

    Current project convention for `brushnet_masks` is 0=hole/unknown and
    1=known/outside. Exp16 loss uses `hole=1` inside the inpainting region.
    """

    if brushnet_masks.ndim not in {4, 5}:
        raise ValueError(f"expected mask as [B,F,1,H,W] or [N,1,H,W], got {tuple(brushnet_masks.shape)}")
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
    mask4 = flatten_video_mask(mask)
    return F.interpolate(mask4.float(), size=size, mode="nearest")


def resize_confidence(confidence: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    conf4 = flatten_video_mask(confidence)
    return F.interpolate(conf4.float(), size=size, mode="bilinear", align_corners=False).clamp(0.0, 1.0)


def boundary_outer_from_hole(hole_mask: torch.Tensor) -> torch.Tensor:
    """Outer boundary = dilate(hole) - hole, matching Exp11 best setting."""

    hole4 = flatten_video_mask(hole_mask)
    binary = (hole4 > 0.5).float()
    dilated = F.max_pool2d(binary, kernel_size=3, stride=1, padding=1)
    return (dilated - binary).clamp(0.0, 1.0)


def compute_prior_confidence_from_gt_error(
    prior_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    hole_mask: torch.Tensor,
    alpha: float = 5.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute C_prior = exp(-alpha * normalized L1 prior error).

    Inputs are expected in the same image scale, typically [-1, 1], with shape
    [B,F,3,H,W]. The returned confidence has shape [B,F,1,H,W].
    """

    if prior_rgb.shape != gt_rgb.shape:
        raise ValueError(f"prior and gt shape mismatch: {tuple(prior_rgb.shape)} vs {tuple(gt_rgb.shape)}")
    if prior_rgb.ndim != 5 or prior_rgb.shape[2] != 3:
        raise ValueError(f"expected RGB video tensor [B,F,3,H,W], got {tuple(prior_rgb.shape)}")
    hole = hole_mask.float()
    if hole.ndim == 4:
        b, f = prior_rgb.shape[:2]
        hole = hole.reshape(b, f, 1, hole.shape[-2], hole.shape[-1])
    if hole.shape[:2] != prior_rgb.shape[:2]:
        raise ValueError(f"mask/video batch mismatch: {tuple(hole.shape)} vs {tuple(prior_rgb.shape)}")
    if hole.shape[-2:] != prior_rgb.shape[-2:]:
        hole_flat = flatten_video_mask(hole)
        hole_flat = F.interpolate(hole_flat, size=prior_rgb.shape[-2:], mode="nearest")
        hole = hole_flat.reshape(prior_rgb.shape[0], prior_rgb.shape[1], 1, prior_rgb.shape[-2], prior_rgb.shape[-1])

    err = (prior_rgb.float() - gt_rgb.float()).abs().mean(dim=2, keepdim=True)
    masked_err = err * (hole > 0.5).float()
    valid = (hole > 0.5)
    if valid.any():
        max_err = masked_err[valid].max().clamp(min=eps)
    else:
        max_err = err.max().clamp(min=eps)
    norm_err = (err / max_err).clamp(0.0, 1.0)
    conf = torch.exp(-float(alpha) * norm_err).clamp(0.0, 1.0)

    conf_inside = conf[valid.expand_as(conf)] if valid.any() else conf.reshape(-1)
    stats = {
        "prior_conf_mean": float(conf_inside.mean().detach().cpu()),
        "prior_conf_p10": float(torch.quantile(conf_inside.float(), 0.10).detach().cpu()),
        "prior_conf_p50": float(torch.quantile(conf_inside.float(), 0.50).detach().cpu()),
        "prior_conf_p90": float(torch.quantile(conf_inside.float(), 0.90).detach().cpu()),
        "prior_conf_mean_inside_mask": float(conf_inside.mean().detach().cpu()),
        "prior_conf_std_inside_mask": float(conf_inside.std(unbiased=False).detach().cpu()),
        "prior_conf_p10_inside_mask": float(torch.quantile(conf_inside.float(), 0.10).detach().cpu()),
        "prior_conf_p50_inside_mask": float(torch.quantile(conf_inside.float(), 0.50).detach().cpu()),
        "prior_conf_p90_inside_mask": float(torch.quantile(conf_inside.float(), 0.90).detach().cpu()),
        "mask_area_ratio": float((hole > 0.5).float().mean().detach().cpu()),
        "confidence_alpha": float(alpha),
    }
    return conf, stats


def weighted_l1(value: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if value.shape != target.shape:
        raise ValueError(f"value/target mismatch: {tuple(value.shape)} vs {tuple(target.shape)}")
    w = weight.to(device=value.device, dtype=value.dtype)
    if w.shape[-2:] != value.shape[-2:]:
        w = F.interpolate(w, size=value.shape[-2:], mode="nearest")
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
    """Recover predicted clean latent x0 from a Diffusers DDPM-style scheduler.

    Supports `epsilon`, `v_prediction`, and `sample` prediction types. If the
    scheduler lacks `alphas_cumprod`, this raises instead of silently falling
    back to a proxy target.
    """

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
    raise RuntimeError(f"unsupported scheduler prediction_type={prediction_type!r}; Exp16 cannot fake prior loss")


def compute_prior_gated_losses(
    z_hat_x0: torch.Tensor,
    z_prior: torch.Tensor,
    z_gt: torch.Tensor,
    brushnet_masks: torch.Tensor,
    prior_confidence: torch.Tensor,
    cfg: PriorConfidenceConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute L_prior/L_gen/L_boundary_extra on real latent targets."""

    latent_hw = z_hat_x0.shape[-2:]
    hole = resize_mask_nearest(brushnet_known_to_hole_mask(brushnet_masks), latent_hw)
    conf = resize_confidence(prior_confidence, latent_hw)
    boundary = boundary_outer_from_hole(hole)
    reliable = (hole * conf).clamp(0.0, 1.0)
    generate = (hole * (1.0 - conf)).clamp(0.0, 1.0)
    hole_sum = hole.float().sum().clamp(min=float(cfg.eps))
    conf_inside = conf[hole > 0.5] if (hole > 0.5).any() else conf.reshape(-1)
    reliable_mass = reliable.float().sum() / hole_sum
    generate_mass = generate.float().sum() / hole_sum

    l_prior = weighted_l1(z_hat_x0, z_prior, reliable, cfg.eps)
    l_gen = weighted_l1(z_hat_x0, z_gt, generate, cfg.eps)
    l_boundary = weighted_l1(z_hat_x0, z_gt, boundary, cfg.eps)
    extra = cfg.lambda_prior * l_prior + cfg.lambda_gen * l_gen + cfg.lambda_boundary_extra * l_boundary

    stats = {
        "L_prior": float(l_prior.detach().cpu()),
        "L_gen": float(l_gen.detach().cpu()),
        "L_boundary_extra": float(l_boundary.detach().cpu()),
        "lambda_prior": cfg.lambda_prior,
        "lambda_gen": cfg.lambda_gen,
        "lambda_boundary_extra": cfg.lambda_boundary_extra,
        "reliable_area_ratio": float((reliable > 1e-4).float().mean().detach().cpu()),
        "generate_area_ratio": float((generate > 1e-4).float().mean().detach().cpu()),
        "boundary_area_ratio": float((boundary > 0.5).float().mean().detach().cpu()),
        "prior_conf_mean": float(conf.mean().detach().cpu()),
        "prior_conf_p10": float(torch.quantile(conf.reshape(-1).float(), 0.10).detach().cpu()),
        "prior_conf_p50": float(torch.quantile(conf.reshape(-1).float(), 0.50).detach().cpu()),
        "prior_conf_p90": float(torch.quantile(conf.reshape(-1).float(), 0.90).detach().cpu()),
        "prior_conf_mean_inside_mask": float(conf_inside.float().mean().detach().cpu()),
        "prior_conf_std_inside_mask": float(conf_inside.float().std(unbiased=False).detach().cpu()),
        "prior_conf_p10_inside_mask": float(torch.quantile(conf_inside.float(), 0.10).detach().cpu()),
        "prior_conf_p50_inside_mask": float(torch.quantile(conf_inside.float(), 0.50).detach().cpu()),
        "prior_conf_p90_inside_mask": float(torch.quantile(conf_inside.float(), 0.90).detach().cpu()),
        "reliable_weight_mass": float(reliable_mass.detach().cpu()),
        "generate_weight_mass": float(generate_mass.detach().cpu()),
        "reliable_generate_mass_sum": float((reliable_mass + generate_mass).detach().cpu()),
        "confidence_alpha": float(cfg.confidence_alpha),
        "prior_target_mode": "latent_x0",
        "confidence_mode": cfg.confidence_mode,
    }
    return extra, stats
