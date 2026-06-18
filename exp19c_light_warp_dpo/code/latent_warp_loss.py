#!/usr/bin/env python3
"""Confidence-gated latent warp consistency for Exp19c.

This module never uses RGB warp targets or GT-derived flow confidence. It only
regularizes the model's predicted clean latents across adjacent frames in
regions where the completed-flow confidence and Exp19 boundary gate are active.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def predict_x0_latent(noisy_latents: torch.Tensor, model_pred: torch.Tensor, timesteps: torch.Tensor, scheduler: Any) -> torch.Tensor:
    """Recover predicted clean latent from a DDPM scheduler prediction."""
    alphas = scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    alpha_t = alphas[timesteps].view(-1, 1, 1, 1)
    sqrt_alpha = alpha_t.sqrt()
    sqrt_beta = (1.0 - alpha_t).sqrt()
    prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if prediction_type == "epsilon":
        return (noisy_latents - sqrt_beta * model_pred) / sqrt_alpha.clamp_min(1e-8)
    if prediction_type == "v_prediction":
        return sqrt_alpha * noisy_latents - sqrt_beta * model_pred
    if prediction_type == "sample":
        return model_pred
    raise ValueError(f"Unsupported scheduler prediction_type for Exp19c x0 recovery: {prediction_type}")


def _resize_flow(flow: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Resize pixel-space flow and scale displacement values."""
    b, t, c, h, w = flow.shape
    target_h, target_w = target_hw
    flat = flow.reshape(b * t, c, h, w)
    out = F.interpolate(flat, size=(target_h, target_w), mode="bilinear", align_corners=False)
    out[:, 0] *= float(target_w) / max(float(w), 1.0)
    out[:, 1] *= float(target_h) / max(float(h), 1.0)
    return out.reshape(b, t, c, target_h, target_w)


def _resize_mask(mask: torch.Tensor, target_hw: tuple[int, int], *, mode: str) -> torch.Tensor:
    b, t, c, h, w = mask.shape
    flat = mask.reshape(b * t, c, h, w)
    if mode == "nearest":
        out = F.interpolate(flat, size=target_hw, mode="nearest")
    else:
        out = F.interpolate(flat, size=target_hw, mode="bilinear", align_corners=False)
    return out.reshape(b, t, c, target_hw[0], target_hw[1])


def _pull_warp(src: torch.Tensor, flow_tgt_to_src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``src`` at target pixels using target->source flow.

    Args:
        src: [N,C,H,W] source feature map.
        flow_tgt_to_src: [N,2,H,W] pixel displacement from target to source.

    Returns:
        warped source and valid in-bound mask.
    """
    n, _c, h, w = src.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=src.device, dtype=src.dtype),
        torch.arange(w, device=src.device, dtype=src.dtype),
        indexing="ij",
    )
    base = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(n, h, w, 2)
    coords = base + flow_tgt_to_src.permute(0, 2, 3, 1)
    valid = (
        (coords[..., 0] >= 0)
        & (coords[..., 0] <= max(w - 1, 1))
        & (coords[..., 1] >= 0)
        & (coords[..., 1] <= max(h - 1, 1))
    )
    norm_x = 2.0 * coords[..., 0] / max(float(w - 1), 1.0) - 1.0
    norm_y = 2.0 * coords[..., 1] / max(float(h - 1), 1.0) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1)
    warped = F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return warped, valid[:, None].to(dtype=src.dtype)


def confidence_gated_latent_warp_loss(
    *,
    z_hat0_flat: torch.Tensor,
    batch: dict[str, torch.Tensor],
    nframes: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute bidirectional latent warp consistency on winner x0 latents."""
    if z_hat0_flat.shape[0] % nframes != 0:
        raise ValueError(f"Cannot reshape z_hat0 {tuple(z_hat0_flat.shape)} with nframes={nframes}")
    b = z_hat0_flat.shape[0] // nframes
    c, h, w = z_hat0_flat.shape[1:]
    z = z_hat0_flat.reshape(b, nframes, c, h, w)
    if nframes < 2:
        zero = z_hat0_flat.new_zeros(())
        return zero, {
            "warp_loss_forward": 0.0,
            "warp_loss_backward": 0.0,
            "warp_valid_ratio": 0.0,
        }

    # Dataset layout: forward[:, t] is t->t+1, backward[:, t+1] is t+1->t.
    fwd = _resize_flow(batch["flow_forward"].float()[:, :-1].to(z.device), (h, w)).to(dtype=z.dtype)
    bwd = _resize_flow(batch["flow_backward"].float()[:, 1:].to(z.device), (h, w)).to(dtype=z.dtype)
    conf = _resize_mask(batch["flow_confidence"].float()[:, :-1].to(z.device), (h, w), mode="bilinear").to(dtype=z.dtype).clamp(0.0, 1.0)
    region = _resize_mask(batch["flow_gate_mask"].float().to(z.device), (h, w), mode="nearest").to(dtype=z.dtype).clamp(0.0, 1.0)
    region_src = region[:, :-1]
    region_tgt = region[:, 1:]

    z_src = z[:, :-1].reshape(b * (nframes - 1), c, h, w)
    z_tgt = z[:, 1:].reshape(b * (nframes - 1), c, h, w)
    fwd_flat = fwd.reshape(b * (nframes - 1), 2, h, w)
    bwd_flat = bwd.reshape(b * (nframes - 1), 2, h, w)
    conf_flat = conf.reshape(b * (nframes - 1), 1, h, w)
    region_src_flat = region_src.reshape(b * (nframes - 1), 1, h, w)
    region_tgt_flat = region_tgt.reshape(b * (nframes - 1), 1, h, w)

    # Pull z_t toward z_{t+1} using backward flow t+1->t.
    warped_fw, valid_fw = _pull_warp(z_src, bwd_flat)
    weight_fw = (conf_flat * region_tgt_flat * valid_fw).clamp(0.0, 1.0)
    loss_fw = (weight_fw * (warped_fw - z_tgt).abs()).sum() / (weight_fw.sum() * c + 1e-8)

    # Pull z_{t+1} toward z_t using forward flow t->t+1.
    warped_bw, valid_bw = _pull_warp(z_tgt, fwd_flat)
    weight_bw = (conf_flat * region_src_flat * valid_bw).clamp(0.0, 1.0)
    loss_bw = (weight_bw * (warped_bw - z_src).abs()).sum() / (weight_bw.sum() * c + 1e-8)

    loss = 0.5 * (loss_fw + loss_bw)
    valid_ratio = 0.5 * (
        float((weight_fw > 1e-6).float().mean().detach().cpu())
        + float((weight_bw > 1e-6).float().mean().detach().cpu())
    )
    stats = {
        "warp_loss_forward": float(loss_fw.detach().float().cpu()),
        "warp_loss_backward": float(loss_bw.detach().float().cpu()),
        "warp_valid_ratio": valid_ratio,
    }
    return loss, stats
