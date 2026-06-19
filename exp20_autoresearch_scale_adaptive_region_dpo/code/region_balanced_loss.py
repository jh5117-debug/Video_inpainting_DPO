"""Region-balanced DPO loss for Exp20.

The functions here operate on already-computed per-pixel error maps and region
maps. They are designed for parity tests first, then imported by the isolated
Exp20 trainers after parity is established.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import torch
import torch.nn.functional as F


EPS = 1e-8


@dataclass(frozen=True)
class DPOHyperParams:
    beta: float = 10.0
    lose_gap_weight: float = 0.25
    lose_gap_clip_tau: float = 1.0
    winner_abs_reg_weight: float = 0.05
    winner_gap_reg_weight: float = 1.0
    winner_gap_reg_margin: float = 0.0


def _reduce_dims(x: torch.Tensor) -> Tuple[int, ...]:
    return tuple(range(1, x.ndim))


def weighted_mean(error: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(device=error.device, dtype=error.dtype)
    while weight.ndim < error.ndim:
        weight = weight.unsqueeze(1)
    num = (error * weight).sum(dim=_reduce_dims(error))
    den = weight.sum(dim=_reduce_dims(weight)).clamp_min(EPS)
    return num / den


def region_means(error: torch.Tensor, region_maps: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for name in ("mask", "boundary", "outside"):
        w = region_maps[name]
        out[name] = weighted_mean(error, w)
    return out


def legacy_global_region_loss(
    error: torch.Tensor,
    region_maps: Mapping[str, torch.Tensor],
    mask_weight: float = 1.0,
    boundary_weight: float = 0.75,
    outside_weight: float = 0.05,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    mask = region_maps["mask"].to(device=error.device, dtype=error.dtype)
    boundary = region_maps["boundary"].to(device=error.device, dtype=error.dtype)
    outside = region_maps["outside"].to(device=error.device, dtype=error.dtype)
    weight = outside_weight * outside + mask_weight * mask + boundary_weight * boundary
    loss = weighted_mean(error, weight)
    stats = region_means(error, region_maps)
    stats["global_weighted"] = loss
    stats["region_weight_sum"] = weight.sum(dim=_reduce_dims(weight))
    return loss, stats


def _valid_alpha(region_maps: Mapping[str, torch.Tensor], alphas: Mapping[str, float], dtype, device) -> Dict[str, torch.Tensor]:
    raw = {}
    for name in ("mask", "boundary", "outside"):
        w = region_maps[name].to(device=device)
        valid = (w.sum(dim=_reduce_dims(w)) > EPS).to(dtype=dtype)
        raw[name] = valid * float(alphas.get(name, 0.0))
    total = sum(raw.values()).clamp_min(EPS)
    return {name: value / total for name, value in raw.items()}


def region_balanced_gap(
    policy_error: torch.Tensor,
    ref_error: torch.Tensor,
    region_maps: Mapping[str, torch.Tensor],
    alphas: Mapping[str, float],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute alpha-normalized sum of per-region log-ratios."""
    policy_means = region_means(policy_error, region_maps)
    ref_means = region_means(ref_error, region_maps)
    valid_alpha = _valid_alpha(region_maps, alphas, policy_error.dtype, policy_error.device)
    gap = torch.zeros_like(next(iter(policy_means.values())))
    stats: Dict[str, torch.Tensor] = {}
    for name in ("mask", "boundary", "outside"):
        g = torch.log((policy_means[name] + EPS) / (ref_means[name] + EPS))
        gap = gap + valid_alpha[name] * g
        stats[f"{name}_policy_mse"] = policy_means[name]
        stats[f"{name}_ref_mse"] = ref_means[name]
        stats[f"{name}_gap"] = g
        stats[f"{name}_alpha_effective"] = valid_alpha[name]
    return gap, stats


def exp11_dpo_from_losses(
    m_w: torch.Tensor,
    m_l: torch.Tensor,
    m_w_ref: torch.Tensor,
    m_l_ref: torch.Tensor,
    hparams: DPOHyperParams = DPOHyperParams(),
    nframes: int | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    raw_win_gap = m_w - m_w_ref
    raw_lose_gap = m_l - m_l_ref
    g_w = torch.log((m_w.clamp(min=0) + EPS) / (m_w_ref.clamp(min=0) + EPS))
    g_l = torch.log((m_l.clamp(min=0) + EPS) / (m_l_ref.clamp(min=0) + EPS))
    g_l_clip = torch.clamp(g_l, max=float(hparams.lose_gap_clip_tau))
    frame_inside_term = -0.5 * float(hparams.beta) * (g_w - float(hparams.lose_gap_weight) * g_l_clip)
    dpo_loss = (-F.logsigmoid(frame_inside_term)).mean()
    winner_abs_reg = m_w.mean()
    winner_gap_reg = F.relu(g_w - float(hparams.winner_gap_reg_margin)).mean()
    total = (
        dpo_loss
        + float(hparams.winner_abs_reg_weight) * winner_abs_reg
        + float(hparams.winner_gap_reg_weight) * winner_gap_reg
    )
    if nframes is None:
        nframes = int(m_w.numel())
    if nframes <= 0 or m_w.numel() % int(nframes) != 0:
        raise ValueError(f"nframes={nframes} must divide pair loss count {m_w.numel()}")
    nframes = int(nframes)
    pair_m_w = m_w.view(-1, nframes).mean(dim=1)
    pair_m_l = m_l.view(-1, nframes).mean(dim=1)
    pair_m_w_ref = m_w_ref.view(-1, nframes).mean(dim=1)
    pair_m_l_ref = m_l_ref.view(-1, nframes).mean(dim=1)
    pair_raw_win_gap = raw_win_gap.view(-1, nframes).mean(dim=1)
    pair_raw_lose_gap = raw_lose_gap.view(-1, nframes).mean(dim=1)
    pair_norm_win_gap = g_w.view(-1, nframes).mean(dim=1)
    pair_norm_lose_gap = g_l.view(-1, nframes).mean(dim=1)
    pair_norm_lose_gap_clipped = g_l_clip.view(-1, nframes).mean(dim=1)
    inside_term = -0.5 * float(hparams.beta) * (
        pair_norm_win_gap - float(hparams.lose_gap_weight) * pair_norm_lose_gap_clipped
    )
    with torch.no_grad():
        correct_mask = inside_term > 0
        winner_improvement = (-pair_norm_win_gap).clamp(min=0)
        loser_degradation = pair_norm_lose_gap_clipped.clamp(min=0)
        loser_dominant_mask = loser_degradation > winner_improvement
        loser_dominant_wins = (correct_mask & loser_dominant_mask).sum().float()
        n_correct = correct_mask.sum().float()
        loser_degrade_ratio = loser_dominant_wins / n_correct.clamp(min=1)

    diagnostics = {
        "loss": total.detach(),
        "dpo_loss": dpo_loss.detach(),
        "m_w": m_w.detach().mean(),
        "m_l": m_l.detach().mean(),
        "m_w_ref": m_w_ref.detach().mean(),
        "m_l_ref": m_l_ref.detach().mean(),
        "norm_win_gap": g_w.detach().mean(),
        "norm_lose_gap": g_l.detach().mean(),
        "norm_lose_gap_clipped": g_l_clip.detach().mean(),
        "winner_abs_reg": winner_abs_reg.detach(),
        "winner_gap_reg": winner_gap_reg.detach(),
        "raw_win_gap": raw_win_gap.detach().mean(),
        "raw_lose_gap": raw_lose_gap.detach().mean(),
        "implicit_acc": (inside_term.detach() > 0).float().mean(),
        "loser_degrade_ratio": loser_degrade_ratio.detach(),
        "loser_degrade_count": loser_dominant_wins.detach(),
        "n_correct": n_correct.detach(),
        "n_total": torch.tensor(float(inside_term.numel()), device=m_w.device),
        "winner_improvement_mean": winner_improvement.detach().mean(),
        "loser_degradation_mean": loser_degradation.detach().mean(),
        "loser_dominant_ratio": loser_degrade_ratio.detach(),
        "mse_w_over_ref_mse_w": ((m_w + EPS) / (m_w_ref + EPS)).detach().mean(),
        "mse_l_over_ref_mse_l": ((m_l + EPS) / (m_l_ref + EPS)).detach().mean(),
        "_inside_term": inside_term.detach(),
        "_winner_improvement": winner_improvement.detach(),
        "_loser_degradation": loser_degradation.detach(),
        "_pair_raw_win_gap": pair_raw_win_gap.detach(),
        "_pair_raw_lose_gap": pair_raw_lose_gap.detach(),
        "_pair_norm_win_gap": pair_norm_win_gap.detach(),
        "_pair_norm_lose_gap": pair_norm_lose_gap.detach(),
        "_pair_norm_lose_gap_clipped": pair_norm_lose_gap_clipped.detach(),
        "_pair_m_w": pair_m_w.detach(),
        "_pair_m_l": pair_m_l.detach(),
        "_pair_m_w_ref": pair_m_w_ref.detach(),
        "_pair_m_l_ref": pair_m_l_ref.detach(),
    }
    return total, diagnostics


def compute_dpo_loss(
    err_w: torch.Tensor,
    err_l: torch.Tensor,
    err_w_ref: torch.Tensor,
    err_l_ref: torch.Tensor,
    region_maps: Mapping[str, torch.Tensor],
    aggregation: str = "legacy_global_weighted_mean",
    hparams: DPOHyperParams = DPOHyperParams(),
    mask_weight: float = 1.0,
    boundary_weight: float = 0.75,
    outside_weight: float = 0.05,
    region_alphas: Mapping[str, float] | None = None,
    nframes: int | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute Exp20 DPO with either legacy global or region-balanced aggregation."""
    if aggregation == "legacy_global_weighted_mean":
        m_w, stats_w = legacy_global_region_loss(err_w, region_maps, mask_weight, boundary_weight, outside_weight)
        m_l, stats_l = legacy_global_region_loss(err_l, region_maps, mask_weight, boundary_weight, outside_weight)
        m_w_ref, stats_w_ref = legacy_global_region_loss(err_w_ref, region_maps, mask_weight, boundary_weight, outside_weight)
        m_l_ref, stats_l_ref = legacy_global_region_loss(err_l_ref, region_maps, mask_weight, boundary_weight, outside_weight)
        loss, diag = exp11_dpo_from_losses(m_w, m_l, m_w_ref, m_l_ref, hparams, nframes=nframes)
    elif aggregation == "region_balanced":
        alphas = region_alphas or {"mask": 1.0, "boundary": boundary_weight, "outside": outside_weight}
        g_w, stats_w = region_balanced_gap(err_w, err_w_ref, region_maps, alphas)
        g_l, stats_l = region_balanced_gap(err_l, err_l_ref, region_maps, alphas)
        g_l_clip = torch.clamp(g_l, max=float(hparams.lose_gap_clip_tau))
        inside_term = -0.5 * float(hparams.beta) * (g_w - float(hparams.lose_gap_weight) * g_l_clip)
        dpo_loss = (-F.logsigmoid(inside_term)).mean()
        m_w, stats_w_global = legacy_global_region_loss(err_w, region_maps, mask_weight, boundary_weight, outside_weight)
        winner_abs_reg = m_w.mean()
        winner_gap_reg = F.relu(g_w - float(hparams.winner_gap_reg_margin)).mean()
        loss = (
            dpo_loss
            + float(hparams.winner_abs_reg_weight) * winner_abs_reg
            + float(hparams.winner_gap_reg_weight) * winner_gap_reg
        )
        diag = {
            "loss": loss.detach(),
            "dpo_loss": dpo_loss.detach(),
            "m_w": m_w.detach().mean(),
            "norm_win_gap": g_w.detach().mean(),
            "norm_lose_gap": g_l.detach().mean(),
            "norm_lose_gap_clipped": g_l_clip.detach().mean(),
            "winner_abs_reg": winner_abs_reg.detach(),
            "winner_gap_reg": winner_gap_reg.detach(),
            "implicit_acc": (inside_term.detach() > 0).float().mean(),
        }
        stats_w.update({f"winner_{k}": v for k, v in stats_w_global.items()})
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    for prefix, stats in (("winner", stats_w), ("loser", stats_l if "stats_l" in locals() else {})):
        for key, value in stats.items():
            if torch.is_tensor(value):
                diag[f"{prefix}_{key}"] = value.detach().mean()
    diag["aggregation"] = aggregation
    diag["boundary_weight"] = torch.tensor(float(boundary_weight), device=err_w.device)
    diag["outside_weight"] = torch.tensor(float(outside_weight), device=err_w.device)
    return loss, diag
