from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F


REGIONS = ("object", "overlap", "affected", "boundary", "outside")


@dataclass(frozen=True)
class AdaptiveLossConfig:
    name: str = "ATS0_Q2_T500_S0"
    beta: float = 0.1
    lambda_max: float = 0.25
    winner_anchor_base: float = 0.10
    object_dpo_base: float = 1.0
    affected_dpo_base: float = 0.25
    overlap_dpo_base: float = 0.0
    boundary_dpo_base: float = 0.0
    outside_dpo_base: float = 0.0
    object_pres_base: float = 0.0
    overlap_pres_base: float = 0.20
    affected_pres_base: float = 0.15
    boundary_pres_base: float = 0.20
    outside_pres_base: float = 0.10
    transition_delta_max: float = 2.0e-5
    outside_delta_max: float = 2.0e-5
    backtracking_scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)
    linear_utility: bool = False
    no_preference_dpo: bool = False
    strict: bool = False
    lr_scale: float = 1.0

    @property
    def dpo_scales(self) -> dict[str, float]:
        return {
            "object": self.object_dpo_base,
            "overlap": self.overlap_dpo_base,
            "affected": self.affected_dpo_base,
            "boundary": self.boundary_dpo_base,
            "outside": self.outside_dpo_base,
        }

    @property
    def preservation_scales(self) -> dict[str, float]:
        return {
            "object": self.object_pres_base,
            "overlap": self.overlap_pres_base,
            "affected": self.affected_pres_base,
            "boundary": self.boundary_pres_base,
            "outside": self.outside_pres_base,
        }


def scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu())
    return float(value)


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    while weight.ndim < pred.ndim:
        weight = weight.unsqueeze(2)
    weight = weight.to(device=pred.device, dtype=pred.dtype)
    diff = (pred - target).float()
    return (diff.square() * weight.float()).sum() / weight.float().sum().clamp_min(1.0)


def flatten_grads(grads: Iterable[torch.Tensor | None]) -> torch.Tensor:
    chunks = [g.detach().float().reshape(-1).cpu() for g in grads if g is not None]
    if not chunks:
        return torch.zeros(1, dtype=torch.float32)
    return torch.cat(chunks)


def grad_stats(winner_grad: torch.Tensor, loser_grad: torch.Tensor, lambda_max: float) -> dict[str, float]:
    winner = winner_grad.float().reshape(-1)
    loser = loser_grad.float().reshape(-1)
    dot = float(torch.dot(winner, loser).item())
    wn = float(winner.norm().item())
    ln = float(loser.norm().item())
    cosine = dot / max(wn * ln, 1.0e-12)
    if not torch.isfinite(torch.tensor(cosine)):
        cosine = 0.0
    if dot <= 0.0:
        lam = 0.0
    else:
        lam = lambda_max * max(0.0, min(1.0, cosine))
    return {
        "gradient_dot": dot,
        "gradient_cosine": cosine,
        "winner_grad_norm": wn,
        "loser_grad_norm": ln,
        "lambda_loser_global": lam,
        "lambda_loser_object": lam,
        "lambda_loser_affected": min(lam, lambda_max * 0.5),
    }


def transition_risk_weights(
    region_deltas: Mapping[str, float],
    config: AdaptiveLossConfig,
) -> dict[str, float | bool]:
    overlap_bad = region_deltas.get("overlap", 0.0) > config.transition_delta_max
    affected_bad = region_deltas.get("affected", 0.0) > config.transition_delta_max
    boundary_bad = region_deltas.get("boundary", 0.0) > config.transition_delta_max
    outside_bad = region_deltas.get("outside", 0.0) > config.outside_delta_max
    any_transition_bad = overlap_bad or affected_bad or boundary_bad
    object_scale = config.object_dpo_base
    if any_transition_bad:
        object_scale *= 0.5
    if outside_bad:
        object_scale *= 0.25
    return {
        "transition_safe_pass": not (any_transition_bad or outside_bad),
        "object_dpo_scale_final": object_scale,
        "affected_dpo_scale_final": config.affected_dpo_base * (0.5 if affected_bad else 1.0),
        "overlap_dpo_scale_final": 0.0,
        "boundary_dpo_scale_final": 0.0,
        "overlap_pres_weight_final": config.overlap_pres_base * (2.0 if overlap_bad else 1.0),
        "affected_pres_weight_final": config.affected_pres_base * (2.0 if affected_bad else 1.0),
        "boundary_pres_weight_final": config.boundary_pres_base * (2.0 if boundary_bad else 1.0),
        "outside_pres_weight_final": config.outside_pres_base * (2.0 if outside_bad else 1.0),
    }


def select_backtracking_scale(
    candidate_deltas: list[dict[str, float]],
    config: AdaptiveLossConfig,
) -> dict[str, Any]:
    for idx, deltas in enumerate(candidate_deltas):
        winner_ok = deltas.get("winner", 0.0) <= config.transition_delta_max
        outside_ok = deltas.get("outside", 0.0) <= config.outside_delta_max
        transition_ok = all(deltas.get(k, 0.0) <= config.transition_delta_max for k in ("overlap", "affected", "boundary"))
        if winner_ok and outside_ok and transition_ok:
            return {
                "finite_diff_pass_loser": True,
                "transition_safe_pass": True,
                "finite_diff_attempts": idx + 1,
                "finite_diff_selected_scale": deltas["scale"],
                "update_rejected": False,
                "object_delta_pred": deltas.get("object", 0.0),
                "overlap_delta_pred": deltas.get("overlap", 0.0),
                "affected_delta_pred": deltas.get("affected", 0.0),
                "boundary_delta_pred": deltas.get("boundary", 0.0),
                "outside_delta_pred": deltas.get("outside", 0.0),
                "winner_delta_pred": deltas.get("winner", 0.0),
                "global_update_scale": deltas["scale"],
            }
    last = candidate_deltas[-1] if candidate_deltas else {"scale": 0.0}
    return {
        "finite_diff_pass_loser": False,
        "transition_safe_pass": False,
        "finite_diff_attempts": len(candidate_deltas),
        "finite_diff_selected_scale": 0.0,
        "update_rejected": True,
        "object_delta_pred": last.get("object", 0.0),
        "overlap_delta_pred": last.get("overlap", 0.0),
        "affected_delta_pred": last.get("affected", 0.0),
        "boundary_delta_pred": last.get("boundary", 0.0),
        "outside_delta_pred": last.get("outside", 0.0),
        "winner_delta_pred": last.get("winner", 0.0),
        "global_update_scale": 0.0,
    }


def build_adaptive_loss(
    winner_region_policy: Mapping[str, torch.Tensor],
    winner_region_reference: Mapping[str, torch.Tensor],
    loser_region_policy: Mapping[str, torch.Tensor],
    loser_region_reference: Mapping[str, torch.Tensor],
    config: AdaptiveLossConfig,
    safe: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    safe = safe or {}
    lambda_loser = float(safe.get("lambda_loser_global", config.lambda_max))
    dpo_scales = config.dpo_scales
    pres_scales = config.preservation_scales

    winner_margin = None
    loser_margin = None
    for region in REGIONS:
        w_gap = winner_region_reference[region].detach() - winner_region_policy[region]
        l_gap = loser_region_reference[region].detach() - loser_region_policy[region]
        w_term = dpo_scales[region] * w_gap
        l_term = dpo_scales[region] * l_gap
        winner_margin = w_term if winner_margin is None else winner_margin + w_term
        loser_margin = l_term if loser_margin is None else loser_margin + l_term

    if config.no_preference_dpo:
        preference = torch.zeros_like(winner_margin)
        effective_loser = torch.zeros_like(loser_margin)
    else:
        effective_loser = loser_margin.detach() + lambda_loser * (loser_margin - loser_margin.detach())
        margin = winner_margin - effective_loser
        preference = -config.beta * margin if config.linear_utility else -F.logsigmoid(config.beta * margin)

    winner_anchor = torch.zeros_like(preference)
    for region in REGIONS:
        winner_anchor = winner_anchor + pres_scales[region] * winner_region_policy[region]
    winner_anchor = winner_anchor + config.winner_anchor_base * winner_region_policy["object"]
    loss = preference + winner_anchor
    info = {
        "loss": loss.detach(),
        "preference_loss": preference.detach(),
        "winner_anchor_loss": winner_anchor.detach(),
        "winner_gap": winner_margin.detach(),
        "loser_gap": loser_margin.detach(),
        "effective_loser_gap": effective_loser.detach(),
        "preference_margin": (winner_margin - effective_loser).detach(),
        "lambda_loser_global": lambda_loser,
        "lambda_loser_object": float(safe.get("lambda_loser_object", lambda_loser)),
        "lambda_loser_affected": float(safe.get("lambda_loser_affected", lambda_loser)),
        "object_dpo_scale": dpo_scales["object"],
        "affected_dpo_scale": dpo_scales["affected"],
        "overlap_dpo_scale": dpo_scales["overlap"],
        "boundary_dpo_scale": dpo_scales["boundary"],
        "outside_dpo_scale": dpo_scales["outside"],
        "winner_anchor_final": config.winner_anchor_base,
        "overlap_pres_final": pres_scales["overlap"],
        "affected_pres_final": pres_scales["affected"],
        "boundary_pres_final": pres_scales["boundary"],
        "outside_pres_final": pres_scales["outside"],
    }
    for region in REGIONS:
        info[f"winner_policy_loss_{region}"] = winner_region_policy[region].detach()
        info[f"winner_reference_loss_{region}"] = winner_region_reference[region].detach()
        info[f"loser_policy_loss_{region}"] = loser_region_policy[region].detach()
        info[f"loser_reference_loss_{region}"] = loser_region_reference[region].detach()
    return loss, info


def config_for_cell(cell: str) -> AdaptiveLossConfig:
    if cell == "ATS_STRICT_Q2_T500_S0":
        return AdaptiveLossConfig(
            name=cell,
            lambda_max=0.10,
            winner_anchor_base=0.15,
            object_dpo_base=0.75,
            affected_dpo_base=0.10,
            overlap_pres_base=0.25,
            affected_pres_base=0.20,
            boundary_pres_base=0.25,
            transition_delta_max=1.0e-5,
            outside_delta_max=1.0e-5,
            strict=True,
        )
    if cell == "ATS_HALFLR_Q2_T500_S0":
        return AdaptiveLossConfig(
            name=cell,
            lambda_max=0.10,
            winner_anchor_base=0.15,
            object_dpo_base=0.75,
            affected_dpo_base=0.10,
            overlap_pres_base=0.25,
            affected_pres_base=0.20,
            boundary_pres_base=0.25,
            transition_delta_max=1.0e-5,
            outside_delta_max=1.0e-5,
            strict=True,
            lr_scale=0.5,
        )
    if cell == "ATS_NODPO_Q2_T500_S0":
        return AdaptiveLossConfig(
            name=cell,
            lambda_max=0.0,
            winner_anchor_base=0.15,
            object_dpo_base=0.0,
            affected_dpo_base=0.0,
            overlap_dpo_base=0.0,
            boundary_dpo_base=0.0,
            overlap_pres_base=0.25,
            affected_pres_base=0.20,
            boundary_pres_base=0.25,
            no_preference_dpo=True,
        )
    if cell == "ATS_LINEAR_Q2_T500_S0":
        return AdaptiveLossConfig(
            name=cell,
            lambda_max=0.10,
            winner_anchor_base=0.15,
            object_dpo_base=0.75,
            affected_dpo_base=0.10,
            overlap_pres_base=0.25,
            affected_pres_base=0.20,
            boundary_pres_base=0.25,
            linear_utility=True,
            strict=True,
        )
    return AdaptiveLossConfig(name=cell)
