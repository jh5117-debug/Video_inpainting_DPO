#!/usr/bin/env python3
"""Formula reference for Exp17 saturation-aware positive DPO.

The actual training implementation lives in ``train_exp17_stage1.py`` because
the copied Exp11 trainer is self-contained. This module is kept small on
purpose: it documents the three Exp17 variants and provides pure tensor helpers
for tests or future refactors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def saturation_weight(margin_pref: torch.Tensor, target_margin: float = 1.0, kappa: float = 5.0) -> torch.Tensor:
    """Smoothly down-weight DPO after the preference margin is already enough."""
    return torch.sigmoid(float(kappa) * (float(target_margin) - margin_pref))


def dpo_positive_components(
    g_w: torch.Tensor,
    g_l_clip: torch.Tensor,
    beta_dpo: float = 10.0,
    lose_gap_weight: float = 0.25,
    margin_pos: float = 0.0,
    target_margin: float = 1.0,
    kappa: float = 5.0,
) -> dict[str, torch.Tensor]:
    """Return Exp17 pairwise and positive-preserving components.

    ``g_w`` and ``g_l_clip`` are normalized winner / clipped loser gaps.
    Smaller ``g_w`` is better for winner preservation. Larger
    ``lose_gap_weight * g_l_clip - g_w`` means the preference margin is already
    satisfied.
    """
    inside = -0.5 * float(beta_dpo) * (g_w - float(lose_gap_weight) * g_l_clip)
    dpo_raw = -F.logsigmoid(inside)
    margin_pref = float(lose_gap_weight) * g_l_clip - g_w
    sat = saturation_weight(margin_pref, target_margin, kappa).detach()
    return {
        "dpo_loss_raw": dpo_raw,
        "dpo_loss_saturated": dpo_raw * sat,
        "positive_loss": F.relu(g_w - float(margin_pos)),
        "margin_pref": margin_pref,
        "sat_weight": sat,
        "saturated_pair": (margin_pref >= float(target_margin)).float(),
    }
