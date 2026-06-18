#!/usr/bin/env python3
"""Diagnostics for Exp19c light latent-warp DPO."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


EXP19C_DIAG_COLUMNS = [
    "step",
    "variant_name",
    "total_loss",
    "dpo_loss",
    "warp_loss",
    "warp_loss_forward",
    "warp_loss_backward",
    "lambda_warp",
    "m_w",
    "m_l",
    "m_w_ref",
    "m_l_ref",
    "norm_win_gap",
    "norm_lose_gap",
    "norm_lose_gap_clipped",
    "winner_abs_reg",
    "winner_gap_reg",
    "loser_dominant_ratio",
    "grad_norm",
    "adapter_grad_norm",
    "base_grad_norm",
    "adapter_residual_norm",
    "adapter_to_base_ratio",
    "residual_scale",
    "confidence_exponent",
    "gate_mean",
    "flow_conf_mean",
    "valid_flow_ratio",
    "warp_valid_ratio",
    "mean_flow_magnitude",
    "motion_score",
    "motion_bin",
    "lr",
]


def append_exp19c_diag_csv(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXP19C_DIAG_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, None) for key in EXP19C_DIAG_COLUMNS})
