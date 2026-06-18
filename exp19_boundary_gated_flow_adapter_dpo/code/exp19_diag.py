#!/usr/bin/env python3
"""Exp19 diagnostic CSV helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


EXP19_DIAG_COLUMNS = [
    "step",
    "variant_name",
    "total_loss",
    "dpo_loss",
    "warp_loss",
    "m_w",
    "m_l",
    "m_w_ref",
    "m_l_ref",
    "raw_win_gap",
    "raw_lose_gap",
    "norm_win_gap",
    "norm_lose_gap",
    "norm_lose_gap_clipped",
    "winner_abs_reg",
    "winner_gap_reg",
    "loser_dominant_ratio",
    "grad_norm",
    "adapter_grad_norm",
    "base_grad_norm",
    "flow_feat_norm",
    "adapter_residual_norm",
    "adapter_to_base_ratio",
    "alpha_scale_1",
    "alpha_scale_2",
    "alpha_scale_3",
    "gate_mean",
    "gate_p10",
    "gate_p50",
    "gate_p90",
    "flow_conf_mean",
    "valid_flow_ratio",
    "mean_flow_magnitude",
    "forward_backward_error",
    "lr",
]


def append_exp19_diag_csv(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXP19_DIAG_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, None) for key in EXP19_DIAG_COLUMNS})


def grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += float(param.grad.detach().float().norm().cpu()) ** 2
    return total ** 0.5
