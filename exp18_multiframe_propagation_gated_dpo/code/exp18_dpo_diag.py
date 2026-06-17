#!/usr/bin/env python3
"""Small diagnostics summarizer for Exp18 DPO CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


FIELDS = [
    "loss",
    "L_base",
    "dpo_loss",
    "L_prop",
    "L_gen",
    "L_boundary",
    "norm_win_gap",
    "norm_lose_gap",
    "norm_lose_gap_clipped",
    "loser_dominant_ratio",
    "grad_norm",
    "propagation_coverage",
    "generate_area_ratio",
    "prop_conf_mean",
    "avg_num_sources_used",
    "propagated_region_psnr",
]


def as_float(value: str | None) -> float | None:
    if value is None or value == "" or value.lower() == "null":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    rows = list(csv.DictReader(csv_path.open()))
    lines = ["# Exp18 DPO Diagnostics Summary", "", f"- csv: `{csv_path}`", f"- rows: `{len(rows)}`", ""]
    if not rows:
        lines.append("No rows.")
    else:
        lines.append("| field | mean | last20_mean | last |")
        lines.append("|---|---:|---:|---:|")
        for field in FIELDS:
            values = [as_float(row.get(field)) for row in rows]
            vals = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
            if vals.size == 0:
                lines.append(f"| {field} | n/a | n/a | n/a |")
                continue
            last20 = vals[-20:]
            lines.append(f"| {field} | {vals.mean():.6f} | {last20.mean():.6f} | {vals[-1]:.6f} |")
        last20_loser = [as_float(row.get("loser_dominant_ratio")) for row in rows[-20:]]
        last20_loser_vals = [v for v in last20_loser if v is not None]
        label = "OK_STABLE"
        if last20_loser_vals and float(np.mean(last20_loser_vals)) > 0.95:
            label = "LOSER_DOMINANT"
        lines.extend(["", f"Label: `{label}`.", ""])
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

