#!/usr/bin/env python3
"""Collect compact DPO diagnostic summaries from experiment registry rows."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import median
from typing import Iterable


STAT_COLUMNS = {
    "dpo_loss": ("first", "last", "mean", "median", "p90", "min", "frac_lt_1e_3"),
    "implicit_acc": ("mean", "frac_gt_0_99"),
    "mse_w": ("first", "last", "mean", "median", "p90", "max"),
    "ref_mse_w": ("first", "last", "mean", "median"),
    "mse_l": ("first", "last", "mean", "median", "p90", "max"),
    "ref_mse_l": ("first", "last", "mean", "median"),
    "win_gap": ("first", "last", "mean", "median", "p90", "max", "frac_gt_0_5"),
    "lose_gap": ("first", "last", "mean", "median", "p90", "max"),
    "winner_abs_reg": ("first", "last", "mean", "median", "p90", "max"),
    "winner_gap_reg": ("first", "last", "mean", "median", "p90", "max"),
    "mse_w_over_ref_mse_w": ("mean", "median", "p90", "max", "frac_gt_5"),
    "mse_l_over_ref_mse_l": ("mean", "median", "p90", "max"),
    "sigma_term": ("mean", "frac_gt_0_99"),
    "kl_divergence": ("mean", "frac_gt_1_0"),
    "loser_dominant_ratio": ("mean", "frac_gt_0_9"),
    "grad_norm": ("mean", "max"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    return parser.parse_args()


def to_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def stat(values: list[float], name: str) -> float | None:
    if not values:
        return None
    if name == "first":
        return values[0]
    if name == "last":
        return values[-1]
    if name == "mean":
        return sum(values) / len(values)
    if name == "median":
        return float(median(values))
    if name == "p90":
        return percentile(values, 0.9)
    if name == "min":
        return min(values)
    if name == "max":
        return max(values)
    if name == "frac_gt_0_99":
        return sum(v > 0.99 for v in values) / len(values)
    if name == "frac_lt_1e_3":
        return sum(v < 1e-3 for v in values) / len(values)
    if name == "frac_gt_0_5":
        return sum(v > 0.5 for v in values) / len(values)
    if name == "frac_gt_5":
        return sum(v > 5.0 for v in values) / len(values)
    if name == "frac_gt_1_0":
        return sum(v > 1.0 for v in values) / len(values)
    if name == "frac_gt_0_9":
        return sum(v > 0.9 for v in values) / len(values)
    raise ValueError(f"unknown stat {name}")


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def resolve_path(path_text: str, base_dir: Path) -> Path | None:
    path_text = (path_text or "").strip()
    if not path_text or path_text.upper() in {"MISSING_DIAG", "MISSING_DPO_DIAG", "REMOTE_ONLY", "PENDING_PAI"}:
        return None
    if path_text.lower() in {"not dpo", "not_dpo", "n/a non-dpo"}:
        return None
    p = Path(path_text)
    if p.is_absolute():
        return p
    return base_dir / p


def collect_file(path: Path) -> tuple[list[dict[str, str]], str]:
    if not path.exists():
        return [], "MISSING_DIAG"
    rows = read_rows(path)
    if not rows:
        return [], "INCOMPLETE"
    return rows, "FOUND"


def values_for(rows: Iterable[dict[str, str]], column: str) -> list[float]:
    out = []
    for row in rows:
        val = to_float(row.get(column))
        if val is not None:
            out.append(val)
    return out


def first_last_step(rows: list[dict[str, str]]) -> tuple[str, str]:
    step_keys = ("global_step", "step", "checkpoint_step")
    vals = []
    for row in rows:
        for key in step_keys:
            val = to_float(row.get(key))
            if val is not None:
                vals.append(int(val))
                break
    if not vals:
        return "", ""
    return str(min(vals)), str(max(vals))


def label_diag(summary: dict[str, str], found_status: str) -> str:
    if found_status == "NOT_DPO":
        return "NOT_DPO"
    if found_status == "MISSING_DIAG":
        return "MISSING_DIAG"
    if found_status == "INCOMPLETE":
        return "INCOMPLETE"
    labels = []

    def get(name: str) -> float:
        return to_float(summary.get(name)) or 0.0

    if get("dpo_loss_median") < 1e-3 or get("dpo_loss_frac_lt_1e_3") > 0.35 or get("implicit_acc_frac_gt_0_99") > 0.35 or get("sigma_term_frac_gt_0_99") > 0.1:
        labels.append("DPO_SATURATED")
    if get("mse_w_over_ref_mse_w_p90") > 5 or get("mse_w_over_ref_mse_w_frac_gt_5") > 0.1 or get("win_gap_p90") > 0.5 or get("win_gap_frac_gt_0_5") > 0.1:
        labels.append("WIN_GAP_EXPLODED")
    if get("loser_dominant_ratio_mean") > 0.9 or get("mse_l_over_ref_mse_l_p90") > 10:
        labels.append("LOSER_DOMINANT")
    if any(x in labels for x in ("DPO_SATURATED", "WIN_GAP_EXPLODED", "LOSER_DOMINANT")):
        labels.append("COLLAPSE_RISK")
    if not labels:
        labels.append("OK_STABLE")
    return ";".join(dict.fromkeys(labels))


def summarize_registry_row(reg_row: dict[str, str], base_dir: Path) -> dict[str, str]:
    diag_path_text = reg_row.get("dpo_diag_csv", "")
    if diag_path_text.strip().lower() in {"not dpo", "not_dpo", "n/a non-dpo"}:
        rows, found_status = [], "NOT_DPO"
    else:
        diag_path = resolve_path(diag_path_text, base_dir)
        rows, found_status = ([], "MISSING_DIAG") if diag_path is None else collect_file(diag_path)
    summary = {
        "experiment_id": reg_row.get("experiment_id", ""),
        "short_name": reg_row.get("short_name", ""),
        "registry_status": reg_row.get("status", ""),
        "dpo_diag_csv": diag_path_text,
        "diag_file_status": found_status,
        "row_count": str(len(rows)) if rows else "0",
        "first_step": "",
        "last_step": "",
    }
    if rows:
        summary["first_step"], summary["last_step"] = first_last_step(rows)
        for column, names in STAT_COLUMNS.items():
            vals = values_for(rows, column)
            for name in names:
                summary[f"{column}_{name}"] = fmt(stat(vals, name))
    for column, names in STAT_COLUMNS.items():
        for name in names:
            summary.setdefault(f"{column}_{name}", "")
    summary["diag_status"] = label_diag(summary, found_status)
    return summary


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    columns = [
        "experiment_id",
        "short_name",
        "diag_file_status",
        "row_count",
        "first_step",
        "last_step",
        "dpo_loss_median",
        "dpo_loss_frac_lt_1e_3",
        "implicit_acc_mean",
        "win_gap_p90",
        "win_gap_frac_gt_0_5",
        "lose_gap_p90",
        "mse_w_over_ref_mse_w_p90",
        "mse_w_over_ref_mse_w_frac_gt_5",
        "mse_l_over_ref_mse_l_p90",
        "sigma_term_frac_gt_0_99",
        "loser_dominant_ratio_mean",
        "diag_status",
    ]
    lines = [
        "# All Experiments DPO Diagnostic Summary",
        "",
        "This table is generated from `experiment_registry/experiment_matrix.csv`.",
        "Experiments without a local diagnostic CSV are explicitly marked `MISSING_DIAG`.",
        "",
        "|" + "|".join(columns) + "|",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join(row.get(c, "") for c in columns) + "|")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_registry_summaries(rows: list[dict[str, str]], registry_path: Path) -> None:
    registry_dir = registry_path.parent
    for row in rows:
        short_name = row.get("short_name", "").strip()
        if not short_name:
            continue
        candidates = [p for p in registry_dir.iterdir() if p.is_dir() and short_name in p.name]
        if not candidates:
            continue
        target = sorted(candidates, key=lambda p: len(p.name))[0] / "dpo_diag_summary.md"
        lines = [
            f"# DPO Diagnostics Summary: {row.get('experiment_id', '')}",
            "",
            f"- short_name: `{short_name}`",
            f"- diag_file_status: `{row.get('diag_file_status', '')}`",
            f"- diag_status: `{row.get('diag_status', '')}`",
            f"- dpo_diag_csv: `{row.get('dpo_diag_csv', '') or 'MISSING_DPO_DIAG'}`",
            f"- row_count: `{row.get('row_count', '0')}`",
            f"- first_step: `{row.get('first_step', '')}`",
            f"- last_step: `{row.get('last_step', '')}`",
            "",
            "## Key Risk Fields",
            "",
            f"- dpo_loss_median: `{row.get('dpo_loss_median', '')}`",
            f"- dpo_loss_frac_lt_1e_3: `{row.get('dpo_loss_frac_lt_1e_3', '')}`",
            f"- implicit_acc_mean: `{row.get('implicit_acc_mean', '')}`",
            f"- win_gap_p90: `{row.get('win_gap_p90', '')}`",
            f"- win_gap_frac_gt_0_5: `{row.get('win_gap_frac_gt_0_5', '')}`",
            f"- mse_w_over_ref_mse_w_p90: `{row.get('mse_w_over_ref_mse_w_p90', '')}`",
            f"- mse_w_over_ref_mse_w_frac_gt_5: `{row.get('mse_w_over_ref_mse_w_frac_gt_5', '')}`",
            f"- mse_l_over_ref_mse_l_p90: `{row.get('mse_l_over_ref_mse_l_p90', '')}`",
            f"- sigma_term_frac_gt_0_99: `{row.get('sigma_term_frac_gt_0_99', '')}`",
            f"- loser_dominant_ratio_mean: `{row.get('loser_dominant_ratio_mean', '')}`",
            "",
            "If `diag_file_status` is `MISSING_DIAG`, this experiment is incomplete as DPO evidence even if videos or metrics exist.",
        ]
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    registry_rows = read_rows(args.registry)
    base_dir = args.registry.parent.parent.resolve()
    summaries = [summarize_registry_row(row, base_dir) for row in registry_rows]
    fieldnames = list(summaries[0].keys()) if summaries else [
        "experiment_id",
        "short_name",
        "diag_file_status",
        "diag_status",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    write_markdown(summaries, args.markdown)
    write_registry_summaries(summaries, args.registry)


if __name__ == "__main__":
    main()
