#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("/home/nvme01/H20_Video_inpainting_DPO_exp56_void_region_safe_h20")
OUT_ROOT = Path("/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp56_void_region_safe_h20")
SRC_EVAL = ROOT / "exp53_void_r1r2_targeted_h20/scripts/eval_exp53b_core_cells.py"
CELLS = ["R5_Q2_T500_S0", "R5_HALF_Q2_T500_S0"]


def mean(rows, key: str) -> float:
    vals = []
    for row in rows:
        val = row.get(key)
        if val in (None, "", "NA"):
            continue
        try:
            vals.append(float(val))
        except ValueError:
            pass
    return float(np.nanmean(vals)) if vals else float("nan")


def exp56_status(metrics: list[dict], diagnostics: list[dict], reviews: list[dict]) -> str:
    full = mean(metrics, "full_psnr_delta")
    outside = mean(metrics, "outside_psnr_delta")
    obj = mean(metrics, "object_psnr_delta")
    boundary = mean(metrics, "boundary_psnr_delta")
    overlap = mean(metrics, "overlap_psnr_delta")
    affected = mean(metrics, "affected_psnr_delta")
    heldout = [d for d in diagnostics if d.get("split") == "heldout4"]
    source = heldout or diagnostics
    loser_ratio = mean(source, "loser_contribution_ratio")
    winner_gap = mean(source, "winner_gap")
    worse = sum(1 for r in reviews if r.get("better_tie_worse") == "worse")
    better_tie = sum(1 for r in reviews if r.get("better_tie_worse") in ("better", "tie"))
    pass_ok = (
        full >= -0.02
        and outside >= -0.02
        and obj >= -0.10
        and boundary >= -0.03
        and overlap >= -0.03
        and affected >= -0.03
        and better_tie >= 3
        and worse <= 1
        and loser_ratio <= 0.25
        and (winner_gap >= 0 or not math.isfinite(winner_gap))
    )
    if pass_ok:
        return "PASS"
    if full >= -0.05 and outside >= -0.05 and worse <= 2:
        return "MIXED"
    return "NEGATIVE"


def main() -> None:
    spec = importlib.util.spec_from_file_location("exp53_eval", SRC_EVAL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SRC_EVAL}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = ROOT
    mod.OUT_ROOT = OUT_ROOT
    mod.VIDEO_ROOT = OUT_ROOT / "r5_video"
    mod.FORWARD_ROOT = OUT_ROOT / "r5_forward"
    mod.REPORTS = ROOT / "reports"
    mod.MANIFEST = ROOT / "manifests/exp50_void_adapter_heldout4_h20.jsonl"
    mod.CELLS = CELLS
    mod.main()

    reports = ROOT / "reports"
    mapping = {
        "exp53b_core_onestep_metrics.csv": "exp56_h20_r5_onestep_metrics.csv",
        "exp53b_core_onestep_visual_review.csv": "exp56_h20_r5_onestep_visual_review.csv",
        "exp53b_core_onestep_diagnostics.csv": "exp56_h20_r5_onestep_diagnostics.csv",
        "exp53b_core_onestep_summary.json": "exp56_h20_r5_onestep_summary.json",
        "exp53b_core_onestep.md": "exp56_h20_r5_onestep.md",
    }
    for src, dst in mapping.items():
        (reports / dst).write_bytes((reports / src).read_bytes())

    with (reports / "exp56_h20_r5_onestep_metrics.csv").open() as f:
        metrics = list(csv.DictReader(f))
    with (reports / "exp56_h20_r5_onestep_diagnostics.csv").open() as f:
        diagnostics = list(csv.DictReader(f))
    with (reports / "exp56_h20_r5_onestep_visual_review.csv").open() as f:
        reviews = list(csv.DictReader(f))
    summary = json.loads((reports / "exp56_h20_r5_onestep_summary.json").read_text())
    cell_summaries = summary.get("cells", {})
    pass_cells: list[str] = []
    mixed_cells: list[str] = []
    negative_cells: list[str] = []
    for cell in CELLS:
        m = [r for r in metrics if r.get("cell") == cell]
        d = [r for r in diagnostics if r.get("cell") == cell]
        v = [r for r in reviews if r.get("cell") == cell]
        status = exp56_status(m, d, v)
        if cell in cell_summaries:
            cell_summaries[cell]["status"] = status
        if status == "PASS":
            pass_cells.append(cell)
        elif status == "MIXED":
            mixed_cells.append(cell)
        else:
            negative_cells.append(cell)
    final_status = (
        "EXP56_H20_R5_ONESTEP_PASS"
        if pass_cells
        else ("EXP56_H20_R5_ONESTEP_MIXED" if mixed_cells else "EXP56_H20_R5_ONESTEP_NEGATIVE")
    )
    summary["status"] = final_status
    summary["best_cell"] = pass_cells[0] if pass_cells else (mixed_cells[0] if mixed_cells else CELLS[0])
    summary["ten_step_run"] = False
    summary["no_vor_eval"] = True
    summary["hard_comp"] = False
    summary["exp56_pass_gate"] = {
        "full_psnr_delta_min": -0.02,
        "outside_psnr_delta_min": -0.02,
        "object_psnr_delta_min": -0.10,
        "boundary_psnr_delta_min": -0.03,
        "overlap_psnr_delta_min": -0.03,
        "affected_psnr_delta_min": -0.03,
        "loser_contribution_ratio_max": 0.25,
    }
    (reports / "exp56_h20_r5_onestep_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = ["# Exp56-H20 R5 One-Step Cells", "", f"Status: `{final_status}`", "", "One optimizer step only. No 10-step.", ""]
    for cell in CELLS:
        cs = summary["cells"][cell]
        lines.extend(
            [
                f"## {cell}",
                f"- status: `{cs['status']}`",
                f"- full PSNR delta: {cs['means']['full_psnr_delta']:.6f}",
                f"- object PSNR delta: {cs['means']['object_psnr_delta']:.6f}",
                f"- overlap PSNR delta: {cs['means']['overlap_psnr_delta']:.6f}",
                f"- affected PSNR delta: {cs['means']['affected_psnr_delta']:.6f}",
                f"- boundary PSNR delta: {cs['means']['boundary_psnr_delta']:.6f}",
                f"- outside PSNR delta: {cs['means']['outside_psnr_delta']:.6f}",
                f"- loser contribution ratio: {cs['diagnostics']['loser_contribution_ratio']:.6f}",
                f"- visual counts: {cs['visual_counts']}",
                f"- contact sheet: `{cs['contact_sheet']}`",
                "",
            ]
        )
    lines.append("No VOR-Eval, hard comp, long training, or 10-step was used.")
    (reports / "exp56_h20_r5_onestep.md").write_text("\n".join(lines) + "\n")
    print(final_status)


if __name__ == "__main__":
    main()
