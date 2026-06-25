#!/usr/bin/env python3
"""Summarize Exp26 VideoPainter step1/step10 search-dev gate.

The script only reads already-generated search-dev outputs and metrics.  It
does not regenerate videos and it does not implement image metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


LOWER_IS_BETTER = {"whole_video_lpips", "ewarp_mask_region", "temporal_diff_delta_vs_gt"}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def summary_path(root: Path, eval_name: str, variant: str) -> Path:
    return root / f"{eval_name}_metric_eval" / f"{variant}_metrics" / "metrics" / "summary.csv"


def per_sample_path(root: Path, eval_name: str, variant: str) -> Path:
    return root / f"{eval_name}_metric_eval" / f"{variant}_metrics" / "metrics" / "per_sample_metrics.csv"


def first_row(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    if not rows:
        raise FileNotFoundError(f"Missing metric summary: {path}")
    return rows[0]


def metric_mean(row: dict[str, str], metric: str) -> float:
    return f(row, f"{metric}_mean")


def paired_stats(
    base_rows: list[dict[str, str]],
    cand_rows: list[dict[str, str]],
    metric: str,
    *,
    seed: int = 20260625,
    n_bootstrap: int = 10000,
) -> dict[str, Any]:
    base = {r["sample_id"]: f(r, metric) for r in base_rows if r.get("status") == "ok"}
    cand = {r["sample_id"]: f(r, metric) for r in cand_rows if r.get("status") == "ok"}
    ids = sorted(set(base) & set(cand))
    deltas = np.array([cand[i] - base[i] for i in ids], dtype=np.float64)
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return {"metric": metric, "n": 0}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        boot[i] = float(rng.choice(deltas, size=deltas.size, replace=True).mean())
    higher_good = metric not in LOWER_IS_BETTER
    wins = deltas > 0 if higher_good else deltas < 0
    prob = float((boot > 0).mean()) if higher_good else float((boot < 0).mean())
    loo = []
    if deltas.size > 1:
        for idx in range(deltas.size):
            loo.append(float(np.delete(deltas, idx).mean()))
    return {
        "metric": metric,
        "n": int(deltas.size),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "win_rate": float(wins.mean()),
        "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
        "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
        "prob_improved": prob,
        "leave_one_out_min": float(min(loo)) if loo else "",
        "leave_one_out_max": float(max(loo)) if loo else "",
    }


def diag_summary(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    if not rows:
        raise FileNotFoundError(f"Missing DPO diagnostics: {path}")
    numeric: dict[str, list[float]] = {}
    for row in rows:
        for key, val in row.items():
            try:
                numeric.setdefault(key, []).append(float(val))
            except Exception:
                pass
    out: dict[str, Any] = {"rows": len(rows)}
    for key in [
        "loss",
        "dpo_loss",
        "implicit_acc",
        "m_w",
        "m_l",
        "m_w_ref",
        "m_l_ref",
        "norm_win_gap",
        "norm_lose_gap",
        "winner_improvement_mean",
        "loser_degradation_mean",
        "loser_dominant_ratio",
        "grad_norm",
    ]:
        vals = numeric.get(key, [])
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            out[f"{key}_mean"] = float(np.nanmean(arr))
            out[f"{key}_last"] = vals[-1]
            out[f"{key}_max"] = float(np.nanmax(arr))
    out["nan_inf_count"] = 0
    for vals in numeric.values():
        out["nan_inf_count"] += int(sum(not math.isfinite(v) for v in vals))
    return out


def gate_decision(step0: dict[str, float], step10: dict[str, float], diag: dict[str, Any], paired: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    psnr_delta = step10["whole_video_psnr"] - step0["whole_video_psnr"]
    lpips_delta = step10["whole_video_lpips"] - step0["whole_video_lpips"]
    tc_delta = step10.get("tc", float("nan")) - step0.get("tc", float("nan"))
    ewarp_delta = step10["ewarp_mask_region"] - step0["ewarp_mask_region"]
    mask_delta = step10["strict_mask_pixel_psnr"] - step0["strict_mask_pixel_psnr"]
    boundary_delta = step10["boundary_pixel_psnr"] - step0["boundary_pixel_psnr"]
    if diag.get("nan_inf_count", 1) != 0:
        reasons.append("non-finite diagnostics present")
    if psnr_delta < -0.02:
        reasons.append(f"PSNR dropped below tolerance ({psnr_delta:+.6f} dB)")
    if lpips_delta > 0.0005:
        reasons.append(f"LPIPS worsened beyond tolerance ({lpips_delta:+.6f})")
    if math.isfinite(tc_delta) and tc_delta < -0.0002:
        reasons.append(f"TC dropped beyond tolerance ({tc_delta:+.6f})")
    if ewarp_delta > 0.03:
        reasons.append(f"Ewarp worsened beyond tolerance ({ewarp_delta:+.6f})")
    positives = 0
    positives += int(mask_delta > 0)
    positives += int(boundary_delta > 0)
    positives += int(lpips_delta < 0)
    positives += int(math.isfinite(tc_delta) and tc_delta > 0)
    positives += int(ewarp_delta < 0)
    if positives < 2:
        reasons.append(f"fewer than two local/perceptual/temporal positives ({positives})")
    psnr_pair = paired.get("whole_video_psnr", {})
    if psnr_pair and psnr_pair.get("win_rate", 0.0) < 0.55:
        reasons.append(f"PSNR per-video win rate below 55% ({psnr_pair.get('win_rate')})")
    if reasons:
        return "VIDEOPAINTER_10STEP_GATE_FAILED", reasons
    return "VIDEOPAINTER_10STEP_GATE_PASSED", ["all pre-registered 10-step tolerances passed"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step0-root", type=Path, required=True)
    parser.add_argument("--step1-root", type=Path, required=True)
    parser.add_argument("--step10-root", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="exp26_vp_10step")
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    roots = {"step0": args.step0_root, "step1": args.step1_root, "step10": args.step10_root}
    aggregate_rows: list[dict[str, Any]] = []
    compact: dict[str, dict[str, float]] = {}
    for step, root in roots.items():
        for variant in ("raw", "comp"):
            row = first_row(summary_path(root, step, variant))
            out = {"step": step, "variant": variant, **row}
            aggregate_rows.append(out)
            if variant == "comp":
                compact[step] = {
                    "whole_video_psnr": metric_mean(row, "whole_video_psnr"),
                    "whole_video_ssim": metric_mean(row, "whole_video_ssim"),
                    "whole_video_lpips": metric_mean(row, "whole_video_lpips"),
                    "ewarp_mask_region": metric_mean(row, "ewarp_mask_region"),
                    "strict_mask_pixel_psnr": metric_mean(row, "strict_mask_pixel_psnr"),
                    "boundary_pixel_psnr": metric_mean(row, "boundary_pixel_psnr"),
                }

    write_csv(args.report_dir / f"{args.output_prefix}_metrics.csv", aggregate_rows)

    base_per = read_csv(per_sample_path(args.step0_root, "step0", "comp"))
    step10_per = read_csv(per_sample_path(args.step10_root, "step10", "comp"))
    paired_list = [
        paired_stats(base_per, step10_per, metric)
        for metric in [
            "whole_video_psnr",
            "whole_video_ssim",
            "whole_video_lpips",
            "strict_mask_pixel_psnr",
            "boundary_pixel_psnr",
            "ewarp_mask_region",
        ]
    ]
    paired = {str(r["metric"]): r for r in paired_list}
    write_csv(args.report_dir / f"{args.output_prefix}_paired_stats.csv", paired_list)

    diag_csv = args.train_root / "dpo_diagnostics.csv"
    diag = diag_summary(diag_csv)
    write_csv(args.report_dir / f"{args.output_prefix}_diagnostics.csv", [diag])

    status, reasons = gate_decision(compact["step0"], compact["step10"], diag, paired)
    summary = {
        "status": status,
        "reasons": reasons,
        "step0_comp": compact["step0"],
        "step1_comp": compact["step1"],
        "step10_comp": compact["step10"],
        "paired_step10_vs_step0": paired,
        "diagnostics": diag,
        "train_root": str(args.train_root),
        "step0_root": str(args.step0_root),
        "step1_root": str(args.step1_root),
        "step10_root": str(args.step10_root),
    }
    (args.report_dir / f"{args.output_prefix}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md = [
        "# Exp26 VideoPainter 10-Step Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Gate Reasons",
        "",
    ]
    md.extend(f"- {r}" for r in reasons)
    md.extend(
        [
            "",
            "## Comp Metrics",
            "",
            "| step | PSNR | SSIM | LPIPS | Ewarp | mask PSNR | boundary PSNR |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for step in ("step0", "step1", "step10"):
        vals = compact[step]
        md.append(
            f"| {step} | {vals['whole_video_psnr']:.6f} | {vals['whole_video_ssim']:.6f} | {vals['whole_video_lpips']:.6f} | {vals['ewarp_mask_region']:.6f} | {vals['strict_mask_pixel_psnr']:.6f} | {vals['boundary_pixel_psnr']:.6f} |"
        )
    d = {k: compact["step10"][k] - compact["step0"][k] for k in compact["step0"]}
    md.extend(
        [
            "",
            "## Step10 - Step0",
            "",
            f"- PSNR: `{d['whole_video_psnr']:+.6f}`",
            f"- SSIM: `{d['whole_video_ssim']:+.6f}`",
            f"- LPIPS: `{d['whole_video_lpips']:+.6f}`",
            f"- Ewarp: `{d['ewarp_mask_region']:+.6f}`",
            f"- strict mask PSNR: `{d['strict_mask_pixel_psnr']:+.6f}`",
            f"- boundary pixel PSNR: `{d['boundary_pixel_psnr']:+.6f}`",
            "",
            "## Diagnostics",
            "",
            f"- rows: `{diag.get('rows')}`",
            f"- max grad norm: `{diag.get('grad_norm_max')}`",
            f"- final loser-dominant ratio: `{diag.get('loser_dominant_ratio_last')}`",
            f"- NaN/Inf count: `{diag.get('nan_inf_count')}`",
            "",
            "This report is a 10-step gate only. It does not start 50-step unless the status is `VIDEOPAINTER_10STEP_GATE_PASSED` and visual review is complete.",
        ]
    )
    (args.report_dir / f"{args.output_prefix}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
