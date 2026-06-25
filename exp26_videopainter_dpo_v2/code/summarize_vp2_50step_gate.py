#!/usr/bin/env python3
"""Summarize Exp26 VideoPainter 50-step gate from existing outputs.

This script only reads already-generated search-dev outputs, diagnostics, and
metric CSVs. It does not regenerate videos and it does not reimplement image
metrics.
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
METRICS = [
    "whole_video_psnr",
    "whole_video_ssim",
    "whole_video_lpips",
    "strict_mask_pixel_psnr",
    "boundary_pixel_psnr",
    "ewarp_mask_region",
]


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


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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


def compact_metrics(row: dict[str, str]) -> dict[str, float]:
    return {metric: metric_mean(row, metric) for metric in METRICS}


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
            out[f"{key}_p95"] = float(np.nanquantile(arr, 0.95))
    out["grad_gt_10_count"] = int(sum(v > 10 for v in numeric.get("grad_norm", [])))
    out["grad_gt_50_count"] = int(sum(v > 50 for v in numeric.get("grad_norm", [])))
    out["grad_gt_100_count"] = int(sum(v > 100 for v in numeric.get("grad_norm", [])))
    out["nan_inf_count"] = 0
    for vals in numeric.values():
        out["nan_inf_count"] += int(sum(not math.isfinite(v) for v in vals))
    return out


def review_summary(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    out: dict[str, Any] = {"rows": len(rows), "classification_counts": {}, "reviewer_pass_count": 0}
    for row in rows:
        cls = row.get("classification") or row.get("final_class") or "unknown"
        out["classification_counts"][cls] = out["classification_counts"].get(cls, 0) + 1
        if str(row.get("reviewer_pass", "")).lower() == "true":
            out["reviewer_pass_count"] += 1
    return out


def load_preflights(train_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in (10, 20, 30, 40, 50):
        path = train_root / f"reload_preflight_step{step}" / "preflight_report.json"
        data = read_json(path) or {}
        rows.append(
            {
                "step": step,
                "path": str(path),
                "exists": path.exists(),
                "status": data.get("status", ""),
                "missing_keys": len(data.get("missing_keys", []) or []),
                "unexpected_keys": len(data.get("unexpected_keys", []) or []),
                "max_reload_diff": data.get("max_reload_diff", data.get("reload_max_abs_diff", "")),
            }
        )
    return rows


def final_gate(compact: dict[str, dict[str, float]], diag: dict[str, Any], paired50: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    positives: list[str] = []
    step0 = compact["step0"]
    step50 = compact["step50"]
    d = {k: step50[k] - step0[k] for k in step0}
    if diag.get("nan_inf_count", 1) != 0:
        reasons.append("non-finite diagnostics present")
    if not (d["whole_video_psnr"] > 0.02 or (d["whole_video_psnr"] >= -0.02 and (d["strict_mask_pixel_psnr"] > 0 or d["boundary_pixel_psnr"] > 0))):
        reasons.append(f"primary PSNR/mask-boundary gate failed ({d['whole_video_psnr']:+.6f} dB)")
    else:
        positives.append("primary PSNR or local metric gate passed")
    psnr_pair = paired50.get("whole_video_psnr", {})
    if psnr_pair.get("win_rate", 0.0) < 0.55:
        reasons.append(f"PSNR per-video win rate below 55% ({psnr_pair.get('win_rate')})")
    if psnr_pair.get("prob_improved", 0.0) < 0.90:
        reasons.append(f"PSNR bootstrap probability below 0.90 ({psnr_pair.get('prob_improved')})")
    if d["whole_video_lpips"] > 0.0005:
        reasons.append(f"LPIPS worsened beyond tolerance ({d['whole_video_lpips']:+.6f})")
    if d["ewarp_mask_region"] > 0.03:
        reasons.append(f"Ewarp worsened beyond tolerance ({d['ewarp_mask_region']:+.6f})")
    if compact["step30"]["whole_video_psnr"] <= step0["whole_video_psnr"] and step50["whole_video_psnr"] <= step0["whole_video_psnr"]:
        reasons.append("checkpoint trend not positive at step30 and step50")
    if reasons:
        return "VIDEOPAINTER_ADAPTER_NEGATIVE", reasons
    return "VIDEOPAINTER_ADAPTER_POSITIVE_METRIC_GATE_PENDING_MANUAL_VISUAL_CONFIRMATION", positives


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step0-root", type=Path, required=True)
    parser.add_argument("--step10-root", type=Path, required=True)
    parser.add_argument("--step30-root", type=Path, required=True)
    parser.add_argument("--step50-root", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="exp26_vp_50step")
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    roots = {
        "step0": args.step0_root,
        "step10": args.step10_root,
        "step30": args.step30_root,
        "step50": args.step50_root,
    }
    aggregate_rows: list[dict[str, Any]] = []
    compact: dict[str, dict[str, float]] = {}
    review_rows: list[dict[str, Any]] = []
    for step, root in roots.items():
        eval_name = step
        for variant in ("raw", "comp"):
            row = first_row(summary_path(root, eval_name, variant))
            aggregate_rows.append({"step": step, "variant": variant, **row})
            if variant == "comp":
                compact[step] = compact_metrics(row)
        review_csv = root / f"{eval_name}_review" / "gate64_visual_review.csv"
        review_rows.append({"step": step, **review_summary(review_csv), "path": str(review_csv)})

    write_csv(args.report_dir / f"{args.output_prefix}_metrics.csv", aggregate_rows)
    write_csv(args.report_dir / f"{args.output_prefix}_visual_review.csv", review_rows)

    base_per = read_csv(per_sample_path(args.step0_root, "step0", "comp"))
    paired_by_step: dict[str, dict[str, dict[str, Any]]] = {}
    paired_rows: list[dict[str, Any]] = []
    for step in ("step10", "step30", "step50"):
        cand_per = read_csv(per_sample_path(roots[step], step, "comp"))
        paired_list = [paired_stats(base_per, cand_per, metric) for metric in METRICS]
        paired_by_step[step] = {str(r["metric"]): r for r in paired_list}
        paired_rows.extend({"step": step, **r} for r in paired_list)
    write_csv(args.report_dir / f"{args.output_prefix}_paired_stats.csv", paired_rows)

    diag = diag_summary(args.train_root / "dpo_diagnostics.csv")
    write_csv(args.report_dir / f"{args.output_prefix}_diagnostics.csv", [diag])
    preflights = load_preflights(args.train_root)
    write_csv(args.report_dir / f"{args.output_prefix}_reload_preflight.csv", preflights)

    status, reasons = final_gate(compact, diag, paired_by_step["step50"])
    deltas = {
        step: {metric: compact[step][metric] - compact["step0"][metric] for metric in METRICS}
        for step in ("step10", "step30", "step50")
    }
    summary = {
        "status": status,
        "reasons": reasons,
        "compact_comp_metrics": compact,
        "deltas_vs_step0": deltas,
        "paired_vs_step0": paired_by_step,
        "diagnostics": diag,
        "reload_preflight": preflights,
        "review_summary": review_rows,
        "train_root": str(args.train_root),
        "step0_root": str(args.step0_root),
        "step10_root": str(args.step10_root),
        "step30_root": str(args.step30_root),
        "step50_root": str(args.step50_root),
    }
    (args.report_dir / f"{args.output_prefix}_statistics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md = [
        "# Exp26 VideoPainter 50-Step Gate",
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
    for step in ("step0", "step10", "step30", "step50"):
        vals = compact[step]
        md.append(
            f"| {step} | {vals['whole_video_psnr']:.6f} | {vals['whole_video_ssim']:.6f} | {vals['whole_video_lpips']:.6f} | {vals['ewarp_mask_region']:.6f} | {vals['strict_mask_pixel_psnr']:.6f} | {vals['boundary_pixel_psnr']:.6f} |"
        )
    md.extend(["", "## Deltas Vs Step0", ""])
    for step in ("step10", "step30", "step50"):
        d = deltas[step]
        md.append(
            f"- {step}: PSNR `{d['whole_video_psnr']:+.6f}`, SSIM `{d['whole_video_ssim']:+.6f}`, LPIPS `{d['whole_video_lpips']:+.6f}`, Ewarp `{d['ewarp_mask_region']:+.6f}`, mask PSNR `{d['strict_mask_pixel_psnr']:+.6f}`, boundary PSNR `{d['boundary_pixel_psnr']:+.6f}`"
        )
    ps = paired_by_step["step50"]["whole_video_psnr"]
    md.extend(
        [
            "",
            "## Step50 Paired PSNR",
            "",
            f"- mean delta: `{ps.get('mean_delta'):+.6f}`",
            f"- win rate: `{ps.get('win_rate'):.6f}`",
            f"- bootstrap 95% CI: `[{ps.get('bootstrap_ci_low'):+.6f}, {ps.get('bootstrap_ci_high'):+.6f}]`",
            f"- probability(delta > 0): `{ps.get('prob_improved'):.6f}`",
            f"- leave-one-out range: `[{ps.get('leave_one_out_min'):+.6f}, {ps.get('leave_one_out_max'):+.6f}]`",
            "",
            "## Diagnostics",
            "",
            f"- rows: `{diag.get('rows')}`",
            f"- max grad norm: `{diag.get('grad_norm_max')}`",
            f"- p95 grad norm: `{diag.get('grad_norm_p95')}`",
            f"- final loser-dominant ratio: `{diag.get('loser_dominant_ratio_last')}`",
            f"- NaN/Inf count: `{diag.get('nan_inf_count')}`",
            "",
            "This report is a 50-step gate only. It does not authorize 100-step or longer training.",
        ]
    )
    (args.report_dir / f"{args.output_prefix}_final.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
