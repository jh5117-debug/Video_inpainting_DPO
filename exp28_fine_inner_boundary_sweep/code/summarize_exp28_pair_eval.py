#!/usr/bin/env python3
"""Summarize Exp23 paired DAVIS50 outputs and training diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


METRIC_MAP = {
    "PSNR": "whole_video_psnr",
    "SSIM": "whole_video_ssim",
    "LPIPS": "whole_video_lpips",
    "TC": "tc",
    "Ewarp": "ewarp",
    "strict_mask_PSNR": "strict_mask_pixel_psnr",
    "mask_region_SSIM": "mask_region_ssim",
    "boundary_PSNR": "boundary_pixel_psnr",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object) -> Optional[float]:
    try:
        val = float(value)  # type: ignore[arg-type]
    except Exception:
        return None
    return val if math.isfinite(val) else None


def finite_mean(values: Iterable[object]) -> float:
    vals = [v for v in (as_float(x) for x in values) if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def finite_max(values: Iterable[object]) -> float:
    vals = [v for v in (as_float(x) for x in values) if v is not None]
    return float(np.max(vals)) if vals else float("nan")


def finite_percentile(values: Iterable[object], pct: float) -> float:
    vals = [v for v in (as_float(x) for x in values) if v is not None]
    return float(np.percentile(vals, pct)) if vals else float("nan")


def collect_eval_rows(eval_root: Path) -> Tuple[List[Dict[str, object]], Dict[str, List[Dict[str, str]]]]:
    summaries: List[Dict[str, object]] = []
    per_video: Dict[str, List[Dict[str, str]]] = {}
    for metrics_dir in sorted(eval_root.glob("*/metrics")):
        label = metrics_dir.parent.name
        summary_csv = metrics_dir / "summary.csv"
        per_video_csv = metrics_dir / "per_video_metrics.csv"
        if not summary_csv.exists() or not per_video_csv.exists():
            continue
        rows = read_csv(summary_csv)
        if rows:
            row: Dict[str, object] = {"label": label}
            row.update(rows[0])
            summaries.append(row)
        per_video[label] = read_csv(per_video_csv)
    return summaries, per_video


def paired_arrays(
    per_video: Dict[str, List[Dict[str, str]]],
    candidate: str,
    fresh: str,
    metric_key: str,
) -> Tuple[np.ndarray, List[str]]:
    cand = {r.get("video", ""): as_float(r.get(metric_key)) for r in per_video.get(candidate, [])}
    base = {r.get("video", ""): as_float(r.get(metric_key)) for r in per_video.get(fresh, [])}
    videos = sorted(v for v in cand if v in base and cand[v] is not None and base[v] is not None)
    deltas = np.array([cand[v] - base[v] for v in videos], dtype=np.float64)
    return deltas, videos


def bootstrap_delta(deltas: np.ndarray, seed: int = 20260619, n: int = 10000) -> Dict[str, object]:
    if deltas.size == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, deltas.size, size=(n, deltas.size))
    means = deltas[idx].mean(axis=1)
    loo = []
    if deltas.size > 1:
        for i in range(deltas.size):
            loo.append(float(np.delete(deltas, i).mean()))
    return {
        "n": int(deltas.size),
        "mean": float(deltas.mean()),
        "median": float(np.median(deltas)),
        "min": float(deltas.min()),
        "max": float(deltas.max()),
        "win_rate": float((deltas > 0).mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "prob_delta_gt_0": float((means > 0).mean()),
        "leave_one_out_min": float(min(loo)) if loo else float("nan"),
        "leave_one_out_max": float(max(loo)) if loo else float("nan"),
    }


def compare_pair(
    per_video: Dict[str, List[Dict[str, str]]],
    candidate: str,
    fresh: str,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    payload: Dict[str, object] = {"candidate": candidate, "fresh": fresh, "metrics": {}}
    for metric_name, metric_key in METRIC_MAP.items():
        deltas, videos = paired_arrays(per_video, candidate, fresh, metric_key)
        stats = bootstrap_delta(deltas)
        stats["metric"] = metric_name
        payload["metrics"][metric_name] = stats
        row = {"candidate": candidate, "fresh": fresh, "metric": metric_name, "videos": len(videos)}
        row.update(stats)
        rows.append(row)
    return rows, payload


def summary_value(row: Dict[str, object], metric_key: str) -> Optional[float]:
    if metric_key == "vfid":
        return as_float(row.get("vfid"))
    return as_float(row.get(f"{metric_key}_mean"))


def compare_summary_pairs(summaries: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_label = {str(row.get("label")): row for row in summaries}
    metric_keys = {
        "PSNR": "whole_video_psnr",
        "SSIM": "whole_video_ssim",
        "LPIPS": "whole_video_lpips",
        "TC": "tc",
        "Ewarp": "ewarp",
        "VFID": "vfid",
        "strict_mask_PSNR": "strict_mask_pixel_psnr",
        "mask_region_SSIM": "mask_region_ssim",
        "boundary_PSNR": "boundary_pixel_psnr",
    }
    rows: List[Dict[str, object]] = []
    for step in [1000, 1500, 2000]:
        for candidate, fresh in [
            (f"candidate_s2_{step}", f"fresh_s2_{step}"),
            (f"candidate_stage1_{step}_sft_s2", f"fresh_stage1_{step}_sft_s2"),
        ]:
            if candidate not in by_label or fresh not in by_label:
                continue
            for metric, key in metric_keys.items():
                cand_val = summary_value(by_label[candidate], key)
                fresh_val = summary_value(by_label[fresh], key)
                if cand_val is None or fresh_val is None:
                    continue
                rows.append(
                    {
                        "candidate": candidate,
                        "fresh": fresh,
                        "metric": metric,
                        "candidate_value": cand_val,
                        "fresh_value": fresh_val,
                        "delta": cand_val - fresh_val,
                    }
                )
    return rows


def summarize_diag_file(path: Path, model: str, stage: str) -> Dict[str, object]:
    if not path.exists():
        return {"model": model, "stage": stage, "status": "missing", "path": str(path)}
    rows = read_csv(path)
    out: Dict[str, object] = {"model": model, "stage": stage, "status": "ok", "rows": len(rows), "path": str(path)}
    for col, prefix in [
        ("loss", "loss"),
        ("total_loss", "total_loss"),
        ("dpo_loss", "dpo_loss"),
        ("implicit_acc", "implicit_acc"),
        ("mse_w", "mse_w"),
        ("mse_l", "mse_l"),
        ("ref_mse_w", "ref_mse_w"),
        ("ref_mse_l", "ref_mse_l"),
        ("raw_win_gap", "raw_win_gap"),
        ("raw_lose_gap", "raw_lose_gap"),
        ("loser_dominant_ratio", "loser_dominant_ratio"),
        ("grad_norm", "grad_norm"),
        ("mse_w_over_ref_mse_w", "mse_w_over_ref_mse_w"),
        ("mse_l_over_ref_mse_l", "mse_l_over_ref_mse_l"),
        ("norm_win_gap", "norm_win_gap"),
        ("norm_lose_gap", "norm_lose_gap"),
        ("norm_lose_gap_clipped", "norm_lose_gap_clipped"),
        ("mask_win_gap", "mask_win_gap"),
        ("mask_lose_gap", "mask_lose_gap"),
        ("boundary_win_gap", "boundary_win_gap"),
        ("boundary_lose_gap", "boundary_lose_gap"),
        ("outside_win_gap", "outside_win_gap"),
        ("outside_lose_gap", "outside_lose_gap"),
    ]:
        if rows and col not in rows[0]:
            continue
        vals = [r.get(col) for r in rows]
        out[f"{prefix}_mean"] = finite_mean(vals)
        out[f"{prefix}_max"] = finite_max(vals)
        out[f"{prefix}_p95"] = finite_percentile(vals, 95)
        out[f"{prefix}_p99"] = finite_percentile(vals, 99)
        out[f"{prefix}_final"] = as_float(rows[-1].get(col)) if rows else None
    nan_inf = 0
    grad_vals = [v for v in (as_float(r.get("grad_norm")) for r in rows) if v is not None]
    out["grad_gt_10_count"] = int(sum(v > 10 for v in grad_vals))
    out["grad_gt_50_count"] = int(sum(v > 50 for v in grad_vals))
    out["grad_gt_100_count"] = int(sum(v > 100 for v in grad_vals))
    for row in rows:
        for value in row.values():
            text = str(value).strip().lower()
            if text in {"nan", "inf", "-inf"}:
                nan_inf += 1
    out["nan_inf_token_count"] = nan_inf
    winner_improvements = []
    loser_degradations = []
    for row in rows:
        win_gap = as_float(row.get("norm_win_gap"))
        lose_gap = as_float(row.get("norm_lose_gap_clipped"))
        if win_gap is None or lose_gap is None:
            continue
        winner_improvements.append(max(0.0, -win_gap))
        loser_degradations.append(max(0.0, lose_gap))
    out["winner_improvement_mean"] = float(np.mean(winner_improvements)) if winner_improvements else float("nan")
    out["loser_degradation_mean"] = float(np.mean(loser_degradations)) if loser_degradations else float("nan")
    out["winner_improvement_final"] = winner_improvements[-1] if winner_improvements else None
    out["loser_degradation_final"] = loser_degradations[-1] if loser_degradations else None
    return out


def _pair_model_names(pair_root: Path) -> List[str]:
    pair_config = pair_root / "pair_config.json"
    if pair_config.exists():
        try:
            payload = json.loads(pair_config.read_text(encoding="utf-8"))
            names = [str(model["name"]) for model in payload.get("models", []) if model.get("name")]
            if names:
                return names
        except Exception:
            pass
    return ["fresh_control_A", "inner2_candidate"]


def summarize_diagnostics(pair_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model in _pair_model_names(pair_root):
        for stage in ["stage1", "stage2"]:
            rows.append(summarize_diag_file(pair_root / model / stage / "dpo_diagnostics.csv", model, stage))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_id", required=True)
    parser.add_argument("--eval_root", required=True, type=Path)
    parser.add_argument("--pair_root", required=True, type=Path)
    parser.add_argument("--report_prefix", required=True, type=Path)
    args = parser.parse_args()

    summaries, per_video = collect_eval_rows(args.eval_root)
    write_csv(Path(f"{args.report_prefix}_davis50_paired_eval.csv"), summaries)
    write_csv(Path(f"{args.report_prefix}_summary_deltas.csv"), compare_summary_pairs(summaries))

    curve_rows: List[Dict[str, object]] = []
    for label_row in summaries:
        label = str(label_row.get("label", ""))
        row = {"label": label}
        row.update(label_row)
        curve_rows.append(row)
    write_csv(Path(f"{args.report_prefix}_checkpoint_curve.csv"), curve_rows)

    stats_rows: List[Dict[str, object]] = []
    bootstrap_payload: Dict[str, object] = {"pair_id": args.pair_id, "comparisons": []}
    for step in [1000, 1500, 2000]:
        rows, payload = compare_pair(per_video, f"candidate_s2_{step}", f"fresh_s2_{step}")
        payload["comparison"] = f"candidate_s2_{step}-fresh_s2_{step}"
        bootstrap_payload["comparisons"].append(payload)
        stats_rows.extend(rows)
        rows, payload = compare_pair(
            per_video,
            f"candidate_stage1_{step}_sft_s2",
            f"fresh_stage1_{step}_sft_s2",
        )
        payload["comparison"] = f"candidate_stage1_{step}_sft_s2-fresh_stage1_{step}_sft_s2"
        bootstrap_payload["comparisons"].append(payload)
        stats_rows.extend(rows)
    write_csv(Path(f"{args.report_prefix}_paired_statistics.csv"), stats_rows)
    Path(f"{args.report_prefix}_bootstrap.json").write_text(
        json.dumps(bootstrap_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    diag_rows = summarize_diagnostics(args.pair_root)
    write_csv(Path(f"{args.report_prefix}_dpo_diag_analysis.csv"), diag_rows)

    main_psnr = next(
        (
            r
            for r in stats_rows
            if r.get("candidate") == "candidate_s2_2000"
            and r.get("fresh") == "fresh_s2_2000"
            and r.get("metric") == "PSNR"
        ),
        {},
    )
    decision = "PAIR001_NEUTRAL"
    mean_delta = as_float(main_psnr.get("mean"))
    prob = as_float(main_psnr.get("prob_delta_gt_0"))
    win_rate = as_float(main_psnr.get("win_rate"))
    if mean_delta is not None and prob is not None and win_rate is not None:
        if mean_delta > 0.02 and prob >= 0.90 and win_rate >= 0.55:
            decision = "PAIR001_POSITIVE_PENDING_PERCEPTUAL_VISUAL_GATES"
        elif mean_delta > 0.0:
            decision = "PAIR001_PARETO_MIXED_OR_NEUTRAL_PENDING_GATES"
        elif mean_delta < -0.02:
            decision = "PAIR001_NEGATIVE_PENDING_GATES"

    lines = [
        "# Exp23 Pair001 DAVIS50 Paired Evaluation",
        "",
        f"pair_id: `{args.pair_id}`",
        f"eval_root: `{args.eval_root}`",
        f"pair_root: `{args.pair_root}`",
        f"main_comparison: `candidate_s2_2000 - fresh_s2_2000`",
        f"preliminary_metric_decision: `{decision}`",
        "",
        "This summary is generated from the fixed DAVIS50 frame-wise evaluator. "
        "Final pair status still requires checkpoint identity, perceptual/temporal gates, and visual review.",
        "",
        "## Output Files",
        "",
        f"- metrics: `{args.report_prefix}_davis50_paired_eval.csv`",
        f"- summary deltas: `{args.report_prefix}_summary_deltas.csv`",
        f"- checkpoint curve: `{args.report_prefix}_checkpoint_curve.csv`",
        f"- paired statistics: `{args.report_prefix}_paired_statistics.csv`",
        f"- bootstrap: `{args.report_prefix}_bootstrap.json`",
        f"- DPO diagnostics: `{args.report_prefix}_dpo_diag_analysis.csv`",
        "",
    ]
    Path(f"{args.report_prefix}_davis50_paired_eval.md").write_text("\n".join(lines), encoding="utf-8")
    Path(f"{args.report_prefix}_checkpoint_curve.md").write_text("\n".join(lines), encoding="utf-8")
    Path(f"{args.report_prefix}_paired_statistics.md").write_text("\n".join(lines), encoding="utf-8")
    Path(f"{args.report_prefix}_dpo_diag_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    Path(f"{args.report_prefix}_final_decision.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
