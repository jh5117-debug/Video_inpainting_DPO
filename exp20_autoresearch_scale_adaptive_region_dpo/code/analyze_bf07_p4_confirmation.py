#!/usr/bin/env python3
"""Analyze Exp20 BF07/P4 equal-step multi-seed confirmation.

This script is read-only with respect to checkpoints and evaluator outputs. It
collects the locked search-dev and shadow-dev metrics, runs paired bootstrap
statistics, summarizes DPO diagnostics, and writes the pre-registered promotion
decision reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METHOD_SPECS: dict[tuple[str, int], tuple[str, str, str]] = {
    ("P0", 20260619): ("EQ_P0", "1d8cd54758b73251", "EQ_P0"),
    ("P4", 20260619): ("EQ_P4", "edbea07bb785e769", "EQ_P4"),
    ("BF07", 20260619): ("EQ_BF07", "2bc98e58514fb1da", "EQ_BF07"),
    ("AD04", 20260619): ("EQ_AD04", "77a0ed002ad3955d", "EQ_AD04"),
    ("P0", 20260620): ("MSEQ_P0_s20260620", "2e233781ed5e1d4c", "MSEQ_P0_s20260620"),
    ("P4", 20260620): ("MSEQ_P4_s20260620", "c6668196e7108e8e", "MSEQ_P4_s20260620"),
    ("BF07", 20260620): ("MSEQ_BF07_s20260620", "b1a2af0e6583af5c", "MSEQ_BF07_s20260620"),
    ("P0", 20260621): ("MSEQ_P0_s20260621", "5f51b788f9bc6ba0", "MSEQ_P0_s20260621"),
    ("P4", 20260621): ("MSEQ_P4_s20260621", "46ae4bf5601388ea", "MSEQ_P4_s20260621"),
    ("BF07", 20260621): ("MSEQ_BF07_s20260621", "b0f4cdd5be0a0136", "MSEQ_BF07_s20260621"),
}

METRICS = {
    "psnr": "whole_video_psnr",
    "ssim": "whole_video_ssim",
    "lpips": "whole_video_lpips",
    "tc": "tc",
    "ewarp": "ewarp",
    "mask_psnr": "strict_mask_pixel_psnr",
    "boundary_psnr": "boundary_pixel_psnr",
}

SUMMARY_METRICS = {
    "psnr": "whole_video_psnr_mean",
    "ssim": "whole_video_ssim_mean",
    "lpips": "whole_video_lpips_mean",
    "vfid": "vfid",
    "tc": "tc_mean",
    "ewarp": "ewarp_mean",
    "mask_psnr": "strict_mask_pixel_psnr_mean",
    "boundary_psnr": "boundary_pixel_psnr_mean",
}

BASELINE_LABELS = {
    "SFT": "SFT48000_baseline",
    "Exp11-S1": "Exp11_outer_b075_S1_plus_SFT_S2",
    "Exp11-S2": "Exp11_outer_b075_S2",
}


def read_csv(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def finite_mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def finite_std(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0 if vals else float("nan")


def trial_dir(root: Path, method: str, seed: int) -> tuple[Path, str]:
    trial_id, hash_id, label = METHOD_SPECS[(method, seed)]
    return root / f"{trial_id}_{hash_id}", label


def eval_dir(root: Path, method: str, seed: int, split: str) -> Path:
    tdir, label = trial_dir(root, method, seed)
    if split == "search":
        return tdir / "eval_dev" / label
    if split == "shadow":
        return tdir / "eval_shadow" / f"{label}_shadow"
    raise ValueError(split)


def load_summary(eval_path: Path) -> dict[str, float]:
    rows = read_csv(eval_path / "metrics" / "backfill_summary.csv")
    if not rows:
        rows = read_csv(eval_path / "metrics" / "summary.csv")
    row = rows[0] if rows else {}
    return {metric: as_float(row.get(key, "")) for metric, key in SUMMARY_METRICS.items()}


def load_per_video(eval_path: Path) -> dict[str, dict[str, float]]:
    rows = read_csv(eval_path / "metrics" / "backfill_per_video_metrics.csv")
    if not rows:
        rows = read_csv(eval_path / "metrics" / "per_video_metrics.csv")
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        video = row.get("video", "")
        if not video:
            continue
        out[video] = {metric: as_float(row.get(key, "")) for metric, key in METRICS.items()}
    return out


def load_baseline_dirs(report_csv: Path) -> dict[str, Path]:
    dirs: dict[str, Path] = {}
    for row in read_csv(report_csv):
        label = row.get("model_label", "")
        for alias, expected in BASELINE_LABELS.items():
            if label == expected:
                dirs[alias] = Path(row.get("result_dir", ""))
    return dirs


def bootstrap_delta(
    cand: dict[str, float],
    base: dict[str, float],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    videos = sorted(set(cand) & set(base))
    deltas = np.array([cand[v] - base[v] for v in videos], dtype=np.float64)
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) == 0:
        return {
            "n_videos": 0,
            "mean_delta": float("nan"),
            "median_delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "prob_delta_gt_0": float("nan"),
            "per_video_win_rate": float("nan"),
            "sign_wins": 0,
            "sign_total": 0,
            "sign_test_one_sided_p": float("nan"),
            "leave_one_out_min": float("nan"),
            "leave_one_out_max": float("nan"),
        }
    rng = np.random.default_rng(seed)
    samples = rng.choice(deltas, size=(n_bootstrap, len(deltas)), replace=True).mean(axis=1)
    sign_wins = int(np.sum(deltas > 0))
    sign_total = int(np.sum(deltas != 0))
    sign_p = float(sum(math.comb(sign_total, k) for k in range(sign_wins, sign_total + 1)) / (2**sign_total)) if sign_total else float("nan")
    if len(deltas) > 1:
        loo = np.array([(np.sum(deltas) - deltas[i]) / (len(deltas) - 1) for i in range(len(deltas))])
        loo_min = float(np.min(loo))
        loo_max = float(np.max(loo))
    else:
        loo_min = loo_max = float(deltas[0])
    return {
        "n_videos": len(deltas),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "prob_delta_gt_0": float(np.mean(samples > 0)),
        "per_video_win_rate": float(np.mean(deltas > 0)),
        "sign_wins": sign_wins,
        "sign_total": sign_total,
        "sign_test_one_sided_p": sign_p,
        "leave_one_out_min": loo_min,
        "leave_one_out_max": loo_max,
    }


def average_per_video(per_video_by_seed: dict[int, dict[str, dict[str, float]]], metric: str) -> dict[str, float]:
    videos: set[str] = set()
    for rows in per_video_by_seed.values():
        videos.update(rows)
    out: dict[str, float] = {}
    for video in videos:
        out[video] = finite_mean([rows.get(video, {}).get(metric, float("nan")) for rows in per_video_by_seed.values()])
    return out


def paired_seed_win_count(summary: dict[tuple[str, str, int], dict[str, float]], split: str, cand: str, base: str) -> int:
    wins = 0
    for seed in (20260619, 20260620, 20260621):
        c = summary.get((split, cand, seed), {}).get("psnr", float("nan"))
        b = summary.get((split, base, seed), {}).get("psnr", float("nan"))
        if math.isfinite(c) and math.isfinite(b) and c > b:
            wins += 1
    return wins


def baseline_seed_win_count(summary: dict[tuple[str, str, int], dict[str, float]], baseline_summary: dict[tuple[str, str], dict[str, float]], split: str, cand: str, baseline: str) -> int:
    wins = 0
    b = baseline_summary.get((split, baseline), {}).get("psnr", float("nan"))
    for seed in (20260619, 20260620, 20260621):
        c = summary.get((split, cand, seed), {}).get("psnr", float("nan"))
        if math.isfinite(c) and math.isfinite(b) and c > b:
            wins += 1
    return wins


def summarize_diagnostics(trial_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in ("P0", "P4", "BF07"):
        for seed in (20260619, 20260620, 20260621):
            tdir, _ = trial_dir(trial_root, method, seed)
            diag = read_csv(tdir / "dpo_diagnostics.csv")
            if not diag:
                continue
            grad_vals = [as_float(r.get("grad_norm", "")) for r in diag]
            loser_vals = [as_float(r.get("loser_dominant_ratio", r.get("loser_degrade_ratio", ""))) for r in diag]
            win_imp = [as_float(r.get("winner_improvement_mean", r.get("_winner_improvement", ""))) for r in diag]
            loser_deg = [as_float(r.get("loser_degradation_mean", r.get("_loser_degradation", ""))) for r in diag]
            norm_win = [as_float(r.get("norm_win_gap", r.get("_pair_norm_win_gap", ""))) for r in diag]
            norm_lose = [as_float(r.get("norm_lose_gap", r.get("_pair_norm_lose_gap", ""))) for r in diag]
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "rows": len(diag),
                    "last_step": diag[-1].get("step", ""),
                    "mean_loser_degrade_ratio": finite_mean(loser_vals),
                    "last_loser_degrade_ratio": loser_vals[-1] if loser_vals else float("nan"),
                    "winner_improvement_mean": finite_mean(win_imp),
                    "loser_degradation_mean": finite_mean(loser_deg),
                    "norm_win_gap_mean": finite_mean(norm_win),
                    "norm_lose_gap_mean": finite_mean(norm_lose),
                    "max_grad_norm": max([v for v in grad_vals if math.isfinite(v)], default=float("nan")),
                    "mean_grad_norm": finite_mean(grad_vals),
                    "gradient_spike_count": sum(1 for v in grad_vals if math.isfinite(v) and v > 10.0),
                    "nan_inf_count": sum(1 for r in diag for v in r.values() if str(v).lower() in {"nan", "inf", "-inf"}),
                    "trial_dir": str(tdir),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-root", default="/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp20_autoresearch_scale_adaptive_region_dpo/trials")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--stats-seed", type=int, default=20260620)
    args = parser.parse_args()

    trial_root = Path(args.trial_root)
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    summary: dict[tuple[str, str, int], dict[str, float]] = {}
    per_video: dict[tuple[str, str, int], dict[str, dict[str, float]]] = {}
    aggregate_rows: list[dict[str, Any]] = []
    for split in ("search", "shadow"):
        for method in ("P0", "P4", "BF07"):
            for seed in (20260619, 20260620, 20260621):
                path = eval_dir(trial_root, method, seed, split)
                if not path.exists():
                    continue
                row = load_summary(path)
                summary[(split, method, seed)] = row
                per_video[(split, method, seed)] = load_per_video(path)
                aggregate_rows.append({"split": split, "method": method, "seed": seed, "eval_dir": str(path), **row})
        ad_path = eval_dir(trial_root, "AD04", 20260619, split)
        if ad_path.exists():
            row = load_summary(ad_path)
            summary[(split, "AD04", 20260619)] = row
            per_video[(split, "AD04", 20260619)] = load_per_video(ad_path)
            aggregate_rows.append({"split": split, "method": "AD04", "seed": 20260619, "eval_dir": str(ad_path), **row})

    baseline_dirs_by_split = {
        "search": load_baseline_dirs(reports / "exp20_dev_baselines.csv"),
        "shadow": load_baseline_dirs(reports / "exp20_shadow_dev_baselines.csv"),
    }
    baseline_summary: dict[tuple[str, str], dict[str, float]] = {}
    baseline_per_video: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for split, dirs in baseline_dirs_by_split.items():
        for alias, path in dirs.items():
            if not path.exists():
                continue
            baseline_summary[(split, alias)] = load_summary(path)
            baseline_per_video[(split, alias)] = load_per_video(path)
            aggregate_rows.append({"split": split, "method": alias, "seed": "baseline", "eval_dir": str(path), **baseline_summary[(split, alias)]})

    write_csv(reports / "exp20_bf07_p4_multiseed_results.csv", aggregate_rows)

    grouped_rows: list[dict[str, Any]] = []
    for split in ("search", "shadow"):
        for method in ("P0", "P4", "BF07"):
            vals = [summary.get((split, method, seed), {}).get("psnr", float("nan")) for seed in (20260619, 20260620, 20260621)]
            grouped_rows.append(
                {
                    "split": split,
                    "method": method,
                    "psnr_mean": finite_mean(vals),
                    "psnr_std": finite_std(vals),
                    "psnr_min": min([v for v in vals if math.isfinite(v)], default=float("nan")),
                    "psnr_max": max([v for v in vals if math.isfinite(v)], default=float("nan")),
                    "available_seeds": sum(1 for v in vals if math.isfinite(v)),
                }
            )

    stats_rows: list[dict[str, Any]] = []
    boot_json: dict[str, Any] = {}
    comparisons = [
        ("P4", "P0"),
        ("BF07", "P0"),
        ("BF07", "P4"),
        ("P4", "Exp11-S1"),
        ("BF07", "Exp11-S1"),
        ("P4", "Exp11-S2"),
        ("BF07", "Exp11-S2"),
    ]
    for split in ("search", "shadow"):
        for cand, base in comparisons:
            if base in {"P0", "P4", "BF07"}:
                cand_video = average_per_video({s: per_video[(split, cand, s)] for s in (20260619, 20260620, 20260621) if (split, cand, s) in per_video}, "psnr")
                base_video = average_per_video({s: per_video[(split, base, s)] for s in (20260619, 20260620, 20260621) if (split, base, s) in per_video}, "psnr")
                seed_wins = paired_seed_win_count(summary, split, cand, base)
            else:
                cand_video = average_per_video({s: per_video[(split, cand, s)] for s in (20260619, 20260620, 20260621) if (split, cand, s) in per_video}, "psnr")
                base_video = {v: vals.get("psnr", float("nan")) for v, vals in baseline_per_video.get((split, base), {}).items()}
                seed_wins = baseline_seed_win_count(summary, baseline_summary, split, cand, base)
            boot = bootstrap_delta(cand_video, base_video, n_bootstrap=args.bootstrap_samples, seed=args.stats_seed)
            row = {
                "split": split,
                "comparison": f"{cand}-{base}",
                "candidate": cand,
                "baseline": base,
                "three_seed_candidate_mean": finite_mean([summary.get((split, cand, s), {}).get("psnr", float("nan")) for s in (20260619, 20260620, 20260621)]),
                "three_seed_candidate_std": finite_std([summary.get((split, cand, s), {}).get("psnr", float("nan")) for s in (20260619, 20260620, 20260621)]),
                "seed_win_count": seed_wins,
                **boot,
            }
            stats_rows.append(row)
            boot_json[f"{split}:{cand}-{base}"] = row

    write_csv(reports / "exp20_bf07_p4_multiseed_statistics.csv", stats_rows)
    (reports / "exp20_bf07_p4_bootstrap.json").write_text(json.dumps(boot_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    diag_rows = summarize_diagnostics(trial_root)
    write_csv(reports / "exp20_bf07_p4_dpo_diagnostics.csv", diag_rows)

    diag_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in diag_rows:
        diag_by_method[str(row["method"])].append(row)

    def diag_mean(method: str, key: str) -> float:
        return finite_mean([as_float(r.get(key, "")) for r in diag_by_method.get(method, [])])

    def split_method_mean(split: str, method: str, metric: str) -> float:
        return finite_mean([summary.get((split, method, s), {}).get(metric, float("nan")) for s in (20260619, 20260620, 20260621)])

    def split_method_std(split: str, method: str, metric: str) -> float:
        return finite_std([summary.get((split, method, s), {}).get(metric, float("nan")) for s in (20260619, 20260620, 20260621)])

    stats_index = {(r["split"], r["comparison"]): r for r in stats_rows}
    decisions: list[dict[str, Any]] = []
    for cand in ("P4", "BF07"):
        search_psnr_delta = split_method_mean("search", cand, "psnr") - baseline_summary.get(("search", "Exp11-S1"), {}).get("psnr", float("nan"))
        shadow_psnr_delta = split_method_mean("shadow", cand, "psnr") - baseline_summary.get(("shadow", "Exp11-S1"), {}).get("psnr", float("nan"))
        shadow_boot = stats_index.get(("shadow", f"{cand}-Exp11-S1"), {})
        search_boot = stats_index.get(("search", f"{cand}-Exp11-S1"), {})
        lpips_delta = split_method_mean("shadow", cand, "lpips") - baseline_summary.get(("shadow", "Exp11-S1"), {}).get("lpips", float("nan"))
        vfid_delta = split_method_mean("shadow", cand, "vfid") - baseline_summary.get(("shadow", "Exp11-S1"), {}).get("vfid", float("nan"))
        tc_delta = split_method_mean("shadow", cand, "tc") - baseline_summary.get(("shadow", "Exp11-S1"), {}).get("tc", float("nan"))
        ewarp_delta = split_method_mean("shadow", cand, "ewarp") - baseline_summary.get(("shadow", "Exp11-S1"), {}).get("ewarp", float("nan"))
        mask_search = split_method_mean("search", cand, "mask_psnr") - baseline_summary.get(("search", "Exp11-S1"), {}).get("mask_psnr", float("nan"))
        mask_shadow = split_method_mean("shadow", cand, "mask_psnr") - baseline_summary.get(("shadow", "Exp11-S1"), {}).get("mask_psnr", float("nan"))
        boundary_search = split_method_mean("search", cand, "boundary_psnr") - baseline_summary.get(("search", "Exp11-S1"), {}).get("boundary_psnr", float("nan"))
        boundary_shadow = split_method_mean("shadow", cand, "boundary_psnr") - baseline_summary.get(("shadow", "Exp11-S1"), {}).get("boundary_psnr", float("nan"))
        loser_delta = diag_mean(cand, "mean_loser_degrade_ratio") - diag_mean("P0", "mean_loser_degrade_ratio")
        checks = {
            "search_psnr_delta_ge_0p03": search_psnr_delta >= 0.03,
            "shadow_psnr_delta_ge_0p02": shadow_psnr_delta >= 0.02,
            "search_seed_wins_ge_2": baseline_seed_win_count(summary, baseline_summary, "search", cand, "Exp11-S1") >= 2,
            "shadow_seed_wins_ge_2": baseline_seed_win_count(summary, baseline_summary, "shadow", cand, "Exp11-S1") >= 2,
            "shadow_per_video_win_rate_ge_0p55": as_float(shadow_boot.get("per_video_win_rate", "")) >= 0.55,
            "shadow_boot_prob_ge_0p95": as_float(shadow_boot.get("prob_delta_gt_0", "")) >= 0.95,
            "lpips_degrade_le_0p0003": lpips_delta <= 0.0003,
            "vfid_degrade_le_0p005": vfid_delta <= 0.005,
            "tc_drop_le_0p0002": tc_delta >= -0.0002,
            "ewarp_degrade_le_0p03": ewarp_delta <= 0.03,
            "mask_or_boundary_positive_both_splits": (mask_search > 0 and mask_shadow > 0) or (boundary_search > 0 and boundary_shadow > 0),
            "loser_degrade_not_worse_than_p0_plus_0p05": loser_delta <= 0.05,
            "leave_one_out_not_negative": as_float(shadow_boot.get("leave_one_out_min", "")) >= 0,
        }
        decisions.append(
            {
                "candidate": cand,
                "search_psnr_delta_vs_exp11_s1": search_psnr_delta,
                "shadow_psnr_delta_vs_exp11_s1": shadow_psnr_delta,
                "shadow_lpips_delta_vs_exp11_s1": lpips_delta,
                "shadow_vfid_delta_vs_exp11_s1": vfid_delta,
                "shadow_tc_delta_vs_exp11_s1": tc_delta,
                "shadow_ewarp_delta_vs_exp11_s1": ewarp_delta,
                "mask_search_delta": mask_search,
                "mask_shadow_delta": mask_shadow,
                "boundary_search_delta": boundary_search,
                "boundary_shadow_delta": boundary_shadow,
                "loser_degrade_delta_vs_p0": loser_delta,
                "stage1_gate_pre_visual": all(checks.values()),
                **{f"check_{k}": v for k, v in checks.items()},
            }
        )

    bf07_vs_p4_shadow = stats_index.get(("shadow", "BF07-P4"), {})
    bf07_replace_checks = {
        "search_mean_psnr_ge_p4_plus_0p01": split_method_mean("search", "BF07", "psnr") - split_method_mean("search", "P4", "psnr") >= 0.01,
        "shadow_mean_psnr_ge_p4_plus_0p01": split_method_mean("shadow", "BF07", "psnr") - split_method_mean("shadow", "P4", "psnr") >= 0.01,
        "seed_wins_ge_2": paired_seed_win_count(summary, "shadow", "BF07", "P4") >= 2,
        "shadow_boot_prob_ge_0p95": as_float(bf07_vs_p4_shadow.get("prob_delta_gt_0", "")) >= 0.95,
        "shadow_per_video_win_rate_ge_0p55": as_float(bf07_vs_p4_shadow.get("per_video_win_rate", "")) >= 0.55,
        "lpips_degrade_le_0p0002": split_method_mean("shadow", "BF07", "lpips") - split_method_mean("shadow", "P4", "lpips") <= 0.0002,
        "vfid_degrade_le_0p003": split_method_mean("shadow", "BF07", "vfid") - split_method_mean("shadow", "P4", "vfid") <= 0.003,
        "tc_drop_le_0p00015": split_method_mean("shadow", "BF07", "tc") - split_method_mean("shadow", "P4", "tc") >= -0.00015,
        "ewarp_not_worse": split_method_mean("shadow", "BF07", "ewarp") - split_method_mean("shadow", "P4", "ewarp") <= 0,
        "loser_degrade_not_worse_than_p4_plus_0p05": diag_mean("BF07", "mean_loser_degrade_ratio") - diag_mean("P4", "mean_loser_degrade_ratio") <= 0.05,
    }

    p4_gate = next((r for r in decisions if r["candidate"] == "P4"), {}).get("stage1_gate_pre_visual", False)
    bf07_gate = next((r for r in decisions if r["candidate"] == "BF07"), {}).get("stage1_gate_pre_visual", False)
    if bf07_gate and all(bf07_replace_checks.values()):
        final_decision = "BF07_SELECTED_PRE_VISUAL"
        enter_500 = True
    elif p4_gate:
        final_decision = "P4_SELECTED_PRE_VISUAL"
        enter_500 = True
    else:
        final_decision = "NO_CANDIDATE_PROMOTED_PRE_VISUAL"
        enter_500 = False

    decision_rows = decisions + [{"candidate": "BF07_REPLACES_P4", **{f"check_{k}": v for k, v in bf07_replace_checks.items()}, "stage1_gate_pre_visual": all(bf07_replace_checks.values())}]
    write_csv(reports / "exp20_multiseed_shadow_promotion_decision.csv", decision_rows)

    lines = [
        "# Exp20 BF07/P4 Multi-Seed Results",
        "",
        "| split | method | PSNR mean | PSNR std | available seeds |",
        "|---|---|---:|---:|---:|",
    ]
    for row in grouped_rows:
        lines.append(f"| {row['split']} | {row['method']} | {as_float(row['psnr_mean']):.6f} | {as_float(row['psnr_std']):.6f} | {row['available_seeds']} |")
    (reports / "exp20_bf07_p4_multiseed_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    stat_lines = [
        "# Exp20 BF07/P4 Paired Statistics",
        "",
        f"- bootstrap samples: `{args.bootstrap_samples}`",
        f"- bootstrap seed: `{args.stats_seed}`",
        "",
        "| split | comparison | mean delta | 95% CI | P(delta>0) | per-video win rate | seed wins | LOO min | LOO max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stats_rows:
        stat_lines.append(
            f"| {row['split']} | {row['comparison']} | {as_float(row['mean_delta']):.6f} | "
            f"[{as_float(row['ci_low']):.6f}, {as_float(row['ci_high']):.6f}] | "
            f"{as_float(row['prob_delta_gt_0']):.4f} | {as_float(row['per_video_win_rate']):.3f} | "
            f"{row['seed_win_count']}/3 | {as_float(row['leave_one_out_min']):.6f} | {as_float(row['leave_one_out_max']):.6f} |"
        )
    (reports / "exp20_bf07_p4_multiseed_statistics.md").write_text("\n".join(stat_lines) + "\n", encoding="utf-8")

    diag_lines = [
        "# Exp20 BF07/P4 DPO Diagnostics",
        "",
        "| method | loser degrade ratio | winner improvement | loser degradation | max grad norm | spike count | NaN/Inf |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("P0", "P4", "BF07"):
        diag_lines.append(
            f"| {method} | {diag_mean(method, 'mean_loser_degrade_ratio'):.6f} | "
            f"{diag_mean(method, 'winner_improvement_mean'):.6f} | {diag_mean(method, 'loser_degradation_mean'):.6f} | "
            f"{max([as_float(r.get('max_grad_norm', '')) for r in diag_by_method.get(method, [])], default=float('nan')):.6f} | "
            f"{sum(int(as_float(r.get('gradient_spike_count', 0))) for r in diag_by_method.get(method, []))} | "
            f"{sum(int(as_float(r.get('nan_inf_count', 0))) for r in diag_by_method.get(method, []))} |"
        )
    (reports / "exp20_bf07_p4_dpo_diagnostics.md").write_text("\n".join(diag_lines) + "\n", encoding="utf-8")

    decision_lines = [
        "# Exp20 Multi-Seed Shadow Promotion Decision",
        "",
        f"- pre-visual decision: `{final_decision}`",
        f"- enter 500-step gate before visual review: `{enter_500}`",
        "- visual-artifact checks are deliberately not auto-passed by this script; final decision must include visual review.",
        "",
        "| candidate | pre-visual gate | search delta vs Exp11-S1 | shadow delta vs Exp11-S1 | loser delta vs P0 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in decisions:
        decision_lines.append(
            f"| {row['candidate']} | {row['stage1_gate_pre_visual']} | "
            f"{as_float(row['search_psnr_delta_vs_exp11_s1']):.6f} | "
            f"{as_float(row['shadow_psnr_delta_vs_exp11_s1']):.6f} | "
            f"{as_float(row['loser_degrade_delta_vs_p0']):.6f} |"
        )
    decision_lines.extend(["", "## BF07 Replaces P4 Checks", ""])
    for key, value in bf07_replace_checks.items():
        decision_lines.append(f"- {key}: `{value}`")
    (reports / "exp20_multiseed_shadow_promotion_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    print(json.dumps({"final_decision_pre_visual": final_decision, "enter_500_pre_visual": enter_500}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
