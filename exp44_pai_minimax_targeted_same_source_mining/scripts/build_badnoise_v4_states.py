#!/usr/bin/env python3
"""Build Exp44 MiniMax bad-noise v4 state metadata from same-source pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--failure-pool", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--reports-dir", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> str:
    data = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def f(row: dict[str, object], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def mse_from_psnr(psnr: float) -> float:
    return 10 ** (-psnr / 10.0)


def residuals_from_candidate(row: dict[str, object], prefix: str = "") -> dict[str, float]:
    key = lambda name: f"{prefix}{name}" if prefix else name
    mask = mse_from_psnr(f(row, key("mask_psnr")))
    boundary = mse_from_psnr(f(row, key("boundary_psnr")))
    outside = mse_from_psnr(f(row, key("outside_psnr")))
    full = mse_from_psnr(f(row, key("full_psnr")))
    local = (mask + boundary) / 2.0
    return {
        "full_residual": full,
        "mask_residual": mask,
        "boundary_residual": boundary,
        "affected_residual": local,
        "outside_residual": outside,
        "local_outside_ratio": local / max(outside, 1e-12),
    }


def classify_h_state(pair: dict[str, object], loser: dict[str, float], winner: dict[str, float], random_local_outside_median: float) -> str:
    margin = loser["affected_residual"] - winner["affected_residual"]
    temporal = f(pair, "failure_temporal_diff_mae")
    ratio = loser["local_outside_ratio"] / max(random_local_outside_median, 1e-12)
    if temporal >= 1.8:
        return "H4_temporal_hard"
    if margin <= 0:
        return "H2_preference_violation"
    if winner["outside_residual"] <= loser["outside_residual"] and ratio >= 1.5:
        return "H3_winner_safe"
    return "H1_local_failure_hard"


def main() -> None:
    args = parse_args()
    pairs = read_jsonl(Path(args.pairs))
    failure_pool = read_jsonl(Path(args.failure_pool))
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    random_ratios = [residuals_from_candidate(row)["local_outside_ratio"] for row in failure_pool]
    random_outside = [residuals_from_candidate(row)["outside_residual"] for row in failure_pool]
    random_local_outside_median = statistics.median(random_ratios)
    random_outside_median = statistics.median(random_outside)

    states: list[dict[str, object]] = []
    for pair in pairs:
        loser = residuals_from_candidate(pair, "failure_")
        winner = residuals_from_candidate(pair, "success_")
        preference_margin = loser["affected_residual"] - winner["affected_residual"]
        gradient_proxy = max(preference_margin, 0.0) * loser["local_outside_ratio"]
        h_state = classify_h_state(pair, loser, winner, random_local_outside_median)
        state = {
            "state_id": f"{pair['pair_id']}__{h_state}",
            "pair_id": pair["pair_id"],
            "split": pair["split"],
            "source_group": pair["source_group"],
            "source_id": pair["source_id"],
            "hard_state": h_state,
            "condition_latent_identity": {
                "condition_path": pair["condition_path"],
                "frame_map": pair["sample_frame_map"],
            },
            "winner_target_identity": {
                "gt_winner_path": pair["gt_winner_path"],
                "pseudo_success_path": pair["pseudo_success_path"],
                "winner_type": pair["winner_type"],
            },
            "loser_failure_identity": {
                "failure_loser_path": pair["failure_loser_path"],
                "failure_loser_frames_dir": pair["failure_loser_frames_dir"],
                "failure_candidate_id": pair["failure_candidate_id"],
            },
            "success_seed_noise": pair["noise_state_success"],
            "failure_seed_noise": pair["noise_state_failure"],
            "scheduler_state": pair["scheduler_state"],
            "timestep_metadata": {
                "source": "official MiniMax inference schedule metadata",
                "num_inference_steps": pair["scheduler_state"]["num_inference_steps"],
                "iterations": pair["scheduler_state"]["iterations"],
                "selected_training_timestep": "to_be_sampled_by_future_runner_from_this state seed",
            },
            "flow_target_metadata": {
                "target": "epsilon_minus_z0",
                "winner_target": "gt_background",
                "loser_target": "same_source_minimax_failure",
            },
            "regions": {
                "mask_path": pair["mask_path"],
                "boundary": "derive from mask dilation/erosion in future runner",
                "affected_region": "derive from abs(condition - gt_winner) when materialized",
                "outside_region": "mask complement",
            },
            "winner_residual": winner,
            "loser_residual": loser,
            "preference_margin": preference_margin,
            "gradient_proxy_norm": gradient_proxy,
            "local_random_gradient_ratio": loser["local_outside_ratio"] / max(random_local_outside_median, 1e-12),
            "outside_risk_vs_random_median": loser["outside_residual"] / max(random_outside_median, 1e-12),
            "winner_risk": winner["outside_residual"] + winner["boundary_residual"],
            "failure_likelihood": min(1.0, max(0.0, loser["local_outside_ratio"] / (loser["local_outside_ratio"] + 10.0))),
            "temporal_flicker_residual": f(pair, "failure_temporal_diff_mae"),
            "success_review_sheet": pair["success_review_sheet"],
            "failure_review_sheet": pair["failure_review_sheet"],
            "success_temporal_strip_16": pair["success_temporal_strip_16"],
            "failure_temporal_strip_16": pair["failure_temporal_strip_16"],
            "training_run": False,
            "optimizer_step": False,
            "vor_eval_used": False,
            "hard_comp_used": False,
            "gradient_proxy_is_backprop_gradient": False,
        }
        states.append(state)

    ratios = [float(row["local_random_gradient_ratio"]) for row in states]
    outside_risks = [float(row["outside_risk_vs_random_median"]) for row in states]
    winner_risks = [float(row["winner_risk"]) for row in states]
    usable_h_states = [
        row
        for row in states
        if float(row["local_random_gradient_ratio"]) >= 1.5
        and float(row["outside_risk_vs_random_median"]) <= 1.2
        and float(row["winner_risk"]) <= statistics.median(winner_risks) * 1.5
    ]
    status = "MINIMAX_BADNOISE_V4_READY" if len(usable_h_states) >= 24 else "MINIMAX_BADNOISE_V4_WEAK"
    manifest_hash = write_jsonl(Path(args.out_manifest), states)

    csv_path = reports_dir / "exp44_badnoise_v4_states.csv"
    fieldnames = [
        "state_id",
        "pair_id",
        "split",
        "source_group",
        "hard_state",
        "preference_margin",
        "gradient_proxy_norm",
        "local_random_gradient_ratio",
        "outside_risk_vs_random_median",
        "winner_risk",
        "temporal_flicker_residual",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in states:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    summary = {
        "status": status,
        "state_count": len(states),
        "usable_h_state_count": len(usable_h_states),
        "minimum_h_state_gate": 24,
        "random_local_outside_median": random_local_outside_median,
        "hard_state_local_random_ratio_median": statistics.median(ratios),
        "hard_state_local_random_ratio_mean": statistics.mean(ratios),
        "outside_risk_median": statistics.median(outside_risks),
        "outside_risk_mean": statistics.mean(outside_risks),
        "winner_risk_median": statistics.median(winner_risks),
        "hard_state_counts": dict(Counter(str(row["hard_state"]) for row in states)),
        "manifest": str(args.out_manifest),
        "manifest_sha256": manifest_hash,
        "training_run": False,
        "optimizer_step": False,
        "vor_eval_used": False,
        "hard_comp_used": False,
        "notes": [
            "Residuals and gradient_proxy_norm are CPU-side metrics proxies, not backprop gradients.",
            "State metadata is complete enough for future runner materialization but no future runner was launched here.",
            "No training, optimizer step, VOR-Eval use, hard comp, or H20 modification occurred.",
        ],
    }
    summary_path = reports_dir / "exp44_badnoise_v4_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    md_path = reports_dir / "exp44_badnoise_v4_states.md"
    md_path.write_text(
        "\n".join(
            [
                "# Exp44 Bad-Noise V4 States",
                "",
                f"- Status: {status}",
                f"- States: {len(states)}",
                f"- Usable H states: {len(usable_h_states)}",
                f"- Minimum gate: 24",
                f"- Median local/random gradient-proxy ratio: {statistics.median(ratios):.6f}",
                f"- Median outside risk vs random: {statistics.median(outside_risks):.6f}",
                "",
                "## Interpretation",
                "",
                "The state pool is MiniMax-native and same-source: each row keeps the success seed/noise, failure seed/noise, scheduler metadata, GT winner identity, pseudo-success identity, loser identity, mask path, and residual proxy diagnostics.",
                "",
                "The gradient proxy is derived from local and outside residuals, not from an optimizer backward pass. This milestone builds data/state metadata only.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
