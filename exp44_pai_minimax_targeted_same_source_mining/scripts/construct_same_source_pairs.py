#!/usr/bin/env python3
"""Construct Exp44 same-source MiniMax success/failure pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


PAIRABLE_SPLITS = {
    "BLENDER_DESERT004": "train",
    "BLENDER_FOREST026": "train",
    "BLENDER_GRASS001": "train",
    "REAL_ENV080_00003": "train",
    "REAL_ENV105_00002": "train",
    "REAL_ENV105_00004": "train",
    "BLENDER_MOUNTAIN002": "search",
    "REAL_ENV059_00001": "search",
    "BLENDER_SCHOOL004": "shadow",
    "REAL_ENV097_00001": "shadow",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--success-manifest", required=True)
    parser.add_argument("--failure-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--max-pairs-per-group", type=int, default=4)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(data, encoding="utf-8")
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def f(row: dict[str, object], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def success_sort_key(row: dict[str, object]) -> tuple[int, float, float, float]:
    label_rank = 0 if row.get("visual_label") == "SUCCESS_CLEAN" else 1
    return (label_rank, -f(row, "full_psnr"), -f(row, "boundary_psnr"), f(row, "outside_mae"))


def failure_sort_key(row: dict[str, object]) -> tuple[float, float, float]:
    # Prefer visible local defects, while avoiding outside-damaged failures.
    return (f(row, "mask_psnr"), -f(row, "outside_psnr"), f(row, "temporal_diff_mae"))


def make_pair(group: str, split: str, idx: int, success: dict[str, object], failure: dict[str, object]) -> dict[str, object]:
    pair_id = f"{group}__pair{idx:03d}__s{success.get('seed_index')}__f{failure.get('seed_index')}"
    gt_winner_path = failure.get("winner_path") or success.get("winner_path", "")
    condition_path = failure.get("condition_path") or success.get("condition_path", "")
    mask_path = failure.get("mask_path") or success.get("mask_path", "")
    sample_id = failure.get("sample_id") or success.get("sample_id", "")
    return {
        "pair_id": pair_id,
        "split": split,
        "source_group": group,
        "source_id": sample_id,
        "condition_path": condition_path,
        "gt_winner_path": gt_winner_path,
        "pseudo_success_path": success.get("raw_output_mp4", ""),
        "pseudo_success_frames_dir": success.get("frames_dir", ""),
        "failure_loser_path": failure.get("raw_output_mp4", ""),
        "failure_loser_frames_dir": failure.get("frames_dir", ""),
        "mask_path": mask_path,
        "affected_map_path": "",
        "success_seed": success.get("seed", ""),
        "failure_seed": failure.get("seed", ""),
        "success_seed_index": success.get("seed_index", ""),
        "failure_seed_index": failure.get("seed_index", ""),
        "noise_state_success": {"seed": success.get("seed", ""), "seed_index": success.get("seed_index", "")},
        "noise_state_failure": {"seed": failure.get("seed", ""), "seed_index": failure.get("seed_index", "")},
        "scheduler_state": {
            "scheduler": "UniPCMultistepScheduler",
            "num_inference_steps": failure.get("num_inference_steps", success.get("num_inference_steps", 12)),
            "iterations": failure.get("iterations", success.get("iterations", 6)),
            "dtype": failure.get("dtype", success.get("dtype", "float16")),
        },
        "sample_frame_map": "17 frames inherited from materialized source; exact frame order unchanged",
        "winner_type": "gt_background",
        "pseudo_success_label": success.get("visual_label", ""),
        "failure_label": failure.get("visual_label", ""),
        "classification": "SAME_SOURCE_GT_WINNER_VS_MINIMAX_MEDIUM_HARD_FAILURE",
        "pairing_rule": "same source group only; GT winner preferred for DPO; pseudo-success retained for Stage2 distillation",
        "success_candidate_id": success.get("candidate_id", ""),
        "failure_candidate_id": failure.get("candidate_id", ""),
        "success_review_sheet": success.get("review_sheet", ""),
        "success_temporal_strip_16": success.get("temporal_strip_16", ""),
        "failure_review_sheet": failure.get("review_sheet", ""),
        "failure_temporal_strip_16": failure.get("temporal_strip_16", ""),
        "success_full_psnr": success.get("full_psnr", ""),
        "success_mask_psnr": success.get("mask_psnr", ""),
        "success_boundary_psnr": success.get("boundary_psnr", ""),
        "success_outside_psnr": success.get("outside_psnr", ""),
        "failure_full_psnr": failure.get("full_psnr", ""),
        "failure_mask_psnr": failure.get("mask_psnr", ""),
        "failure_boundary_psnr": failure.get("boundary_psnr", ""),
        "failure_outside_psnr": failure.get("outside_psnr", ""),
        "vor_eval_used": False,
        "hard_comp_used": False,
        "training_run": False,
        "optimizer_step": False,
    }


def main() -> None:
    args = parse_args()
    successes = read_jsonl(Path(args.success_manifest))
    failures = read_jsonl(Path(args.failure_manifest))
    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    success_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    failure_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in successes:
        success_by_group[str(row.get("scene_group", ""))].append(row)
    for row in failures:
        failure_by_group[str(row.get("scene_group", ""))].append(row)

    pairs: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    for group in sorted(set(success_by_group) & set(failure_by_group)):
        split = PAIRABLE_SPLITS.get(group)
        if split is None:
            continue
        group_success = sorted(success_by_group[group], key=success_sort_key)
        group_failure = sorted(failure_by_group[group], key=failure_sort_key)
        group_pair_count = min(len(group_success) * len(group_failure), args.max_pairs_per_group)
        made = 0
        used_combos: set[tuple[str, str]] = set()
        for offset in range(max(len(group_success), len(group_failure), args.max_pairs_per_group)):
            if made >= group_pair_count:
                break
            success = group_success[offset % len(group_success)]
            failure = group_failure[offset % len(group_failure)]
            combo = (str(success.get("candidate_id", "")), str(failure.get("candidate_id", "")))
            if combo in used_combos:
                continue
            used_combos.add(combo)
            pairs.append(make_pair(group, split, made, success, failure))
            made += 1
        group_rows.append(
            {
                "scene_group": group,
                "split": split,
                "success_usable": len(group_success),
                "failure_medium_hard": len(group_failure),
                "pairs_built": made,
                "max_pairs_per_group": args.max_pairs_per_group,
            }
        )

    split_rows = {
        "train": [row for row in pairs if row["split"] == "train"],
        "search": [row for row in pairs if row["split"] == "search"],
        "shadow": [row for row in pairs if row["split"] == "shadow"],
    }
    hashes = {
        "exp44_same_source_pairs_all.jsonl": write_jsonl(out_dir / "exp44_same_source_pairs_all.jsonl", pairs),
        "exp44_same_source_pairs_train_candidates.jsonl": write_jsonl(out_dir / "exp44_same_source_pairs_train_candidates.jsonl", split_rows["train"]),
        "exp44_same_source_pairs_search_candidates.jsonl": write_jsonl(out_dir / "exp44_same_source_pairs_search_candidates.jsonl", split_rows["search"]),
        "exp44_same_source_pairs_shadow_candidates.jsonl": write_jsonl(out_dir / "exp44_same_source_pairs_shadow_candidates.jsonl", split_rows["shadow"]),
    }

    report_csv = reports_dir / "exp44_same_source_pair_construction.csv"
    pair_fields = [
        "pair_id",
        "split",
        "source_group",
        "source_id",
        "success_candidate_id",
        "failure_candidate_id",
        "pseudo_success_label",
        "failure_label",
        "winner_type",
        "success_full_psnr",
        "success_mask_psnr",
        "failure_full_psnr",
        "failure_mask_psnr",
        "condition_path",
        "gt_winner_path",
        "pseudo_success_path",
        "failure_loser_path",
        "mask_path",
    ]
    with report_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields)
        writer.writeheader()
        for row in pairs:
            writer.writerow({key: row.get(key, "") for key in pair_fields})

    group_csv = reports_dir / "exp44_same_source_pair_group_yield.csv"
    with group_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scene_group", "split", "success_usable", "failure_medium_hard", "pairs_built", "max_pairs_per_group"])
        writer.writeheader()
        writer.writerows(group_rows)

    split_counts = Counter(row["split"] for row in pairs)
    split_groups = defaultdict(set)
    for row in pairs:
        split_groups[str(row["split"])].add(str(row["source_group"]))
    overlap_ok = not (split_groups["train"] & split_groups["search"] or split_groups["train"] & split_groups["shadow"] or split_groups["search"] & split_groups["shadow"])
    status = "MINIMAX_SAME_SOURCE_PAIR_GATE_PASSED" if len(pairs) >= 24 and overlap_ok else "MINIMAX_SAME_SOURCE_PAIR_YIELD_INSUFFICIENT"
    summary = {
        "status": status,
        "same_source_pair_count": len(pairs),
        "target_pair_count": 48,
        "minimum_pair_gate": 24,
        "max_pairs_per_group": args.max_pairs_per_group,
        "split_counts": dict(split_counts),
        "split_groups": {key: sorted(value) for key, value in split_groups.items()},
        "split_group_overlap_ok": overlap_ok,
        "group_rows": group_rows,
        "manifest_hashes": hashes,
        "training_run": False,
        "optimizer_step": False,
        "vor_eval_used": False,
        "hard_comp_used": False,
        "notes": [
            "Pairs use only same source groups.",
            "GT background is the primary DPO winner; pseudo-success is retained only for Stage2 distillation fields.",
            "No training, optimizer step, bad-noise mining, or Stage2 handoff is performed in this milestone.",
        ],
    }
    summary_path = reports_dir / "exp44_same_source_pair_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    md_path = reports_dir / "exp44_same_source_pair_construction.md"
    md_path.write_text(
        "\n".join(
            [
                "# Exp44 Same-Source Pair Construction",
                "",
                f"- Status: {status}",
                f"- Usable same-source pairs: {len(pairs)}",
                f"- Minimum gate: 24",
                f"- Target: 48",
                f"- Max pairs per group: {args.max_pairs_per_group}",
                f"- Split counts: train={split_counts.get('train', 0)}, search={split_counts.get('search', 0)}, shadow={split_counts.get('shadow', 0)}",
                f"- Split group overlap ok: {overlap_ok}",
                "",
                "## Pairing Rule",
                "",
                "Winner is the GT background path for DPO preference construction. The MiniMax successful-removal output is preserved as `pseudo_success_path` for possible Stage2-style distillation, but it is not treated as GT.",
                "",
                "Loser is a same-source MiniMax raw output labeled `FAILURE_MEDIUM_HARD` after visual relabeling. Cross-source pairing is not used.",
                "",
                "## Safety",
                "",
                "No training, optimizer step, bad-noise scan, Stage2 handoff, hard comp, or VOR-Eval use occurred in this milestone.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
