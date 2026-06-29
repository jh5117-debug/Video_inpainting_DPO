#!/usr/bin/env python3
"""Build Exp44 Stage2-style MiniMax handoff manifests.

This script is intentionally metadata-only. It creates dataset-view manifests
for a future H20 runner without decoding videos, launching training, or updating
any model weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PREFERRED = {"train": 64, "search": 24, "shadow": 24}
MINIMUM = {"train": 32, "search": 16, "shadow": 16}
SPLITS = ("train", "search", "shadow")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return sha256_file(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_exists(path_value: str) -> bool:
    return bool(path_value) and Path(path_value).exists()


def scene_overlap(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
    groups = {
        split: {row.get("source_group", "") for row in rows if row.get("source_group")}
        for split, rows in rows_by_split.items()
    }
    return {
        "train_search": sorted(groups["train"] & groups["search"]),
        "train_shadow": sorted(groups["train"] & groups["shadow"]),
        "search_shadow": sorted(groups["search"] & groups["shadow"]),
    }


def common_fields(pair: dict[str, Any], view: str) -> dict[str, Any]:
    return {
        "dataset_view": view,
        "pair_id": pair["pair_id"],
        "split": pair["split"],
        "source_group": pair["source_group"],
        "source_id": pair["source_id"],
        "condition_path": pair["condition_path"],
        "mask_path": pair["mask_path"],
        "affected_map": "derive_abs_condition_minus_gt_winner",
        "frame_map": pair.get("sample_frame_map", ""),
        "scheduler_state": pair.get("scheduler_state", {}),
        "raw_output_primary": True,
        "hard_comp_used": False,
        "vor_eval_used": False,
        "training_run": False,
        "optimizer_step": False,
        "loss_regions": [
            "mask",
            "boundary",
            "affected",
            "outside_preservation",
        ],
        "source_branch": "research/exp44-pai-minimax-targeted-same-source-mining-20260629",
    }


def gt_distill_row(pair: dict[str, Any], index: int, training_unlocked: bool) -> dict[str, Any]:
    row = common_fields(pair, "gt_distillation")
    row.update(
        {
            "row_id": f"gt_{pair['split']}_{index:04d}_{pair['pair_id']}",
            "target_path": pair["gt_winner_path"],
            "target_type": "gt_background",
            "pseudo_success_path": pair.get("pseudo_success_path", ""),
            "pseudo_success_label": pair.get("pseudo_success_label", ""),
            "training_unlocked": training_unlocked,
            "preferred_first_h20_experiment": False,
        }
    )
    return row


def pseudo_success_row(
    pair: dict[str, Any], index: int, training_unlocked: bool
) -> dict[str, Any]:
    row = common_fields(pair, "pseudo_success_distillation")
    row.update(
        {
            "row_id": f"pseudo_{pair['split']}_{index:04d}_{pair['pair_id']}",
            "target_path": pair["pseudo_success_path"],
            "target_frames_dir": pair.get("pseudo_success_frames_dir", ""),
            "target_type": "visually_approved_minimax_pseudo_success",
            "pseudo_success_label": pair.get("pseudo_success_label", ""),
            "pseudo_success_visual_approved": pair.get("pseudo_success_label")
            in {"SUCCESS_CLEAN", "SUCCESS_USABLE"},
            "gt_background_path": pair["gt_winner_path"],
            "training_unlocked": training_unlocked,
            "preferred_first_h20_experiment": True,
        }
    )
    return row


def preference_row(
    pair: dict[str, Any],
    state: dict[str, Any] | None,
    index: int,
    training_unlocked: bool,
) -> dict[str, Any]:
    row = common_fields(pair, "same_source_preference")
    clean_success = pair.get("pseudo_success_label") == "SUCCESS_CLEAN"
    row.update(
        {
            "row_id": f"pref_{pair['split']}_{index:04d}_{pair['pair_id']}",
            "winner_path": pair["gt_winner_path"],
            "winner_type": "gt_background",
            "alternate_success_clean_winner_path": pair.get("pseudo_success_path", "")
            if clean_success
            else "",
            "loser_path": pair["failure_loser_path"],
            "loser_frames_dir": pair.get("failure_loser_frames_dir", ""),
            "loser_type": "same_source_minimax_failure_medium_hard",
            "failure_label": pair.get("failure_label", ""),
            "pseudo_success_path": pair.get("pseudo_success_path", ""),
            "pseudo_success_label": pair.get("pseudo_success_label", ""),
            "badnoise_state_id": state.get("state_id", "") if state else "",
            "badnoise_hard_state": state.get("hard_state", "") if state else "",
            "badnoise_gradient_proxy_norm": state.get("gradient_proxy_norm")
            if state
            else None,
            "badnoise_local_random_gradient_ratio": state.get(
                "local_random_gradient_ratio"
            )
            if state
            else None,
            "badnoise_outside_risk_vs_random_median": state.get(
                "outside_risk_vs_random_median"
            )
            if state
            else None,
            "badnoise_gradient_proxy_is_backprop_gradient": False,
            "training_unlocked": training_unlocked,
            "preferred_first_h20_experiment": False,
        }
    )
    return row


def validate_rows(rows: list[dict[str, Any]], required_paths: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts["rows"] += 1
        for key in required_paths:
            if path_exists(str(row.get(key, ""))):
                counts[f"{key}_exists"] += 1
            else:
                counts[f"{key}_missing"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--badnoise", required=True, type=Path)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--reports-dir", required=True, type=Path)
    args = parser.parse_args()

    pairs = read_jsonl(args.pairs)
    states = {row["pair_id"]: row for row in read_jsonl(args.badnoise)}
    pairs_by_split: dict[str, list[dict[str, Any]]] = {
        split: [row for row in pairs if row.get("split") == split] for split in SPLITS
    }
    counts = {split: len(rows) for split, rows in pairs_by_split.items()}
    meets_preferred = all(counts[split] >= PREFERRED[split] for split in SPLITS)
    meets_minimum = all(counts[split] >= MINIMUM[split] for split in SPLITS)
    training_unlocked = meets_minimum
    status = (
        "MINIMAX_STAGE2_DATA_HANDOFF_READY"
        if meets_minimum
        else "MINIMAX_STAGE2_DATA_HANDOFF_PARTIAL"
    )
    training_status = "TRAINING_UNLOCKED" if training_unlocked else "TRAINING_NOT_UNLOCKED"

    outputs: dict[str, dict[str, Any]] = {}
    all_view_rows: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for split in SPLITS:
        split_pairs = pairs_by_split[split]
        gt_rows = [gt_distill_row(pair, i, training_unlocked) for i, pair in enumerate(split_pairs)]
        pseudo_rows = [
            pseudo_success_row(pair, i, training_unlocked)
            for i, pair in enumerate(split_pairs)
            if pair.get("pseudo_success_label") in {"SUCCESS_CLEAN", "SUCCESS_USABLE"}
        ]
        pref_rows = [
            preference_row(pair, states.get(pair["pair_id"]), i, training_unlocked)
            for i, pair in enumerate(split_pairs)
        ]
        all_view_rows["gt_distillation"][split] = gt_rows
        all_view_rows["pseudo_success_distillation"][split] = pseudo_rows
        all_view_rows["same_source_preference"][split] = pref_rows

    manifest_names = {
        "gt_distillation": "exp44_stage2_gt_distill",
        "pseudo_success_distillation": "exp44_stage2_pseudo_success",
        "same_source_preference": "exp44_stage2_same_source_preference",
    }
    for view, split_rows in all_view_rows.items():
        for split, rows in split_rows.items():
            path = args.manifest_dir / f"{manifest_names[view]}_{split}.jsonl"
            sha = write_jsonl(path, rows)
            outputs[f"{view}_{split}"] = {
                "path": str(path),
                "rows": len(rows),
                "sha256": sha,
            }

    overlap = scene_overlap(pairs_by_split)
    path_checks = {
        "gt_distillation": {
            split: validate_rows(rows, ["condition_path", "target_path", "mask_path"])
            for split, rows in all_view_rows["gt_distillation"].items()
        },
        "pseudo_success_distillation": {
            split: validate_rows(rows, ["condition_path", "target_path", "mask_path"])
            for split, rows in all_view_rows["pseudo_success_distillation"].items()
        },
        "same_source_preference": {
            split: validate_rows(rows, ["condition_path", "winner_path", "loser_path", "mask_path"])
            for split, rows in all_view_rows["same_source_preference"].items()
        },
    }
    usable_state_count = sum(
        1
        for pair in pairs
        if states.get(pair["pair_id"], {}).get("local_random_gradient_ratio", 0) >= 1.5
    )
    nas_mount_available = Path("/mnt/nas").exists()
    summary = {
        "status": status,
        "training_status": training_status,
        "training_unlocked": training_unlocked,
        "reason": "split counts are below minimum train32/search16/shadow16"
        if not training_unlocked
        else "minimum split counts met",
        "preferred_counts": PREFERRED,
        "minimum_counts": MINIMUM,
        "pair_counts": counts,
        "total_pairs": len(pairs),
        "scene_group_overlap": overlap,
        "outputs": outputs,
        "path_checks": {
            view: {
                split: dict(counter) for split, counter in split_counts.items()
            }
            for view, split_counts in path_checks.items()
        },
        "path_check_context": {
            "current_session_has_mnt_nas": nas_mount_available,
            "interpretation": (
                "Path existence is not validated in this session because /mnt/nas is unavailable; H20 must verify paths before running."
                if not nas_mount_available
                else "Path existence was checked in this session."
            ),
        },
        "badnoise_states_available": len(states),
        "badnoise_states_matched_to_pairs": sum(1 for pair in pairs if pair["pair_id"] in states),
        "usable_state_count_ratio_ge_1p5": usable_state_count,
        "first_h20_experiment": "pseudo-success SFT 30-step",
        "do_not_start_first": "GT-only SFT",
        "precision_recommendation": "fp32 loss reduction; fp16/bf16 only after one-batch preflight confirms finite loss",
        "raw_output_primary": True,
        "hard_comp_used": False,
        "vor_eval_used": False,
        "training_run": False,
        "optimizer_step": False,
    }

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.reports_dir / "exp44_stage2_dataset_handoff_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = args.reports_dir / "exp44_stage2_dataset_handoff.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["view", "split", "rows", "sha256", "path"],
        )
        writer.writeheader()
        for key, payload in outputs.items():
            view, split = key.rsplit("_", 1)
            writer.writerow(
                {
                    "view": view,
                    "split": split,
                    "rows": payload["rows"],
                    "sha256": payload["sha256"],
                    "path": payload["path"],
                }
            )

    path_validation_note = (
        "Path existence could not be validated in this Codex session because "
        "`/mnt/nas` is not mounted. H20 must verify every manifest path before "
        "running the dataloader or runner."
        if not nas_mount_available
        else "Manifest paths were checked for existence in this session."
    )

    report_path = args.reports_dir / "exp44_stage2_dataset_handoff.md"
    report_path.write_text(
        "\n".join(
            [
                "# Exp44 Stage2 Dataset Handoff",
                "",
                f"- Status: `{status}`",
                f"- Training status: `{training_status}`",
                f"- Pair counts train/search/shadow: `{counts['train']}` / `{counts['search']}` / `{counts['shadow']}`",
                f"- Minimum train/search/shadow: `{MINIMUM['train']}` / `{MINIMUM['search']}` / `{MINIMUM['shadow']}`",
                f"- Preferred train/search/shadow: `{PREFERRED['train']}` / `{PREFERRED['search']}` / `{PREFERRED['shadow']}`",
                f"- Scene overlap train/search: `{len(overlap['train_search'])}`",
                f"- Scene overlap train/shadow: `{len(overlap['train_shadow'])}`",
                f"- Scene overlap search/shadow: `{len(overlap['search_shadow'])}`",
                f"- Bad-noise states matched to pairs: `{summary['badnoise_states_matched_to_pairs']}`",
                f"- Usable local/random ratio states: `{usable_state_count}`",
                "",
                "## View Counts",
                "",
                "| view | train | search | shadow |",
                "| --- | ---: | ---: | ---: |",
                f"| GT distillation | {outputs['gt_distillation_train']['rows']} | {outputs['gt_distillation_search']['rows']} | {outputs['gt_distillation_shadow']['rows']} |",
                f"| pseudo-success distillation | {outputs['pseudo_success_distillation_train']['rows']} | {outputs['pseudo_success_distillation_search']['rows']} | {outputs['pseudo_success_distillation_shadow']['rows']} |",
                f"| same-source preference | {outputs['same_source_preference_train']['rows']} | {outputs['same_source_preference_search']['rows']} | {outputs['same_source_preference_shadow']['rows']} |",
                "",
                "## Interpretation",
                "",
                "The handoff is partial because the same-source pair pool passes the Exp44",
                "minimum pair gate but does not reach the dataset minimum",
                "train32/search16/shadow16. H20 may use these manifests for a",
                "bounded debug/preflight run, but this package must not be described as",
                "formal training-unlocked evidence.",
                "",
                "The first H20 experiment should be pseudo-success SFT 30-step. Do not",
                "start with GT-only SFT first.",
                "",
                path_validation_note,
                "",
                "No training, optimizer step, VOR-Eval use, hard comp, or model update",
                "occurred in this milestone.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    handoff_path = args.reports_dir / "exp44_h20_handoff_instructions.md"
    checksum_lines = [
        f"- `{payload['path']}`: `{payload['sha256']}` ({payload['rows']} rows)"
        for payload in outputs.values()
    ]
    handoff_path.write_text(
        "\n".join(
            [
                "# Exp44 H20 Handoff Instructions",
                "",
                "- Source branch: `research/exp44-pai-minimax-targeted-same-source-mining-20260629`",
                "- Pull/checkout this branch in an isolated H20 worktree.",
                "- Pull only Exp44 manifests, reports, and helper metadata listed below.",
                "- Do not modify PAI Exp44 outputs from H20.",
                "- Do not start GT-only SFT first.",
                "- First H20 experiment: pseudo-success SFT 30-step using the",
                "  pseudo-success distillation train/search/shadow manifests.",
                "- Precision recommendation: fp32 loss reduction; use fp16/bf16 only",
                "  after a one-batch preflight verifies finite latents, finite flow",
                "  target `epsilon - z0`, finite loss, and no NaN/Inf.",
                "- Raw output remains primary; no hard comp may be used to claim gain.",
                "- This handoff is partial and marked `TRAINING_NOT_UNLOCKED` because",
                "  train/search/shadow counts are below 32/16/16.",
                "- This Codex session did not have `/mnt/nas` mounted, so H20 must",
                "  verify that every absolute condition/winner/mask/output path exists",
                "  before launching any dataloader or runner.",
                "",
                "## Manifest Checksums",
                "",
                *checksum_lines,
                "",
                "## H20 Must Also Pull",
                "",
                "- `reports/exp44_stage2_dataset_handoff.md`",
                "- `reports/exp44_stage2_dataset_handoff.csv`",
                "- `reports/exp44_stage2_dataset_handoff_summary.json`",
                "- `reports/exp44_badnoise_v4_summary.json`",
                "- `reports/exp44_badnoise_v4_states.csv`",
                "- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_badnoise_v4_states.jsonl`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
