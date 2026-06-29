#!/usr/bin/env python3
"""Build Exp45 formal Stage2 handoff splits from same-source candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


MINIMUM = {"train": 32, "search": 16, "shadow": 16}
PREFERRED = {"train": 64, "search": 24, "shadow": 24}
SPLITS = ("train", "search", "shadow")

SPLIT_GROUPS = {
    "train": {
        "BLENDER_DESERT004",
        "BLENDER_FOREST026",
        "BLENDER_GRASS001",
        "REAL_ENV080_00003",
        "REAL_ENV105_00002",
        "REAL_ENV105_00004",
    },
    "search": {
        "BLENDER_MOUNTAIN002",
        "REAL_ENV059_00001",
        "REAL_ENV068_00002",
    },
    "shadow": {
        "BLENDER_SCHOOL004",
        "REAL_ENV097_00001",
        "REAL_ENV105_00001",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(data, encoding="utf-8")
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def f(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def group_of(row: dict[str, Any]) -> str:
    return str(row.get("scene_group") or row.get("source_group") or str(row.get("sample_id", "")).rsplit("_", 1)[0])


def row_key(row: dict[str, Any]) -> str:
    return str(row.get("candidate_id") or row.get("raw_output_mp4") or row.get("frames_dir"))


def split_for_group(group: str) -> str | None:
    for split, groups in SPLIT_GROUPS.items():
        if group in groups:
            return split
    return None


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = row_key(row)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def success_sort(row: dict[str, Any]) -> tuple[int, float, float, float]:
    label = row.get("visual_label")
    rank = 0 if label == "SUCCESS_CLEAN" else 1
    return (rank, -f(row, "full_psnr"), -f(row, "boundary_psnr"), f(row, "outside_mae"))


def failure_sort(row: dict[str, Any]) -> tuple[float, float, float]:
    return (f(row, "mask_psnr"), -f(row, "outside_psnr"), f(row, "temporal_diff_mae"))


def make_pair(group: str, split: str, idx: int, success: dict[str, Any], failure: dict[str, Any]) -> dict[str, Any]:
    success_id = str(success.get("candidate_id", ""))
    failure_id = str(failure.get("candidate_id", ""))
    pair_id = f"{group}__exp45_pair{idx:03d}__s{success.get('seed_index')}__f{failure.get('seed_index')}"
    return {
        "pair_id": pair_id,
        "split": split,
        "source_group": group,
        "source_id": failure.get("sample_id") or success.get("sample_id", ""),
        "condition_path": failure.get("condition_path") or success.get("condition_path", ""),
        "gt_winner_path": failure.get("winner_path") or success.get("winner_path", ""),
        "pseudo_success_path": success.get("raw_output_mp4", ""),
        "pseudo_success_frames_dir": success.get("frames_dir", ""),
        "failure_loser_path": failure.get("raw_output_mp4", ""),
        "failure_loser_frames_dir": failure.get("frames_dir", ""),
        "mask_path": failure.get("mask_path") or success.get("mask_path", ""),
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
        "classification": "EXP45_SAME_SOURCE_GT_WINNER_VS_MINIMAX_MEDIUM_HARD_FAILURE",
        "pairing_rule": "same source group only; GT winner preferred for DPO; pseudo-success retained for Stage2 distillation",
        "success_candidate_id": success_id,
        "failure_candidate_id": failure_id,
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
        "review_evidence_paths": {
            "success_review_sheet": success.get("review_sheet", ""),
            "success_temporal_strip_16": success.get("temporal_strip_16", ""),
            "failure_review_sheet": failure.get("review_sheet", ""),
            "failure_temporal_strip_16": failure.get("temporal_strip_16", ""),
        },
        "source_branch": "research/exp45-pai-minimax-pair-scaleup-20260629",
        "vor_eval_used": False,
        "hard_comp_used": False,
        "training_run": False,
        "optimizer_step": False,
    }


def path_exists(value: Any) -> bool:
    return bool(value) and Path(str(value)).exists()


def common_fields(pair: dict[str, Any], view: str, training_unlocked: bool) -> dict[str, Any]:
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
        "training_unlocked": training_unlocked,
        "source_branch": "research/exp45-pai-minimax-pair-scaleup-20260629",
        "loss_regions": ["mask", "boundary", "affected", "outside_preservation"],
    }


def gt_row(pair: dict[str, Any], idx: int, training_unlocked: bool) -> dict[str, Any]:
    row = common_fields(pair, "gt_distillation", training_unlocked)
    row.update(
        {
            "row_id": f"gt_{pair['split']}_{idx:04d}_{pair['pair_id']}",
            "target_path": pair["gt_winner_path"],
            "target_type": "gt_background",
            "pseudo_success_path": pair.get("pseudo_success_path", ""),
            "pseudo_success_label": pair.get("pseudo_success_label", ""),
            "preferred_first_h20_experiment": False,
        }
    )
    return row


def pseudo_row(pair: dict[str, Any], idx: int, training_unlocked: bool) -> dict[str, Any]:
    row = common_fields(pair, "pseudo_success_distillation", training_unlocked)
    row.update(
        {
            "row_id": f"pseudo_{pair['split']}_{idx:04d}_{pair['pair_id']}",
            "target_path": pair["pseudo_success_path"],
            "target_frames_dir": pair.get("pseudo_success_frames_dir", ""),
            "target_type": "visually_approved_minimax_pseudo_success",
            "pseudo_success_label": pair.get("pseudo_success_label", ""),
            "pseudo_success_visual_approved": pair.get("pseudo_success_label") in {"SUCCESS_CLEAN", "SUCCESS_USABLE"},
            "gt_background_path": pair["gt_winner_path"],
            "preferred_first_h20_experiment": True,
        }
    )
    return row


def pref_row(pair: dict[str, Any], idx: int, training_unlocked: bool) -> dict[str, Any]:
    row = common_fields(pair, "same_source_preference", training_unlocked)
    clean_success = pair.get("pseudo_success_label") == "SUCCESS_CLEAN"
    row.update(
        {
            "row_id": f"pref_{pair['split']}_{idx:04d}_{pair['pair_id']}",
            "winner_path": pair["gt_winner_path"],
            "winner_type": "gt_background",
            "alternate_success_clean_winner_path": pair.get("pseudo_success_path", "") if clean_success else "",
            "loser_path": pair["failure_loser_path"],
            "loser_frames_dir": pair.get("failure_loser_frames_dir", ""),
            "loser_type": "same_source_minimax_failure_medium_hard",
            "failure_label": pair.get("failure_label", ""),
            "pseudo_success_path": pair.get("pseudo_success_path", ""),
            "pseudo_success_label": pair.get("pseudo_success_label", ""),
            "badnoise_state_id": "",
            "badnoise_hard_state": "",
            "badnoise_gradient_proxy_is_backprop_gradient": False,
            "preferred_first_h20_experiment": False,
        }
    )
    return row


def validate(rows: list[dict[str, Any]], keys: list[str]) -> Counter[str]:
    counter: Counter[str] = Counter(rows=len(rows))
    for row in rows:
        for key in keys:
            counter[f"{key}_{'exists' if path_exists(row.get(key)) else 'missing'}"] += 1
    return counter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--success-manifest", action="append", required=True, type=Path)
    parser.add_argument("--failure-manifest", action="append", required=True, type=Path)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--reports-dir", required=True, type=Path)
    parser.add_argument("--preferred", action="store_true")
    args = parser.parse_args()

    successes = dedupe([row for path in args.success_manifest for row in read_jsonl(path)])
    failures = dedupe([row for path in args.failure_manifest for row in read_jsonl(path)])
    success_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failure_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in successes:
        success_by_group[group_of(row)].append(row)
    for row in failures:
        failure_by_group[group_of(row)].append(row)

    target = PREFERRED if args.preferred else MINIMUM
    pairs_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    group_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for group in sorted(SPLIT_GROUPS[split]):
            group_success = sorted(success_by_group.get(group, []), key=success_sort)
            group_failure = sorted(failure_by_group.get(group, []), key=failure_sort)
            made = 0
            used: set[tuple[str, str]] = set()
            for success in group_success:
                for failure in group_failure:
                    if len(pairs_by_split[split]) >= target[split]:
                        break
                    combo = (row_key(success), row_key(failure))
                    if combo in used:
                        continue
                    used.add(combo)
                    pairs_by_split[split].append(make_pair(group, split, made, success, failure))
                    made += 1
                if len(pairs_by_split[split]) >= target[split]:
                    break
            group_rows.append(
                {
                    "source_group": group,
                    "split": split,
                    "success_usable": len(group_success),
                    "failure_medium_hard": len(group_failure),
                    "pairs_built": made,
                }
            )

    all_pairs = [row for split in SPLITS for row in pairs_by_split[split]]
    counts = {split: len(rows) for split, rows in pairs_by_split.items()}
    ready = all(counts[split] >= MINIMUM[split] for split in SPLITS)
    status = "MINIMAX_STAGE2_FORMAL_DATA_READY" if ready else "MINIMAX_STAGE2_FORMAL_DATA_PARTIAL"
    training_status = "TRAINING_UNLOCKED_FOR_H20_HANDOFF" if ready else "TRAINING_NOT_UNLOCKED"

    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    hashes["exp45_same_source_pairs_all.jsonl"] = write_jsonl(args.manifest_dir / "exp45_same_source_pairs_all.jsonl", all_pairs)
    for split, rows in pairs_by_split.items():
        hashes[f"exp45_same_source_pairs_{split}.jsonl"] = write_jsonl(args.manifest_dir / f"exp45_same_source_pairs_{split}.jsonl", rows)

    view_rows: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for split, rows in pairs_by_split.items():
        view_rows["gt_distillation"][split] = [gt_row(pair, idx, ready) for idx, pair in enumerate(rows)]
        view_rows["pseudo_success_distillation"][split] = [pseudo_row(pair, idx, ready) for idx, pair in enumerate(rows)]
        view_rows["same_source_preference"][split] = [pref_row(pair, idx, ready) for idx, pair in enumerate(rows)]

    names = {
        "gt_distillation": "exp45_stage2_gt_distill",
        "pseudo_success_distillation": "exp45_stage2_pseudosuccess",
        "same_source_preference": "exp45_stage2_preference",
    }
    outputs: dict[str, dict[str, Any]] = {}
    for view, split_rows in view_rows.items():
        for split, rows in split_rows.items():
            path = args.manifest_dir / f"{names[view]}_{split}.jsonl"
            sha = write_jsonl(path, rows)
            outputs[f"{view}_{split}"] = {"path": str(path), "rows": len(rows), "sha256": sha}

    with (args.reports_dir / "exp45_stage2_formal_handoff.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["pair_id", "split", "source_group", "source_id", "success_candidate_id", "failure_candidate_id", "pseudo_success_label", "failure_label", "condition_path", "gt_winner_path", "pseudo_success_path", "failure_loser_path", "mask_path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in all_pairs:
            writer.writerow({key: pair.get(key, "") for key in fields})

    with (args.reports_dir / "exp45_stage2_formal_group_yield.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["source_group", "split", "success_usable", "failure_medium_hard", "pairs_built"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(group_rows)

    path_checks = {
        "gt_distillation": {split: dict(validate(rows, ["condition_path", "target_path", "mask_path"])) for split, rows in view_rows["gt_distillation"].items()},
        "pseudo_success_distillation": {split: dict(validate(rows, ["condition_path", "target_path", "mask_path"])) for split, rows in view_rows["pseudo_success_distillation"].items()},
        "same_source_preference": {split: dict(validate(rows, ["condition_path", "winner_path", "loser_path", "mask_path"])) for split, rows in view_rows["same_source_preference"].items()},
    }
    summary = {
        "status": status,
        "training_status": training_status,
        "training_run": False,
        "optimizer_step": False,
        "vor_eval_used": False,
        "hard_comp_used": False,
        "minimum_counts": MINIMUM,
        "preferred_counts": PREFERRED,
        "requested_counts": target,
        "split_counts": counts,
        "total_pairs": len(all_pairs),
        "success_rows_input": len(successes),
        "failure_rows_input": len(failures),
        "group_rows": group_rows,
        "manifest_hashes": hashes,
        "outputs": outputs,
        "path_checks": path_checks,
        "scene_group_overlap": {
            "train_search": sorted(SPLIT_GROUPS["train"] & SPLIT_GROUPS["search"]),
            "train_shadow": sorted(SPLIT_GROUPS["train"] & SPLIT_GROUPS["shadow"]),
            "search_shadow": sorted(SPLIT_GROUPS["search"] & SPLIT_GROUPS["shadow"]),
        },
        "first_h20_experiment": "pseudo-success SFT 30-step, then 100-step only if 30-step gate passes",
        "do_not_start_first": "GT-only SFT",
    }
    (args.reports_dir / "exp45_stage2_formal_handoff_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.reports_dir / "exp45_stage2_formal_handoff.md").write_text(
        "\n".join(
            [
                "# Exp45 Stage2 Formal Handoff",
                "",
                f"Status: `{status}`",
                "",
                f"- Pair count: `{len(all_pairs)}`",
                f"- Split counts: train/search/shadow = `{counts['train']}/{counts['search']}/{counts['shadow']}`",
                f"- Formal minimum: `{MINIMUM['train']}/{MINIMUM['search']}/{MINIMUM['shadow']}`",
                f"- Preferred: `{PREFERRED['train']}/{PREFERRED['search']}/{PREFERRED['shadow']}`",
                "- Scene overlap: `0` by locked split group sets",
                "- VOR-Eval used: `false`",
                "- hard comp used: `false`",
                "- training run: `false`",
                "- optimizer step: `false`",
                "",
                "H20 should mirror the package and run pseudo-success SFT 30-step first. Do not start GT-only SFT first.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
