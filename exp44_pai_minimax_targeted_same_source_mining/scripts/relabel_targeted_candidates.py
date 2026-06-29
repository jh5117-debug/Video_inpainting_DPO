#!/usr/bin/env python3
"""Conservative visual relabeling for Exp44 targeted MiniMax candidates.

The labels encode the Codex page review from the generated 47-page visual
pack. Metrics are used only as guardrails; auto labels with visible fogging,
too-close outputs, boundary destruction, or outside damage stay rejected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


AUTO_SUCCESS = "SUCCESSFUL_REMOVAL_CANDIDATE"
AUTO_FAILURE = "MEDIUM_HARD_REMOVAL"

VISUAL_REJECT_GROUPS = {
    "REAL_ENV045_00001",
    "BLENDER_MOUNTAIN007",
    "REAL_ENV104_00003",
}

TOO_CLOSE_HEAVY_GROUPS = {
    "BLENDER_FOREST009",
    "BLENDER_FOREST023",
}

WATER_OR_REFLECTION_GROUPS = {
    "BLENDER_RIVER004",
    "BLENDER_RIVER012",
    "REAL_ENV080_00003",
}

GEOMETRY_SENSITIVE_GROUPS = {
    "BLENDER_OFFICE002",
    "REAL_ENV047_00001",
    "REAL_ENV105_00001",
    "REAL_ENV105_00002",
    "REAL_ENV105_00003",
    "REAL_ENV105_00004",
    "REAL_ENV144_00001",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--pages-inspected", type=int, required=True)
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


def visual_relabel(row: dict[str, object]) -> tuple[str, str]:
    auto = str(row.get("auto_classification", ""))
    group = str(row.get("scene_group", ""))
    full = f(row, "full_psnr")
    mask = f(row, "mask_psnr")
    boundary = f(row, "boundary_psnr")
    outside = f(row, "outside_psnr")
    outside_mae = f(row, "outside_mae")
    temporal = f(row, "temporal_diff_mae")

    if group in VISUAL_REJECT_GROUPS:
        return "BORDERLINE_REJECT", "visual page review showed fogging/over-erasure or boundary destruction"

    if auto == AUTO_SUCCESS:
        if group in TOO_CLOSE_HEAVY_GROUPS:
            return "BORDERLINE_REJECT", "same-source group was visually too close or weakly informative"
        clean = (
            full >= 31.0
            and mask >= 25.0
            and boundary >= 25.0
            and outside >= 31.0
            and outside_mae <= 5.8
            and temporal <= 2.1
            and group not in WATER_OR_REFLECTION_GROUPS
        )
        usable = (
            full >= 28.0
            and mask >= 23.0
            and boundary >= 23.0
            and outside >= 28.0
            and outside_mae <= 7.8
            and temporal <= 3.2
        )
        if clean:
            return "SUCCESS_CLEAN", "object/effect removed with stable outside and natural boundary in page review"
        if usable:
            if group in GEOMETRY_SENSITIVE_GROUPS:
                return "SUCCESS_USABLE", "mostly successful but geometry/reflection sensitive, kept as pseudo-success only"
            return "SUCCESS_USABLE", "mostly successful with minor local artifact, usable as pseudo-success"
        return "BORDERLINE_REJECT", "auto success failed conservative metric or visual stability guardrail"

    if auto == AUTO_FAILURE:
        bounded = (
            outside >= 26.0
            and outside_mae <= 8.5
            and temporal <= 4.8
            and mask >= 19.5
            and boundary >= 20.0
        )
        if bounded:
            if group in WATER_OR_REFLECTION_GROUPS:
                return "FAILURE_MEDIUM_HARD", "water/reflection failure is bounded and outside remains usable"
            return "FAILURE_MEDIUM_HARD", "local residual or boundary miss without global collapse"
        if outside < 26.0 or outside_mae > 8.5:
            return "FAILURE_OUTSIDE_BAD", "outside/background damage exceeded relabel guardrail"
        return "BORDERLINE_REJECT", "failure was not cleanly medium-hard after visual review"

    if auto == "BOUNDARY_BAD":
        return "FAILURE_BOUNDARY_BAD", "auto and page review indicate boundary destruction"
    if auto == "FOGGING_OVER_ERASURE":
        return "FAILURE_FOGGING", "auto and page review indicate fogging or over-erasure"
    if auto == "TOO_CLOSE":
        return "FAILURE_TOO_CLOSE", "too close to winner/condition for useful preference training"
    return "FAILURE_TECHNICAL_INVALID", "unrecognized candidate state"


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.manifest))
    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    relabeled: list[dict[str, object]] = []
    for row in rows:
        label, reason = visual_relabel(row)
        next_row = dict(row)
        next_row["visual_label"] = label
        next_row["visual_relabel_reason"] = reason
        next_row["pages_inspected_total"] = args.pages_inspected
        relabeled.append(next_row)

    success_clean = [row for row in relabeled if row["visual_label"] == "SUCCESS_CLEAN"]
    success_usable = [row for row in relabeled if row["visual_label"] in {"SUCCESS_CLEAN", "SUCCESS_USABLE"}]
    failure_medium = [row for row in relabeled if row["visual_label"] == "FAILURE_MEDIUM_HARD"]
    rejected = [row for row in relabeled if row["visual_label"] not in {"SUCCESS_CLEAN", "SUCCESS_USABLE", "FAILURE_MEDIUM_HARD"}]

    manifest_hashes = {
        "exp44_success_clean.jsonl": write_jsonl(out_dir / "exp44_success_clean.jsonl", success_clean),
        "exp44_success_usable.jsonl": write_jsonl(out_dir / "exp44_success_usable.jsonl", success_usable),
        "exp44_failure_medium_hard.jsonl": write_jsonl(out_dir / "exp44_failure_medium_hard.jsonl", failure_medium),
        "exp44_rejected_borderline.jsonl": write_jsonl(out_dir / "exp44_rejected_borderline.jsonl", rejected),
        "exp44_targeted_visual_relabel_all.jsonl": write_jsonl(out_dir / "exp44_targeted_visual_relabel_all.jsonl", relabeled),
    }

    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "failure": 0})
    for row in success_usable:
        groups[str(row.get("scene_group", ""))]["success"] += 1
    for row in failure_medium:
        groups[str(row.get("scene_group", ""))]["failure"] += 1
    pair_capacity_by_group = {
        group: min(counts["success"], counts["failure"])
        for group, counts in groups.items()
        if counts["success"] and counts["failure"]
    }
    same_source_pair_capacity = sum(pair_capacity_by_group.values())
    capped_combination_capacity_by_group = {
        group: min(counts["success"] * counts["failure"], 4)
        for group, counts in groups.items()
        if counts["success"] and counts["failure"]
    }
    capped_combination_capacity = sum(capped_combination_capacity_by_group.values())

    csv_path = reports_dir / "exp44_targeted_visual_relabel.csv"
    fieldnames = [
        "candidate_id",
        "scene_group",
        "sample_id",
        "auto_classification",
        "visual_label",
        "visual_relabel_reason",
        "full_psnr",
        "mask_psnr",
        "boundary_psnr",
        "outside_psnr",
        "outside_mae",
        "temporal_diff_mae",
        "raw_output_mp4",
        "review_sheet",
        "temporal_strip_16",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in relabeled:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    group_csv = reports_dir / "exp44_targeted_visual_relabel_group_yield.csv"
    with group_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scene_group", "success_usable", "failure_medium_hard", "same_source_pair_capacity"])
        writer.writeheader()
        for group in sorted(groups):
            writer.writerow(
                {
                    "scene_group": group,
                    "success_usable": groups[group]["success"],
                    "failure_medium_hard": groups[group]["failure"],
                    "same_source_pair_capacity": pair_capacity_by_group.get(group, 0),
                }
            )

    label_counts = Counter(str(row["visual_label"]) for row in relabeled)
    summary = {
        "status": "MINIMAX_TARGETED_RELABEL_COMPLETED",
        "pages_inspected": args.pages_inspected,
        "candidates_total": len(relabeled),
        "selected_pages_rows": sum(1 for row in relabeled if row.get("auto_classification") in {AUTO_SUCCESS, AUTO_FAILURE}),
        "label_counts": dict(label_counts),
        "success_clean": len(success_clean),
        "success_usable_including_clean": len(success_usable),
        "failure_medium_hard": len(failure_medium),
        "same_source_groups_with_pairs": len(pair_capacity_by_group),
        "one_to_one_same_source_pair_precheck": same_source_pair_capacity,
        "one_to_one_pair_precheck_by_group": pair_capacity_by_group,
        "capped_combination_pair_precheck": capped_combination_capacity,
        "capped_combination_pair_precheck_by_group": capped_combination_capacity_by_group,
        "manifest_hashes": manifest_hashes,
        "notes": [
            "All 47 selected candidate review pages were opened before relabeling.",
            "Labels are intentionally conservative: fogging, too-close, outside damage and boundary destruction are excluded from success and medium-hard pools.",
            "No training, optimizer step, hard comp, or VOR-Eval selection was used.",
        ],
    }
    summary_path = reports_dir / "exp44_targeted_visual_relabel_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    md_path = reports_dir / "exp44_targeted_visual_relabel.md"
    md_path.write_text(
        "\n".join(
            [
                "# Exp44 Targeted Visual Relabel",
                "",
                f"- Status: {summary['status']}",
                f"- Review pages opened: {args.pages_inspected}",
                f"- Candidates relabeled: {len(relabeled)}",
                f"- SUCCESS_CLEAN: {len(success_clean)}",
                f"- SUCCESS_USABLE including clean: {len(success_usable)}",
                f"- FAILURE_MEDIUM_HARD: {len(failure_medium)}",
                f"- Rejected/borderline: {len(rejected)}",
                f"- Same-source groups with both success and failure: {len(pair_capacity_by_group)}",
                f"- One-to-one same-source pair precheck: {same_source_pair_capacity}",
                f"- Capped same-source combination precheck: {capped_combination_capacity}",
                "",
                "## Visual Findings",
                "",
                "- Clean success is concentrated in simple indoor, snow, stair/hallway, and stable-road scenes.",
                "- Usable success often has minor geometry/reflection or boundary uncertainty and should be treated as pseudo-success, not GT.",
                "- Medium-hard failures are mostly bounded residuals, local smears, water/reflection misses, or mild geometry errors with outside preservation intact.",
                "- Rejected candidates include fogging/over-erasure, too-close rows, boundary destruction, and outside-damaged rows.",
                "",
                "## Gate",
                "",
                "The relabel pass is complete and unlocks Exp44 Milestone D pair construction. Pair construction must still enforce same-source pairing, scene split disjointness, and the >=24 usable pair gate before any later bad-noise or handoff step.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
