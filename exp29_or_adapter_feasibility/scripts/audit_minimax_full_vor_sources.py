#!/usr/bin/env python3
"""Build the Exp29 MiniMax full-VOR source audit from existing Exp25 indexes.

This script is intentionally metadata-only. It never opens the VOR tar files
and never writes to Exp25. Mask size/effect/motion fields are reported as
unknown when the source index does not contain them; they must be measured
during the later first-pass generation/materialization milestone.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            row["_source_file"] = str(path)
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_source_type(row: dict[str, Any]) -> str:
    explicit = row.get("source_type")
    if explicit in {"REAL", "BLENDER"}:
        return explicit
    sample_id = str(row.get("sample_id", ""))
    if sample_id.startswith("REAL_"):
        return "REAL"
    if sample_id.startswith("BLENDER_"):
        return "BLENDER"
    return "UNKNOWN"


def infer_scene_group(row: dict[str, Any]) -> str:
    if row.get("scene_group"):
        return str(row["scene_group"])
    sample_id = str(row.get("sample_id", ""))
    parts = sample_id.split("_")
    if sample_id.startswith("REAL_") and len(parts) >= 3:
        return "_".join(parts[:3])
    if sample_id.startswith("BLENDER_") and len(parts) >= 2:
        return "_".join(parts[:2])
    return sample_id


def nonempty_path(row: dict[str, Any], key: str) -> bool:
    value = row.get(key)
    return isinstance(value, str) and bool(value.strip())


def load_exclusions(paths: list[Path]) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    sample_ids: set[str] = set()
    scene_groups: set[str] = set()
    evidence: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            sample_id = str(row.get("sample_id", ""))
            scene_group = infer_scene_group(row)
            if sample_id:
                sample_ids.add(sample_id)
            if scene_group:
                scene_groups.add(scene_group)
            evidence.append(
                {
                    "path": str(path),
                    "sample_id": sample_id,
                    "scene_group": scene_group,
                }
            )
    return sample_ids, scene_groups, evidence


def group_candidates(rows: list[dict[str, Any]], exclude_samples: set[str], exclude_groups: set[str]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    audit_rows: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        scene_group = infer_scene_group(row)
        source_type = infer_source_type(row)
        reason = ""
        accepted = True

        if not sample_id or not scene_group:
            accepted = False
            reason = "missing_sample_or_scene_group"
        elif sample_id in seen_samples:
            accepted = False
            reason = "duplicate_sample_id"
        elif sample_id in exclude_samples or scene_group in exclude_groups:
            accepted = False
            reason = "excluded_previous_exp29_sample_or_scene"
        elif "VOR-Eval" in " ".join(str(row.get(k, "")) for k in ("condition_member_path", "winner_member_path", "mask_member_path")):
            accepted = False
            reason = "vor_eval_path"
        elif source_type not in {"REAL", "BLENDER"}:
            accepted = False
            reason = "unknown_source_type"
        elif not all(nonempty_path(row, key) for key in ("condition_member_path", "winner_member_path", "mask_member_path")):
            accepted = False
            reason = "missing_member_path"
        elif row.get("condition_role") != "FG_BG" or row.get("winner_role") != "BG" or row.get("mask_role") != "MASK":
            accepted = False
            reason = "unexpected_triplet_roles"

        if sample_id:
            seen_samples.add(sample_id)
        if accepted:
            grouped[scene_group].append(row)

        audit_rows.append(
            {
                "sample_id": sample_id,
                "scene_group": scene_group,
                "source_type": source_type,
                "condition_member_path": row.get("condition_member_path", ""),
                "winner_member_path": row.get("winner_member_path", ""),
                "mask_member_path": row.get("mask_member_path", ""),
                "accepted_for_group_pool": accepted,
                "reject_reason": reason,
                "mask_bucket": "unknown_pending_materialization",
                "effect_type": "unknown_pending_metadata",
                "motion_bucket": "unknown_pending_metadata",
            }
        )
    return grouped, audit_rows


def choose_balanced_groups(grouped: dict[str, list[dict[str, Any]]], target: int) -> list[dict[str, Any]]:
    by_type: dict[str, list[tuple[str, dict[str, Any]]]] = {"REAL": [], "BLENDER": []}
    for scene_group in sorted(grouped):
        rows = sorted(grouped[scene_group], key=lambda r: str(r.get("sample_id", "")))
        if not rows:
            continue
        representative = rows[0]
        source_type = infer_source_type(representative)
        if source_type in by_type:
            by_type[source_type].append((scene_group, representative))

    per_type_target = target // 2
    selected: list[tuple[str, dict[str, Any]]] = []
    for source_type in ("REAL", "BLENDER"):
        selected.extend(by_type[source_type][:per_type_target])

    if len(selected) < target:
        selected_groups = {group for group, _ in selected}
        leftovers: list[tuple[str, dict[str, Any]]] = []
        for source_type in ("REAL", "BLENDER"):
            leftovers.extend(item for item in by_type[source_type][per_type_target:] if item[0] not in selected_groups)
        selected.extend(leftovers[: target - len(selected)])

    out: list[dict[str, Any]] = []
    for rank, (scene_group, row) in enumerate(selected):
        source_type = infer_source_type(row)
        out.append(
            {
                "selection_rank": rank,
                "sample_id": row.get("sample_id"),
                "scene_group": scene_group,
                "source_type": source_type,
                "condition_member_path": row.get("condition_member_path"),
                "winner_member_path": row.get("winner_member_path"),
                "mask_member_path": row.get("mask_member_path"),
                "condition_role": row.get("condition_role"),
                "winner_role": row.get("winner_role"),
                "mask_role": row.get("mask_role"),
                "task": row.get("task", "object_removal"),
                "comp_mode": row.get("comp_mode", "none"),
                "hard_comp": bool(row.get("hard_comp", False)),
                "member_paths_resolvable_by_index": True,
                "requires_selective_extraction": True,
                "decode_verified": False,
                "mask_nonempty_verified": False,
                "can_materialize_17f": "pending_selective_extraction",
                "mask_bucket": "unknown_pending_materialization",
                "effect_type": "unknown_pending_metadata",
                "motion_bucket": "unknown_pending_metadata",
                "vor_eval_used": False,
                "eligible_for_generation": True,
                "eligible_for_training": False,
                "selection_rule": "balanced_REAL_BLENDER_scene_group_unique_from_full_vor_index",
            }
        )
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Exp29 MiniMax Full-VOR Source Audit",
        "",
        f"Status: `{summary['status']}`",
        "",
        "## Inputs",
        "",
        f"- Full metadata index: `{summary['full_index_path']}`",
        f"- Full metadata SHA256: `{summary['full_index_sha256']}`",
        f"- Previous source32 excluded rows: {summary['previous_source32_rows']}",
        f"- EffectErase smoke excluded rows: {summary['effecterase_excluded_rows']}",
        "",
        "## Counts",
        "",
        f"- Raw rows read: {summary['raw_rows']}",
        f"- Unique valid candidate scene groups after exclusions: {summary['valid_candidate_groups']}",
        f"- Selected source groups: {summary['selected_groups']}",
        f"- Selected manifest SHA256: `{summary['selected_manifest_sha256']}`",
        "",
        "## Balance",
        "",
        f"- Selected source type counts: `{summary['selected_source_counts']}`",
        f"- Mask buckets: `unknown_pending_materialization` because the full metadata index does not contain mask-area fields.",
        f"- Effect labels: `unknown_pending_metadata` because the full metadata index does not contain effect-type fields.",
        f"- Motion labels: `unknown_pending_metadata` because the full metadata index does not contain motion fields.",
        "",
        "## Interpretation",
        "",
        "This milestone does not claim MiniMax micro-data quality. It only fixes the",
        "previous source-pool blocker by deriving a scene-disjoint candidate pool from",
        "the full VOR Train metadata index. Mask non-emptiness, 17-frame decode,",
        "medium-hard quality, and defect labels must be measured during the next",
        "first-pass generation/materialization milestone before any training or recipe",
        "gate is allowed.",
        "",
        "No VOR-Eval rows are used, no MiniMax generation was launched, and no training",
        "manifest was created by this audit.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-index", type=Path, required=True)
    parser.add_argument("--previous-source32", type=Path, action="append", default=[])
    parser.add_argument("--effecterase-manifest", type=Path, action="append", default=[])
    parser.add_argument("--target-groups", type=int, default=192)
    parser.add_argument("--min-groups", type=int, default=128)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.full_index)
    previous_samples, previous_groups, previous_evidence = load_exclusions(args.previous_source32)
    effect_samples, effect_groups, effect_evidence = load_exclusions(args.effecterase_manifest)
    exclude_samples = previous_samples | effect_samples
    exclude_groups = previous_groups | effect_groups

    grouped, audit_rows = group_candidates(rows, exclude_samples, exclude_groups)
    selected = choose_balanced_groups(grouped, args.target_groups)
    status = "MINIMAX_FULL_VOR_SOURCE_AUDIT_READY" if len(grouped) >= args.min_groups and len(selected) >= args.min_groups else "MINIMAX_FULL_VOR_SOURCE_AUDIT_INSUFFICIENT"

    write_jsonl(args.out_manifest, selected)
    write_csv(args.out_csv, audit_rows)
    selected_sha = sha256_file(args.out_manifest)

    selected_source_counts = Counter(row["source_type"] for row in selected)
    valid_group_source_counts = Counter(infer_source_type(rows[0]) for rows in grouped.values() if rows)
    summary: dict[str, Any] = {
        "status": status,
        "full_index_path": str(args.full_index),
        "full_index_sha256": sha256_file(args.full_index),
        "raw_rows": len(rows),
        "raw_sample_ids": len({str(row.get("sample_id", "")) for row in rows if row.get("sample_id")}),
        "raw_scene_groups": len({infer_scene_group(row) for row in rows if infer_scene_group(row)}),
        "previous_source32_rows": len(previous_evidence),
        "previous_source32_sample_ids": len(previous_samples),
        "previous_source32_scene_groups": len(previous_groups),
        "effecterase_excluded_rows": len(effect_evidence),
        "effecterase_excluded_sample_ids": len(effect_samples),
        "effecterase_excluded_scene_groups": len(effect_groups),
        "valid_candidate_groups": len(grouped),
        "valid_group_source_counts": dict(valid_group_source_counts),
        "selected_groups": len(selected),
        "selected_source_counts": dict(selected_source_counts),
        "target_groups": args.target_groups,
        "min_groups": args.min_groups,
        "selected_manifest": str(args.out_manifest),
        "selected_manifest_sha256": selected_sha,
        "audit_csv": str(args.out_csv),
        "mask_bucket_status": "unknown_pending_materialization",
        "effect_label_status": "unknown_pending_metadata",
        "motion_label_status": "unknown_pending_metadata",
        "generation_launched": False,
        "training_manifest_created": False,
        "vor_eval_used": False,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.out_md, summary)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
