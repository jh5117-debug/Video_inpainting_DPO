#!/usr/bin/env python3
"""Preregister Exp30 Smoke32 v3 confirmation sources.

Smoke32 is a confirmation step after Smoke16 v3.  It selects exactly sixteen
new scene groups from the locked VOR OR source-pool v2, excluding the repaired
Smoke16 sources and known technical-invalid rows from the pre-inference repair
audit.  This script is metadata-only: it does not extract videos or run model
inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


KNOWN_INVALID_SAMPLE_IDS = {
    "BLENDER_CARTOON006_00001",
    "REAL_ENV044_00004_001_01",
    "REAL_ENV046_00001_001_01",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pool", type=Path, action="append", required=True)
    parser.add_argument("--exclude-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_scene(row: dict) -> str:
    return str(row.get("scene_group") or row.get("sample_id"))


def select_confirmation_rows(rows: list[dict], excluded_scenes: set[str], count: int) -> list[dict]:
    target_by_type = {"BLENDER": count // 2, "REAL": count - count // 2}
    selected: list[dict] = []
    seen_scenes = set(excluded_scenes)
    selected_by_type = Counter()

    for source_type in ("BLENDER", "REAL"):
        for row in rows:
            if selected_by_type[source_type] >= target_by_type[source_type]:
                break
            if row.get("source_type") != source_type:
                continue
            sample_id = str(row.get("sample_id"))
            scene = row_scene(row)
            if sample_id in KNOWN_INVALID_SAMPLE_IDS or scene in seen_scenes:
                continue
            locked = dict(row)
            locked["smoke32_index"] = len(selected)
            locked["accepted_split"] = "smoke32_v3"
            locked["selection_reason"] = (
                "deterministic_balanced_confirmation_from_source_pool_v2_"
                "excluding_smoke16_and_known_invalid"
            )
            locked["task"] = "object_removal"
            locked["hard_comp"] = False
            locked["comp_mode"] = "none"
            locked["condition_source_role"] = "V_obj"
            locked["winner_source_role"] = "V_bg"
            locked["mask_source_role"] = "foreground_object_mask"
            selected.append(locked)
            selected_by_type[source_type] += 1
            seen_scenes.add(scene)

    if len(selected) != count:
        raise RuntimeError(f"selected {len(selected)} rows, expected {count}")
    return selected


def main() -> int:
    args = parse_args()
    all_rows: list[dict] = []
    source_pool_shas = {}
    for path in args.source_pool:
        pool_rows = read_jsonl(path)
        for row in pool_rows:
            row["_source_pool_path"] = str(path)
        all_rows.extend(pool_rows)
        source_pool_shas[str(path)] = sha256_file(path)

    excluded_rows = read_jsonl(args.exclude_manifest)
    excluded_scenes = {row_scene(row) for row in excluded_rows}
    selected = select_confirmation_rows(all_rows, excluded_scenes, args.count)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_sha = sha256_file(args.output_manifest)

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "smoke32_index",
        "sample_id",
        "scene_group",
        "source_type",
        "mask_bucket",
        "effect_type",
        "condition_member_path",
        "winner_member_path",
        "mask_member_path",
        "_source_pool_path",
        "selection_reason",
    ]
    with args.summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    summary = {
        "status": "EXP30_SMOKE32_V3_PREREGISTERED",
        "rows": len(selected),
        "scene_groups": len({row_scene(row) for row in selected}),
        "source_type_counts": dict(Counter(row.get("source_type") for row in selected)),
        "mask_bucket_counts": dict(Counter(row.get("mask_bucket") for row in selected)),
        "effect_type_counts": dict(Counter(row.get("effect_type") for row in selected)),
        "excluded_smoke16_rows": len(excluded_rows),
        "excluded_smoke16_scene_groups": len(excluded_scenes),
        "known_invalid_sample_ids": sorted(KNOWN_INVALID_SAMPLE_IDS),
        "source_pool_sha256": source_pool_shas,
        "manifest": str(args.output_manifest),
        "manifest_sha256": manifest_sha,
        "vor_eval_used": False,
        "model_outputs_generated": False,
        "gpu_used": False,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Exp30 Smoke32 V3 Preregistration",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"- Locked confirmation rows: {summary['rows']}",
        f"- Scene groups: {summary['scene_groups']}",
        f"- Source type counts: `{summary['source_type_counts']}`",
        f"- Excluded Smoke16 scene groups: {summary['excluded_smoke16_scene_groups']}",
        f"- Known invalid sample IDs excluded: `{summary['known_invalid_sample_ids']}`",
        f"- Output manifest: `{args.output_manifest}`",
        f"- Output manifest SHA256: `{manifest_sha}`",
        "",
        "This preregistration performs no extraction, no model inference, no visual",
        "selection, no VOR-Eval access, and no training.  It only unlocks Smoke32",
        "materialization and candidate generation after the manifest is committed.",
    ]
    args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
