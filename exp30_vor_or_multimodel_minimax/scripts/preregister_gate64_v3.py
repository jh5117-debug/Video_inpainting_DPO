#!/usr/bin/env python3
"""Preregister Exp30 Gate64 v3 source groups.

This is metadata-only.  It selects scene-disjoint VOR-OR source groups after
Smoke16/Smoke32 have both passed, and it does not extract videos or run model
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
    p = argparse.ArgumentParser()
    p.add_argument("--source-pool", type=Path, action="append", required=True)
    p.add_argument("--exclude-manifest", type=Path, action="append", required=True)
    p.add_argument("--output-manifest", type=Path, required=True)
    p.add_argument("--summary-md", type=Path, required=True)
    p.add_argument("--summary-csv", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, required=True)
    p.add_argument("--count", type=int, default=64)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene(row: dict) -> str:
    return str(row.get("scene_group") or row.get("sample_id"))


def select_rows(rows: list[dict], excluded_scenes: set[str], count: int) -> list[dict]:
    target = {"BLENDER": count // 2, "REAL": count - count // 2}
    selected: list[dict] = []
    selected_by_type = Counter()
    seen = set(excluded_scenes)
    for source_type in ("BLENDER", "REAL"):
        for row in rows:
            if selected_by_type[source_type] >= target[source_type]:
                break
            if row.get("source_type") != source_type:
                continue
            sample_id = str(row.get("sample_id"))
            row_scene = scene(row)
            if sample_id in KNOWN_INVALID_SAMPLE_IDS or row_scene in seen:
                continue
            locked = dict(row)
            locked["gate64_index"] = len(selected)
            locked["accepted_split"] = "gate64_v3"
            locked["selection_reason"] = (
                "deterministic_balanced_gate64_from_source_pool_v2_"
                "excluding_smoke16_smoke32_and_known_invalid"
            )
            locked["task"] = "object_removal"
            locked["hard_comp"] = False
            locked["comp_mode"] = "none"
            locked["condition_source_role"] = "V_obj"
            locked["winner_source_role"] = "V_bg"
            locked["mask_source_role"] = "foreground_object_mask"
            selected.append(locked)
            selected_by_type[source_type] += 1
            seen.add(row_scene)
    if len(selected) != count:
        raise RuntimeError(f"selected {len(selected)} rows, expected {count}; counts={dict(selected_by_type)}")
    return selected


def main() -> int:
    args = parse_args()
    rows: list[dict] = []
    pool_shas = {}
    for path in args.source_pool:
        pool_rows = read_jsonl(path)
        for row in pool_rows:
            row["_source_pool_path"] = str(path)
        rows.extend(pool_rows)
        pool_shas[str(path)] = sha256_file(path)

    excluded_rows: list[dict] = []
    for path in args.exclude_manifest:
        excluded_rows.extend(read_jsonl(path))
    excluded_scenes = {scene(row) for row in excluded_rows}

    selected = select_rows(rows, excluded_scenes, args.count)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_sha = sha256_file(args.output_manifest)

    fieldnames = [
        "gate64_index",
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
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    summary = {
        "status": "EXP30_GATE64_V3_PREREGISTERED",
        "rows": len(selected),
        "scene_groups": len({scene(row) for row in selected}),
        "source_type_counts": dict(Counter(row.get("source_type") for row in selected)),
        "mask_bucket_counts": dict(Counter(row.get("mask_bucket") for row in selected)),
        "effect_type_counts": dict(Counter(row.get("effect_type") for row in selected)),
        "excluded_rows": len(excluded_rows),
        "excluded_scene_groups": len(excluded_scenes),
        "known_invalid_sample_ids": sorted(KNOWN_INVALID_SAMPLE_IDS),
        "source_pool_sha256": pool_shas,
        "manifest": str(args.output_manifest),
        "manifest_sha256": manifest_sha,
        "vor_eval_used": False,
        "model_outputs_generated": False,
        "gpu_used": False,
        "gate64_pool_ready": False,
        "training_unlocked": False,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Exp30 Gate64 V3 Preregistration",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"- Locked rows: {summary['rows']}",
        f"- Scene groups: {summary['scene_groups']}",
        f"- Source type counts: `{summary['source_type_counts']}`",
        f"- Excluded prior smoke scene groups: {summary['excluded_scene_groups']}",
        f"- Manifest: `{args.output_manifest}`",
        f"- Manifest SHA256: `{manifest_sha}`",
        "",
        "This milestone performs no extraction, no model inference, no visual",
        "selection, no VOR-Eval access, and no training.  It only locks the",
        "limited Gate64 source rows after Smoke16/Smoke32 v3 passed.",
    ]
    args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
