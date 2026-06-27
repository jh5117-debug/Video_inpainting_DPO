#!/usr/bin/env python3
"""Repair Gate64 v3 manifest after pre-inference materialization failures."""

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
    p.add_argument("--original-manifest", type=Path, required=True)
    p.add_argument("--failed-jsonl", type=Path, required=True)
    p.add_argument("--source-pool", type=Path, action="append", required=True)
    p.add_argument("--exclude-manifest", type=Path, action="append", required=True)
    p.add_argument("--output-manifest", type=Path, required=True)
    p.add_argument("--replacement-manifest", type=Path, required=True)
    p.add_argument("--summary-md", type=Path, required=True)
    p.add_argument("--summary-csv", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, required=True)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene(row: dict) -> str:
    return str(row.get("scene_group") or row.get("sample_id"))


def main() -> int:
    args = parse_args()
    original = read_jsonl(args.original_manifest)
    failed_rows = read_jsonl(args.failed_jsonl)
    failed_ids = {str(row["sample_id"]) for row in failed_rows}
    original_by_id = {str(row["sample_id"]): row for row in original}
    missing = sorted(failed_ids - set(original_by_id))
    if missing:
        raise RuntimeError(f"failed rows missing from original manifest: {missing}")

    exclude_scenes = set()
    exclude_sample_ids = set()
    for path in args.exclude_manifest:
        for row in read_jsonl(path):
            exclude_scenes.add(scene(row))
            if row.get("sample_id"):
                exclude_sample_ids.add(str(row["sample_id"]))
    used_scenes = {scene(row) for row in original}
    used_sample_ids = {str(row["sample_id"]) for row in original}
    used_scenes.update(exclude_scenes)

    pool_rows: list[dict] = []
    pool_shas = {}
    for path in args.source_pool:
        rows = read_jsonl(path)
        for row in rows:
            row["_source_pool_path"] = str(path)
        pool_rows.extend(rows)
        pool_shas[str(path)] = sha256_file(path)

    replacement_by_failed: dict[str, dict] = {}
    for failed in failed_rows:
        failed_id = str(failed["sample_id"])
        failed_original = original_by_id[failed_id]
        wanted_type = failed_original.get("source_type")
        for row in pool_rows:
            sample_id = str(row.get("sample_id"))
            row_scene = scene(row)
            if row.get("source_type") != wanted_type:
                continue
            if (
                sample_id in KNOWN_INVALID_SAMPLE_IDS
                or sample_id in exclude_sample_ids
                or sample_id in used_sample_ids
                or row_scene in used_scenes
            ):
                continue
            repl = dict(row)
            repl["gate64_index"] = failed_original.get("gate64_index")
            repl["accepted_split"] = "gate64_v3_repaired"
            repl["selection_reason"] = (
                "deterministic_same_source_type_pre_inference_replacement_"
                "after_empty_mask_materialization_failure"
            )
            repl["replaces_sample_id"] = failed_id
            repl["repair_error"] = failed.get("error", "")
            repl["task"] = "object_removal"
            repl["hard_comp"] = False
            repl["comp_mode"] = "none"
            repl["condition_source_role"] = "V_obj"
            repl["winner_source_role"] = "V_bg"
            repl["mask_source_role"] = "foreground_object_mask"
            replacement_by_failed[failed_id] = repl
            used_sample_ids.add(sample_id)
            used_scenes.add(row_scene)
            break
        if failed_id not in replacement_by_failed:
            raise RuntimeError(f"no replacement found for {failed_id} ({wanted_type})")

    final_rows: list[dict] = []
    for row in original:
        sample_id = str(row["sample_id"])
        final_rows.append(replacement_by_failed.get(sample_id, row))
    final_rows = sorted(final_rows, key=lambda row: int(row.get("gate64_index", 0)))
    replacements = [replacement_by_failed[str(row["sample_id"])] for row in failed_rows]

    for path, rows in [(args.output_manifest, final_rows), (args.replacement_manifest, replacements)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    fieldnames = [
        "failed_sample_id",
        "failed_source_type",
        "replacement_sample_id",
        "replacement_scene_group",
        "replacement_source_type",
        "gate64_index",
        "repair_error",
        "replacement_source_pool",
    ]
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for failed in failed_rows:
            failed_id = str(failed["sample_id"])
            failed_original = original_by_id[failed_id]
            repl = replacement_by_failed[failed_id]
            writer.writerow(
                {
                    "failed_sample_id": failed_id,
                    "failed_source_type": failed_original.get("source_type"),
                    "replacement_sample_id": repl.get("sample_id"),
                    "replacement_scene_group": scene(repl),
                    "replacement_source_type": repl.get("source_type"),
                    "gate64_index": repl.get("gate64_index"),
                    "repair_error": failed.get("error", ""),
                    "replacement_source_pool": repl.get("_source_pool_path", ""),
                }
            )

    summary = {
        "status": "EXP30_GATE64_V3_MANIFEST_REPAIRED_PRE_INFERENCE",
        "original_rows": len(original),
        "failed_rows": len(failed_rows),
        "replacement_rows": len(replacements),
        "final_rows": len(final_rows),
        "final_scene_groups": len({scene(row) for row in final_rows}),
        "final_source_type_counts": dict(Counter(row.get("source_type") for row in final_rows)),
        "replacement_source_type_counts": dict(Counter(row.get("source_type") for row in replacements)),
        "source_pool_sha256": pool_shas,
        "original_manifest": str(args.original_manifest),
        "failed_jsonl": str(args.failed_jsonl),
        "output_manifest": str(args.output_manifest),
        "replacement_manifest": str(args.replacement_manifest),
        "output_manifest_sha256": sha256_file(args.output_manifest),
        "replacement_manifest_sha256": sha256_file(args.replacement_manifest),
        "model_outputs_generated_before_repair": False,
        "training_unlocked": False,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Exp30 Gate64 V3 Manifest Repair",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"- Original rows: {summary['original_rows']}",
        f"- Pre-inference materialization failures: {summary['failed_rows']}",
        f"- Replacement rows: {summary['replacement_rows']}",
        f"- Final rows: {summary['final_rows']}",
        f"- Final source type counts: `{summary['final_source_type_counts']}`",
        f"- Final manifest: `{args.output_manifest}`",
        f"- Final manifest SHA256: `{summary['output_manifest_sha256']}`",
        "",
        "The repair happened before any Gate64 model output, visual selection,",
        "adapter gate, or training.  Failed rows are preserved in the report and",
        "not silently reused.",
    ]
    args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
