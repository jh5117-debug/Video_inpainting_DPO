#!/usr/bin/env python3
"""Replace pre-inference technical-invalid smoke16 rows deterministically."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pool", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--invalid-sample-id", action="append", default=[])
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    invalid = set(args.invalid_sample_id)
    pool = read_jsonl(args.source_pool)
    base = read_jsonl(args.base_manifest)
    kept = [row for row in base if row["sample_id"] not in invalid]
    used_scenes = {row.get("scene_group") for row in kept}
    used_samples = {row["sample_id"] for row in kept}
    needed = Counter(row.get("source_type", "UNKNOWN") for row in base if row["sample_id"] in invalid)
    replacements: list[dict] = []
    for source_type, count in needed.items():
        for row in pool:
            if len([r for r in replacements if r.get("source_type") == source_type]) >= count:
                break
            if row.get("source_type") != source_type:
                continue
            if row["sample_id"] in used_samples or row["sample_id"] in invalid:
                continue
            if row.get("scene_group") in used_scenes:
                continue
            rep = dict(row)
            rep["replacement_for"] = "technical_invalid_pre_inference"
            rep["selection_reason"] = "deterministic_same_source_type_replacement_after_decode_or_empty_mask_failure"
            replacements.append(rep)
            used_samples.add(rep["sample_id"])
            used_scenes.add(rep.get("scene_group"))
    if len(replacements) != sum(needed.values()):
        raise RuntimeError(f"needed {sum(needed.values())} replacements, found {len(replacements)}")
    final = kept + replacements
    final = sorted(final, key=lambda r: (str(r.get("source_type")), int(r.get("smoke16_index", 9999)), str(r.get("sample_id"))))
    for idx, row in enumerate(final):
        row["smoke16_index"] = idx
        row["accepted_split"] = "smoke16_v2_final"
        row["task"] = "object_removal"
        row["hard_comp"] = False
        row["comp_mode"] = "none"
        row["condition_source_role"] = "V_obj"
        row["winner_source_role"] = "V_bg"
        row["mask_source_role"] = "foreground_object_mask"
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as f:
        for row in final:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_id",
            "scene_group",
            "source_type",
            "replacement_for",
            "condition_member_path",
            "winner_member_path",
            "mask_member_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in final:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    summary = {
        "status": "EXP30_SMOKE16_V2_MANIFEST_REPAIRED_PRE_INFERENCE",
        "base_manifest": str(args.base_manifest),
        "invalid_sample_ids": sorted(invalid),
        "replacement_sample_ids": [row["sample_id"] for row in replacements],
        "rows": len(final),
        "scene_groups": len({row.get("scene_group") for row in final}),
        "source_type_counts": dict(Counter(row.get("source_type") for row in final)),
        "manifest": str(args.output_manifest),
        "manifest_sha256": sha256_file(args.output_manifest),
        "model_outputs_seen": False,
        "replacement_rule": "same-source-type next available row from source-pool v2, scene-disjoint, before any candidate/model output review",
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.summary_md.write_text(
        "# Exp30 Smoke16 V2 Manifest Repair\n\n"
        f"Status: `{summary['status']}`\n\n"
        f"- Invalid sample IDs: `{summary['invalid_sample_ids']}`\n"
        f"- Replacement sample IDs: `{summary['replacement_sample_ids']}`\n"
        f"- Rows: {summary['rows']}\n"
        f"- Source type counts: `{summary['source_type_counts']}`\n"
        f"- Final manifest: `{args.output_manifest}`\n"
        f"- Final manifest SHA256: `{summary['manifest_sha256']}`\n\n"
        "The repair happened before any model candidate output review. It is a "
        "technical decode/non-empty-mask replacement, not result-based source "
        "selection.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
