#!/usr/bin/env python3
"""Preregister the Exp30 smoke16 rows from the locked source-pool v2.

This is metadata-only. It does not extract videos and does not run model
inference. The output JSONL is also the exact triplet list passed to the safe
VOR selective extractor.
"""

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
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_balanced(rows: list[dict], count: int) -> list[dict]:
    by_type = {
        "BLENDER": [r for r in rows if r.get("source_type") == "BLENDER"],
        "REAL": [r for r in rows if r.get("source_type") == "REAL"],
    }
    selected: list[dict] = []
    seen_scenes: set[str] = set()
    while len(selected) < count and (by_type["BLENDER"] or by_type["REAL"]):
        for source_type in ("BLENDER", "REAL"):
            while by_type[source_type]:
                row = by_type[source_type].pop(0)
                scene = str(row.get("scene_group") or row.get("sample_id"))
                if scene in seen_scenes:
                    continue
                selected.append(dict(row))
                seen_scenes.add(scene)
                break
            if len(selected) >= count:
                break
    if len(selected) != count:
        raise RuntimeError(f"selected {len(selected)} rows, expected {count}")
    return selected


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.source_pool)
    selected = select_balanced(rows, args.count)
    for idx, row in enumerate(selected):
        row["smoke16_index"] = idx
        row["accepted_split"] = "smoke16_v2"
        row["selection_reason"] = "deterministic_balanced_source_type_from_source_pool_v2"
        row["task"] = "object_removal"
        row["hard_comp"] = False
        row["comp_mode"] = "none"
        row["condition_source_role"] = "V_obj"
        row["winner_source_role"] = "V_bg"
        row["mask_source_role"] = "foreground_object_mask"

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_sha = sha256_file(args.output_manifest)

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "smoke16_index",
            "sample_id",
            "scene_group",
            "source_type",
            "mask_bucket",
            "effect_type",
            "condition_member_path",
            "winner_member_path",
            "mask_member_path",
            "selection_reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    summary = {
        "status": "EXP30_SMOKE16_V2_PREREGISTERED",
        "source_pool": str(args.source_pool),
        "source_pool_sha256": sha256_file(args.source_pool),
        "rows": len(selected),
        "scene_groups": len({r.get("scene_group") for r in selected}),
        "source_type_counts": dict(Counter(r.get("source_type") for r in selected)),
        "mask_bucket_counts": dict(Counter(r.get("mask_bucket") for r in selected)),
        "effect_type_counts": dict(Counter(r.get("effect_type") for r in selected)),
        "manifest": str(args.output_manifest),
        "manifest_sha256": manifest_sha,
        "vor_eval_used": False,
        "model_outputs_generated": False,
        "gpu_used": False,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Exp30 Multi-Model OR Smoke16 V2 Preregistration",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"- Source pool: `{args.source_pool}`",
        f"- Source pool SHA256: `{summary['source_pool_sha256']}`",
        f"- Locked smoke rows: {summary['rows']}",
        f"- Scene groups: {summary['scene_groups']}",
        f"- Source types: `{summary['source_type_counts']}`",
        f"- Mask buckets: `{summary['mask_bucket_counts']}`",
        f"- Effect labels: `{summary['effect_type_counts']}`",
        f"- Output manifest: `{args.output_manifest}`",
        f"- Output manifest SHA256: `{manifest_sha}`",
        "",
        "This preregistration does not extract videos, run inference, inspect",
        "model outputs, use VOR-Eval, or unlock Gate64 by itself.",
    ]
    args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
