#!/usr/bin/env python3
"""Select stratified VOR triplets for semantic extraction audit."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--triplet-jsonl", type=Path, required=True)
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260623)
    p.add_argument("--output-jsonl", type=Path, default=Path("exp25_vor_or_preference_data/manifests/vor_triplet_audit64.jsonl"))
    p.add_argument("--sample-ids-txt", type=Path, default=Path("exp25_vor_or_preference_data/manifests/vor_triplet_audit64_sample_ids.txt"))
    p.add_argument("--report-md", type=Path, default=Path("reports/vor_triplet_audit64_selection.md"))
    return p.parse_args()


def source_type(sample_id: str) -> str:
    if sample_id.startswith("REAL_"):
        return "REAL"
    if sample_id.startswith("BLENDER_"):
        return "BLENDER"
    return "OTHER"


def main() -> int:
    args = parse_args()
    rows = []
    with args.triplet_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                row["source_type"] = source_type(row["sample_id"])
                rows.append(row)
    if len(rows) < args.count:
        raise SystemExit(f"Need {args.count} triplets, found {len(rows)}")

    rng = random.Random(args.seed)
    by_source_group: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_source_group[row["source_type"]][row["scene_group"]].append(row)

    targets = {
        "REAL": round(args.count * sum(1 for r in rows if r["source_type"] == "REAL") / len(rows)),
        "BLENDER": 0,
    }
    targets["BLENDER"] = args.count - targets["REAL"]
    if "OTHER" in by_source_group:
        targets["OTHER"] = min(len(by_source_group["OTHER"]), max(1, args.count // 16))

    selected: list[dict] = []
    used_ids: set[str] = set()
    for src, target in targets.items():
        groups = list(by_source_group.get(src, {}))
        rng.shuffle(groups)
        cursor = 0
        while len([r for r in selected if r["source_type"] == src]) < target and groups:
            group = groups[cursor % len(groups)]
            candidates = [r for r in by_source_group[src][group] if r["sample_id"] not in used_ids]
            if candidates:
                row = rng.choice(candidates)
                selected.append(row)
                used_ids.add(row["sample_id"])
            cursor += 1
            if cursor > len(groups) * 4 and len([r for r in selected if r["source_type"] == src]) < target:
                remaining = [r for r in rows if r["source_type"] == src and r["sample_id"] not in used_ids]
                if not remaining:
                    break
                row = rng.choice(remaining)
                selected.append(row)
                used_ids.add(row["sample_id"])

    while len(selected) < args.count:
        row = rng.choice([r for r in rows if r["sample_id"] not in used_ids])
        selected.append(row)
        used_ids.add(row["sample_id"])

    selected = selected[: args.count]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    args.sample_ids_txt.parent.mkdir(parents=True, exist_ok=True)
    args.sample_ids_txt.write_text("\n".join(r["sample_id"] for r in selected) + "\n", encoding="utf-8")

    source_counts = defaultdict(int)
    for row in selected:
        source_counts[row["source_type"]] += 1
    group_count = len({r["scene_group"] for r in selected})
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(
        "# VOR Triplet Audit64 Selection\n\n"
        f"- triplet_jsonl: `{args.triplet_jsonl}`\n"
        f"- selected: {len(selected)}\n"
        f"- seed: {args.seed}\n"
        f"- source_counts: {dict(source_counts)}\n"
        f"- scene_groups: {group_count}\n"
        f"- output_jsonl: `{args.output_jsonl}`\n"
        f"- sample_ids_txt: `{args.sample_ids_txt}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected": len(selected), "source_counts": dict(source_counts), "scene_groups": group_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
