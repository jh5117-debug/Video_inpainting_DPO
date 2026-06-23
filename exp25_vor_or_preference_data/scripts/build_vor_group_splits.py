#!/usr/bin/env python3
"""Build VOR Train source/search/shadow splits with scene-group isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--triplet-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("exp25_vor_or_preference_data/manifests"))
    p.add_argument("--train-count", type=int, default=4096)
    p.add_argument("--search-dev-count", type=int, default=256)
    p.add_argument("--shadow-dev-count", type=int, default=256)
    p.add_argument("--gate-count", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260623)
    p.add_argument("--exclude-sample-ids", type=Path, help="Optional text file of sample ids to exclude from all outputs.")
    p.add_argument("--report-md", type=Path, default=Path("reports/vor_group_split_audit.md"))
    p.add_argument("--report-json", type=Path, default=Path("reports/vor_group_split_audit.json"))
    return p.parse_args()


def source_type(row: dict) -> str:
    sid = str(row.get("sample_id", ""))
    if sid.startswith("REAL_"):
        return "REAL"
    if sid.startswith("BLENDER_"):
        return "BLENDER"
    return "OTHER"


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("source_type") is None:
                row["source_type"] = source_type(row)
            if not row.get("scene_group"):
                raise ValueError(f"missing scene_group for {row.get('sample_id')}")
            rows.append(row)
    return rows


def load_excludes(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assign_group_pool(
    groups_by_source: dict[str, list[str]],
    rows_by_group: dict[str, list[dict]],
    used_groups: set[str],
    count: int,
    rng: random.Random,
    preferred_source_ratio: dict[str, float],
) -> tuple[set[str], list[dict]]:
    picked_groups: set[str] = set()
    picked_rows: list[dict] = []
    target_by_source = {k: int(round(count * v)) for k, v in preferred_source_ratio.items()}
    remaining = count
    for source, source_target in sorted(target_by_source.items()):
        candidates = [g for g in groups_by_source.get(source, []) if g not in used_groups]
        rng.shuffle(candidates)
        for group in candidates:
            if len([r for r in picked_rows if r["source_type"] == source]) >= source_target and remaining <= count // 4:
                break
            picked_groups.add(group)
            used_groups.add(group)
            picked_rows.extend(rows_by_group[group])
            remaining = max(0, count - len(picked_rows))
            if len([r for r in picked_rows if r["source_type"] == source]) >= source_target and len(picked_rows) >= count:
                break
    if len(picked_rows) < count:
        candidates = [g for source_groups in groups_by_source.values() for g in source_groups if g not in used_groups]
        rng.shuffle(candidates)
        for group in candidates:
            picked_groups.add(group)
            used_groups.add(group)
            picked_rows.extend(rows_by_group[group])
            if len(picked_rows) >= count:
                break
    if len(picked_rows) < count:
        raise RuntimeError(f"not enough isolated groups to sample {count} rows; got {len(picked_rows)}")
    rng.shuffle(picked_rows)
    return picked_groups, picked_rows[:count]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def sample_rows_by_source(rows: list[dict], count: int, ratios: dict[str, float], rng: random.Random) -> list[dict]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source_type"]].append(row)
    for source_rows in by_source.values():
        rng.shuffle(source_rows)
    targets = {src: int(round(count * ratios.get(src, 0.0))) for src in by_source}
    while sum(targets.values()) < count:
        src = max(by_source, key=lambda s: ratios.get(s, 0.0) - targets.get(s, 0) / max(1, count))
        targets[src] += 1
    while sum(targets.values()) > count:
        src = max(targets, key=targets.get)
        targets[src] -= 1
    picked: list[dict] = []
    for src, target in targets.items():
        picked.extend(by_source[src][:target])
    if len(picked) < count:
        remainder = [r for src_rows in by_source.values() for r in src_rows if r not in picked]
        rng.shuffle(remainder)
        picked.extend(remainder[: count - len(picked)])
    rng.shuffle(picked)
    return picked[:count]


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    excludes = load_excludes(args.exclude_sample_ids)
    rows = [r for r in load_rows(args.triplet_jsonl) if r.get("sample_id") not in excludes]
    rows_by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_group[row["scene_group"]].append(row)
    groups_by_source: dict[str, list[str]] = defaultdict(list)
    for group, group_rows in rows_by_group.items():
        src = Counter(r["source_type"] for r in group_rows).most_common(1)[0][0]
        groups_by_source[src].append(group)
    for group_list in groups_by_source.values():
        group_list.sort()

    source_counts = Counter(r["source_type"] for r in rows)
    total = sum(source_counts.values())
    ratios = {k: v / total for k, v in source_counts.items()}
    used_groups: set[str] = set()

    search_groups, search_rows = assign_group_pool(groups_by_source, rows_by_group, used_groups, args.search_dev_count, rng, ratios)
    shadow_groups, shadow_rows = assign_group_pool(groups_by_source, rows_by_group, used_groups, args.shadow_dev_count, rng, ratios)
    train_groups, train_rows = assign_group_pool(groups_by_source, rows_by_group, used_groups, args.train_count, rng, ratios)
    gate_rows = sample_rows_by_source(train_rows, args.gate_count, ratios, rng)

    outputs = {
        "train_source_pool": args.output_dir / "vor_train_source_pool_4096.jsonl",
        "search_dev": args.output_dir / "vor_search_dev_256.jsonl",
        "shadow_dev": args.output_dir / "vor_shadow_dev_256.jsonl",
        "gate128": args.output_dir / "vor_gate128.jsonl",
    }
    write_jsonl(outputs["train_source_pool"], train_rows)
    write_jsonl(outputs["search_dev"], search_rows)
    write_jsonl(outputs["shadow_dev"], shadow_rows)
    write_jsonl(outputs["gate128"], gate_rows)

    group_sets = {
        "train_source_pool": {r["scene_group"] for r in train_rows},
        "search_dev": {r["scene_group"] for r in search_rows},
        "shadow_dev": {r["scene_group"] for r in shadow_rows},
        "gate128": {r["scene_group"] for r in gate_rows},
    }
    overlap = {
        "train_search": sorted(group_sets["train_source_pool"] & group_sets["search_dev"]),
        "train_shadow": sorted(group_sets["train_source_pool"] & group_sets["shadow_dev"]),
        "search_shadow": sorted(group_sets["search_dev"] & group_sets["shadow_dev"]),
    }
    summary = {
        "seed": args.seed,
        "input": str(args.triplet_jsonl),
        "input_sha256": sha256(args.triplet_jsonl),
        "total_triplets": len(rows),
        "excluded_sample_ids": len(excludes),
        "total_scene_groups": len(rows_by_group),
        "source_counts": dict(source_counts),
        "outputs": {k: str(v) for k, v in outputs.items()},
        "counts": {k: sum(1 for _ in v.open()) for k, v in outputs.items()},
        "scene_group_counts": {k: len(v) for k, v in group_sets.items()},
        "group_overlap_counts": {k: len(v) for k, v in overlap.items()},
        "group_overlaps": overlap,
        "train_source_counts": dict(Counter(r["source_type"] for r in train_rows)),
        "search_dev_source_counts": dict(Counter(r["source_type"] for r in search_rows)),
        "shadow_dev_source_counts": dict(Counter(r["source_type"] for r in shadow_rows)),
        "gate128_source_counts": dict(Counter(r["source_type"] for r in gate_rows)),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VOR Group-Level Split Audit\n\n",
        f"- input_triplets: {len(rows)}\n",
        f"- input_sha256: `{summary['input_sha256']}`\n",
        f"- scene_groups: {len(rows_by_group)}\n",
        f"- source_counts: `{dict(source_counts)}`\n",
        f"- output_counts: `{summary['counts']}`\n",
        f"- scene_group_counts: `{summary['scene_group_counts']}`\n",
        f"- group_overlap_counts: `{summary['group_overlap_counts']}`\n",
        f"- gate128_source_counts: `{summary['gate128_source_counts']}`\n",
        "\nVOR-Eval is not read by this script and remains excluded from train/search/shadow construction.\n",
    ]
    args.report_md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
