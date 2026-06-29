#!/usr/bin/env python3
"""Build Exp45 second-pass MiniMax targeted mining manifest.

The script is metadata-only: it selects locked VOR-Train source rows from the
Exp44 source manifest and assigns deterministic new seeds beyond the seeds
already mined in Exp44. It does not run inference or training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_GROUPS = (
    "BLENDER_MOUNTAIN002",
    "REAL_ENV059_00001",
    "BLENDER_SCHOOL004",
    "REAL_ENV097_00001",
    "REAL_ENV068_00002",
    "REAL_ENV105_00001",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(data, encoding="utf-8")
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def write_json(path: Path, obj: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    path.write_text(data, encoding="utf-8")
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def group_of(row: dict[str, Any]) -> str:
    return str(row.get("scene_group") or row.get("source_group") or str(row.get("sample_id", "")).rsplit("_", 1)[0])


def seed_index(row: dict[str, Any]) -> int | None:
    value = row.get("seed_index")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_groups(value: str) -> list[str]:
    groups = [part.strip() for part in value.split(",") if part.strip()]
    return groups or list(DEFAULT_GROUPS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp44-source-manifest", required=True, type=Path)
    parser.add_argument("--exp44-all-candidates", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--new-seeds-per-group", type=int, default=12)
    args = parser.parse_args()

    source_rows = read_jsonl(args.exp44_source_manifest)
    all_candidates = read_jsonl(args.exp44_all_candidates)
    selected_groups = parse_groups(args.groups)

    source_by_group: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        group = group_of(row)
        source_by_group.setdefault(group, row)

    used_seed_indices: dict[str, set[int]] = defaultdict(set)
    used_seeds: dict[str, set[int]] = defaultdict(set)
    for row in all_candidates:
        group = group_of(row)
        idx = seed_index(row)
        if idx is not None:
            used_seed_indices[group].add(idx)
        try:
            used_seeds[group].add(int(row.get("seed")))
        except (TypeError, ValueError):
            pass

    out_rows: list[dict[str, Any]] = []
    missing_groups: list[str] = []
    for priority, group in enumerate(selected_groups, 1):
        source = source_by_group.get(group)
        if source is None:
            missing_groups.append(group)
            continue
        seed_start = int(source.get("exp44_seed_start"))
        used = used_seed_indices.get(group, set())
        next_idx = (max(used) + 1) if used else len(source.get("exp44_seeds", []))
        seeds: list[int] = []
        idx = next_idx
        while len(seeds) < args.new_seeds_per_group:
            seed = seed_start + idx
            if seed not in used_seeds[group]:
                seeds.append(seed)
            idx += 1
        row = dict(source)
        row["exp44_seeds"] = seeds
        row["exp44_initial_new_candidate_budget"] = len(seeds)
        row["exp44_max_new_candidate_budget"] = len(seeds)
        row["exp44_bucket"] = "EXP45_pair_scaleup_second_pass"
        row["exp44_goal"] = "increase same-source success/failure density for formal train32/search16/shadow16 split"
        row["exp44_priority"] = priority
        row["exp45_scaleup_source"] = True
        row["exp45_selected_group"] = group
        row["exp45_new_seed_start_index"] = next_idx
        row["exp45_new_seed_count"] = len(seeds)
        row["vor_eval_used"] = False
        out_rows.append(row)

    manifest_sha = write_jsonl(args.output_manifest, out_rows)
    summary = {
        "status": "EXP45_SCALEUP_SOURCE_MANIFEST_READY" if out_rows else "EXP45_SCALEUP_SOURCE_MANIFEST_BLOCKED",
        "source_manifest": str(args.exp44_source_manifest),
        "source_manifest_sha256": sha256_file(args.exp44_source_manifest),
        "all_candidates": str(args.exp44_all_candidates),
        "all_candidates_sha256": sha256_file(args.exp44_all_candidates),
        "output_manifest": str(args.output_manifest),
        "output_manifest_sha256": manifest_sha,
        "groups_requested": selected_groups,
        "groups_selected": [group_of(row) for row in out_rows],
        "missing_groups": missing_groups,
        "rows": len(out_rows),
        "new_seeds_per_group": args.new_seeds_per_group,
        "candidate_budget": sum(len(row["exp44_seeds"]) for row in out_rows),
        "training_run": False,
        "optimizer_step": False,
        "vor_eval_used": False,
        "hard_comp_used": False,
    }
    write_json(args.output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
