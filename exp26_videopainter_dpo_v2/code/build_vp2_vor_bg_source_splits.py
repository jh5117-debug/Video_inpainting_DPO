#!/usr/bin/env python3
"""Build VideoPainter 49F source splits from VOR-Train BG fallback rows.

This does not extract videos, generate masks, or generate losers. It records a
formal source decision when sparse YouTube-VOS frames are insufficient.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-pool", type=Path, required=True)
    p.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    p.add_argument("--output-dir", type=Path, default=Path("exp26_videopainter_dpo_v2/manifests"))
    p.add_argument("--train-count", type=int, default=128)
    p.add_argument("--search-count", type=int, default=32)
    p.add_argument("--shadow-count", type=int, default=32)
    p.add_argument("--seed", type=int, default=20260623)
    p.add_argument("--frames", type=int, default=49)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    train_pool = read_jsonl(args.train_pool)
    excluded_samples: set[str] = set()
    excluded_groups: set[str] = set()
    for path in args.exclude_manifest:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            excluded_samples.add(str(row.get("sample_id", "")))
            excluded_groups.add(str(row.get("scene_group", "")))

    candidates_by_group: dict[str, list[dict]] = defaultdict(list)
    for row in train_pool:
        sample_id = str(row["sample_id"])
        group = str(row["scene_group"])
        if sample_id in excluded_samples or group in excluded_groups:
            continue
        winner_member = str(row["winner_member_path"])
        if not winner_member.startswith("VOR-Train/BG/"):
            continue
        candidates_by_group[group].append(
            {
                "sample_id": f"vp2_vor_bg_49f_{sample_id}",
                "source_sample_id": sample_id,
                "video_id": sample_id,
                "scene_group": group,
                "source_dataset": "VOR-Train-BG",
                "source_type": row.get("source_type", ""),
                "winner_member_path": winner_member,
                "winner_role": "BG",
                "num_frames": args.frames,
                "frame_mapping": "first_49_real_frames_after_extraction",
                "mask_generation": "moving_partial_mask_pending",
                "condition_definition": "winner * (1 - generated_moving_br_mask)",
                "first_frame_gt": True,
                "formal_49f": True,
                "plumbing_only_13f": False,
                "status": "SOURCE_ONLY_PENDING_EXTRACTION_AND_MASK_GENERATION",
                "identity_hash": sha256_text(sample_id + "|" + group + "|" + winner_member),
            }
        )

    rng = random.Random(args.seed)
    candidates: list[dict] = []
    for group in sorted(candidates_by_group):
        rows = candidates_by_group[group]
        rows.sort(key=lambda r: r["source_sample_id"])
        candidates.append(rng.choice(rows))
    rng.shuffle(candidates)
    need = args.train_count + args.search_count + args.shadow_count
    if len(candidates) < need:
        raise RuntimeError(
            f"need {need} scene-disjoint VOR-BG source candidates, "
            f"found {len(candidates)} unique scene groups"
        )

    train = candidates[: args.train_count]
    search = candidates[args.train_count : args.train_count + args.search_count]
    shadow = candidates[args.train_count + args.search_count : need]
    for split_name, rows in [("train_source_vor_bg", train), ("search_dev_vor_bg", search), ("shadow_dev_vor_bg", shadow)]:
        for row in rows:
            row["split"] = split_name

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": args.output_dir / f"vp2_vor_bg_train_source_{args.train_count}.jsonl",
        "search": args.output_dir / f"vp2_vor_bg_search_dev_{args.search_count}.jsonl",
        "shadow": args.output_dir / f"vp2_vor_bg_shadow_dev_{args.shadow_count}.jsonl",
    }
    write_jsonl(paths["train"], train)
    write_jsonl(paths["search"], search)
    write_jsonl(paths["shadow"], shadow)

    group_sets = {
        "train": {r["scene_group"] for r in train},
        "search": {r["scene_group"] for r in search},
        "shadow": {r["scene_group"] for r in shadow},
    }
    stats = {
        "status": "VOR_BG_SOURCE_SPLIT_LOCKED_PENDING_EXTRACTION_MASKS",
        "train_pool": str(args.train_pool),
        "exclude_manifests": [str(p) for p in args.exclude_manifest],
        "seed": args.seed,
        "frames": args.frames,
        "candidate_count_after_exclusions": sum(len(v) for v in candidates_by_group.values()),
        "unique_scene_group_count_after_exclusions": len(candidates),
        "train_count": len(train),
        "search_count": len(search),
        "shadow_count": len(shadow),
        "train_search_group_overlap": len(group_sets["train"] & group_sets["search"]),
        "train_shadow_group_overlap": len(group_sets["train"] & group_sets["shadow"]),
        "search_shadow_group_overlap": len(group_sets["search"] & group_sets["shadow"]),
        "paths": {k: str(v) for k, v in paths.items()},
        "manifest_sha256": {k: sha256_text(v.read_text(encoding="utf-8")) for k, v in paths.items()},
        "notes": "VOR-Train BG selected as fallback clean BR source; VOR-Eval excluded by source construction.",
    }
    stats_path = args.output_dir / "vp2_vor_bg_49f_source_split_statistics.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = args.output_dir / "vp2_vor_bg_49f_source_split_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "sample_id", "source_sample_id", "scene_group", "winner_member_path", "identity_hash"])
        writer.writeheader()
        for row in train + search + shadow:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames or []})
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
