#!/usr/bin/env python3
"""Build formal 49-frame VideoPainter source splits from YouTube-VOS.

This creates source manifests only. It does not generate losers or run DPO.
The manifests are grouped by video id, require at least 49 real frames and
masks, and keep DAVIS names out of the selection audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--youtubevos-root", type=Path, required=True)
    p.add_argument("--davis-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("exp26_videopainter_dpo_v2/manifests"))
    p.add_argument("--train-count", type=int, default=512)
    p.add_argument("--search-count", type=int, default=64)
    p.add_argument("--shadow-count", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260623)
    p.add_argument("--frames", type=int, default=49)
    return p.parse_args()


def image_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def image_count(path: Path) -> int:
    if not path.is_dir():
        return 0
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file() and Path(entry.name).suffix.lower() in IMG_EXTS:
                total += 1
    return total


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def davis_ids(root: Path) -> set[str]:
    ids: set[str] = set()
    for sub in ["JPEGImages_432_240", "JPEGImages", "test_masks"]:
        d = root / sub
        if d.is_dir():
            ids.update(p.name for p in d.iterdir() if p.is_dir())
    return ids


def build_candidates(youtubevos_root: Path, davis: set[str], frames: int) -> tuple[list[dict], list[dict]]:
    image_root = youtubevos_root / "JPEGImages"
    mask_root = youtubevos_root / "Annotations"
    candidates: list[dict] = []
    failures: list[dict] = []
    for video_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        video_id = video_dir.name
        mask_dir = mask_root / video_id
        frame_count = image_count(video_dir)
        mask_count = image_count(mask_dir)
        if video_id in davis:
            failures.append({"video_id": video_id, "reason": "davis_name_overlap"})
            continue
        if frame_count < frames or mask_count < frames:
            failures.append({"video_id": video_id, "reason": "insufficient_49f", "frames": frame_count, "masks": mask_count})
            continue
        frame_files = image_files(video_dir)
        mask_files = image_files(mask_dir)
        frame_names = [p.name for p in frame_files[:frames]]
        mask_names = [p.name for p in mask_files[:frames]]
        candidates.append(
            {
                "sample_id": f"ytvos49_{video_id}",
                "video_id": video_id,
                "scene_group": video_id,
                "source_dataset": "youtubevos_2019_train",
                "win_video_path": str(video_dir),
                "condition_video_path": str(video_dir),
                "mask_path": str(mask_dir),
                "num_frames": frames,
                "frame_indices": list(range(frames)),
                "frame_names": frame_names,
                "mask_names": mask_names,
                "prompt": "",
                "first_frame_gt": True,
                "formal_49f": True,
                "plumbing_only_13f": False,
                "status": "SOURCE_ONLY_PENDING_SELF_LOSER",
                "identity_hash": sha256_text(video_id + "|" + "|".join(frame_names) + "|" + "|".join(mask_names)),
            }
        )
    return candidates, failures


def main() -> int:
    args = parse_args()
    davis = davis_ids(args.davis_root)
    candidates, failures = build_candidates(args.youtubevos_root, davis, args.frames)
    need = args.train_count + args.search_count + args.shadow_count
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failure_path = args.output_dir / "vp2_49f_source_failures.csv"
    with failure_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "reason", "frames", "masks"])
        writer.writeheader()
        for row in failures:
            writer.writerow(row)
    frame_counts = [int(row.get("frames", 0) or 0) for row in failures if row.get("reason") == "insufficient_49f"]
    mask_counts = [int(row.get("masks", 0) or 0) for row in failures if row.get("reason") == "insufficient_49f"]
    pre_stats = {
        "youtubevos_root": str(args.youtubevos_root),
        "davis_root": str(args.davis_root),
        "seed": args.seed,
        "frames": args.frames,
        "required_valid_candidates": need,
        "valid_candidates": len(candidates),
        "failed_candidates": len(failures),
        "max_frame_count_seen": max(frame_counts) if frame_counts else None,
        "max_mask_count_seen": max(mask_counts) if mask_counts else None,
        "failure_path": str(failure_path),
    }
    if len(candidates) < need:
        pre_stats["status"] = "BLOCKED_INSUFFICIENT_49F_SOURCE"
        pre_stats["reason"] = f"need {need} valid 49f videos, found {len(candidates)}"
        stats_path = args.output_dir / "vp2_49f_source_split_statistics.json"
        stats_path.write_text(json.dumps(pre_stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(pre_stats, indent=2, sort_keys=True))
        return 2
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    train = candidates[: args.train_count]
    search = candidates[args.train_count : args.train_count + args.search_count]
    shadow = candidates[args.train_count + args.search_count : need]
    for split_name, rows in [("train_source_512", train), ("search_dev_64", search), ("shadow_dev_64", shadow)]:
        for row in rows:
            row["split"] = split_name
    paths = {
        "train": args.output_dir / "vp2_train_source_512.jsonl",
        "search": args.output_dir / "vp2_search_dev_64.jsonl",
        "shadow": args.output_dir / "vp2_shadow_dev_64.jsonl",
    }
    write_jsonl(paths["train"], train)
    write_jsonl(paths["search"], search)
    write_jsonl(paths["shadow"], shadow)
    split_groups = {
        "train": {r["scene_group"] for r in train},
        "search": {r["scene_group"] for r in search},
        "shadow": {r["scene_group"] for r in shadow},
    }
    stats = {
        "youtubevos_root": str(args.youtubevos_root),
        "davis_root": str(args.davis_root),
        "seed": args.seed,
        "frames": args.frames,
        "valid_candidates": len(candidates),
        "failed_candidates": len(failures),
        "status": "LOCKED_49F_SOURCE_SPLITS",
        "max_frame_count_seen": max(frame_counts) if frame_counts else None,
        "max_mask_count_seen": max(mask_counts) if mask_counts else None,
        "train_count": len(train),
        "search_count": len(search),
        "shadow_count": len(shadow),
        "train_search_overlap": len(split_groups["train"] & split_groups["search"]),
        "train_shadow_overlap": len(split_groups["train"] & split_groups["shadow"]),
        "search_shadow_overlap": len(split_groups["search"] & split_groups["shadow"]),
        "davis_overlap": sum(1 for r in train + search + shadow if r["video_id"] in davis),
        "manifest_sha256": {k: sha256_text(v.read_text(encoding="utf-8")) for k, v in paths.items()},
        "paths": {k: str(v) for k, v in paths.items()},
    }
    stats_path = args.output_dir / "vp2_49f_source_split_statistics.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
