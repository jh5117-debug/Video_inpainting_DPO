#!/usr/bin/env python3
"""Build canonical VOR OR triplets from full member indexes."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-index", type=Path, required=True)
    p.add_argument("--mask-index", type=Path, required=True)
    p.add_argument("--output-jsonl", type=Path, default=Path("exp25_vor_or_preference_data/manifests/vor_train_metadata_index.jsonl"))
    p.add_argument("--summary-md", type=Path, default=Path("reports/vor_train_pairing_rule.md"))
    p.add_argument("--sample-limit", type=int, default=0)
    return p.parse_args()


def basename_no_ext(path: str) -> str:
    return Path(path).stem


def role_from_path(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    if "FG_BG" in parts:
        return "FG_BG"
    if "BG" in parts:
        return "BG"
    if "MASK" in parts or (parts and parts[0] == "MASK"):
        return "MASK"
    return ""


def scene_group_from_id(sample_id: str) -> str:
    # Examples observed:
    #   REAL_ENV900_00001_001_03 -> REAL_ENV900_00001
    #   BLENDER_BEACH038_06047 -> BLENDER_BEACH038
    parts = sample_id.split("_")
    if sample_id.startswith("REAL_") and len(parts) >= 3:
        return "_".join(parts[:3])
    if sample_id.startswith("BLENDER_") and len(parts) >= 2:
        return "_".join(parts[:2])
    if len(parts) > 1:
        return "_".join(parts[:-1])
    return sample_id


def read_role_map(index_paths: list[Path]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for path in index_paths:
        with path.open() as f:
            for row in csv.DictReader(f):
                if row.get("type") != "file":
                    continue
                member = row.get("member_path", "")
                if not member.lower().endswith((".mp4", ".mov", ".avi")):
                    continue
                role = role_from_path(member)
                if not role:
                    continue
                sid = basename_no_ext(member)
                out.setdefault(sid, {})[role] = member
    return out


def main() -> int:
    args = parse_args()
    role_map = read_role_map([args.train_index, args.mask_index])
    rows = []
    incomplete = []
    for sid in sorted(role_map):
        roles = role_map[sid]
        missing = [r for r in ["FG_BG", "BG", "MASK"] if r not in roles]
        if missing:
            incomplete.append({"sample_id": sid, "missing": missing, "roles": sorted(roles)})
            continue
        row = {
            "sample_id": sid,
            "scene_group": scene_group_from_id(sid),
            "condition_role": "FG_BG",
            "winner_role": "BG",
            "mask_role": "MASK",
            "condition_member_path": roles["FG_BG"],
            "winner_member_path": roles["BG"],
            "mask_member_path": roles["MASK"],
            "task": "object_removal",
            "hard_comp": False,
            "comp_mode": "none",
        }
        rows.append(row)
        if args.sample_limit and len(rows) >= args.sample_limit:
            break
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    groups = {r["scene_group"] for r in rows}
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(
        "# VOR Train Pairing Rule\n\n"
        "Pairing is by exact video basename across `VOR-Train/FG_BG`, "
        "`VOR-Train/BG`, and `MASK`.\n\n"
        f"- complete_triplets: {len(rows)}\n"
        f"- incomplete_video_ids: {len(incomplete)}\n"
        f"- scene_groups: {len(groups)}\n"
        "- condition = FG_BG / V_obj\n"
        "- winner = BG / V_bg\n"
        "- mask = MASK / foreground object mask\n"
    )
    print(json.dumps({"complete_triplets": len(rows), "incomplete_video_ids": len(incomplete), "scene_groups": len(groups)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
