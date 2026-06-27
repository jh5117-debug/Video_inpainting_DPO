#!/usr/bin/env python3
"""Build Exp30 VOR-OR source-pool v2 from the full metadata index.

This is metadata-only sampling. It does not scan tar archives and does not
extract videos.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


KNOWN_BAD_SAMPLE_IDS = {"BLENDER_RIVER007_00001"}
SPLIT_TARGETS = {
    "primary": 128,
    "reserve": 128,
    "reserve2": 128,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_source_type(sample_id: str) -> str:
    if sample_id.startswith("REAL_"):
        return "REAL"
    if sample_id.startswith("BLENDER_"):
        return "BLENDER"
    return "UNKNOWN"


def scene_from_sample(sample_id: str) -> str:
    if sample_id.startswith("BLENDER_"):
        parts = sample_id.split("_")
        return "_".join(parts[:2]) if len(parts) >= 2 else sample_id
    if sample_id.startswith("REAL_"):
        parts = sample_id.split("_")
        return "_".join(parts[:3]) if len(parts) >= 3 else sample_id
    return sample_id


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_exclusions(paths: list[Path]) -> tuple[set[str], set[str], list[dict]]:
    sample_ids: set[str] = set()
    scene_groups: set[str] = set()
    sources: list[dict] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".jsonl":
                rows = iter_jsonl(path)
            elif path.suffix.lower() == ".csv":
                rows = csv.DictReader(path.open(newline=""))
            else:
                continue
            for row in rows:
                sample_id = (
                    row.get("sample_id")
                    or row.get("source_sample_id")
                    or row.get("source_id")
                    or row.get("id")
                    or ""
                )
                scene_group = row.get("scene_group") or row.get("source_group") or ""
                if sample_id and not scene_group:
                    scene_group = scene_from_sample(sample_id)
                if sample_id:
                    sample_ids.add(sample_id)
                if scene_group:
                    scene_groups.add(scene_group)
            sources.append({"path": str(path), "exists": True})
        except Exception as exc:  # noqa: BLE001 - report and continue
            sources.append({"path": str(path), "exists": True, "error": repr(exc)})
    return sample_ids, scene_groups, sources


def load_metadata(path: Path) -> list[dict]:
    rows: list[dict] = []
    for row in iter_jsonl(path):
        sample_id = row["sample_id"]
        if sample_id in KNOWN_BAD_SAMPLE_IDS:
            continue
        source_type = infer_source_type(sample_id)
        rows.append(
            {
                "sample_id": sample_id,
                "scene_group": row.get("scene_group") or scene_from_sample(sample_id),
                "source_type": source_type,
                "effect_type": "unknown",
                "mask_bucket": "unknown",
                "mask_area_stats": None,
                "condition_member_path": row["condition_member_path"],
                "winner_member_path": row["winner_member_path"],
                "mask_member_path": row["mask_member_path"],
                "basename": sample_id,
                "num_frames_estimate": None,
                "affected_map_status": "unknown_metadata_only",
                "selection_reason": "",
                "prior_diagnostic_seen": False,
                "visual_preview_status": "metadata_only_visual_preview_pending",
            }
        )
    return rows


def representative_by_scene(rows: list[dict]) -> dict[str, dict]:
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_scene[row["scene_group"]].append(row)
    reps = {}
    for scene_group, group_rows in by_scene.items():
        reps[scene_group] = sorted(group_rows, key=lambda r: r["sample_id"])[0]
    return reps


def take_balanced(candidates: list[dict], total: int) -> list[dict]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(candidates, key=lambda r: (r["source_type"], r["scene_group"], r["sample_id"])):
        by_type[row["source_type"]].append(row)

    selected: list[dict] = []
    target_each = total // 2
    for source_type in ("BLENDER", "REAL"):
        selected.extend(by_type[source_type][:target_each])
        by_type[source_type] = by_type[source_type][target_each:]

    if len(selected) < total:
        leftovers = []
        for rows in by_type.values():
            leftovers.extend(rows)
        leftovers = sorted(leftovers, key=lambda r: (r["source_type"], r["scene_group"], r["sample_id"]))
        selected.extend(leftovers[: total - len(selected)])

    return selected[:total]


def assign_splits(candidates: list[dict]) -> tuple[dict[str, list[dict]], list[dict]]:
    remaining = list(candidates)
    splits: dict[str, list[dict]] = {}
    for split, target in SPLIT_TARGETS.items():
        selected = take_balanced(remaining, target)
        selected_keys = {row["scene_group"] for row in selected}
        for row in selected:
            row["accepted_split"] = split
            row["selection_reason"] = "balanced_metadata_sampling_from_full_vor_index"
        splits[split] = selected
        remaining = [row for row in remaining if row["scene_group"] not in selected_keys]
    rejected = []
    for row in remaining:
        rejected.append(
            {
                "sample_id": row["sample_id"],
                "scene_group": row["scene_group"],
                "source_type": row["source_type"],
                "reason": "not_selected_after_primary_reserve_reserve2_targets",
            }
        )
    return splits, rejected


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return sha256_file(path)


def write_outputs(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    reports = out / "reports"
    manifests = out / "manifests"
    reports.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    exclude_paths = [Path(p) for p in args.exclude_file]
    excluded_sample_ids, excluded_scene_groups, exclusion_sources = load_exclusions(exclude_paths)

    metadata_rows = load_metadata(Path(args.metadata_index))
    reps_all = representative_by_scene(metadata_rows)

    strict_candidates = []
    prior_seen_count = 0
    rejected_prior = []
    for scene_group, row in sorted(reps_all.items()):
        if scene_group in excluded_scene_groups or row["sample_id"] in excluded_sample_ids:
            prior_seen_count += 1
            rejected_prior.append(
                {
                    "sample_id": row["sample_id"],
                    "scene_group": scene_group,
                    "source_type": row["source_type"],
                    "reason": "prior_diagnostic_seen_strict_exclusion",
                }
            )
            continue
        strict_candidates.append(row)

    relaxed_used = False
    candidate_pool = strict_candidates
    if len(candidate_pool) < 384:
        relaxed_used = True
        candidate_pool = list(reps_all.values())
        for row in candidate_pool:
            if row["scene_group"] in excluded_scene_groups or row["sample_id"] in excluded_sample_ids:
                row["prior_diagnostic_seen"] = True

    splits, rejected_unselected = assign_splits(candidate_pool)
    rejected = rejected_prior + rejected_unselected

    primary = splits["primary"]
    reserve = splits["reserve"]
    reserve2 = splits["reserve2"]

    status = "VOR_OR_SOURCE_POOL_V2_READY"
    if len(primary) < 128 or len(reserve) < 64:
        status = "VOR_OR_SOURCE_POOL_V2_BLOCKED"

    manifest_info = {}
    manifest_info["primary_sha256"] = write_jsonl(
        manifests / "vor_or_source_pool_v2_primary128.jsonl", primary
    )
    manifest_info["reserve_sha256"] = write_jsonl(
        manifests / "vor_or_source_pool_v2_reserve128.jsonl", reserve
    )
    manifest_info["reserve2_sha256"] = write_jsonl(
        manifests / "vor_or_source_pool_v2_reserve2_128.jsonl", reserve2
    )
    manifest_info["rejected_sha256"] = write_jsonl(
        manifests / "vor_or_source_pool_v2_rejected.jsonl", rejected
    )

    def counts(rows: list[dict], key: str) -> dict:
        return dict(Counter(row.get(key, "unknown") for row in rows))

    summary = {
        "status": status,
        "metadata_index": args.metadata_index,
        "metadata_sha256": sha256_file(Path(args.metadata_index)),
        "metadata_rows_after_quarantine": len(metadata_rows),
        "scene_groups_total_after_quarantine": len(reps_all),
        "strict_exclusion_scene_groups": len(excluded_scene_groups),
        "strict_exclusion_sample_ids": len(excluded_sample_ids),
        "strict_candidates": len(strict_candidates),
        "relaxed_diagnostic_exclusion_used": relaxed_used,
        "prior_diagnostic_seen_count": prior_seen_count,
        "primary_count": len(primary),
        "reserve_count": len(reserve),
        "reserve2_count": len(reserve2),
        "rejected_count": len(rejected),
        "primary_source_type_counts": counts(primary, "source_type"),
        "reserve_source_type_counts": counts(reserve, "source_type"),
        "reserve2_source_type_counts": counts(reserve2, "source_type"),
        "primary_mask_bucket_counts": counts(primary, "mask_bucket"),
        "reserve_mask_bucket_counts": counts(reserve, "mask_bucket"),
        "effect_type_counts_primary": counts(primary, "effect_type"),
        "manifest_sha256": manifest_info,
        "exclusion_sources": exclusion_sources,
        "archive_scanned": False,
        "videos_extracted": False,
        "visual_preview_status": "metadata_only_visual_preview_pending",
    }

    (reports / "exp30_vor_or_source_pool_v2_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    with (reports / "exp30_vor_or_source_pool_v2_sampling.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "accepted_split",
                "sample_id",
                "scene_group",
                "source_type",
                "effect_type",
                "mask_bucket",
                "condition_member_path",
                "winner_member_path",
                "mask_member_path",
                "prior_diagnostic_seen",
                "visual_preview_status",
            ],
        )
        writer.writeheader()
        for split in ("primary", "reserve", "reserve2"):
            for row in splits[split]:
                writer.writerow({field: row.get(field, "") for field in writer.fieldnames})

    with (reports / "exp30_vor_or_source_pool_v2_preview_review.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "rows",
                "review_status",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "primary128",
                "rows": len(primary),
                "review_status": "VISUAL_PREVIEW_PENDING",
                "reason": "metadata-only source-pool lock; videos not extracted yet",
            }
        )
        writer.writerow(
            {
                "split": "reserve128",
                "rows": len(reserve),
                "review_status": "VISUAL_PREVIEW_PENDING",
                "reason": "metadata-only source-pool lock; videos not extracted yet",
            }
        )

    md = [
        "# Exp30 VOR-OR Source Pool V2 Sampling",
        "",
        f"Status: `{status}`",
        "",
        "## Inputs",
        "",
        f"- Metadata index: `{args.metadata_index}`",
        f"- Metadata SHA256: `{summary['metadata_sha256']}`",
        f"- Metadata rows after known-bad quarantine: {len(metadata_rows)}",
        f"- Scene groups after known-bad quarantine: {len(reps_all)}",
        "",
        "## Exclusion",
        "",
        f"- Strict excluded scene groups: {len(excluded_scene_groups)}",
        f"- Strict excluded sample ids: {len(excluded_sample_ids)}",
        f"- Strict candidates after exclusion: {len(strict_candidates)}",
        f"- Relaxed diagnostic exclusion used: {relaxed_used}",
        "",
        "## Selected Pools",
        "",
        f"- Primary rows: {len(primary)}",
        f"- Reserve rows: {len(reserve)}",
        f"- Reserve2 rows: {len(reserve2)}",
        f"- Primary source type counts: `{summary['primary_source_type_counts']}`",
        f"- Reserve source type counts: `{summary['reserve_source_type_counts']}`",
        f"- Reserve2 source type counts: `{summary['reserve2_source_type_counts']}`",
        f"- Mask buckets: metadata unavailable, recorded as `unknown` rather than inferred.",
        f"- Effect labels: metadata unavailable, recorded as `unknown` rather than inferred.",
        "",
        "## Preview",
        "",
        "No videos were extracted in this milestone. Preview status is",
        "`metadata_only_visual_preview_pending`; full video evidence remains required",
        "before any smoke pass, data-ready, or adapter-positive claim.",
        "",
        "## Decision",
        "",
    ]
    if status == "VOR_OR_SOURCE_POOL_V2_READY":
        md.append(
            "Source-pool v2 meets the count gate: primary128 exists and reserve "
            "contains at least 64 rows. It unlocks multi-model OR smoke16, but "
            "does not itself constitute video-review or data-ready evidence."
        )
    else:
        md.append("Source-pool v2 remains blocked before any downstream smoke.")
    md.append("")
    (reports / "exp30_vor_or_source_pool_v2_sampling.md").write_text("\n".join(md))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exclude-file", action="append", default=[])
    args = parser.parse_args()
    write_outputs(args)


if __name__ == "__main__":
    main()
