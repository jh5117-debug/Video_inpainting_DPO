#!/usr/bin/env python3
"""Recover and audit the full VOR-OR valid triplet index for Exp30.

The script reads the existing Exp25 metadata/member indexes only. It does not
open VOR tar archives and does not extract videos.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


REQUIRED_FIELDS = {
    "sample_id",
    "scene_group",
    "condition_member_path",
    "winner_member_path",
    "mask_member_path",
    "condition_role",
    "winner_role",
    "mask_role",
    "task",
    "comp_mode",
    "hard_comp",
}

KNOWN_BAD_SAMPLE_IDS = {"BLENDER_RIVER007_00001"}


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


def expected_paths(sample_id: str) -> tuple[str, str, str]:
    return (
        f"VOR-Train/FG_BG/{sample_id}.mp4",
        f"VOR-Train/BG/{sample_id}.mp4",
        f"MASK/{sample_id}.mp4",
    )


def audit_metadata(path: Path) -> dict:
    row_count = 0
    valid_count = 0
    quarantined_count = 0
    invalid_rows: list[dict] = []
    source_type_counts: Counter[str] = Counter()
    scene_groups: Counter[str] = Counter()
    field_missing_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    sample_ids: set[str] = set()
    duplicate_sample_ids: list[str] = []
    examples: list[dict] = []

    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row_count += 1
            row = json.loads(line)
            if len(examples) < 5:
                examples.append(row)

            missing = sorted(REQUIRED_FIELDS.difference(row))
            for field in missing:
                field_missing_counts[field] += 1

            sample_id = row.get("sample_id", "")
            if sample_id in sample_ids:
                duplicate_sample_ids.append(sample_id)
            sample_ids.add(sample_id)
            scene_group = row.get("scene_group", "")
            source_type = infer_source_type(sample_id)
            source_type_counts[source_type] += 1
            scene_groups[scene_group] += 1

            condition_expected, winner_expected, mask_expected = expected_paths(sample_id)
            reasons = []
            if missing:
                reasons.append("missing_required_fields")
            if row.get("condition_member_path") != condition_expected:
                reasons.append("condition_path_mismatch")
            if row.get("winner_member_path") != winner_expected:
                reasons.append("winner_path_mismatch")
            if row.get("mask_member_path") != mask_expected:
                reasons.append("mask_path_mismatch")
            if row.get("condition_role") != "FG_BG":
                reasons.append("condition_role_not_fg_bg")
            if row.get("winner_role") != "BG":
                reasons.append("winner_role_not_bg")
            if row.get("mask_role") != "MASK":
                reasons.append("mask_role_not_mask")
            if row.get("task") != "object_removal":
                reasons.append("task_not_object_removal")
            if row.get("hard_comp") is not False:
                reasons.append("hard_comp_not_false")
            if "VOR-Eval" in " ".join(str(row.get(k, "")) for k in row):
                reasons.append("vor_eval_path")

            role_counts[f"condition={row.get('condition_role')}"] += 1
            role_counts[f"winner={row.get('winner_role')}"] += 1
            role_counts[f"mask={row.get('mask_role')}"] += 1

            if sample_id in KNOWN_BAD_SAMPLE_IDS:
                quarantined_count += 1
                reasons.append("known_bad_quarantined")

            if reasons:
                if len(invalid_rows) < 100:
                    invalid_rows.append(
                        {
                            "line_no": line_no,
                            "sample_id": sample_id,
                            "scene_group": scene_group,
                            "reasons": reasons,
                        }
                    )
                if reasons == ["known_bad_quarantined"]:
                    continue
                continue

            valid_count += 1

    return {
        "row_count": row_count,
        "valid_count": valid_count,
        "quarantined_count": quarantined_count,
        "invalid_rows_sample": invalid_rows,
        "invalid_count_including_quarantine": row_count - valid_count,
        "source_type_counts": dict(source_type_counts),
        "scene_group_count": len(scene_groups),
        "scene_group_row_count_min": min(scene_groups.values()) if scene_groups else 0,
        "scene_group_row_count_max": max(scene_groups.values()) if scene_groups else 0,
        "field_missing_counts": dict(field_missing_counts),
        "role_counts": dict(role_counts),
        "duplicate_sample_id_count": len(duplicate_sample_ids),
        "duplicate_sample_ids_sample": duplicate_sample_ids[:20],
        "examples": examples,
    }


def audit_member_index(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {"available": False}

    row_count = 0
    file_count = 0
    unsafe_count = 0
    type_counts: Counter[str] = Counter()
    prefix_counts: Counter[str] = Counter()
    fields: list[str] | None = None
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        for row in reader:
            row_count += 1
            type_counts[row.get("type", "")] += 1
            if row.get("unsafe_reason"):
                unsafe_count += 1
            member = row.get("member_path", "")
            if row.get("type") == "file":
                file_count += 1
                if member.startswith("VOR-Train/FG_BG/"):
                    prefix_counts["VOR-Train/FG_BG"] += 1
                elif member.startswith("VOR-Train/BG/"):
                    prefix_counts["VOR-Train/BG"] += 1
                elif member.startswith("MASK/"):
                    prefix_counts["MASK"] += 1
                else:
                    prefix_counts["OTHER_FILE"] += 1
    return {
        "available": True,
        "row_count": row_count,
        "file_count": file_count,
        "unsafe_count": unsafe_count,
        "fields": fields,
        "type_counts": dict(type_counts),
        "prefix_counts": dict(prefix_counts),
    }


def write_outputs(args: argparse.Namespace, metadata_audit: dict, member_audit: dict) -> None:
    out = Path(args.output_dir)
    reports = out / "reports"
    manifests = out / "manifests"
    reports.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(args.metadata_index)
    member_path = Path(args.member_index) if args.member_index else None
    metadata_sha = sha256_file(metadata_path)
    member_sha = sha256_file(member_path) if member_path and member_path.exists() else None

    status = "FULL_VOR_VALID_TRIPLET_INDEX_READY"
    if metadata_audit["valid_count"] < 57750:
        status = "FULL_VOR_INDEX_BLOCKED"
    if metadata_audit["duplicate_sample_id_count"] != 0:
        status = "FULL_VOR_INDEX_BLOCKED"
    if metadata_audit["invalid_count_including_quarantine"] != metadata_audit["quarantined_count"]:
        status = "FULL_VOR_INDEX_BLOCKED"

    summary = {
        "status": status,
        "metadata_index": str(metadata_path),
        "metadata_sha256": metadata_sha,
        "metadata_rows": metadata_audit["row_count"],
        "valid_triplet_count": metadata_audit["valid_count"],
        "quarantined_known_bad_count": metadata_audit["quarantined_count"],
        "member_index": str(member_path) if member_path else None,
        "member_index_sha256": member_sha,
        "metadata_audit": metadata_audit,
        "member_index_audit": member_audit,
        "pairing_rule": {
            "condition": "VOR-Train/FG_BG/<sample_id>.mp4",
            "winner": "VOR-Train/BG/<sample_id>.mp4",
            "mask": "MASK/<sample_id>.mp4",
        },
        "archive_scanned": False,
        "videos_extracted": False,
        "vor_eval_used": False,
    }

    ref = {
        "status": status,
        "source_of_truth": "Exp25 full VOR-Train metadata index",
        "metadata_index_path": str(metadata_path),
        "metadata_index_sha256": metadata_sha,
        "metadata_rows": metadata_audit["row_count"],
        "valid_triplet_count": metadata_audit["valid_count"],
        "quarantined_sample_ids": sorted(KNOWN_BAD_SAMPLE_IDS),
        "member_index_path": str(member_path) if member_path else None,
        "member_index_sha256": member_sha,
        "pairing_rule": summary["pairing_rule"],
        "source_type_rule": "Infer from sample_id prefix REAL_ or BLENDER_",
    }

    (reports / "exp30_full_vor_index_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (manifests / "vor_or_full_valid_triplet_index_ref.json").write_text(
        json.dumps(ref, indent=2, sort_keys=True) + "\n"
    )

    with (reports / "exp30_full_vor_index_recovery.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        writer.writerow(["status", status])
        writer.writerow(["metadata_index", metadata_path])
        writer.writerow(["metadata_sha256", metadata_sha])
        writer.writerow(["metadata_rows", metadata_audit["row_count"]])
        writer.writerow(["valid_triplet_count", metadata_audit["valid_count"]])
        writer.writerow(["quarantined_known_bad_count", metadata_audit["quarantined_count"]])
        writer.writerow(["scene_group_count", metadata_audit["scene_group_count"]])
        writer.writerow(["source_type_counts", json.dumps(metadata_audit["source_type_counts"], sort_keys=True)])
        writer.writerow(["member_index", str(member_path) if member_path else ""])
        writer.writerow(["member_index_sha256", member_sha or ""])
        writer.writerow(["member_prefix_counts", json.dumps(member_audit.get("prefix_counts", {}), sort_keys=True)])

    md = [
        "# Exp30 Full VOR Index Recovery",
        "",
        f"Status: `{status}`",
        "",
        "## Source Of Truth",
        "",
        f"- Metadata index: `{metadata_path}`",
        f"- Metadata SHA256: `{metadata_sha}`",
        f"- Metadata rows: {metadata_audit['row_count']}",
        f"- Valid triplets after quarantining known bad rows: {metadata_audit['valid_count']}",
        f"- Known bad quarantined rows: {metadata_audit['quarantined_count']}",
        f"- Scene groups: {metadata_audit['scene_group_count']}",
        f"- Source type counts: `{metadata_audit['source_type_counts']}`",
        "",
        "## Pairing Rule",
        "",
        "- condition: `VOR-Train/FG_BG/<sample_id>.mp4`",
        "- winner: `VOR-Train/BG/<sample_id>.mp4`",
        "- mask: `MASK/<sample_id>.mp4`",
        "",
        "## Member Index",
        "",
        f"- Member index: `{member_path}`" if member_path else "- Member index: not provided",
        f"- Member SHA256: `{member_sha}`" if member_sha else "- Member SHA256: unavailable",
        f"- Member audit: `{member_audit}`",
        "",
        "## Safety",
        "",
        "- VOR tar archives scanned: no",
        "- Videos extracted: no",
        "- VOR-Eval used: no",
        "- Exp25 worktree modified: no",
        "",
        "## Decision",
        "",
    ]
    if status == "FULL_VOR_VALID_TRIPLET_INDEX_READY":
        md.append(
            "The full VOR metadata index is a usable source-of-truth for Exp30 "
            "source-pool v2 sampling. The previous 192-triplet result was an "
            "exact-extraction-cache subset, not the full VOR-Train inventory."
        )
    else:
        md.append(
            "The full VOR metadata index could not be promoted. See the JSON "
            "summary for invalid-row diagnostics."
        )
    md.append("")
    (reports / "exp30_full_vor_index_recovery.md").write_text("\n".join(md))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-index", required=True)
    parser.add_argument("--member-index")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    metadata_path = Path(args.metadata_index)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    member_path = Path(args.member_index) if args.member_index else None
    metadata_audit = audit_metadata(metadata_path)
    member_audit = audit_member_index(member_path)
    write_outputs(args, metadata_audit, member_audit)


if __name__ == "__main__":
    main()
