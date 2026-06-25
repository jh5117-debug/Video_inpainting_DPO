#!/usr/bin/env python3
"""Select Exp25 DE-B Gate16 sources from the VOR train source pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-source-pool", type=Path, default=Path("exp25_vor_or_preference_data/manifests/vor_train_source_pool_4096.jsonl"))
    p.add_argument("--search-dev", type=Path, default=Path("exp25_vor_or_preference_data/manifests/vor_search_dev_256.jsonl"))
    p.add_argument("--shadow-dev", type=Path, default=Path("exp25_vor_or_preference_data/manifests/vor_shadow_dev_256.jsonl"))
    p.add_argument("--root-cause-manifest", type=Path, default=Path("/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_vor_or_preference_data/root_cause_matrix_20260625/root_cause_sample12_manifest.jsonl"))
    p.add_argument("--gate32-materialized", type=Path, default=Path("/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/materialized/gate32_canonical_d0_24f/gate32_materialized.jsonl"))
    p.add_argument("--output-manifest", type=Path, required=True)
    p.add_argument("--audit-json", type=Path, required=True)
    p.add_argument("--audit-csv", type=Path, required=True)
    p.add_argument("--audit-md", type=Path, required=True)
    p.add_argument("--limit", type=int, default=16)
    p.add_argument("--per-source-type", type=int, default=8)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ids_and_groups(rows: Iterable[dict]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    groups: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id") or row.get("source_sample_id") or "")
        if sample_id:
            ids.add(sample_id)
        source_sample_id = str(row.get("source_sample_id") or "")
        if source_sample_id:
            ids.add(source_sample_id)
        group = str(row.get("scene_group") or "")
        if group:
            groups.add(group)
    return ids, groups


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    train_rows = read_jsonl(args.train_source_pool)
    search_rows = read_jsonl(args.search_dev)
    shadow_rows = read_jsonl(args.shadow_dev)
    root_rows = read_jsonl(args.root_cause_manifest)
    gate32_rows = read_jsonl(args.gate32_materialized)

    excluded_ids: set[str] = set()
    excluded_groups: set[str] = set()
    for rows in (search_rows, shadow_rows, root_rows, gate32_rows):
        ids, groups = ids_and_groups(rows)
        excluded_ids.update(ids)
        excluded_groups.update(groups)

    selected: list[dict] = []
    used_groups: set[str] = set()
    source_counts: Counter[str] = Counter()
    target_source_counts = {"REAL": args.per_source_type, "BLENDER": args.per_source_type}

    def is_available(row: dict) -> bool:
        sid = str(row.get("sample_id", ""))
        group = str(row.get("scene_group", ""))
        if not sid or not group:
            return False
        return sid not in excluded_ids and group not in excluded_groups and group not in used_groups

    def add_row(row: dict) -> None:
        group = str(row.get("scene_group", ""))
        out = dict(row)
        out.update(
            {
                "gate": "EXP25_DIFFUSERASER_DEB_GATE16",
                "condition_source_role": "V_obj",
                "winner_source_role": "V_bg",
                "mask_source_role": "foreground_object_mask",
                "loser_stack_id": "DE-B_sft_raw6_d8_propainter",
                "hard_comp": False,
                "pcm_mode": "none",
                "prior_mode": "propainter",
                "no_pcm_steps": 6,
                "guidance": 0.0,
                "mask_dilation_iter": 8,
            }
        )
        selected.append(out)
        used_groups.add(group)
        source_counts[str(row.get("source_type", ""))] += 1

    for desired in ("REAL", "BLENDER"):
        for row in train_rows:
            if source_counts[desired] >= target_source_counts[desired] or len(selected) >= args.limit:
                break
            if row.get("source_type") != desired or not is_available(row):
                continue
            add_row(row)

    for row in train_rows:
        if len(selected) >= args.limit:
            break
        if not is_available(row):
            continue
        add_row(row)

    if len(selected) != args.limit:
        raise SystemExit(f"selected {len(selected)} rows, expected {args.limit}; counts={dict(source_counts)}")

    balance_status = "ideal_8_8" if all(source_counts[k] == target_source_counts[k] for k in target_source_counts) else "best_available_after_exclusions"
    balance_notes = (
        "REAL/BLENDER target is 8/8. Current source pool has fewer eligible rows for at least one source type after "
        "root-cause/search-dev/shadow-dev/Gate32 scene-group exclusions, so the selector filled to 16 with the best "
        "available disjoint sources and records the realized counts."
        if balance_status != "ideal_8_8"
        else "REAL/BLENDER balanced 8/8; mask size and motion/effect labels unavailable in current manifest, so scene-group/source balance is enforced here and dense review records quality buckets."
    )

    write_jsonl(args.output_manifest, selected)
    digest = sha256_file(args.output_manifest)
    audit = {
        "status": "GATE16_MANIFEST_LOCKED",
        "output_manifest": str(args.output_manifest),
        "manifest_sha256": digest,
        "rows": len(selected),
        "source_counts": dict(Counter(row.get("source_type", "") for row in selected)),
        "scene_groups": len({row.get("scene_group", "") for row in selected}),
        "excluded_sample_ids": len(excluded_ids),
        "excluded_scene_groups": len(excluded_groups),
        "root_cause_overlap": sorted({row["sample_id"] for row in selected} & {str(r.get("sample_id", "")) for r in root_rows}),
        "search_dev_overlap_groups": sorted({row["scene_group"] for row in selected} & {str(r.get("scene_group", "")) for r in search_rows}),
        "shadow_dev_overlap_groups": sorted({row["scene_group"] for row in selected} & {str(r.get("scene_group", "")) for r in shadow_rows}),
        "gate32_overlap_groups": sorted({row["scene_group"] for row in selected} & {str(r.get("scene_group", "")) for r in gate32_rows}),
        "balance_status": balance_status,
        "balance_notes": balance_notes,
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.audit_csv.open("w", encoding="utf-8", newline="") as fh:
        fields = ["sample_id", "source_type", "scene_group", "winner_member_path", "condition_member_path", "mask_member_path", "loser_stack_id", "hard_comp"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({key: row.get(key, "") for key in fields})
    args.audit_md.write_text(
        "\n".join(
            [
                "# Exp25 DE-B Gate16 Source Audit",
                "",
                f"- status: `{audit['status']}`",
                f"- manifest: `{args.output_manifest}`",
                f"- manifest_sha256: `{digest}`",
                f"- rows: `{len(selected)}`",
                f"- source_counts: `{audit['source_counts']}`",
                f"- scene_groups: `{audit['scene_groups']}`",
                f"- root_cause_overlap: `{audit['root_cause_overlap']}`",
                f"- search_dev_overlap_groups: `{audit['search_dev_overlap_groups']}`",
                f"- shadow_dev_overlap_groups: `{audit['shadow_dev_overlap_groups']}`",
                f"- gate32_overlap_groups: `{audit['gate32_overlap_groups']}`",
                "",
                "Stack locked to `DE-B_sft_raw6_d8_propainter`: no PCM, ProPainter prior, raw6, mask dilation 8, hard_comp=false.",
                "",
                audit["balance_notes"],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
