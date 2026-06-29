#!/usr/bin/env python3
"""Build an Exp45 H20 handoff filelist without touching H20.

The script reads Exp44 repository-side manifests and records every absolute
path a later H20 session may need to mirror. It computes SHA256 only for files
that are visible in the current PAI/session filesystem.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MANIFESTS = [
    "exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_stage2_pseudo_success_train.jsonl",
    "exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_stage2_pseudo_success_search.jsonl",
    "exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_stage2_pseudo_success_shadow.jsonl",
    "exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_same_source_pairs_all.jsonl",
    "exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_badnoise_v4_states.jsonl",
]

LOCAL_REPO_ARTIFACTS = [
    "exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_stage2_pseudo_success_train.jsonl",
    "exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_stage2_pseudo_success_search.jsonl",
    "exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_stage2_pseudo_success_shadow.jsonl",
    "exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_stage2_gt_distill_train.jsonl",
    "exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_stage2_gt_distill_search.jsonl",
    "exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_stage2_gt_distill_shadow.jsonl",
    "exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_stage2_same_source_preference_train.jsonl",
    "exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_stage2_same_source_preference_search.jsonl",
    "exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_stage2_same_source_preference_shadow.jsonl",
    "exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_badnoise_v4_states.jsonl",
    "reports/exp44_stage2_dataset_handoff.md",
    "reports/exp44_stage2_dataset_handoff.csv",
    "reports/exp44_stage2_dataset_handoff_summary.json",
    "reports/exp44_badnoise_v4_states.md",
    "reports/exp44_badnoise_v4_states.csv",
    "reports/exp44_badnoise_v4_summary.json",
]

PATH_KEYS = {
    "condition_path",
    "gt_background_path",
    "gt_winner_path",
    "mask_path",
    "target_frames_dir",
    "target_path",
    "pseudo_success_path",
    "pseudo_success_frames_dir",
    "failure_loser_path",
    "failure_loser_frames_dir",
    "success_review_sheet",
    "failure_review_sheet",
    "success_temporal_strip_16",
    "failure_temporal_strip_16",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def collect_paths(obj: Any, out: dict[str, set[str]], source: str, parent_key: str = "") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            collect_paths(value, out, source, key)
    elif isinstance(obj, list):
        for value in obj:
            collect_paths(value, out, source, parent_key)
    elif isinstance(obj, str):
        if obj.startswith("/mnt/") and (parent_key in PATH_KEYS or parent_key.endswith("_path") or parent_key.endswith("_dir")):
            out.setdefault(obj, set()).add(source)


def classify_path(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    exists = path.exists()
    is_file = path.is_file()
    is_dir = path.is_dir()
    size = path.stat().st_size if is_file else None
    digest = sha256_file(path) if is_file else ""
    if is_file:
        status = "file_ready"
    elif is_dir:
        status = "directory_ready_sha256_not_applicable"
    else:
        status = "missing_in_current_session"
    return {
        "path": path_text,
        "exists": exists,
        "is_file": is_file,
        "is_dir": is_dir,
        "size_bytes": size,
        "sha256": digest,
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument(
        "--source-root",
        default="/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp44_pai_minimax_targeted_same_source_mining/targeted_mining_20260629_161742",
    )
    parser.add_argument(
        "--h20-target-root",
        default="/home/hj/H20_Video_inpainting_DPO_h20_mirror/exp44_pai_minimax_targeted_same_source_mining/targeted_mining_20260629_161742",
    )
    args = parser.parse_args()

    repo = Path(args.repo_root)
    reports = repo / args.reports_dir
    reports.mkdir(parents=True, exist_ok=True)

    path_sources: dict[str, set[str]] = {}
    manifest_rows = 0
    for rel in DEFAULT_MANIFESTS:
        manifest = repo / rel
        for row in iter_jsonl(manifest):
            manifest_rows += 1
            collect_paths(row, path_sources, rel)

    records = []
    for path_text, sources in sorted(path_sources.items()):
        record = classify_path(path_text)
        record["sources"] = sorted(sources)
        records.append(record)

    local_records = []
    for rel in LOCAL_REPO_ARTIFACTS:
        path = repo / rel
        local_records.append(
            {
                "path": rel,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
                "sha256": sha256_file(path) if path.exists() and path.is_file() else "",
            }
        )

    filelist_path = reports / "exp45_h20_required_filelist.txt"
    with filelist_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record["path"] + "\n")

    sha_path = reports / "exp45_h20_required_sha256.txt"
    with sha_path.open("w", encoding="utf-8") as handle:
        handle.write("# sha256 status path\n")
        for record in records:
            sha = record["sha256"] or "NA"
            handle.write(f"{sha} {record['status']} {record['path']}\n")
        handle.write("\n# repository-side manifests and reports\n")
        for record in local_records:
            sha = record["sha256"] or "NA"
            status = "file_ready" if record["exists"] else "missing"
            handle.write(f"{sha} {status} {record['path']}\n")

    total_ready_size = sum(record["size_bytes"] or 0 for record in records)
    missing_count = sum(1 for record in records if record["status"] == "missing_in_current_session")
    file_ready_count = sum(1 for record in records if record["status"] == "file_ready")
    dir_ready_count = sum(1 for record in records if record["status"] == "directory_ready_sha256_not_applicable")
    summary = {
        "status": "EXP45_H20_FILELIST_PARTIAL_SOURCE_ROOT_UNAVAILABLE" if missing_count else "EXP45_H20_FILELIST_READY",
        "source_root": args.source_root,
        "h20_target_root": args.h20_target_root,
        "manifest_rows_scanned": manifest_rows,
        "required_path_count": len(records),
        "file_ready_count": file_ready_count,
        "directory_ready_count": dir_ready_count,
        "missing_in_current_session": missing_count,
        "total_ready_size_bytes": total_ready_size,
        "current_session_mnt_nas_available": Path("/mnt/nas").exists(),
        "current_session_mnt_workspace_available": Path("/mnt/workspace").exists(),
        "pai_does_not_execute_h20_mirror": True,
        "rsync_template_for_h20_session": (
            f"rsync -aH --info=progress2 <PAI_HOST>:{args.source_root}/ "
            f"{args.h20_target_root}/"
        ),
        "filelist_path": str(filelist_path),
        "sha256_path": str(sha_path),
        "local_repo_artifacts": local_records,
    }

    json_path = reports / "exp45_h20_handoff_package.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = reports / "exp45_h20_required_filelist.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "status", "exists", "is_file", "is_dir", "size_bytes", "sha256", "sources"],
        )
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["sources"] = ";".join(record["sources"])
            writer.writerow(row)

    md_path = reports / "exp45_h20_handoff_package.md"
    md_path.write_text(
        "\n".join(
            [
                "# Exp45 H20 Handoff Package",
                "",
                f"Status: `{summary['status']}`",
                "",
                "## Scope",
                "",
                "- PAI generated this filelist only.",
                "- PAI did not copy files to H20.",
                "- PAI did not validate H20 paths.",
                "- PAI did not run training or optimizer steps.",
                "",
                "## Source And Target",
                "",
                f"- PAI source root: `{args.source_root}`",
                f"- expected H20 target root: `{args.h20_target_root}`",
                f"- required path count: `{len(records)}`",
                f"- paths ready in this session: `{file_ready_count}` files, `{dir_ready_count}` directories",
                f"- paths missing in this session: `{missing_count}`",
                f"- total ready file size bytes: `{total_ready_size}`",
                "",
                "## Files",
                "",
                f"- filelist: `reports/{filelist_path.name}`",
                f"- sha256/status list: `reports/{sha_path.name}`",
                f"- csv: `reports/{csv_path.name}`",
                f"- json: `reports/{json_path.name}`",
                "",
                "## H20 Mirror Command Template",
                "",
                "Run this only from a later H20 session, not from PAI:",
                "",
                "```bash",
                summary["rsync_template_for_h20_session"],
                "```",
                "",
                "## Current Blocker",
                "",
                (
                    "The current session cannot see `/mnt/nas` or `/mnt/workspace`, so "
                    "absolute PAI/NAS artifacts are recorded as missing here and their "
                    "SHA256 cannot be computed in this session."
                    if missing_count
                    else "All required paths were visible in this session."
                ),
                "",
                "## Next H20 Experiment",
                "",
                "After a separate H20 session mirrors and validates the files, the first "
                "training experiment should be pseudo-success SFT 30-step. Do not start "
                "GT-only SFT first.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
