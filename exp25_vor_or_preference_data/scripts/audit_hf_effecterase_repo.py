#!/usr/bin/env python3
"""Audit EffectErase remote inventory and required VOR core files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

try:
    from effecterase_common import (
        REPO_ID,
        REPO_TYPE,
        REPORTS,
        RUNTIME,
        continuity_report,
        ensure_dirs,
        required_group,
        required_inventory,
        total_bytes,
    )
except ModuleNotFoundError:
    from .effecterase_common import (
        REPO_ID,
        REPO_TYPE,
        REPORTS,
        RUNTIME,
        continuity_report,
        ensure_dirs,
        required_group,
        required_inventory,
        total_bytes,
    )


def file_record(sibling) -> dict:
    lfs = getattr(sibling, "lfs", None) or {}
    return {
        "filename": sibling.rfilename,
        "size": int(getattr(sibling, "size", 0) or 0),
        "blob_id": getattr(sibling, "blob_id", "") or "",
        "lfs_sha256": lfs.get("sha256", "") if isinstance(lfs, dict) else "",
        "lfs_size": lfs.get("size", "") if isinstance(lfs, dict) else "",
        "required_group": required_group(sibling.rfilename) or "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--hf-home", default="/home/hj/.cache/huggingface_effecterase_auth")
    parser.add_argument("--probe-readme", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    api = HfApi()
    info = api.dataset_info(args.repo_id, files_metadata=True)
    revision = info.sha
    files = [file_record(s) for s in info.siblings]
    required = required_inventory(
        {"filename": f["filename"], "size": f["size"], "lfs_sha256": f["lfs_sha256"], "blob_id": f["blob_id"]}
        for f in files
    )
    continuity = continuity_report(required)
    for group, report in continuity.items():
        if report["count"] <= 0 or not report["contiguous"] or report["duplicates"]:
            raise SystemExit(f"Invalid part inventory for {group}: {report}")
    if any(int(f.get("size") or 0) <= 0 for f in required):
        bad = [f["filename"] for f in required if int(f.get("size") or 0) <= 0]
        raise SystemExit(f"Required files with non-positive size: {bad}")

    inventory = {
        "repo_id": args.repo_id,
        "repo_type": REPO_TYPE,
        "revision": revision,
        "file_count": len(files),
        "required_file_count": len(required),
        "required_total_bytes": total_bytes(required),
        "largest_required_part_bytes": max(int(f["size"]) for f in required),
        "continuity": continuity,
        "all_files": files,
        "required_files": required,
    }
    (REPORTS / "effecterase_remote_inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    with (REPORTS / "effecterase_remote_inventory.csv").open("w", newline="") as f:
        fieldnames = ["filename", "size", "required_group", "lfs_sha256", "blob_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in files:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    md = [
        "# EffectErase Remote Inventory",
        "",
        f"- repo: `{args.repo_id}`",
        f"- revision: `{revision}`",
        f"- total files: {len(files)}",
        f"- required core files: {len(required)}",
        f"- required total bytes: {total_bytes(required)}",
        f"- largest required part bytes: {max(int(f['size']) for f in required)}",
        "",
        "## Continuity",
    ]
    for group, report in continuity.items():
        md.append(f"- {group}: count={report['count']} contiguous={report['contiguous']} duplicates={report['duplicates']} missing={report['missing']}")
    md += ["", "## Required Files"]
    for f in required:
        md.append(f"- `{f['filename']}` size={f['size']} group={f['group']}")
    (REPORTS / "effecterase_remote_inventory.md").write_text("\n".join(md) + "\n")
    (RUNTIME / "dataset_revision.txt").write_text(revision + "\n")
    (RUNTIME / "required_files.json").write_text(json.dumps(required, indent=2, sort_keys=True) + "\n")

    if args.probe_readme:
        probe_dir = Path("exp25_vor_or_preference_data/runtime/readme_probe")
        probe_dir.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(
            repo_id=args.repo_id,
            repo_type=REPO_TYPE,
            revision=revision,
            filename="README.md",
            local_dir=str(probe_dir),
        )
        (REPORTS / "effecterase_readme_probe.md").write_text(f"# README Probe\n\nDownloaded `{path}` for revision `{revision}`.\n")
    print(json.dumps({"revision": revision, "required_file_count": len(required), "required_total_bytes": total_bytes(required)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
