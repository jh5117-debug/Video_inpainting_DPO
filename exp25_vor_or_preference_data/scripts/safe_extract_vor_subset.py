#!/usr/bin/env python3
"""Selective VOR extraction by sample id.

This script streams split archives once and only writes members whose inferred
sample id is requested.  It is intended for VOR-Eval full extraction and small
train/mask subsets, not for materializing the whole 60K train set.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from vor_archive_utils import (
    DEFAULT_PAI_ARCHIVE_ROOT,
    append_csv,
    atomic_write_json,
    group_parts,
    open_tar_stream,
    safe_output_path,
    sample_id_from_member,
    unsafe_member_reason,
)


FIELDS = ["group", "member_path", "sample_id", "type", "size", "status", "reason", "output_path"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--archive-dir", type=Path, default=DEFAULT_PAI_ARCHIVE_ROOT)
    p.add_argument("--groups", nargs="+", required=True, choices=["VOR-Eval", "VOR-Train-MASK", "VOR-Train"])
    p.add_argument("--sample-ids", type=Path, help="JSON/JSONL/TXT sample ids. Omit with --extract-all.")
    p.add_argument("--extract-all", action="store_true")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--manifest-csv", type=Path, default=Path("reports/vor_selective_extraction.csv"))
    p.add_argument("--state-json", type=Path, default=Path("exp25_vor_or_preference_data/runtime/vor_selective_extraction_state.json"))
    p.add_argument("--max-members", type=int, default=0)
    return p.parse_args()


def load_sample_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    text = path.read_text().strip()
    if not text:
        return set()
    if path.suffix == ".json":
        obj = json.loads(text)
        if isinstance(obj, list):
            return {str(x) for x in obj}
        if isinstance(obj, dict) and "sample_ids" in obj:
            return {str(x) for x in obj["sample_ids"]}
    ids: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            row = json.loads(line)
            ids.add(str(row.get("sample_id") or row.get("id")))
        else:
            ids.add(line)
    return ids


def extract_group(args: argparse.Namespace, group: str, sample_ids: set[str]) -> dict:
    parts = group_parts(args.archive_dir, group)
    out = {"group": group, "seen": 0, "matched": 0, "written": 0, "skipped": 0, "unsafe": 0, "error": ""}
    try:
        with open_tar_stream(parts) as tar:
            for idx, member in enumerate(tar):
                if args.max_members and idx >= args.max_members:
                    break
                out["seen"] += 1
                sid = sample_id_from_member(member.name)
                should_extract = args.extract_all or sid in sample_ids
                reason = unsafe_member_reason(member)
                if reason:
                    out["unsafe"] += 1
                    append_csv(args.manifest_csv, FIELDS, {"group": group, "member_path": member.name, "sample_id": sid, "type": "unsafe", "size": member.size, "status": "SKIP", "reason": reason})
                    continue
                if not should_extract:
                    out["skipped"] += 1
                    continue
                out["matched"] += 1
                dest = safe_output_path(args.output_root / group, member.name)
                status = "SKIP"
                if member.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                    status = "DIR"
                elif member.isfile():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    src = tar.extractfile(member)
                    if src is None:
                        status = "NO_FILEOBJ"
                    else:
                        tmp = dest.with_suffix(dest.suffix + ".tmp")
                        with tmp.open("wb") as f:
                            shutil.copyfileobj(src, f, length=8 * 1024 * 1024)
                        tmp.replace(dest)
                        status = "OK"
                        out["written"] += 1
                append_csv(args.manifest_csv, FIELDS, {"group": group, "member_path": member.name, "sample_id": sid, "type": "file" if member.isfile() else "dir", "size": member.size, "status": status, "reason": "", "output_path": str(dest)})
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)
    return out


def main() -> int:
    args = parse_args()
    if not args.extract_all and not args.sample_ids:
        raise SystemExit("--sample-ids is required unless --extract-all is set")
    sample_ids = load_sample_ids(args.sample_ids)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.manifest_csv.exists():
        args.manifest_csv.unlink()
    state = {"archive_dir": str(args.archive_dir), "output_root": str(args.output_root), "extract_all": args.extract_all, "sample_id_count": len(sample_ids), "groups": []}
    for group in args.groups:
        group_state = extract_group(args, group, sample_ids)
        state["groups"].append(group_state)
        atomic_write_json(args.state_json, state)
        if group_state["error"]:
            break
    state["ok"] = all(not g["error"] for g in state["groups"])
    atomic_write_json(args.state_json, state)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0 if state["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
