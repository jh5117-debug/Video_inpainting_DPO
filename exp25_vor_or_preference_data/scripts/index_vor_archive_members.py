#!/usr/bin/env python3
"""Build a resumable member index for VOR split tar.gz archives."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vor_archive_utils import (
    DEFAULT_PAI_ARCHIVE_ROOT,
    MemberRecord,
    append_csv,
    atomic_write_json,
    group_parts,
    open_tar_stream,
    read_json,
)


FIELDS = ["group", "member_index", "member_path", "sample_id", "type", "size", "mtime", "unsafe_reason"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--archive-dir", type=Path, default=DEFAULT_PAI_ARCHIVE_ROOT)
    p.add_argument("--groups", nargs="+", default=["VOR-Eval"], choices=["VOR-Eval", "VOR-Train-MASK", "VOR-Train"])
    p.add_argument("--output-csv", type=Path, default=Path("reports/vor_archive_inventory.csv"))
    p.add_argument("--state-json", type=Path, default=Path("exp25_vor_or_preference_data/runtime/vor_archive_member_index_state.json"))
    p.add_argument("--heartbeat-json", type=Path, default=Path("exp25_vor_or_preference_data/runtime/vor_archive_member_index_heartbeat.json"))
    p.add_argument("--max-members", type=int, default=0, help="0 means no limit")
    p.add_argument("--metadata-only", action="store_true", help="Only keep likely metadata/index members for huge Train scans")
    p.add_argument("--resume", action="store_true", help="Skip groups marked completed in state-json and append to existing CSV")
    p.add_argument("--heartbeat-every", type=int, default=1000)
    return p.parse_args()


def keep_member(path: str, metadata_only: bool) -> bool:
    if not metadata_only:
        return True
    lower = path.lower()
    return any(token in lower for token in ["metadata", "meta", "annotation", "json", "csv", "txt", "split", "index"])


def write_heartbeat(args: argparse.Namespace, payload: dict) -> None:
    payload = dict(payload)
    payload["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_write_json(args.heartbeat_json, payload)


def index_group(args: argparse.Namespace, group: str) -> dict:
    parts = group_parts(args.archive_dir, group)
    state = {
        "group": group,
        "parts": [p.name for p in parts],
        "seen": 0,
        "written": 0,
        "unsafe": 0,
        "bytes": 0,
        "completed": False,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "completed_at": "",
        "error": "",
    }
    try:
        with open_tar_stream(parts) as tar:
            for idx, member in enumerate(tar):
                if args.max_members and idx >= args.max_members:
                    break
                state["seen"] += 1
                rec = MemberRecord.from_tarinfo(group, idx, member)
                if rec.unsafe_reason:
                    state["unsafe"] += 1
                state["bytes"] += rec.size
                if keep_member(rec.member_path, args.metadata_only):
                    append_csv(args.output_csv, FIELDS, rec.to_dict())
                    state["written"] += 1
                if args.heartbeat_every > 0 and state["seen"] % args.heartbeat_every == 0:
                    write_heartbeat(args, {"status": "RUNNING", **state})
        state["completed"] = not bool(state["error"])
        state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    except Exception as exc:  # noqa: BLE001
        state["error"] = repr(exc)
    write_heartbeat(args, {"status": "COMPLETED" if state["completed"] else "ERROR", **state})
    return state


def main() -> int:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    previous = read_json(args.state_json, {"groups": []}) or {"groups": []}
    previous_by_group = {
        g.get("group"): g for g in previous.get("groups", []) if g.get("completed") and not g.get("error")
    }
    if args.output_csv.exists() and not args.resume:
        args.output_csv.unlink()
    all_state = {"archive_dir": str(args.archive_dir), "groups": []}
    for group in args.groups:
        if args.resume and group in previous_by_group:
            group_state = dict(previous_by_group[group])
            group_state["skipped_by_resume"] = True
            write_heartbeat(args, {"status": "SKIPPED_COMPLETED_GROUP", **group_state})
        else:
            group_state = index_group(args, group)
        all_state["groups"].append(group_state)
        atomic_write_json(args.state_json, all_state)
        if group_state["error"]:
            break
    ok = all(not g["error"] for g in all_state["groups"])
    all_state["ok"] = ok
    atomic_write_json(args.state_json, all_state)
    print(json.dumps(all_state, indent=2, sort_keys=True))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
