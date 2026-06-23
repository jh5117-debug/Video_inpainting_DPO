#!/usr/bin/env python3
"""Lightweight VOR archive audit.

By default this checks part presence, continuity, expected byte counts, and
transfer-manifest status without streaming the 363GB archive payload.  Pass
``--stream-probe`` to open each split gzip tar stream and inspect the first few
members for readability and path safety.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from vor_archive_utils import (
    DEFAULT_PAI_ARCHIVE_ROOT,
    GROUP_PREFIX,
    atomic_write_json,
    continuity_for_parts,
    group_for_name,
    group_parts,
    open_tar_stream,
    unsafe_member_reason,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--archive-dir", type=Path, default=DEFAULT_PAI_ARCHIVE_ROOT)
    p.add_argument("--required-files", type=Path, default=Path("exp25_vor_or_preference_data/runtime/required_files.json"))
    p.add_argument("--transfer-manifest", type=Path, default=Path("exp25_vor_or_preference_data/runtime/transfer_manifest.csv"))
    p.add_argument("--state-json", type=Path, default=Path("exp25_vor_or_preference_data/runtime/vor_archive_state.json"))
    p.add_argument("--report-md", type=Path, default=Path("reports/vor_archive_integrity.md"))
    p.add_argument("--stream-probe", action="store_true")
    p.add_argument("--probe-members", type=int, default=20)
    return p.parse_args()


def load_required(path: Path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def load_transfer_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def probe_group(parts: list[Path], group: str, max_members: int) -> dict:
    out = {"opened": False, "members": 0, "unsafe": [], "first_members": [], "error": ""}
    try:
        with open_tar_stream(parts) as tar:
            out["opened"] = True
            for idx, member in enumerate(tar):
                if idx >= max_members:
                    break
                reason = unsafe_member_reason(member)
                rec = {"path": member.name, "size": member.size, "type": member.type.decode("latin1") if isinstance(member.type, bytes) else str(member.type), "unsafe_reason": reason}
                out["first_members"].append(rec)
                if reason:
                    out["unsafe"].append(rec)
                out["members"] += 1
    except Exception as exc:  # noqa: BLE001 - report audit failure, do not hide it
        out["error"] = repr(exc)
    return out


def main() -> int:
    args = parse_args()
    required = load_required(args.required_files)
    if not required:
        required = [
            {
                "filename": row.get("filename", ""),
                "group": row.get("group") or group_for_name(row.get("filename", "")),
                "size": int(row.get("size") or 0),
            }
            for row in load_transfer_manifest(args.transfer_manifest)
            if row.get("status") == "VERIFIED" and group_for_name(row.get("filename", "")) in set(GROUP_PREFIX)
        ]
    expected_by_name = {Path(r["filename"]).name: int(r.get("size") or 0) for r in required}
    groups = {}
    for group in GROUP_PREFIX:
        parts = group_parts(args.archive_dir, group)
        continuity = continuity_for_parts(parts)
        expected_parts = [r for r in required if r.get("group") == group]
        expected_bytes = sum(int(r.get("size") or 0) for r in expected_parts)
        size_mismatch = []
        for part in parts:
            expected = expected_by_name.get(part.name)
            actual = part.stat().st_size if part.exists() else None
            if expected is not None and actual != expected:
                size_mismatch.append({"file": part.name, "expected": expected, "actual": actual})
        groups[group] = {
            "expected_count": len(expected_parts),
            "expected_bytes": expected_bytes,
            "actual_count": len(parts),
            "actual_bytes": continuity["total_bytes"],
            "continuity": continuity,
            "size_mismatch": size_mismatch,
        }
        if args.stream_probe and parts:
            groups[group]["stream_probe"] = probe_group(parts, group, args.probe_members)

    readme = args.archive_dir / "README.md"
    state = {
        "archive_dir": str(args.archive_dir),
        "readme_exists": readme.exists(),
        "readme_size": readme.stat().st_size if readme.exists() else 0,
        "groups": groups,
        "ok_lightweight": all(
            g["expected_count"] == g["actual_count"]
            and g["expected_bytes"] == g["actual_bytes"]
            and g["continuity"]["contiguous"]
            and not g["size_mismatch"]
            for g in groups.values()
        ),
    }
    atomic_write_json(args.state_json, state)

    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# VOR Archive Integrity Audit", ""]
    lines += [f"- archive_dir: `{args.archive_dir}`", f"- lightweight_ok: `{state['ok_lightweight']}`", f"- stream_probe: `{args.stream_probe}`", ""]
    lines.append("| group | expected parts | actual parts | contiguous | expected bytes | actual bytes | size mismatches |")
    lines.append("| --- | ---: | ---: | --- | ---: | ---: | ---: |")
    for group, g in groups.items():
        lines.append(
            f"| {group} | {g['expected_count']} | {g['actual_count']} | {g['continuity']['contiguous']} | "
            f"{g['expected_bytes']} | {g['actual_bytes']} | {len(g['size_mismatch'])} |"
        )
    if args.stream_probe:
        lines += ["", "## Stream Probe"]
        for group, g in groups.items():
            probe = g.get("stream_probe", {})
            lines.append(f"- {group}: opened={probe.get('opened')} members={probe.get('members')} unsafe={len(probe.get('unsafe', []))} error=`{probe.get('error', '')}`")
    args.report_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0 if state["ok_lightweight"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
