#!/usr/bin/env python3
"""Choose a safe HAL staging root for one-part-at-a-time VOR transfer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from effecterase_common import DEFAULT_HAL_STAGING, REPORTS, RUNTIME, ensure_dirs, free_bytes, read_json
except ModuleNotFoundError:
    from .effecterase_common import DEFAULT_HAL_STAGING, REPORTS, RUNTIME, ensure_dirs, free_bytes, read_json


def candidate_paths(default: Path) -> list[Path]:
    paths = [default, Path("/home/hj"), Path("/tmp/hj_exp25_effecterase"), Path("/scratch/hj"), Path("/mnt/scratch/hj")]
    for root in [Path("/home")]:
        for p in root.glob("nvme*"):
            paths.append(p / "hj")
            paths.append(p)
    return list(dict.fromkeys(paths))


def writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".exp25_write_probe"
        probe.write_text("ok")
        probe.unlink()
        return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default="reports/effecterase_remote_inventory.json")
    parser.add_argument("--default", default=str(DEFAULT_HAL_STAGING))
    args = parser.parse_args()
    ensure_dirs()
    inv = read_json(Path(args.inventory))
    if not inv:
        raise SystemExit("Missing inventory. Run audit_hf_effecterase_repo.py first.")
    largest = int(inv["largest_required_part_bytes"])
    required = int(largest * 2.2 + 10 * 1024**3)
    rows = []
    selected = None
    for path in candidate_paths(Path(args.default)):
        row = {"path": str(path), "exists": path.exists(), "writable": False, "free_bytes": 0, "selected": False, "reason": ""}
        if str(path).startswith("/home/") and path.parts[:3] not in [("/", "home", "hj")]:
            row["reason"] = "not user-owned namespace"
        elif writable(path):
            row["writable"] = True
            row["free_bytes"] = free_bytes(path)
            if selected is None and row["free_bytes"] >= required:
                selected = path
                row["selected"] = True
                row["reason"] = "first safe candidate with required free bytes"
        rows.append(row)
    if selected is None:
        (REPORTS / "effecterase_hal_staging_candidates.json").write_text(json.dumps(rows, indent=2) + "\n")
        raise SystemExit(f"BLOCKED_HAL_SINGLE_PART_SPACE largest={largest} required={required}")
    selected.mkdir(parents=True, exist_ok=True)
    out = {
        "path": str(selected),
        "realpath": os.path.realpath(selected),
        "free_bytes": free_bytes(selected),
        "largest_part_bytes": largest,
        "required_bytes": required,
        "selection_reason": "safe writable root with single-part cache margin",
        "candidates": rows,
    }
    (RUNTIME / "selected_hal_staging.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    (REPORTS / "effecterase_hal_staging_candidates.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
