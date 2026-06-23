#!/usr/bin/env python3
"""Validate a selectively extracted VOR subset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from vor_archive_utils import atomic_write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--extraction-manifest", type=Path, default=Path("reports/vor_selective_extraction.csv"))
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--report-md", type=Path, default=Path("reports/vor_selective_extraction.md"))
    p.add_argument("--state-json", type=Path, default=Path("exp25_vor_or_preference_data/runtime/vor_extracted_subset_validation.json"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = list(csv.DictReader(args.extraction_manifest.open())) if args.extraction_manifest.exists() else []
    ok_rows = [r for r in rows if r.get("status") == "OK"]
    missing = []
    for row in ok_rows:
        path = Path(row.get("output_path", ""))
        if not path.exists():
            missing.append(row)
    sample_ids = sorted({r.get("sample_id", "") for r in ok_rows if r.get("sample_id")})
    state = {
        "output_root": str(args.output_root),
        "manifest_rows": len(rows),
        "written_files": len(ok_rows),
        "sample_count": len(sample_ids),
        "missing_files": len(missing),
        "ok": not missing and args.output_root.exists(),
    }
    atomic_write_json(args.state_json, state)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(
        "# VOR Selective Extraction Validation\n\n"
        f"- output_root: `{args.output_root}`\n"
        f"- manifest_rows: {len(rows)}\n"
        f"- written_files: {len(ok_rows)}\n"
        f"- sample_count: {len(sample_ids)}\n"
        f"- missing_files: {len(missing)}\n"
        f"- ok: `{state['ok']}`\n"
    )
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0 if state["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
