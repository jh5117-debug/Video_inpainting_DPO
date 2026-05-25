#!/usr/bin/env python3
"""Rewrite generated-loser JSONL manifest path prefixes.

Use this after moving H20-generated data to PAI, for example:

python tools/rewrite_generated_loser_manifest_paths.py \
  --input h20_selected_primary_fullmask.jsonl \
  --output pai_selected_primary_fullmask.jsonl \
  --map /home/nvme01/H20_Video_inpainting_DPO=/mnt/nas/hj/H20_Video_inpainting_DPO
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_mapping(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--map must be OLD_PREFIX=NEW_PREFIX")
    old, new = value.split("=", 1)
    old = old.rstrip("/")
    new = new.rstrip("/")
    if not old or not new:
        raise argparse.ArgumentTypeError("--map prefixes cannot be empty")
    return old, new


def rewrite_value(value: Any, mappings: list[tuple[str, str]]) -> tuple[Any, int]:
    if isinstance(value, str):
        for old, new in mappings:
            if value == old:
                return new, 1
            if value.startswith(old + "/"):
                return new + value[len(old) :], 1
        return value, 0
    if isinstance(value, list):
        changed = 0
        out = []
        for item in value:
            new_item, n = rewrite_value(item, mappings)
            changed += n
            out.append(new_item)
        return out, changed
    if isinstance(value, dict):
        changed = 0
        out = {}
        for key, item in value.items():
            new_item, n = rewrite_value(item, mappings)
            changed += n
            out[key] = new_item
        return out, changed
    return value, 0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"[error] invalid JSONL {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rewrite path prefixes in generated-loser JSONL manifests.")
    parser.add_argument("--input", required=True, help="Input JSONL manifest")
    parser.add_argument("--output", default=None, help="Output JSONL manifest. Required unless --in_place.")
    parser.add_argument("--map", action="append", required=True, type=parse_mapping, dest="mappings", help="OLD_PREFIX=NEW_PREFIX")
    parser.add_argument("--in_place", action="store_true", help="Rewrite the input file in place")
    parser.add_argument("--dry_run", action="store_true", help="Print counts without writing")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    in_path = Path(args.input)
    out_path = in_path if args.in_place else Path(args.output or "")
    if not args.in_place and not args.output:
        raise SystemExit("[error] --output is required unless --in_place")

    rows = read_jsonl(in_path)
    rewritten = []
    changed_values = 0
    changed_rows = 0
    for row in rows:
        new_row, n = rewrite_value(row, args.mappings)
        changed_values += n
        changed_rows += int(n > 0)
        rewritten.append(new_row)

    summary = {
        "input": str(in_path),
        "output": str(out_path),
        "rows": len(rows),
        "changed_rows": changed_rows,
        "changed_values": changed_values,
        "mappings": args.mappings,
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.dry_run:
        write_jsonl(out_path, rewritten)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
