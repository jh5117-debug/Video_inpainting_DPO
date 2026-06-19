"""Evaluation result parser for Exp20 trial outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_summary_csv(path: Path) -> dict[str, float | str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {path}")
    row = rows[-1]
    out: dict[str, float | str] = {}
    for key, value in row.items():
        if value is None or value == "":
            continue
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    summary = read_summary_csv(Path(args.summary_csv))
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
