#!/usr/bin/env python
"""Summarize VBench JSON outputs into compact JSON/CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def extract_average(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)) and value:
        return extract_average(value[0])
    if isinstance(value, dict):
        for key in ("avg", "average", "mean", "score"):
            if key in value:
                return extract_average(value[key])
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--output_csv", type=Path, default=None)
    args = parser.parse_args()

    with args.result_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for dim, value in sorted(raw.items()):
        avg = extract_average(value)
        rows.append({"dimension": dim, "score": avg})

    valid = [r["score"] for r in rows if r["score"] is not None]
    summary = {
        "source": str(args.result_json),
        "num_dimensions": len(rows),
        "num_valid": len(valid),
        "mean_score": sum(valid) / len(valid) if valid else None,
        "dimensions": rows,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dimension", "score"])
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
