#!/usr/bin/env python3
"""Exp18 eval launcher guard.

The actual DAVIS metric backend remains the project metric wrapper. This script
only records which variant outputs are available and writes a small status
report so blocked variants are not silently reported as results.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--variants", nargs="+", default=["exp18a", "exp18b", "exp18c_oracle"])
    args = parser.parse_args()
    root = Path(args.eval_root)
    lines = ["# Exp18 DAVIS Eval Status", "", f"- eval_root: `{root}`", ""]
    lines.append("| variant | status | path |")
    lines.append("|---|---|---|")
    for variant in args.variants:
        path = root / variant
        status = "READY_FOR_METRIC_WRAPPER" if path.exists() else "MISSING_OUTPUT"
        lines.append(f"| {variant} | {status} | `{path}` |")
    lines.extend(
        [
            "",
            "This guard does not compute metrics by itself. Use the existing fixed",
            "DAVIS raw6 hard-comp metric wrapper once variant prediction frames exist.",
        ]
    )
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

