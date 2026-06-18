#!/usr/bin/env python3
"""Exp19 Stage2 adapter trainer entrypoint guard.

The full trainer must not run until the architecture preflight confirms a safe
multi-scale adapter path. This entrypoint exists so launchers fail explicitly
instead of accidentally running upstream Exp11/official training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from run_exp19_architecture_preflight import main as preflight_main  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--report_path", default="reports/exp19_preflight_report.md")
    parser.add_argument("--allow_mid_only", action="store_true")
    args, _unknown = parser.parse_known_args()
    if not args.preflight_only:
        print(
            "Exp19 full trainer is intentionally blocked until a clean Exp19-only "
            "multi-scale UNet residual interface and inference wrapper exist.",
            file=sys.stderr,
        )
    sys.argv = [sys.argv[0], "--report_path", args.report_path]
    if args.allow_mid_only:
        sys.argv.append("--allow_mid_only")
    return preflight_main()


if __name__ == "__main__":
    raise SystemExit(main())
