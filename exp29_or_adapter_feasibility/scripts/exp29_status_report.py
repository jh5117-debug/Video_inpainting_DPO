#!/usr/bin/env python3
"""Print the current Exp29 status file path.

This utility exists so the Exp29 scripts directory has a Python entry point
for the standard milestone compile check.
"""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    print(root / "experiment_registry" / "exp29_or_adapter_feasibility" / "status.md")


if __name__ == "__main__":
    main()
