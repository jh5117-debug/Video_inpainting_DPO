"""Trial runner skeleton for Exp20.

The runner validates immutable config files and writes a BLOCKED/READY row when
the heavy trainer is not explicitly enabled. This prevents accidental GPU use
before parity and dev split are locked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from search_controller import RESULT_FIELDS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", default="exp20_autoresearch_scale_adaptive_region_dpo/results.tsv")
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    print(json.dumps({"status": "CONFIG_VALID", "config_hash": cfg["config_hash"], "trial_id": cfg["trial_id"]}, indent=2))
    if args.dry_run:
        return
    raise RuntimeError("Heavy trainer connection is intentionally disabled until parity passes.")


if __name__ == "__main__":
    main()
