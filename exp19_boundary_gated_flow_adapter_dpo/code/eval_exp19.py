#!/usr/bin/env python3
"""Exp19 eval guard.

Existing DiffuEraser eval scripts cannot load external flow adapters. A real
Exp19 eval must use an Exp19 inference wrapper that computes completed flow per
video, enables the adapter, and still calls the project metric backend.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_path", default="reports/exp19_eval_wrapper_status.md")
    args = parser.parse_args()
    report = Path(args.report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# Exp19 Eval Wrapper Status",
                "",
                "```text",
                "BLOCKED_PENDING_EXP19_INFERENCE_WRAPPER",
                "```",
                "",
                "The existing `tools/run_davis50_framewise_protocol_eval.py` can load",
                "standard DiffuEraser `last_weights`, but it cannot load external",
                "flow encoder / adapter weights or pass flow tensors into the UNet.",
                "",
                "Do not evaluate Exp19 by silently falling back to Exp11 weights.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[exp19-eval] wrote {report}")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
