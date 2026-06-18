#!/usr/bin/env python3
"""Static architecture preflight for Exp19.

This guard prevents an unsafe training launch when the requested multi-scale
adapter cannot be injected without changing shared model code.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.unet_motion_model import UNetMotionModel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_path", default="reports/exp19_preflight_report.md")
    parser.add_argument("--allow_mid_only", action="store_true")
    args = parser.parse_args()

    source = inspect.getsource(UNetMotionModel.forward)
    has_down = "down_block_additional_residuals" in source
    has_mid = "mid_block_additional_residual" in source
    has_controlnet_double_path = (
        "if is_controlnet:" in source
        and "if down_block_additional_residuals is not None:" in source
        and source.count("down_block_additional_residuals") >= 6
    )
    status = "PASS_MID_ONLY_ALLOWED" if args.allow_mid_only else "BLOCKED_MULTI_SCALE_INJECTION_UNSAFE"
    exit_code = 0 if args.allow_mid_only else 3

    lines = [
        "# Exp19 Architecture Preflight",
        "",
        f"- UNetMotionModel.forward has down residual argument: `{has_down}`",
        f"- UNetMotionModel.forward has mid residual argument: `{has_mid}`",
        f"- detected shared down-residual double-add risk: `{has_controlnet_double_path}`",
        f"- allow_mid_only: `{args.allow_mid_only}`",
        f"- status: `{status}`",
        "",
        "## Decision",
        "",
    ]
    if args.allow_mid_only:
        lines.extend(
            [
                "A mid-block-only adapter could be run with the existing shared model,",
                "but that is not the requested Exp19 multi-scale flow adapter. Treat",
                "this only as a debugging path, not as the official Exp19 gate.",
            ]
        )
    else:
        lines.extend(
            [
                "Do not launch Exp19 training from this shared UNetMotionModel path.",
                "",
                "Reason: the requested multi-scale flow adapter needs clean additive",
                "down/mid residual semantics. The current shared forward has a",
                "ControlNet-style branch and an unconditional second",
                "`down_block_additional_residuals` addition. Passing both down and",
                "mid residuals would double-add the down residuals. Passing only down",
                "residuals falls into the legacy T2I-adapter path with a different",
                "shape contract. Reducing Exp19 to mid-block-only would violate the",
                "requested method definition.",
                "",
                "Allowed next implementation path: copy `libs/unet_motion_model.py`",
                "into `exp19_boundary_gated_flow_adapter_dpo/code/`, implement a",
                "clean Exp19-only residual interface there, and write a matching",
                "Exp19 inference wrapper. Until that exists, training is blocked.",
            ]
        )
    report = Path(args.report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[exp19-preflight] wrote {report} status={status}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
