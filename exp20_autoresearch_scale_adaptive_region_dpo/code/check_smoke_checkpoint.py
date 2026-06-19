#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.brushnet_CA import BrushNetModel
from libs.unet_2d_condition import UNet2DConditionModel


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--last-weights", required=True)
    parser.add_argument("--sft-weights", required=True)
    parser.add_argument("--report", default="reports/exp20_checkpoint_reload_audit.md")
    args = parser.parse_args()

    last = Path(args.last_weights)
    sft = Path(args.sft_weights)
    if last.resolve() == sft.resolve():
        raise RuntimeError("last_weights resolves to SFT weights; this would be fallback")
    for rel in ["unet_main/config.json", "brushnet/config.json"]:
        if not (last / rel).exists():
            raise RuntimeError(f"missing checkpoint file: {last / rel}")

    unet = UNet2DConditionModel.from_pretrained(str(last), subfolder="unet_main")
    brushnet = BrushNetModel.from_pretrained(str(last), subfolder="brushnet")
    payload = {
        "status": "CHECKPOINT_RELOAD_PASSED",
        "last_weights": str(last),
        "sft_weights": str(sft),
        "last_equals_sft_realpath": False,
        "unet_param_count": count_params(unet),
        "brushnet_param_count": count_params(brushnet),
        "unet_class": unet.__class__.__name__,
        "brushnet_class": brushnet.__class__.__name__,
    }
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Exp20 Checkpoint Reload Audit\n\n"
        + "```json\n"
        + json.dumps(payload, indent=2, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
