#!/usr/bin/env python
# coding=utf-8
"""Compatibility wrapper for exporting a Stage 1 accelerator checkpoint.

Example:
  python tools/save_checkpoint_stage1.py \
    --checkpoint_dir experiments/sft/stage1/run/checkpoint-26000 \
    --base_model_path weights/stable-diffusion-v1-5 \
    --brushnet_path weights/diffuEraser \
    --output_dir weights/diffuEraser/converted_weights_step26000
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.convert_checkpoint import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--stage", "1", *sys.argv[1:]]
    main()
