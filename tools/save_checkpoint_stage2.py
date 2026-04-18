#!/usr/bin/env python
# coding=utf-8
"""Compatibility wrapper for exporting a Stage 2 accelerator checkpoint.

Example:
  python tools/save_checkpoint_stage2.py \
    --checkpoint_dir experiments/sft/stage2/run/checkpoint-48000 \
    --base_model_path weights/stable-diffusion-v1-5 \
    --brushnet_path weights/diffuEraser \
    --motion_adapter_path weights/animatediff-motion-adapter-v1-5-2 \
    --pretrained_stage1 weights/diffuEraser/converted_stage1 \
    --output_dir weights/diffuEraser/converted_weights_step48000
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.convert_checkpoint import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--stage", "2", *sys.argv[1:]]
    main()
