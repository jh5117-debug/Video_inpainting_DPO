#!/usr/bin/env python
"""Smoke-test the VideoDPO full-mask DiffuEraser dataset adapter."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer

from training.dpo.dataset.videodpo_fullmask_dataset import VideoDPOFullMaskDiffuEraserDataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo_data_root", required=True)
    parser.add_argument("--base_model_name_or_path", required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_height", type=int, default=None)
    parser.add_argument("--train_width", type=int, default=None)
    parser.add_argument("--nframes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--videodpo_frame_stride", type=int, default=1)
    parser.add_argument("--videodpo_clip_length", type=float, default=1.0)
    parser.add_argument("--videodpo_full_mask_value", type=float, default=0.0)
    parser.add_argument("--proportion_empty_prompts", type=float, default=0.0)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )
    dataset = VideoDPOFullMaskDiffuEraserDataset(args, tokenizer, dpo_data_root=args.dpo_data_root)
    sample = dataset[args.index]
    print(f"[smoke] len={len(dataset)} index={args.index}")
    for key, value in sample.items():
        if hasattr(value, "shape"):
            print(f"[smoke] {key}: shape={tuple(value.shape)} dtype={value.dtype} min={value.min().item():.4f} max={value.max().item():.4f}")
        else:
            print(f"[smoke] {key}: {type(value).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
