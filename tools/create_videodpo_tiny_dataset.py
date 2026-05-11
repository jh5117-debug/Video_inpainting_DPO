#!/usr/bin/env python
"""Create a tiny VideoDPO-style pair dataset for adapter smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
import cv2


def _make_clip(path: Path, seed: int, num_frames: int, size: int) -> None:
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        8.0,
        (size, size),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open mp4 writer: {path}")
    yy, xx = np.mgrid[:size, :size]
    try:
        for idx in range(num_frames):
            base = np.zeros((size, size, 3), dtype=np.uint8)
            base[..., 0] = (xx + idx * 7 + seed * 11) % 255
            base[..., 1] = (yy + idx * 5 + seed * 17) % 255
            base[..., 2] = ((xx // 2 + yy // 3) + idx * 13) % 255
            noise = rng.integers(0, 12, size=base.shape, dtype=np.uint8)
            rgb = np.clip(base + noise, 0, 255)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--size", type=int, default=96)
    args = parser.parse_args()

    root = Path(args.output_dir).expanduser().resolve()
    dataset_root = root / "data" / "vidpro-vc2-dpo-tiny"
    clips_root = dataset_root / "clips"
    clips_root.mkdir(parents=True, exist_ok=True)

    winner = clips_root / "winner.mp4"
    loser = clips_root / "loser.mp4"
    _make_clip(winner, seed=1, num_frames=args.num_frames, size=args.size)
    _make_clip(loser, seed=2, num_frames=args.num_frames, size=args.size)

    metadata = []
    for clip in [winner, loser]:
        metadata.append(
            {
                "basic": {
                    "clip_path": str(clip.relative_to(dataset_root)),
                    "clip_duration": float(args.num_frames) / 8.0,
                },
                "misc": {
                    "frame_caption": ["a colorful synthetic smoke-test video"],
                },
            }
        )
    with (dataset_root / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    pairs = [
        {
            "video1": 0,
            "video2": 1,
            "frame_caption": "a colorful synthetic smoke-test video",
        }
    ]
    with (dataset_root / "pair.json").open("w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)

    yaml_path = root / "train_data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"META": ["data/vidpro-vc2-dpo-tiny"]}, f, sort_keys=False)

    print(f"[tiny-videodpo] root={root}")
    print(f"[tiny-videodpo] train_data={yaml_path}")
    print(f"[tiny-videodpo] dataset_root={dataset_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
