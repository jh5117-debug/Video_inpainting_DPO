#!/usr/bin/env python
"""Prepare VideoDPO inference outputs for VBench standard-mode evaluation.

VideoDPO's VC2 inference script writes videos as ``0001.mp4``, ``0002.mp4``,
... in prompt-file order.  VBench standard mode expects files named
``<prompt>-<sample_idx>.mp4``.  This utility creates a symlink/copy tree with
the names VBench expects, without touching the raw generated files.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


VIDEO_EXTS = (".mp4", ".gif")


def load_prompts(path: Path, limit: int = 0) -> list[str]:
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            prompt = line.strip()
            if prompt:
                prompts.append(prompt)
            if limit and len(prompts) >= limit:
                break
    return prompts


def find_raw_video(raw_dir: Path, index: int) -> Path | None:
    stem = f"{index + 1:04d}"
    for ext in VIDEO_EXTS:
        candidate = raw_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True, type=Path)
    parser.add_argument("--prompts_file", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--sample_index", required=True, type=int)
    parser.add_argument("--prompt_limit", type=int, default=0)
    parser.add_argument("--mode", choices=["symlink", "copy", "hardlink"], default="symlink")
    parser.add_argument("--strict", action="store_true", help="Fail if any expected raw video is missing.")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_file, args.prompt_limit)
    if not prompts:
        raise RuntimeError(f"No prompts found in {args.prompts_file}")

    missing = []
    written = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for idx, prompt in enumerate(prompts):
        src = find_raw_video(args.raw_dir, idx)
        if src is None:
            missing.append(f"{idx + 1:04d}")
            continue
        dst = args.output_dir / f"{prompt}-{args.sample_index}{src.suffix.lower()}"
        link_or_copy(src, dst, args.mode)
        written += 1

    print(
        f"[prepare-vbench] raw_dir={args.raw_dir} sample={args.sample_index} "
        f"written={written} missing={len(missing)} output={args.output_dir}"
    )
    if missing:
        print("[prepare-vbench] missing raw indices:", ",".join(missing[:50]))
        if args.strict:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
