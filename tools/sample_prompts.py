#!/usr/bin/env python
"""Sample prompt lines from a text file with a fixed seed."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def read_prompts(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("count", type=int)
    parser.add_argument("seed", type=int)
    args = parser.parse_args()

    prompts = read_prompts(args.input)
    if args.count < 1:
        raise SystemExit("-- count must be >= 1")
    if args.count > len(prompts):
        raise SystemExit(f"count={args.count} exceeds prompt count={len(prompts)}")

    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(prompts)), args.count))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(prompts[i] + "\n" for i in indices), encoding="utf-8")
    print(f"[sample-prompts] input={args.input} output={args.output} count={args.count} seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
