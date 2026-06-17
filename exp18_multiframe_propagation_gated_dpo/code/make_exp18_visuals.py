#!/usr/bin/env python3
"""Build a lightweight Exp18 visual evidence index.

This script intentionally creates symlinks instead of copying videos/images.
Actual side-by-side generation is delegated to the existing project evaluators.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for file in sorted(eval_root.rglob("*.mp4")):
        rel = file.relative_to(eval_root)
        dst = out / "videos" / "__".join(rel.parts)
        safe_symlink(file, dst)
        rows.append({"asset_type": "video", "source": str(file), "link": str(dst)})
    for file in sorted(eval_root.rglob("*.jpg")):
        rel = file.relative_to(eval_root)
        dst = out / "contact_sheets" / "__".join(rel.parts)
        safe_symlink(file, dst)
        rows.append({"asset_type": "jpg", "source": str(file), "link": str(dst)})
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["asset_type", "source", "link"])
        writer.writeheader()
        writer.writerows(rows)
    (out / "README.md").write_text(
        "\n".join(
            [
                "# Exp18 Visual Evidence Index",
                "",
                f"- eval_root: `{eval_root}`",
                f"- output_dir: `{out}`",
                f"- assets: `{len(rows)}`",
                "",
                "This folder uses symlinks to avoid duplicating large videos.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(os.fspath(out))


if __name__ == "__main__":
    main()

