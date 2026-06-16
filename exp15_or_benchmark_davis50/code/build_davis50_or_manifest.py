#!/usr/bin/env python3
"""Build/audit the DAVIS50-only object-removal manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(path: Path) -> List[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def read_names_from_or150(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = [row["video"] for row in rows if row.get("dataset") == "DAVIS50_OR"]
    names = sorted(dict.fromkeys(names))
    if len(names) != 50:
        raise RuntimeError(f"expected 50 DAVIS50_OR names from {path}, got {len(names)}")
    return names


def first_resolution(files: List[Path]) -> Tuple[int, int]:
    if not files:
        return 0, 0
    with Image.open(files[0]) as img:
        return img.size


def write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_manifest", default="exp15_or_benchmark/manifests/or150_manifest.csv")
    parser.add_argument("--pai_davis_root", default="/mnt/nas/hj/data/external/davis_2017_full_resolution_or_eval50/DAVIS")
    parser.add_argument("--out_csv", default="exp15_or_benchmark_davis50/manifests/davis50_or_manifest.csv")
    parser.add_argument("--report_md", default="reports/exp15_davis50_or_dataset_audit.md")
    args = parser.parse_args()

    names = read_names_from_or150(Path(args.source_manifest))
    davis_root = Path(args.pai_davis_root)
    rows = []
    missing = []
    total_frames = 0
    for name in names:
        frame_dir = davis_root / "JPEGImages" / "Full-Resolution" / name
        mask_dir = davis_root / "Annotations" / "Full-Resolution" / name
        frame_files = image_files(frame_dir) if frame_dir.is_dir() else []
        mask_files = image_files(mask_dir) if mask_dir.is_dir() else []
        n = min(len(frame_files), len(mask_files))
        width, height = first_resolution(frame_files)
        if n == 0:
            missing.append(name)
        total_frames += n
        rows.append(
            {
                "video_name": name,
                "frame_dir": str(frame_dir),
                "mask_dir": str(mask_dir),
                "num_frames": n,
                "resolution": f"{width}x{height}" if width and height else "",
                "selected_reason": "same video name as current DAVIS50 protocol",
                "notes": "DAVIS2017 foreground mask; nonzero means object to remove",
            }
        )

    write_csv(
        Path(args.out_csv),
        rows,
        ["video_name", "frame_dir", "mask_dir", "num_frames", "resolution", "selected_reason", "notes"],
    )

    report = Path(args.report_md)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# Exp15 DAVIS50 OR Dataset Audit",
                "",
                f"- DAVIS root: `{davis_root}`",
                f"- selected videos: `{len(rows)}`",
                f"- total aligned frames: `{total_frames}`",
                f"- missing/empty videos: `{', '.join(missing) if missing else 'none'}`",
                f"- manifest: `{args.out_csv}`",
                "",
                "Mask semantics: DAVIS2017 foreground annotation; nonzero pixels are objects to remove.",
                "",
                "This is DAVIS50-only. YouTubeVOS100 and OR150 are intentionally not used in this run.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[manifest] rows={len(rows)} frames={total_frames} out={args.out_csv}")
    if missing:
        raise SystemExit(f"missing/empty DAVIS videos: {missing}")


if __name__ == "__main__":
    main()
