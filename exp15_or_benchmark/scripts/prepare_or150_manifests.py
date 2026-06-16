#!/usr/bin/env python3
"""Prepare manifests for the Exp15 object-removal benchmark.

The script is intentionally lightweight: it does not copy video frames. It
creates path manifests and an rsync file list for the DAVIS2017 full-resolution
foreground-mask subset matching the current DAVIS50 protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(path: Path) -> List[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def read_davis50_names(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        names = sorted({row["video"] for row in reader if row.get("video")})
    if len(names) != 50:
        raise RuntimeError(f"expected 50 DAVIS names from {path}, got {len(names)}")
    return names


def read_youtubevos100_names(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        names = [row["video_id"] for row in reader if row.get("video_id")]
    if len(names) != 100:
        raise RuntimeError(f"expected 100 YouTubeVOS names from {path}, got {len(names)}")
    return names


def write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--davis2017_root", default="/home/hj/Video_inpainting_DPO/data/external/davis_2017_full_resolution/DAVIS")
    parser.add_argument("--davis50_source_csv", default="reports/videopainter_adapter_gate2000_davis_per_video.csv")
    parser.add_argument("--youtubevos100_manifest", default="/mnt/workspace/hj/nas_hj/data/external/youtubevos_432_240_eval100/sample_manifest.csv")
    parser.add_argument("--pai_davis_target", default="/mnt/nas/hj/data/external/davis_2017_full_resolution_or_eval50/DAVIS")
    parser.add_argument("--pai_youtubevos100_root", default="/mnt/workspace/hj/nas_hj/data/external/youtubevos_432_240_eval100")
    parser.add_argument("--out_dir", default="exp15_or_benchmark/manifests")
    args = parser.parse_args()

    davis_root = Path(args.davis2017_root)
    out_dir = Path(args.out_dir)
    davis_names = read_davis50_names(Path(args.davis50_source_csv))

    davis_rows = []
    rsync_paths: List[str] = ["ImageSets/2017/val.txt", "README.md", "SOURCES.md"]
    for name in davis_names:
        src_frames = davis_root / "JPEGImages" / "Full-Resolution" / name
        src_masks = davis_root / "Annotations" / "Full-Resolution" / name
        if not src_frames.is_dir() or not src_masks.is_dir():
            raise FileNotFoundError(f"missing DAVIS2017 frames/masks for {name}")
        frame_files = image_files(src_frames)
        mask_files = image_files(src_masks)
        n = min(len(frame_files), len(mask_files))
        davis_rows.append(
            {
                "dataset": "DAVIS50_OR",
                "video": name,
                "frames": n,
                "hal_frame_dir": str(src_frames),
                "hal_mask_dir": str(src_masks),
                "pai_frame_dir": f"{args.pai_davis_target}/JPEGImages/Full-Resolution/{name}",
                "pai_mask_dir": f"{args.pai_davis_target}/Annotations/Full-Resolution/{name}",
                "mask_semantics": "DAVIS2017 foreground annotation; nonzero means object to remove",
            }
        )
        rsync_paths.append(f"JPEGImages/Full-Resolution/{name}/")
        rsync_paths.append(f"Annotations/Full-Resolution/{name}/")

    write_csv(
        out_dir / "davis50_or_manifest.csv",
        davis_rows,
        ["dataset", "video", "frames", "hal_frame_dir", "hal_mask_dir", "pai_frame_dir", "pai_mask_dir", "mask_semantics"],
    )
    (out_dir / "davis50_or_rsync_files.txt").write_text("\n".join(rsync_paths) + "\n", encoding="utf-8")

    yt_manifest = Path(args.youtubevos100_manifest)
    yt_rows = []
    if yt_manifest.exists():
        for name in read_youtubevos100_names(yt_manifest):
            frame_dir = f"{args.pai_youtubevos100_root}/JPEGImages_432_240/{name}"
            mask_dir = f"{args.pai_youtubevos100_root}/test_masks/{name}"
            yt_rows.append(
                {
                    "dataset": "YouTubeVOS100_OR",
                    "video": name,
                    "pai_frame_dir": frame_dir,
                    "pai_mask_dir": mask_dir,
                    "mask_semantics": "existing YouTubeVOS test mask; nonzero means object to remove",
                }
            )
    else:
        print(f"[warn] YouTubeVOS100 manifest not found on this machine: {yt_manifest}")

    write_csv(
        out_dir / "youtubevos100_or_manifest.csv",
        yt_rows,
        ["dataset", "video", "pai_frame_dir", "pai_mask_dir", "mask_semantics"],
    )

    combined = []
    for row in davis_rows:
        combined.append(
            {
                "dataset": row["dataset"],
                "video": row["video"],
                "frame_dir": row["pai_frame_dir"],
                "mask_dir": row["pai_mask_dir"],
                "mask_semantics": row["mask_semantics"],
            }
        )
    for row in yt_rows:
        combined.append(
            {
                "dataset": row["dataset"],
                "video": row["video"],
                "frame_dir": row["pai_frame_dir"],
                "mask_dir": row["pai_mask_dir"],
                "mask_semantics": row["mask_semantics"],
            }
        )
    write_csv(out_dir / "or150_manifest.csv", combined, ["dataset", "video", "frame_dir", "mask_dir", "mask_semantics"])

    report = {
        "davis50_count": len(davis_rows),
        "youtubevos100_count": len(yt_rows),
        "combined_count": len(combined),
        "davis2017_root": str(davis_root),
        "pai_davis_target": args.pai_davis_target,
        "pai_youtubevos100_root": args.pai_youtubevos100_root,
    }
    (out_dir / "or150_manifest_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
