#!/usr/bin/env python3
"""Inspect generated-loser manifest media paths.

The generated-loser manifests usually point to frame directories, but this
utility also accepts video files for future exported artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"[error] invalid JSONL {path}:{line_no}: {exc}") from exc
    return rows


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def inspect_frame_dir(path: Path, expect_frames: int, expect_height: int, expect_width: int) -> list[str]:
    errors = []
    frames = image_files(path)
    if len(frames) != expect_frames:
        errors.append(f"{path}: frames={len(frames)} expected={expect_frames}")
    for frame in frames[:expect_frames]:
        try:
            with Image.open(frame) as im:
                width, height = im.size
        except Exception as exc:
            errors.append(f"{frame}: unreadable image: {exc}")
            continue
        if width != expect_width or height != expect_height:
            errors.append(f"{frame}: resolution={height}x{width} expected={expect_height}x{expect_width}")
    return errors


def inspect_video_file(path: Path, expect_frames: int, expect_height: int, expect_width: int) -> list[str]:
    try:
        import cv2
    except Exception as exc:
        return [f"{path}: cv2 import failed for video inspection: {exc}"]

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [f"{path}: cannot open video"]
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    errors = []
    if frames != expect_frames:
        errors.append(f"{path}: frames={frames} expected={expect_frames}")
    if width != expect_width or height != expect_height:
        errors.append(f"{path}: resolution={height}x{width} expected={expect_height}x{expect_width}")
    return errors


def inspect_path(path_value: str, expect_frames: int, expect_height: int, expect_width: int) -> list[str]:
    if not path_value:
        return []
    path = Path(path_value)
    if path.is_dir():
        return inspect_frame_dir(path, expect_frames, expect_height, expect_width)
    if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
        return inspect_video_file(path, expect_frames, expect_height, expect_width)
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [f"{path}: expected frame directory or video, got single image file"]
    return [f"{path}: missing or unsupported media path"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate generated-loser manifest media paths.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--path_fields",
        default="final_loser_video_path,raw_loser_video_path,win_video_path,mask_path",
        help="Comma-separated manifest fields to inspect when present.",
    )
    parser.add_argument("--expect_frames", type=int, default=16)
    parser.add_argument("--expect_height", type=int, default=320)
    parser.add_argument("--expect_width", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warn_prefix", action="append", default=[], help="Report paths beginning with this prefix.")
    parser.add_argument("--max_errors", type=int, default=50)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = read_jsonl(Path(args.manifest))
    if args.limit > 0:
        rows = rows[: args.limit]
    fields = [x.strip() for x in args.path_fields.split(",") if x.strip()]

    checked_paths: set[str] = set()
    field_counts: Counter[str] = Counter()
    prefix_hits: Counter[str] = Counter()
    errors: list[str] = []

    for row in rows:
        for field in fields:
            value = str(row.get(field) or "")
            if not value or value in checked_paths:
                continue
            checked_paths.add(value)
            field_counts[field] += 1
            for prefix in args.warn_prefix:
                if value == prefix or value.startswith(prefix.rstrip("/") + "/"):
                    prefix_hits[prefix] += 1
            errors.extend(inspect_path(value, args.expect_frames, args.expect_height, args.expect_width))
            if len(errors) >= args.max_errors:
                break
        if len(errors) >= args.max_errors:
            break

    summary = {
        "manifest": args.manifest,
        "rows_checked": len(rows),
        "unique_paths_checked": len(checked_paths),
        "field_counts": dict(field_counts),
        "warn_prefix_hits": dict(prefix_hits),
        "errors": errors[: args.max_errors],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
