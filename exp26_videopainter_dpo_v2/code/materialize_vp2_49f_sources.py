#!/usr/bin/env python3
"""Materialize formal VideoPainter 49-frame VOR-BG sources.

This script is intentionally strict: formal mode requires exactly 49 decoded,
unique source frames selected by the locked sampler. It never pads, loops,
duplicates, interpolates, or falls back to the 13-frame plumbing path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import cv2


IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_member_path(row: dict, source_root: Path) -> Path:
    for key in ("winner_path", "video_path", "source_video_path", "local_video_path"):
        value = row.get(key)
        if value:
            path = Path(value)
            if path.exists():
                return path
            candidate = source_root / value
            if candidate.exists():
                return candidate
    member = row.get("winner_member_path")
    if not member:
        raise FileNotFoundError(f"row has no local path or winner_member_path: {row.get('sample_id')}")
    candidate = source_root / member
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"missing source video for {row.get('sample_id')}: {candidate}")


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"frame_count": count, "fps": fps, "width": width, "height": height}


def official_indices(frame_count: int, num_frames: int, stride: int, offset: int) -> list[int]:
    if num_frames != 49:
        raise ValueError("formal VideoPainter mode requires num_frames=49")
    indices = [offset + i * stride for i in range(num_frames)]
    if len(indices) != len(set(indices)):
        raise ValueError("official sampler produced duplicate frame indices")
    if not indices or indices[-1] >= frame_count:
        raise ValueError(
            f"video has {frame_count} frames; cannot read 49 unique frames "
            f"with stride={stride}, offset={offset}"
        )
    return indices


def duplicate_groups(values: list[str]) -> list[list[int]]:
    groups: dict[str, list[int]] = {}
    for idx, value in enumerate(values):
        groups.setdefault(value, []).append(idx)
    return [items for items in groups.values() if len(items) > 1]


def summarize_groups(groups: list[list[int]], limit: int = 8) -> str:
    return ";".join(",".join(str(i) for i in group[:12]) for group in groups[:limit])


def decode_indices(video_path: Path, indices: list[int], output_dir: Path) -> tuple[list[str], list[str], dict]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    hashes: list[str] = []
    paths: list[str] = []
    for out_idx, src_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(src_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"failed to decode frame {src_idx} from {video_path}")
        path = output_dir / f"{out_idx:05d}.png"
        cv2.imwrite(str(path), frame)
        hashes.append(sha256_file(path))
        paths.append(str(path))
    cap.release()
    dup_groups = duplicate_groups(hashes)
    diagnostics = {
        "frame_hash_unique_count": len(set(hashes)),
        "frame_hash_duplicate_group_count": len(dup_groups),
        "frame_hash_duplicate_groups": summarize_groups(dup_groups),
        "pixel_duplicate_policy": "allowed_static_pixels_unique_indices",
    }
    return paths, hashes, diagnostics


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--status-csv", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    if args.limit > 0:
        rows = rows[: args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    materialized: list[dict] = []
    status_rows: list[dict] = []
    for row in rows:
        sample_id = row["sample_id"]
        try:
            video_path = resolve_member_path(row, args.source_root)
            info = video_info(video_path)
            indices = official_indices(info["frame_count"], args.num_frames, args.stride, args.offset)
            frame_dir = args.output_root / sample_id / "frames"
            frame_paths, frame_hashes, frame_hash_diagnostics = decode_indices(video_path, indices, frame_dir)
            out_row = dict(row)
            out_row.update(
                {
                    "status": "FORMAL_49F_MATERIALIZED",
                    "formal_49f": True,
                    "plumbing_only_13f": False,
                    "source_video_path": str(video_path),
                    "frame_dir": str(frame_dir),
                    "frame_indices": indices,
                    "frame_paths": frame_paths,
                    "frame_hashes": frame_hashes,
                    **frame_hash_diagnostics,
                    "original_frame_count": info["frame_count"],
                    "original_fps": info["fps"],
                    "width": info["width"],
                    "height": info["height"],
                    "sampler": {"name": "first_49_unique_frames", "stride": args.stride, "offset": args.offset},
                }
            )
            materialized.append(out_row)
            status = {
                "sample_id": sample_id,
                "status": "OK",
                "source_video_path": str(video_path),
                "original_frame_count": info["frame_count"],
                "fps": info["fps"],
                "width": info["width"],
                "height": info["height"],
                "frame_count": len(indices),
                **frame_hash_diagnostics,
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001 - resumable status table records per-row failures.
            status = {
                "sample_id": sample_id,
                "status": "FAILED",
                "source_video_path": "",
                "original_frame_count": "",
                "fps": "",
                "width": "",
                "height": "",
                "frame_count": 0,
                "error": repr(exc),
            }
        status_rows.append(status)

    write_jsonl(args.output_manifest, materialized)
    args.status_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.status_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(status_rows[0].keys()) if status_rows else ["sample_id", "status"])
        writer.writeheader()
        writer.writerows(status_rows)
    print(json.dumps({"ok": len(materialized), "failed": len(status_rows) - len(materialized), "manifest": str(args.output_manifest)}, indent=2))
    return 0 if materialized else 2


if __name__ == "__main__":
    raise SystemExit(main())
