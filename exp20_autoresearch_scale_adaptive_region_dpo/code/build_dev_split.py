#!/usr/bin/env python3
"""Build the locked Exp20 engineering dev split.

The split is selected from the full YouTubeVOS 432x240 pool while excluding:

- the Exp11/Exp20 DPO training source video ids;
- the locked YouTubeVOS100 final-eval ids;
- DAVIS ids, when present.

The script writes a JSONL manifest, overlap/audit reports, and optionally a
symlinked evaluator root so the existing frame-wise DiffuEraser evaluator can
consume exactly the selected videos without modifying source data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
MASK_EXTS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class VideoStats:
    video_id: str
    frame_dir: Path
    mask_dir: Path
    num_frames: int
    num_masks: int
    selected_frames: list[str]
    selected_masks: list[str]
    frame_hash: str
    mask_hash: str
    mask_area_ratio: float
    boundary_length: float
    motion_proxy: float
    mask_motion_overlap: float
    mask_bucket: str = ""
    motion_bucket: str = ""


def sha256_files(paths: Iterable[Path], chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
    return h.hexdigest()


def read_train_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            vid = row.get("source_video_id") or row.get("exp8c_gtwin_video_id")
            if vid:
                ids.add(str(vid))
    return ids


def read_csv_ids(path: Path, key: str = "video_id") -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {str(row[key]) for row in reader if row.get(key)}


def read_jsonl_ids(path: Path, key: str = "video_id") -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get(key):
                ids.add(str(row[key]))
    return ids


def list_names(path: Path, exts: set[str]) -> list[Path]:
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts])


def bucketize(values: list[float]) -> list[str]:
    if not values:
        return []
    q1, q2 = np.quantile(np.asarray(values, dtype=np.float64), [1.0 / 3.0, 2.0 / 3.0])
    out = []
    for value in values:
        if value <= q1:
            out.append("low")
        elif value <= q2:
            out.append("medium")
        else:
            out.append("high")
    return out


def boundary_length(mask01: np.ndarray) -> float:
    mask_u8 = mask01.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    edge = cv2.dilate(mask_u8, kernel, iterations=1) - cv2.erode(mask_u8, kernel, iterations=1)
    return float(edge.mean())


def compute_stats(video_id: str, frame_dir: Path, mask_dir: Path, video_length: int) -> VideoStats | None:
    frames = list_names(frame_dir, IMAGE_EXTS)
    masks = list_names(mask_dir, MASK_EXTS)
    n = min(len(frames), len(masks), video_length)
    if n < max(4, min(video_length, 8)):
        return None
    frames = frames[:n]
    masks = masks[:n]

    area_vals: list[float] = []
    boundary_vals: list[float] = []
    motion_vals: list[float] = []
    mask_motion_vals: list[float] = []
    prev_gray = None
    prev_mask = None
    for frame_path, mask_path in zip(frames, masks):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if frame is None or mask is None:
            return None
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask01 = mask > 127
        area_vals.append(float(mask01.mean()))
        boundary_vals.append(boundary_length(mask01))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if prev_gray is not None:
            diff = np.abs(gray - prev_gray)
            motion_vals.append(float(diff.mean()))
            overlap = mask01 | (prev_mask if prev_mask is not None else mask01)
            mask_motion_vals.append(float(diff[overlap].mean()) if np.any(overlap) else 0.0)
        prev_gray = gray
        prev_mask = mask01

    return VideoStats(
        video_id=video_id,
        frame_dir=frame_dir,
        mask_dir=mask_dir,
        num_frames=len(list_names(frame_dir, IMAGE_EXTS)),
        num_masks=len(list_names(mask_dir, MASK_EXTS)),
        selected_frames=[p.name for p in frames],
        selected_masks=[p.name for p in masks],
        frame_hash="PENDING_SELECTED_ONLY",
        mask_hash="PENDING_SELECTED_ONLY",
        mask_area_ratio=float(np.mean(area_vals)),
        boundary_length=float(np.mean(boundary_vals)),
        motion_proxy=float(np.mean(motion_vals)) if motion_vals else 0.0,
        mask_motion_overlap=float(np.mean(mask_motion_vals)) if mask_motion_vals else 0.0,
    )


def symlink_dir(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and Path(os.readlink(dst)) == src:
            return
        raise FileExistsError(f"Refusing to overwrite existing dev symlink path: {dst}")
    os.symlink(src, dst)


def select_balanced(candidates: list[VideoStats], count: int) -> list[VideoStats]:
    if len(candidates) <= count:
        return candidates
    mask_buckets = bucketize([c.mask_area_ratio for c in candidates])
    motion_buckets = bucketize([c.motion_proxy for c in candidates])
    enriched = [
        VideoStats(
            **{
                **c.__dict__,
                "mask_bucket": mb,
                "motion_bucket": mob,
            }
        )
        for c, mb, mob in zip(candidates, mask_buckets, motion_buckets)
    ]
    selected: list[VideoStats] = []
    used: set[str] = set()
    for mb in ("low", "medium", "high"):
        for mob in ("low", "medium", "high"):
            group = [c for c in enriched if c.mask_bucket == mb and c.motion_bucket == mob and c.video_id not in used]
            group.sort(key=lambda c: (-c.mask_motion_overlap, c.video_id))
            if group:
                selected.append(group[0])
                used.add(group[0].video_id)
            if len(selected) >= count:
                return selected
    remaining = [c for c in enriched if c.video_id not in used]
    remaining.sort(key=lambda c: (-c.mask_motion_overlap, -c.motion_proxy, c.video_id))
    for item in remaining:
        selected.append(item)
        used.add(item.video_id)
        if len(selected) >= count:
            break
    return selected


def spread_sample(items: list[str], max_count: int) -> list[str]:
    if max_count <= 0 or len(items) <= max_count:
        return items
    if max_count == 1:
        return [items[0]]
    idxs = np.linspace(0, len(items) - 1, max_count)
    seen: set[int] = set()
    out: list[str] = []
    for idx in idxs:
        i = int(round(float(idx)))
        if i in seen:
            continue
        seen.add(i)
        out.append(items[i])
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--youtube-root", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--youtubevos100-manifest", required=True)
    parser.add_argument("--davis-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--overlap-report", required=True)
    parser.add_argument("--overlap-csv", required=True)
    parser.add_argument("--statistics-report", required=True)
    parser.add_argument("--exclude-manifest", action="append", default=[])
    parser.add_argument("--split-version", default="dev_boundary_search_v1")
    parser.add_argument("--materialized-root", default="")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--video-length", type=int, default=24)
    parser.add_argument("--max-candidates-to-score", type=int, default=120)
    args = parser.parse_args()

    youtube_root = Path(args.youtube_root)
    frame_root = youtube_root / "JPEGImages_432_240"
    mask_root = youtube_root / "test_masks"
    if not frame_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError(f"Expected JPEGImages_432_240/test_masks under {youtube_root}")

    train_ids = read_train_ids(Path(args.train_manifest))
    youtubevos100_ids = read_csv_ids(Path(args.youtubevos100_manifest), key="video_id")
    extra_excluded_ids: set[str] = set()
    for exclude_path in args.exclude_manifest:
        extra_excluded_ids |= read_jsonl_ids(Path(exclude_path), key="video_id")
    davis_ids = set()
    davis_frame_root = Path(args.davis_root) / "JPEGImages_432_240"
    if davis_frame_root.is_dir():
        davis_ids = {p.name for p in davis_frame_root.iterdir() if p.is_dir()}

    all_ids = sorted({p.name for p in frame_root.iterdir() if p.is_dir() and (mask_root / p.name).is_dir()})
    excluded = train_ids | youtubevos100_ids | davis_ids | extra_excluded_ids
    eligible_ids = [video_id for video_id in all_ids if video_id not in excluded]
    ids_to_score = spread_sample(eligible_ids, args.max_candidates_to_score)
    candidates: list[VideoStats] = []
    failed: list[dict[str, object]] = []
    for video_id in ids_to_score:
        try:
            stats = compute_stats(video_id, frame_root / video_id, mask_root / video_id, args.video_length)
        except Exception as exc:
            failed.append({"video_id": video_id, "reason": repr(exc)})
            continue
        if stats is None:
            failed.append({"video_id": video_id, "reason": "insufficient_or_unreadable_frames"})
            continue
        candidates.append(stats)

    selected = select_balanced(candidates, args.count)
    selected_with_hashes: list[VideoStats] = []
    for item in selected:
        frames = [item.frame_dir / name for name in item.selected_frames]
        masks = [item.mask_dir / name for name in item.selected_masks]
        selected_with_hashes.append(
            VideoStats(
                **{
                    **item.__dict__,
                    "frame_hash": sha256_files(frames),
                    "mask_hash": sha256_files(masks),
                }
            )
        )
    selected = selected_with_hashes
    materialized_root = Path(args.materialized_root) if args.materialized_root else None
    if materialized_root:
        for item in selected:
            symlink_dir(item.frame_dir, materialized_root / "JPEGImages_432_240" / item.video_id)
            symlink_dir(item.mask_dir, materialized_root / "test_masks" / item.video_id)

    manifest_path = Path(args.output_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with manifest_path.open("w", encoding="utf-8") as handle:
        for idx, item in enumerate(selected):
            row = {
                "split_version": args.split_version,
                "index": idx,
                "video_id": item.video_id,
                "source_dataset": "youtubevos_432_240",
                "frame_start": 0,
                "frame_end_exclusive": min(args.video_length, len(item.selected_frames)),
                "frame_dir": str(item.frame_dir),
                "mask_dir": str(item.mask_dir),
                "gt_dir": str(item.frame_dir),
                "num_frames_total": item.num_frames,
                "num_masks_total": item.num_masks,
                "frame_hash": item.frame_hash,
                "mask_hash": item.mask_hash,
                "mask_area_ratio": item.mask_area_ratio,
                "boundary_length": item.boundary_length,
                "motion_proxy": item.motion_proxy,
                "mask_motion_overlap": item.mask_motion_overlap,
                "mask_bucket": item.mask_bucket,
                "motion_bucket": item.motion_bucket,
                "selected_reason": f"balanced_{item.mask_bucket}_mask_{item.motion_bucket}_motion_non_train_non_final",
                "notes": "selected from full YouTubeVOS after excluding Exp11/Exp20 training source ids, YouTubeVOS100 final ids, DAVIS ids, and requested exclude manifests",
            }
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    overlap_rows = [
        {
            "set_name": "Exp11_Exp20_train_source_ids",
            "count": len(train_ids),
            "selected_overlap_count": len({r["video_id"] for r in rows} & train_ids),
        },
        {
            "set_name": "YouTubeVOS100_final_ids",
            "count": len(youtubevos100_ids),
            "selected_overlap_count": len({r["video_id"] for r in rows} & youtubevos100_ids),
        },
        {
            "set_name": "DAVIS50_ids_name_space",
            "count": len(davis_ids),
            "selected_overlap_count": len({r["video_id"] for r in rows} & davis_ids),
        },
        {
            "set_name": "extra_exclude_manifest_ids",
            "count": len(extra_excluded_ids),
            "selected_overlap_count": len({r["video_id"] for r in rows} & extra_excluded_ids),
        },
    ]
    write_csv(Path(args.overlap_csv), overlap_rows)

    status = "DEV_SPLIT_LOCKED_NO_OVERLAP" if all(int(r["selected_overlap_count"]) == 0 for r in overlap_rows) else "INTERNAL_ENGINEERING_DEV_WITH_OVERLAP"
    report = [
        "# Exp20 Dev Overlap Audit",
        "",
        f"- status: `{status}`",
        f"- source_pool: `{youtube_root}`",
        f"- total_youtubevos_videos_with_masks: `{len(all_ids)}`",
        f"- train_source_ids: `{len(train_ids)}`",
        f"- youtubevos100_final_ids: `{len(youtubevos100_ids)}`",
        f"- davis_name_space_ids: `{len(davis_ids)}`",
        f"- extra_exclude_manifest_ids: `{len(extra_excluded_ids)}`",
        f"- eligible_candidates_after_exclusion: `{len(eligible_ids)}`",
        f"- scored_candidates: `{len(candidates)}`",
        f"- max_candidates_to_score: `{args.max_candidates_to_score}`",
        f"- selected_videos: `{len(rows)}`",
        f"- split_version: `{args.split_version}`",
        f"- manifest: `{manifest_path}`",
        f"- manifest_sha256: `{manifest_sha}`",
    ]
    if materialized_root:
        report.append(f"- materialized_eval_root: `{materialized_root}`")
    report.extend(["", "## Overlap Counts", "", "| Set | Set Size | Selected Overlap |", "|---|---:|---:|"])
    for row in overlap_rows:
        report.append(f"| {row['set_name']} | {row['count']} | {row['selected_overlap_count']} |")
    report.append("")
    Path(args.overlap_report).write_text("\n".join(report) + "\n", encoding="utf-8")

    values = {
        "selected_count": len(rows),
        "manifest_sha256": manifest_sha,
        "mask_area_ratio_mean": float(np.mean([r["mask_area_ratio"] for r in rows])) if rows else math.nan,
        "mask_area_ratio_min": float(np.min([r["mask_area_ratio"] for r in rows])) if rows else math.nan,
        "mask_area_ratio_max": float(np.max([r["mask_area_ratio"] for r in rows])) if rows else math.nan,
        "motion_proxy_mean": float(np.mean([r["motion_proxy"] for r in rows])) if rows else math.nan,
        "motion_proxy_min": float(np.min([r["motion_proxy"] for r in rows])) if rows else math.nan,
        "motion_proxy_max": float(np.max([r["motion_proxy"] for r in rows])) if rows else math.nan,
    }
    stats_lines = ["# Exp20 Dev Split Statistics", ""]
    for key, value in values.items():
        stats_lines.append(f"- {key}: `{value}`")
    stats_lines.extend(["", "## Selected Videos", "", "| video_id | mask_bucket | motion_bucket | mask_area | motion | mask_motion |", "|---|---|---|---:|---:|---:|"])
    for row in rows:
        stats_lines.append(
            f"| {row['video_id']} | {row['mask_bucket']} | {row['motion_bucket']} | "
            f"{float(row['mask_area_ratio']):.6f} | {float(row['motion_proxy']):.6f} | {float(row['mask_motion_overlap']):.6f} |"
        )
    Path(args.statistics_report).write_text("\n".join(stats_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "selected": len(rows), "manifest_sha256": manifest_sha}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
