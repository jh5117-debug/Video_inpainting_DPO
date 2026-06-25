#!/usr/bin/env python3
"""Audit Gate64 formal-49F failures caused by duplicate decoded frames.

The Gate64 materializer is intentionally strict, but a duplicate-image failure
can come from different places:

- the source video really contains identical pixel frames in the selected span;
- OpenCV random seeking returns repeated frames even when sequential decode is
  unique;
- metadata/frame-count probing is inconsistent with actual decoding.

This audit is read-only. It inspects failed rows from
``gate64_materialized_49f_status.csv`` and writes CSV/JSON/Markdown evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2


ERROR_PATH_RE = re.compile(r"for\s+(.+?\.mp4)")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_json(cmd: list[str]) -> dict[str, Any]:
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def ffprobe_stream(path: Path) -> dict[str, Any]:
    try:
        return run_json(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames,avg_frame_rate,r_frame_rate,duration",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ]
        )
    except Exception as exc:  # noqa: BLE001 - audit records probe failures.
        return {"error": repr(exc)}


def ffprobe_frames(path: Path, limit: int = 80) -> dict[str, Any]:
    try:
        data = run_json(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_entries",
                "frame=best_effort_timestamp_time,pkt_pts_time,pkt_dts_time,pict_type,key_frame,coded_picture_number,display_picture_number",
                "-of",
                "json",
                str(path),
            ]
        )
        frames = data.get("frames", [])
        data["frames"] = frames[:limit]
        data["frame_count_reported_by_show_frames"] = len(frames)
        return data
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc), "frames": [], "frame_count_reported_by_show_frames": 0}


def hash_frame(frame: Any) -> str:
    return hashlib.sha256(frame.tobytes()).hexdigest()


def decode_sequential(path: Path, max_frames: int = 260) -> tuple[list[str], list[dict[str, Any]]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    hashes: list[str] = []
    meta: list[dict[str, Any]] = []
    idx = 0
    while idx < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        hashes.append(hash_frame(frame))
        meta.append(
            {
                "idx": idx,
                "pos_frames_after_read": float(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                "pos_msec_after_read": float(cap.get(cv2.CAP_PROP_POS_MSEC)),
            }
        )
        idx += 1
    cap.release()
    return hashes, meta


def decode_seek(path: Path, indices: list[int]) -> tuple[list[str], list[dict[str, Any]]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    hashes: list[str] = []
    meta: list[dict[str, Any]] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            meta.append({"requested_idx": idx, "ok": False})
            hashes.append("")
            continue
        hashes.append(hash_frame(frame))
        meta.append(
            {
                "requested_idx": idx,
                "ok": True,
                "pos_frames_after_read": float(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                "pos_msec_after_read": float(cap.get(cv2.CAP_PROP_POS_MSEC)),
            }
        )
    cap.release()
    return hashes, meta


def duplicate_groups(hashes: list[str]) -> list[list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, value in enumerate(hashes):
        if value:
            groups[value].append(idx)
    return [items for items in groups.values() if len(items) > 1]


def summarize_groups(groups: list[list[int]], limit: int = 5) -> str:
    if not groups:
        return ""
    return ";".join(",".join(str(i) for i in g[:8]) for g in groups[:limit])


def extract_error_path(row: dict[str, str]) -> str:
    for key in ("source_video_path", "error"):
        value = row.get(key, "")
        if value and value.endswith(".mp4") and Path(value).is_absolute():
            return value
    match = ERROR_PATH_RE.search(row.get("error", ""))
    return match.group(1) if match else ""


def resolve_video_path(raw: str, run_root: Path) -> Path:
    path = Path(raw)
    if path.exists():
        return path
    marker = "/source_videos/"
    if marker in raw:
        suffix = raw.split(marker, 1)[1]
        candidate = run_root / "source_videos" / suffix
        if candidate.exists():
            return candidate
    return path


def classify(
    seq_hashes: list[str],
    seek_hashes: list[str],
    target_indices: list[int],
    ff_frame_count: int,
) -> str:
    target_seq = [seq_hashes[i] for i in target_indices if i < len(seq_hashes)]
    seq_unique = len(set(target_seq))
    seek_unique = len(set(h for h in seek_hashes if h))
    if len(target_seq) < len(target_indices):
        return "INSUFFICIENT_DECODED_FRAMES"
    if seq_unique == len(target_indices) and seek_unique < len(target_indices):
        return "OPENCV_RANDOM_SEEK_DUPLICATE"
    if seq_unique < len(target_indices) and seek_unique < len(target_indices):
        return "SOURCE_PIXEL_DUPLICATE_IN_SELECTED_49F"
    if ff_frame_count and ff_frame_count < len(target_indices):
        return "FFPROBE_INSUFFICIENT_FRAMES"
    if seq_unique == len(target_indices) and seek_unique == len(target_indices):
        return "NO_DUPLICATE_REPRODUCED"
    return "UNCLASSIFIED_DUPLICATE_FAILURE"


def audit_row(row: dict[str, str], run_root: Path, indices: list[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_id = row["sample_id"]
    raw_path = extract_error_path(row)
    video_path = resolve_video_path(raw_path, run_root)
    detail: dict[str, Any] = {
        "sample_id": sample_id,
        "raw_error": row.get("error", ""),
        "raw_video_path": raw_path,
        "resolved_video_path": str(video_path),
        "exists": video_path.exists(),
    }
    if not video_path.exists():
        summary = {
            "sample_id": sample_id,
            "classification": "SOURCE_VIDEO_MISSING",
            "resolved_video_path": str(video_path),
            "exists": False,
        }
        return summary, detail

    stream = ffprobe_stream(video_path)
    frames = ffprobe_frames(video_path)
    seq_hashes, seq_meta = decode_sequential(video_path)
    seek_hashes, seek_meta = decode_seek(video_path, indices)
    target_seq_hashes = [seq_hashes[i] for i in indices if i < len(seq_hashes)]

    seq_groups = duplicate_groups(target_seq_hashes)
    seek_groups = duplicate_groups(seek_hashes)
    ff_count = int(frames.get("frame_count_reported_by_show_frames") or 0)
    classification = classify(seq_hashes, seek_hashes, indices, ff_count)

    stream0 = (stream.get("streams") or [{}])[0] if isinstance(stream, dict) else {}
    summary = {
        "sample_id": sample_id,
        "classification": classification,
        "resolved_video_path": str(video_path),
        "exists": True,
        "width": stream0.get("width", ""),
        "height": stream0.get("height", ""),
        "nb_frames": stream0.get("nb_frames", ""),
        "avg_frame_rate": stream0.get("avg_frame_rate", ""),
        "duration": stream0.get("duration", ""),
        "ffprobe_show_frames_count": ff_count,
        "cv2_sequential_decoded_count": len(seq_hashes),
        "selected_count": len(indices),
        "selected_sequential_unique_count": len(set(target_seq_hashes)),
        "selected_seek_unique_count": len(set(h for h in seek_hashes if h)),
        "selected_sequential_duplicate_group_count": len(seq_groups),
        "selected_seek_duplicate_group_count": len(seek_groups),
        "selected_sequential_duplicate_groups": summarize_groups(seq_groups),
        "selected_seek_duplicate_groups": summarize_groups(seek_groups),
        "recommendation": recommendation_for(classification),
    }
    detail.update(
        {
            "ffprobe_stream": stream,
            "ffprobe_frames_head": frames,
            "sequential_meta_head": seq_meta[:80],
            "seek_meta": seek_meta,
            "selected_indices": indices,
            "selected_sequential_duplicate_groups": seq_groups,
            "selected_seek_duplicate_groups": seek_groups,
            "classification": classification,
        }
    )
    return summary, detail


def recommendation_for(classification: str) -> str:
    if classification == "OPENCV_RANDOM_SEEK_DUPLICATE":
        return "fix_materializer_to_sequential_decode_or_verify_seek_before_rejecting"
    if classification == "SOURCE_PIXEL_DUPLICATE_IN_SELECTED_49F":
        return "treat_as_source_static_duplicate; decide whether formal mode permits pixel-identical real frames"
    if classification == "INSUFFICIENT_DECODED_FRAMES":
        return "reject_source_or_adjust_source_pool; not a model failure"
    if classification == "NO_DUPLICATE_REPRODUCED":
        return "rerun_materializer_with_debug_logging; failure not reproduced"
    return "manual_review_required"


def write_markdown(path: Path, rows: list[dict[str, Any]], run_root: Path) -> None:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["classification"])] += 1
    lines = [
        "# Exp26 Gate64 Duplicate Source Audit",
        "",
        f"Run root: `{run_root}`",
        "",
        "## Summary",
        "",
    ]
    for key in sorted(counts):
        lines.append(f"- `{key}`: {counts[key]}")
    lines.extend(
        [
            "",
            "## Failed Source Rows",
            "",
            "| sample_id | classification | seq unique | seek unique | duplicate groups | recommendation |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {sample_id} | {classification} | {selected_sequential_unique_count} | "
            "{selected_seek_unique_count} | {selected_sequential_duplicate_groups} | "
            "{recommendation} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `SOURCE_PIXEL_DUPLICATE_IN_SELECTED_49F` means sequential decoding of the first 49 source frames already contains pixel-identical frames.",
            "- `OPENCV_RANDOM_SEEK_DUPLICATE` means sequential decode is unique but OpenCV random seeking repeated frames; this is a materializer implementation issue.",
            "- This audit does not alter Gate64 outputs and does not start VideoPainter DPO.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--status-csv", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    status_csv = args.status_csv or args.run_root / "gate64_materialized_49f_status.csv"
    rows = read_csv(status_csv)
    failed = [row for row in rows if row.get("status") != "OK"]
    indices = [args.offset + i * args.stride for i in range(args.num_frames)]
    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for row in failed:
        summary, detail = audit_row(row, args.run_root, indices)
        summaries.append(summary)
        details.append(detail)

    write_csv(args.output_csv, summaries)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(details, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(args.output_md, summaries, args.run_root)
    print(json.dumps({"failed_rows": len(failed), "audited": len(summaries), "csv": str(args.output_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
