#!/usr/bin/env python3
"""Deep audit for Exp26 Gate64 formal-source duplicate-frame failures.

This read-only audit answers a narrower question than the original guard:
whether a rejected source has 49 real decoded positions with unique, monotonic
timestamps. Pixel-identical static frames are recorded as a diagnostic, but are
not by themselves treated as padding/loop/interpolation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import av
import cv2

from audit_gate64_duplicate_sources import extract_error_path, read_csv, resolve_video_path, write_csv


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_text(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        return proc.returncode, proc.stdout
    except Exception as exc:  # noqa: BLE001 - dependency failures are audit evidence.
        return 127, repr(exc)


def pyav_decode(path: Path, max_frames: int = 260) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for idx, frame in enumerate(container.decode(stream)):
            if idx >= max_frames:
                break
            arr = frame.to_ndarray(format="rgb24")
            rows.append(
                {
                    "index": idx,
                    "pts": frame.pts,
                    "time": float(frame.time) if frame.time is not None else None,
                    "width": frame.width,
                    "height": frame.height,
                    "hash": sha_bytes(arr.tobytes()),
                }
            )
    return rows


def cv2_seek_hashes(path: Path, indices: list[int]) -> list[dict[str, Any]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    rows: list[dict[str, Any]] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            rows.append({"requested_index": idx, "ok": False, "hash": ""})
            continue
        rows.append(
            {
                "requested_index": idx,
                "ok": True,
                "pos_frames_after_read": float(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                "pos_msec_after_read": float(cap.get(cv2.CAP_PROP_POS_MSEC)),
                "hash": sha_bytes(frame.tobytes()),
            }
        )
    cap.release()
    return rows


def duplicate_groups(values: list[Any]) -> list[list[int]]:
    groups: dict[Any, list[int]] = defaultdict(list)
    for idx, value in enumerate(values):
        if value not in (None, ""):
            groups[value].append(idx)
    return [v for v in groups.values() if len(v) > 1]


def groups_short(groups: list[list[int]], limit: int = 5) -> str:
    return ";".join(",".join(str(i) for i in g[:10]) for g in groups[:limit])


def classify(selected: list[dict[str, Any]], cv_seek: list[dict[str, Any]], indices: list[int]) -> tuple[str, bool, str]:
    if len(selected) < len(indices):
        return "D_DECODER_RETURNED_FEWER_THAN_49_REAL_POSITIONS", False, "reject_source_or_use_replacement"
    pts = [row["pts"] for row in selected]
    times = [row["time"] for row in selected]
    hashes = [row["hash"] for row in selected]
    unique_pts = len(set(pts)) == len(pts)
    monotonic_pts = all(a is None or b is None or b > a for a, b in zip(pts, pts[1:]))
    unique_time = len(set(times)) == len(times) if all(t is not None for t in times) else True
    monotonic_time = all(a is None or b is None or b > a for a, b in zip(times, times[1:]))
    pixel_unique = len(set(hashes)) == len(hashes)
    seek_ok = all(row.get("ok") for row in cv_seek)
    seek_unique = len({row.get("hash") for row in cv_seek if row.get("hash")}) == len(indices)

    if not unique_pts or not monotonic_pts or not unique_time or not monotonic_time:
        return "C_PTS_OR_TIMESTAMP_REPEAT_OR_NONMONOTONIC", False, "reject_source_or_inspect_container"
    if not seek_ok:
        return "D_RANDOM_SEEK_DECODE_FAILED", False, "fix_materializer_or_replacement"
    if pixel_unique and not seek_unique:
        return "G_RANDOM_SEEK_HASH_DUPLICATE_PYAV_UNIQUE", True, "fix_materializer_seek_guard"
    if not pixel_unique:
        return "F_STATIC_PIXEL_DUPLICATE_BUT_UNIQUE_TIMESTAMPS", True, "guard_misclassified_static_pixels_as_invalid"
    return "NO_DUPLICATE_REPRODUCED_FORMAL_VALID", True, "rerun_materializer_with_timestamp_based_guard"


def audit_row(row: dict[str, str], run_root: Path, indices: list[int], detail_dir: Path) -> dict[str, Any]:
    sample_id = row["sample_id"]
    raw_path = extract_error_path(row)
    path = resolve_video_path(raw_path, run_root)
    out: dict[str, Any] = {
        "sample_id": sample_id,
        "raw_video_path": raw_path,
        "resolved_video_path": str(path),
        "exists": path.exists(),
        "original_error": row.get("error", ""),
    }
    if not path.exists():
        out.update({"classification": "SOURCE_VIDEO_MISSING", "formal_valid": False, "recommendation": "replacement"})
        return out

    pyav_rows = pyav_decode(path, max(indices) + 1)
    selected = [pyav_rows[i] for i in indices if i < len(pyav_rows)]
    pyav_rows_2 = pyav_decode(path, max(indices) + 1)
    selected_2 = [pyav_rows_2[i] for i in indices if i < len(pyav_rows_2)]
    cv_seek = cv2_seek_hashes(path, indices)
    classification, formal_valid, recommendation = classify(selected, cv_seek, indices)

    hashes = [r["hash"] for r in selected]
    pts = [r["pts"] for r in selected]
    times = [r["time"] for r in selected]
    pyav_repeat_deterministic = [r["hash"] for r in selected] == [r["hash"] for r in selected_2]
    ffprobe_rc, ffprobe_out = run_text(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time,pkt_pts_time,pkt_dts_time,coded_picture_number,display_picture_number",
            "-of",
            "json",
            str(path),
        ]
    )
    framemd5_rc, framemd5_out = run_text(["ffmpeg", "-v", "error", "-i", str(path), "-f", "framemd5", "-"])

    detail_dir.mkdir(parents=True, exist_ok=True)
    detail_path = detail_dir / f"{sample_id}.json"
    detail = {
        "sample_id": sample_id,
        "path": str(path),
        "selected_indices": indices,
        "selected_pyav": selected,
        "selected_pyav_second_decode": selected_2,
        "cv2_seek": cv_seek,
        "pixel_duplicate_groups": duplicate_groups(hashes),
        "pts_duplicate_groups": duplicate_groups(pts),
        "time_duplicate_groups": duplicate_groups(times),
        "ffprobe_rc": ffprobe_rc,
        "ffprobe_output_head": ffprobe_out[:8000],
        "framemd5_rc": framemd5_rc,
        "framemd5_output_head": framemd5_out[:8000],
    }
    detail_path.write_text(json.dumps(detail, indent=2, sort_keys=True), encoding="utf-8")

    out.update(
        {
            "classification": classification,
            "formal_valid": formal_valid,
            "recommendation": recommendation,
            "pyav_decoded_count": len(pyav_rows),
            "selected_count": len(selected),
            "selected_unique_hashes": len(set(hashes)),
            "selected_unique_pts": len(set(pts)),
            "selected_unique_times": len(set(times)) if all(t is not None for t in times) else "",
            "pixel_duplicate_groups": groups_short(duplicate_groups(hashes)),
            "pts_duplicate_groups": groups_short(duplicate_groups(pts)),
            "time_duplicate_groups": groups_short(duplicate_groups(times)),
            "pyav_repeat_deterministic": pyav_repeat_deterministic,
            "cv2_seek_ok_count": sum(1 for r in cv_seek if r.get("ok")),
            "cv2_seek_unique_hashes": len({r.get("hash") for r in cv_seek if r.get("hash")}),
            "ffprobe_available": ffprobe_rc == 0,
            "ffmpeg_framemd5_available": framemd5_rc == 0,
            "detail_json": str(detail_path),
        }
    )
    return out


def write_markdown(path: Path, rows: list[dict[str, Any]], run_root: Path) -> None:
    counts = Counter(str(r["classification"]) for r in rows)
    formal_valid = sum(1 for r in rows if str(r.get("formal_valid")).lower() in {"true", "1"})
    lines = [
        "# Exp26 Gate64 Deep Duplicate Source Audit",
        "",
        f"Run root: `{run_root}`",
        "",
        "This audit treats 49 unique decoded indices with unique monotonic PTS/time as formal-valid even when two static frames are pixel-identical.",
        "",
        "## Summary",
        "",
        f"- audited failures: `{len(rows)}`",
        f"- formal-valid after timestamp/index audit: `{formal_valid}`",
    ]
    for key in sorted(counts):
        lines.append(f"- `{key}`: `{counts[key]}`")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| sample_id | classification | formal_valid | unique_hashes | unique_pts | pixel dup groups | ffprobe | framemd5 | recommendation |",
            "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {sample_id} | {classification} | {formal_valid} | {selected_unique_hashes} | {selected_unique_pts} | {pixel_duplicate_groups} | {ffprobe_available} | {ffmpeg_framemd5_available} | {recommendation} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Dependency Note",
            "",
            "If `ffprobe_available` or `ffmpeg_framemd5_available` is false, the row still has PyAV timestamp/index evidence and OpenCV seek evidence; the missing CLI tool is tracked as an environment issue, not as source invalidity.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--status-csv", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--detail-dir", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    status_csv = args.status_csv or args.run_root / "gate64_materialized_49f_status.csv"
    rows = read_csv(status_csv)
    failed = [row for row in rows if row.get("status") != "OK"]
    indices = [args.offset + i * args.stride for i in range(args.num_frames)]
    summaries = [audit_row(row, args.run_root, indices, args.detail_dir) for row in failed]
    write_csv(args.output_csv, summaries)
    args.output_json.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(args.output_md, summaries, args.run_root)
    print(json.dumps({"failed_rows": len(failed), "csv": str(args.output_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
