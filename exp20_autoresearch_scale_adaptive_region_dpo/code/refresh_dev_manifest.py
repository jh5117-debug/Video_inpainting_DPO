#!/usr/bin/env python3
"""Refresh an existing Exp20 dev manifest to a fixed frame length."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np


def list_files(path: Path, exts: set[str]) -> list[Path]:
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts])


def sha256_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def boundary_length(mask01: np.ndarray) -> float:
    kernel = np.ones((3, 3), dtype=np.uint8)
    m = mask01.astype(np.uint8)
    return float((cv2.dilate(m, kernel) - cv2.erode(m, kernel)).mean())


def refresh_row(row: dict[str, object], video_length: int) -> dict[str, object]:
    frame_dir = Path(str(row["frame_dir"]))
    mask_dir = Path(str(row["mask_dir"]))
    frames = list_files(frame_dir, {".jpg", ".jpeg", ".png"})[:video_length]
    masks = list_files(mask_dir, {".png", ".jpg", ".jpeg"})[:video_length]
    if len(frames) < video_length or len(masks) < video_length:
        raise ValueError(f"{row['video_id']} has too few frames/masks for length={video_length}")
    areas: list[float] = []
    boundaries: list[float] = []
    motions: list[float] = []
    mask_motions: list[float] = []
    prev_gray = None
    prev_mask = None
    for frame_path, mask_path in zip(frames, masks):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if frame is None or mask is None:
            raise ValueError(f"bad read {frame_path} {mask_path}")
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask01 = mask > 127
        areas.append(float(mask01.mean()))
        boundaries.append(boundary_length(mask01))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if prev_gray is not None:
            diff = np.abs(gray - prev_gray)
            motions.append(float(diff.mean()))
            overlap = mask01 | prev_mask
            mask_motions.append(float(diff[overlap].mean()) if np.any(overlap) else 0.0)
        prev_gray = gray
        prev_mask = mask01
    out = dict(row)
    out.update(
        {
            "frame_end_exclusive": video_length,
            "frame_hash": sha256_files(frames),
            "mask_hash": sha256_files(masks),
            "mask_area_ratio": float(np.mean(areas)),
            "boundary_length": float(np.mean(boundaries)),
            "motion_proxy": float(np.mean(motions)) if motions else 0.0,
            "mask_motion_overlap": float(np.mean(mask_motions)) if mask_motions else 0.0,
            "notes": str(row.get("notes", "")) + f"; refreshed_to_{video_length}_frames_for_diffueraser_min_duration",
        }
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--overlap-report", required=True)
    parser.add_argument("--statistics-report", required=True)
    parser.add_argument("--video-length", type=int, default=24)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    refreshed = [refresh_row(row, args.video_length) for row in rows]
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in refreshed), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()

    report = Path(args.overlap_report)
    text = report.read_text(encoding="utf-8")
    text = re.sub(r"manifest_sha256: `[^`]+`", f"manifest_sha256: `{manifest_sha}`", text)
    if "- video_length:" in text:
        text = re.sub(r"- video_length: `[^`]+`", f"- video_length: `{args.video_length}`", text)
    else:
        text = text.replace("- selected_videos: `16`", f"- selected_videos: `16`\n- video_length: `{args.video_length}`")
    report.write_text(text, encoding="utf-8")

    stats = Path(args.statistics_report)
    lines = [
        "# Exp20 Dev Split Statistics",
        "",
        f"- selected_count: `{len(refreshed)}`",
        f"- manifest_sha256: `{manifest_sha}`",
        f"- video_length: `{args.video_length}`",
        f"- mask_area_ratio_mean: `{float(np.mean([r['mask_area_ratio'] for r in refreshed]))}`",
        f"- mask_area_ratio_min: `{float(np.min([r['mask_area_ratio'] for r in refreshed]))}`",
        f"- mask_area_ratio_max: `{float(np.max([r['mask_area_ratio'] for r in refreshed]))}`",
        f"- motion_proxy_mean: `{float(np.mean([r['motion_proxy'] for r in refreshed]))}`",
        f"- motion_proxy_min: `{float(np.min([r['motion_proxy'] for r in refreshed]))}`",
        f"- motion_proxy_max: `{float(np.max([r['motion_proxy'] for r in refreshed]))}`",
        "",
        "## Selected Videos",
        "",
        "| video_id | mask_bucket | motion_bucket | mask_area | motion | mask_motion |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in refreshed:
        lines.append(
            f"| {row['video_id']} | {row['mask_bucket']} | {row['motion_bucket']} | "
            f"{float(row['mask_area_ratio']):.6f} | {float(row['motion_proxy']):.6f} | "
            f"{float(row['mask_motion_overlap']):.6f} |"
        )
    stats.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"selected": len(refreshed), "video_length": args.video_length, "manifest_sha256": manifest_sha}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
