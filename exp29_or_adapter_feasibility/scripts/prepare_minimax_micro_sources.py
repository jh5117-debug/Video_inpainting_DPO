#!/usr/bin/env python3
"""Prepare fixed VOR OR source frames for the MiniMax micro data gate.

This script is CPU-only. It selects 32 source triplets from the already audited
VOR triplet audit CSV, materializes the first N aligned frames, and writes a
locked source manifest. It does not run MiniMax inference and does not create
training pairs by itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-sources", type=int, default=32)
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed-rule", default="20260626,20260627,20260628")
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bucket_mask(area: float) -> str:
    if area < 0.035:
        return "small"
    if area < 0.10:
        return "medium"
    return "large"


def center_crop_resize(img: np.ndarray, width: int, height: int, interpolation: int) -> np.ndarray:
    h, w = img.shape[:2]
    target_ratio = width / height
    ratio = w / h
    if ratio > target_ratio:
        new_w = int(round(h * target_ratio))
        x0 = max(0, (w - new_w) // 2)
        img = img[:, x0 : x0 + new_w]
    elif ratio < target_ratio:
        new_h = int(round(w / target_ratio))
        y0 = max(0, (h - new_h) // 2)
        img = img[y0 : y0 + new_h, :]
    return cv2.resize(img, (width, height), interpolation=interpolation)


def read_video_frames(path: Path, num_frames: int, width: int, height: int, gray: bool = False) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    frames: list[np.ndarray] = []
    while len(frames) < num_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = center_crop_resize(frame, width, height, cv2.INTER_NEAREST)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = center_crop_resize(frame, width, height, cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()
    if len(frames) != num_frames:
        raise RuntimeError(f"expected {num_frames} frames from {path}, got {len(frames)}")
    return frames


def save_rgb(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))


def save_mask(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = (frame > 20).astype(np.uint8) * 255
    cv2.imwrite(str(path), mask)


def image_strip(frames: Iterable[np.ndarray], labels: Iterable[str]) -> np.ndarray:
    rows = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for frame, label in zip(frames, labels):
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        tile = frame.copy()
        cv2.putText(tile, label, (8, 24), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(tile)
    return np.concatenate(rows, axis=1)


def select_rows(rows: list[dict[str, str]], num_sources: int) -> list[dict[str, str]]:
    ok_rows = [
        r
        for r in rows
        if r.get("status") == "OK"
        and r.get("aligned_frames") == "True"
        and r.get("aligned_size") == "True"
        and Path(r["fg_bg_path"]).exists()
        and Path(r["bg_path"]).exists()
        and Path(r["mask_path"]).exists()
    ]
    for row in ok_rows:
        row["mask_bucket"] = bucket_mask(float(row["mask_area_mean"]))
        row["selection_key"] = sha256_text(row["sample_id"])

    selected: list[dict[str, str]] = []
    targets = [
        ("REAL", "small"),
        ("REAL", "medium"),
        ("REAL", "large"),
        ("BLENDER", "small"),
        ("BLENDER", "medium"),
        ("BLENDER", "large"),
    ]
    seen: set[str] = set()
    while len(selected) < num_sources:
        advanced = False
        for source_type, bucket in targets:
            candidates = sorted(
                [
                    r
                    for r in ok_rows
                    if r["sample_id"] not in seen
                    and r.get("source_type") == source_type
                    and r.get("mask_bucket") == bucket
                ],
                key=lambda r: r["selection_key"],
            )
            if candidates and len(selected) < num_sources:
                picked = candidates[0]
                selected.append(picked)
                seen.add(picked["sample_id"])
                advanced = True
        if not advanced:
            remaining = sorted([r for r in ok_rows if r["sample_id"] not in seen], key=lambda r: r["selection_key"])
            if not remaining:
                break
            picked = remaining[0]
            selected.append(picked)
            seen.add(picked["sample_id"])
    if len(selected) != num_sources:
        raise RuntimeError(f"selected {len(selected)} sources, expected {num_sources}")
    return selected


def main() -> None:
    args = parse_args()
    audit_csv = Path(args.audit_csv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    sources_root = output_root / "materialized_sources"
    evidence_root = output_root / "source_evidence"
    with audit_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    selected = select_rows(rows, args.num_sources)
    seeds = [int(part) for part in args.seed_rule.split(",") if part.strip()]

    manifest_rows = []
    for idx, row in enumerate(selected):
        sample_id = row["sample_id"]
        sample_root = sources_root / sample_id
        fg_frames = read_video_frames(Path(row["fg_bg_path"]), args.num_frames, args.width, args.height, gray=False)
        bg_frames = read_video_frames(Path(row["bg_path"]), args.num_frames, args.width, args.height, gray=False)
        mask_frames = read_video_frames(Path(row["mask_path"]), args.num_frames, args.width, args.height, gray=True)
        for frame_idx, (fg, bg, mask) in enumerate(zip(fg_frames, bg_frames, mask_frames)):
            save_rgb(sample_root / "fg_bg_frames" / f"{frame_idx:05d}.png", fg)
            save_rgb(sample_root / "bg_frames" / f"{frame_idx:05d}.png", bg)
            save_mask(sample_root / "masks" / f"{frame_idx:05d}.png", mask)
        strip = image_strip(
            [fg_frames[0], bg_frames[0], mask_frames[0], fg_frames[len(fg_frames) // 2], bg_frames[len(bg_frames) // 2], mask_frames[len(mask_frames) // 2]],
            ["fg_bg_0", "bg_0", "mask_0", "fg_bg_mid", "bg_mid", "mask_mid"],
        )
        evidence_path = evidence_root / f"{idx:03d}_{sample_id}.jpg"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(evidence_path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "scene_group": row["scene_group"],
                "source_type": row["source_type"],
                "mask_bucket": row["mask_bucket"],
                "mask_area_mean": float(row["mask_area_mean"]),
                "masked_absdiff_mean": float(row["masked_absdiff_mean"]),
                "fg_bg_video_path": row["fg_bg_path"],
                "bg_video_path": row["bg_path"],
                "mask_video_path": row["mask_path"],
                "condition_frame_dir": str(sample_root / "fg_bg_frames"),
                "winner_frame_dir": str(sample_root / "bg_frames"),
                "mask_frame_dir": str(sample_root / "masks"),
                "num_frames": args.num_frames,
                "width": args.width,
                "height": args.height,
                "candidate_seeds": seeds,
                "selection_source": str(audit_csv),
                "selection_rule": "deterministic_balanced_source_type_and_mask_bucket",
                "evidence_sheet": str(evidence_path),
            }
        )

    manifest_path = output_root / "minimax_micro_source32.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "status": "MINIMAX_MICRO_SOURCE32_MATERIALIZED",
        "num_sources": len(manifest_rows),
        "num_frames": args.num_frames,
        "seed_rule": seeds,
        "source_types": {k: sum(1 for row in manifest_rows if row["source_type"] == k) for k in sorted({row["source_type"] for row in manifest_rows})},
        "mask_buckets": {k: sum(1 for row in manifest_rows if row["mask_bucket"] == k) for k in sorted({row["mask_bucket"] for row in manifest_rows})},
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    (output_root / "minimax_micro_source32_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
