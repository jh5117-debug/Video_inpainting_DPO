#!/usr/bin/env python3
"""Generate one deterministic 49-frame moving BR mask per materialized source."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def stable_seed(sample_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{sample_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def choose_area_ratio(rng: random.Random, sample_id: str) -> tuple[str, float]:
    buckets = [
        ("small", 0.06, 0.11),
        ("medium", 0.12, 0.20),
        ("large", 0.21, 0.32),
    ]
    bucket = buckets[int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:2], 16) % len(buckets)]
    return bucket[0], rng.uniform(bucket[1], bucket[2])


def moving_mask_sequence(
    *,
    sample_id: str,
    num_frames: int,
    height: int,
    width: int,
    seed: int,
    first_frame_gt: bool,
) -> tuple[list[np.ndarray], dict]:
    if num_frames != 49:
        raise ValueError(f"VideoPainter formal masks require 49 frames, got {num_frames}")
    rng = random.Random(stable_seed(sample_id, seed))
    area_bucket, target_area = choose_area_ratio(rng, sample_id)
    aspect = rng.uniform(0.65, 1.55)
    ellipse_h = int(max(16, math.sqrt(target_area * height * width / aspect)))
    ellipse_w = int(max(16, ellipse_h * aspect))
    ellipse_w = min(ellipse_w, max(16, width - 2))
    ellipse_h = min(ellipse_h, max(16, height - 2))
    margin_x = max(ellipse_w // 2 + 2, 4)
    margin_y = max(ellipse_h // 2 + 2, 4)
    x0 = rng.randint(margin_x, max(margin_x, width - margin_x))
    y0 = rng.randint(margin_y, max(margin_y, height - margin_y))
    dx = rng.uniform(-0.28, 0.28) * width
    dy = rng.uniform(-0.22, 0.22) * height
    angle0 = rng.uniform(0, 180)
    angle_delta = rng.uniform(-35, 35)
    masks: list[np.ndarray] = []
    centers: list[list[float]] = []
    area_curve: list[float] = []
    for t in range(num_frames):
        if t == 0 and first_frame_gt:
            mask = np.zeros((height, width), dtype=np.uint8)
            centers.append([float(x0), float(y0)])
            area_curve.append(0.0)
            masks.append(mask)
            continue
        phase = t / max(1, num_frames - 1)
        wobble = math.sin(phase * math.pi * 2.0)
        cx = int(np.clip(x0 + dx * phase + wobble * 0.04 * width, margin_x, width - margin_x))
        cy = int(np.clip(y0 + dy * phase - wobble * 0.03 * height, margin_y, height - margin_y))
        scale = 1.0 + 0.08 * math.sin(phase * math.pi)
        axes = (max(4, int(ellipse_w * scale / 2)), max(4, int(ellipse_h * scale / 2)))
        angle = angle0 + angle_delta * phase
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, thickness=-1)
        centers.append([float(cx), float(cy)])
        area_curve.append(float((mask > 0).mean()))
        masks.append(mask)
    meta = {
        "area_bucket": area_bucket,
        "target_area_ratio": target_area,
        "area_mean": float(np.mean(area_curve)),
        "area_min": float(np.min(area_curve)),
        "area_max": float(np.max(area_curve)),
        "centroid_start": centers[0],
        "centroid_end": centers[-1],
        "centroid_motion_px": float(np.linalg.norm(np.array(centers[-1]) - np.array(centers[0]))),
        "first_frame_gt": first_frame_gt,
    }
    return masks, meta


def infer_size(row: dict) -> tuple[int, int]:
    frame_paths = row.get("frame_paths") or []
    if not frame_paths:
        raise ValueError(f"row has no frame_paths: {row.get('sample_id')}")
    img = cv2.imread(frame_paths[0], cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read frame: {frame_paths[0]}")
    return int(img.shape[0]), int(img.shape[1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--status-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--first-frame-gt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = read_jsonl(args.materialized_manifest)
    if args.limit > 0:
        rows = rows[: args.limit]
    output_rows: list[dict] = []
    status_rows: list[dict] = []
    for row in rows:
        sample_id = row["sample_id"]
        try:
            frame_paths = row.get("frame_paths") or []
            if len(frame_paths) != 49:
                raise ValueError(f"expected 49 materialized frames, got {len(frame_paths)}")
            height, width = infer_size(row)
            masks, meta = moving_mask_sequence(
                sample_id=sample_id,
                num_frames=49,
                height=height,
                width=width,
                seed=args.seed,
                first_frame_gt=args.first_frame_gt,
            )
            mask_dir = args.output_root / sample_id / "masks"
            if mask_dir.exists():
                shutil.rmtree(mask_dir)
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask_hashes: list[str] = []
            for idx, mask in enumerate(masks):
                path = mask_dir / f"{idx:05d}.png"
                cv2.imwrite(str(path), mask)
                mask_hashes.append(sha256_file(path))
            out_row = dict(row)
            out_row.update(
                {
                    "status": "FORMAL_49F_MASK_READY",
                    "mask_dir": str(mask_dir),
                    "mask_hashes": mask_hashes,
                    "mask_generation": "exp26_moving_br_mask_v1",
                    "mask_generator_seed": args.seed,
                    "first_frame_gt": args.first_frame_gt,
                    "condition_definition": "winner * (1 - generated_moving_br_mask)",
                    "winner_role": "BG",
                    "loser_role": "VideoPainter self-model output pending",
                    "mask_meta": meta,
                }
            )
            output_rows.append(out_row)
            status = {
                "sample_id": sample_id,
                "status": "OK",
                "first_frame_sum": int(masks[0].sum()),
                "area_mean": meta["area_mean"],
                "centroid_motion_px": meta["centroid_motion_px"],
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001
            status = {
                "sample_id": sample_id,
                "status": "FAILED",
                "first_frame_sum": "",
                "area_mean": "",
                "centroid_motion_px": "",
                "error": repr(exc),
            }
        status_rows.append(status)

    write_jsonl(args.output_manifest, output_rows)
    args.status_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.status_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(status_rows[0].keys()) if status_rows else ["sample_id", "status"])
        writer.writeheader()
        writer.writerows(status_rows)
    print(json.dumps({"ok": len(output_rows), "failed": len(status_rows) - len(output_rows), "manifest": str(args.output_manifest)}, indent=2))
    return 0 if output_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
