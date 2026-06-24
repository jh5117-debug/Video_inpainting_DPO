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


AREA_BUCKETS = {
    "small": (0.08, 0.14),
    "medium": (0.18, 0.27),
    "large": (0.28, 0.36),
}

MOTION_BUCKETS = {
    "low": (0.02, 0.06),
    "medium": (0.08, 0.16),
    "high": (0.18, 0.28),
}

DEFORMATION_BUCKETS = {
    "slow": (0.02, 0.06, 0.0, 12.0),
    "moderate": (0.06, 0.12, 12.0, 35.0),
}

MASK_PROFILES = {
    "irregular_freeform",
    "object_like_polygon",
    "soft_blob",
    "edge_touch_freeform",
    "ellipse_circle_subset",
    "thin_structure_freeform",
}


def choose_area_ratio(rng: random.Random, sample_id: str, area_bucket: str | None = None) -> tuple[str, float]:
    if area_bucket in AREA_BUCKETS:
        lo, hi = AREA_BUCKETS[str(area_bucket)]
        return str(area_bucket), rng.uniform(lo, hi)
    names = list(AREA_BUCKETS)
    name = names[int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:2], 16) % len(names)]
    lo, hi = AREA_BUCKETS[name]
    return name, rng.uniform(lo, hi)


def _unit_polygon(n: int, rng: random.Random, jitter: float = 0.2) -> np.ndarray:
    angles = np.linspace(0, math.tau, n, endpoint=False)
    rng.shuffle(angles)
    angles = np.sort(angles)
    pts = []
    for angle in angles:
        radius = 1.0 + rng.uniform(-jitter, jitter)
        pts.append([math.cos(angle) * radius, math.sin(angle) * radius])
    return np.asarray(pts, dtype=np.float32)


def _transform_points(points: np.ndarray, cx: float, cy: float, rx: float, ry: float, angle_deg: float) -> np.ndarray:
    theta = math.radians(angle_deg)
    rot = np.asarray([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=np.float32)
    scaled = points * np.asarray([rx, ry], dtype=np.float32)
    transformed = scaled @ rot.T + np.asarray([cx, cy], dtype=np.float32)
    return np.round(transformed).astype(np.int32)


def _draw_polygon(mask: np.ndarray, points: np.ndarray) -> None:
    cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 255)


def _draw_irregular(mask: np.ndarray, cx: float, cy: float, rx: float, ry: float, angle: float, base: np.ndarray) -> None:
    pts = _transform_points(base, cx, cy, rx, ry, angle)
    _draw_polygon(mask, pts)
    kernel = np.ones((5, 5), np.uint8)
    mask[:] = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _draw_soft_blob(mask: np.ndarray, cx: float, cy: float, rx: float, ry: float, angle: float, base: np.ndarray) -> None:
    pts = _transform_points(base, cx, cy, rx, ry, angle)
    _draw_polygon(mask, pts)
    k = max(7, int(min(mask.shape[:2]) * 0.035) | 1)
    blurred = cv2.GaussianBlur(mask, (k, k), 0)
    mask[:] = (blurred > 80).astype(np.uint8) * 255


def _draw_thin(mask: np.ndarray, cx: float, cy: float, rx: float, ry: float, angle: float, base: np.ndarray) -> None:
    pts = _transform_points(base, cx, cy, max(rx, ry) * 1.15, max(4.0, min(rx, ry) * 0.32), angle)
    cv2.polylines(mask, [pts.reshape(-1, 1, 2)], False, 255, thickness=max(5, int(min(rx, ry) * 0.32)))
    cv2.dilate(mask, np.ones((3, 3), np.uint8), dst=mask)


def _draw_profile(mask: np.ndarray, profile: str, cx: float, cy: float, rx: float, ry: float, angle: float, base: np.ndarray) -> None:
    if profile == "ellipse_circle_subset":
        cv2.ellipse(mask, (int(cx), int(cy)), (max(4, int(rx)), max(4, int(ry))), angle, 0, 360, 255, thickness=-1)
    elif profile == "soft_blob":
        _draw_soft_blob(mask, cx, cy, rx, ry, angle, base)
    elif profile == "thin_structure_freeform":
        _draw_thin(mask, cx, cy, rx, ry, angle, base)
    elif profile in {"irregular_freeform", "edge_touch_freeform", "object_like_polygon"}:
        _draw_irregular(mask, cx, cy, rx, ry, angle, base)
    else:
        raise ValueError(f"unknown mask_profile: {profile}")


def _edge_touch(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def _mask_compactness(mask: np.ndarray) -> float:
    area = float((mask > 0).sum())
    if area <= 0:
        return 0.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    if perimeter <= 0:
        return 0.0
    return float(4.0 * math.pi * area / (perimeter * perimeter))


def moving_mask_sequence(
    *,
    sample_id: str,
    num_frames: int,
    height: int,
    width: int,
    seed: int,
    first_frame_gt: bool,
    mask_profile: str = "ellipse_circle_subset",
    area_bucket: str | None = None,
    motion_bucket: str = "medium",
    deformation_bucket: str = "moderate",
    edge_touch_target: bool = False,
) -> tuple[list[np.ndarray], dict]:
    if num_frames != 49:
        raise ValueError(f"VideoPainter formal masks require 49 frames, got {num_frames}")
    if mask_profile not in MASK_PROFILES:
        raise ValueError(f"unknown mask_profile: {mask_profile}")
    rng = random.Random(stable_seed(sample_id, seed))
    area_bucket, target_area = choose_area_ratio(rng, sample_id, area_bucket)
    if mask_profile == "thin_structure_freeform":
        aspect = rng.uniform(2.2, 4.0)
    elif mask_profile == "object_like_polygon":
        aspect = rng.uniform(0.75, 1.35)
    else:
        aspect = rng.uniform(0.65, 1.75)
    shape_h = int(max(16, math.sqrt(target_area * height * width / aspect)))
    shape_w = int(max(16, shape_h * aspect))
    shape_w = min(shape_w, max(16, width - 2))
    shape_h = min(shape_h, max(16, height - 2))
    margin_x = max(shape_w // 2 + 2, 4)
    margin_y = max(shape_h // 2 + 2, 4)
    if edge_touch_target or mask_profile == "edge_touch_freeform":
        side = rng.choice(["left", "right", "top", "bottom"])
        if side == "left":
            x0 = max(1, margin_x // 3)
            y0 = rng.randint(margin_y, max(margin_y, height - margin_y))
        elif side == "right":
            x0 = width - max(2, margin_x // 3)
            y0 = rng.randint(margin_y, max(margin_y, height - margin_y))
        elif side == "top":
            x0 = rng.randint(margin_x, max(margin_x, width - margin_x))
            y0 = max(1, margin_y // 3)
        else:
            x0 = rng.randint(margin_x, max(margin_x, width - margin_x))
            y0 = height - max(2, margin_y // 3)
    else:
        x0 = rng.randint(margin_x, max(margin_x, width - margin_x))
        y0 = rng.randint(margin_y, max(margin_y, height - margin_y))
    motion_lo, motion_hi = MOTION_BUCKETS.get(motion_bucket, MOTION_BUCKETS["medium"])
    motion_frac = rng.uniform(motion_lo, motion_hi)
    direction = rng.uniform(0, math.tau)
    dx = math.cos(direction) * motion_frac * width
    dy = math.sin(direction) * motion_frac * height
    angle0 = rng.uniform(0, 180)
    scale_lo, scale_hi, angle_lo, angle_hi = DEFORMATION_BUCKETS.get(deformation_bucket, DEFORMATION_BUCKETS["moderate"])
    scale_amp = rng.uniform(scale_lo, scale_hi)
    angle_delta = rng.choice([-1.0, 1.0]) * rng.uniform(angle_lo, angle_hi)
    if mask_profile == "thin_structure_freeform":
        base = np.asarray([[-1.0, -0.15], [-0.35, 0.1], [0.2, -0.08], [1.0, 0.15]], dtype=np.float32)
    else:
        n_points = 7 if mask_profile == "object_like_polygon" else 12
        jitter = 0.12 if mask_profile == "object_like_polygon" else 0.35
        base = _unit_polygon(n_points, rng, jitter=jitter)
    masks: list[np.ndarray] = []
    centers: list[list[float]] = []
    area_curve: list[float] = []
    compactness_curve: list[float] = []
    edge_touch_frames = 0
    for t in range(num_frames):
        if t == 0 and first_frame_gt:
            mask = np.zeros((height, width), dtype=np.uint8)
            centers.append([float(x0), float(y0)])
            area_curve.append(0.0)
            compactness_curve.append(0.0)
            masks.append(mask)
            continue
        phase = t / max(1, num_frames - 1)
        wobble = math.sin(phase * math.pi * 2.0)
        if edge_touch_target or mask_profile == "edge_touch_freeform":
            cx = float(np.clip(x0 + dx * phase + wobble * 0.04 * width, 1, width - 2))
            cy = float(np.clip(y0 + dy * phase - wobble * 0.03 * height, 1, height - 2))
        else:
            cx = float(np.clip(x0 + dx * phase + wobble * 0.04 * width, margin_x, width - margin_x))
            cy = float(np.clip(y0 + dy * phase - wobble * 0.03 * height, margin_y, height - margin_y))
        scale = 1.0 + scale_amp * math.sin(phase * math.pi)
        rx = max(4.0, shape_w * scale / 2.0)
        ry = max(4.0, shape_h * scale / 2.0)
        angle = angle0 + angle_delta * phase
        mask = np.zeros((height, width), dtype=np.uint8)
        _draw_profile(mask, mask_profile, cx, cy, rx, ry, angle, base)
        centers.append([float(cx), float(cy)])
        area_curve.append(float((mask > 0).mean()))
        compactness_curve.append(_mask_compactness(mask))
        edge_touch_frames += int(_edge_touch(mask))
        masks.append(mask)
    meta = {
        "mask_profile": mask_profile,
        "area_bucket": area_bucket,
        "motion_bucket": motion_bucket,
        "deformation_bucket": deformation_bucket,
        "edge_touch_target": edge_touch_target,
        "target_area_ratio": target_area,
        "area_mean": float(np.mean(area_curve)),
        "area_min": float(np.min(area_curve)),
        "area_max": float(np.max(area_curve)),
        "compactness_mean": float(np.mean(compactness_curve)),
        "edge_touch_frames": int(edge_touch_frames),
        "centroid_start": centers[0],
        "centroid_end": centers[-1],
        "centroid_motion_px": float(np.linalg.norm(np.array(centers[-1]) - np.array(centers[0]))),
        "centroid_motion_fraction": float(np.linalg.norm(np.array(centers[-1]) - np.array(centers[0])) / max(height, width)),
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
                mask_profile=str(row.get("mask_profile", "ellipse_circle_subset")),
                area_bucket=row.get("area_bucket"),
                motion_bucket=str(row.get("motion_bucket", "medium")),
                deformation_bucket=str(row.get("deformation_bucket", "moderate")),
                edge_touch_target=bool(row.get("edge_touch_target", False)),
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
                "mask_profile": meta["mask_profile"],
                "area_bucket": meta["area_bucket"],
                "motion_bucket": meta["motion_bucket"],
                "deformation_bucket": meta["deformation_bucket"],
                "edge_touch_target": meta["edge_touch_target"],
                "first_frame_sum": int(masks[0].sum()),
                "area_mean": meta["area_mean"],
                "area_min": meta["area_min"],
                "area_max": meta["area_max"],
                "compactness_mean": meta["compactness_mean"],
                "edge_touch_frames": meta["edge_touch_frames"],
                "centroid_motion_px": meta["centroid_motion_px"],
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001
            status = {
                "sample_id": sample_id,
                "status": "FAILED",
                "mask_profile": row.get("mask_profile", ""),
                "area_bucket": row.get("area_bucket", ""),
                "motion_bucket": row.get("motion_bucket", ""),
                "deformation_bucket": row.get("deformation_bucket", ""),
                "edge_touch_target": row.get("edge_touch_target", ""),
                "first_frame_sum": "",
                "area_mean": "",
                "area_min": "",
                "area_max": "",
                "compactness_mean": "",
                "edge_touch_frames": "",
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
