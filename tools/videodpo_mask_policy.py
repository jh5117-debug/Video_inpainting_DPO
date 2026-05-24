#!/usr/bin/env python3
"""VideoDPO canonical partial-mask policy utilities."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml


DEFAULT_POLICY = Path("configs/generation/videodpo_partialmask_policy_v1_medium_hard_k4.yaml")


@dataclass
class MaskPolicy:
    policy_name: str
    num_masks_per_video: int
    seed: int
    height: int
    width: int
    num_frames: int
    mask_area_min: float
    mask_area_max: float
    mask_margin_ratio: float
    mask_static_prob: float
    mask_speed_min: float
    mask_speed_max: float
    mask_center_jitter_ratio: float
    mask_motion_box_ratio: float
    mask_dilation_iter: int
    mask_shape: str
    mask_location: str
    mask_motion: str


def load_policy(path: str | Path = DEFAULT_POLICY) -> MaskPolicy:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    canonical = data.get("canonical", {})
    return MaskPolicy(
        policy_name=str(data["policy_name"]),
        num_masks_per_video=int(data.get("num_masks_per_video", 4)),
        seed=int(data.get("seed", 20260524)),
        height=int(canonical.get("height", 320)),
        width=int(canonical.get("width", 512)),
        num_frames=int(canonical.get("num_frames", 16)),
        mask_area_min=float(data.get("mask_area_min", 0.20)),
        mask_area_max=float(data.get("mask_area_max", 0.30)),
        mask_margin_ratio=float(data.get("mask_margin_ratio", 0.10)),
        mask_static_prob=float(data.get("mask_static_prob", 0.50)),
        mask_speed_min=float(data.get("mask_speed_min", 0.50)),
        mask_speed_max=float(data.get("mask_speed_max", 1.50)),
        mask_center_jitter_ratio=float(data.get("mask_center_jitter_ratio", 0.04)),
        mask_motion_box_ratio=float(data.get("mask_motion_box_ratio", 0.16)),
        mask_dilation_iter=int(data.get("mask_dilation_iter", 0)),
        mask_shape=str(data.get("mask_shape", "irregular_polygon")),
        mask_location=str(data.get("mask_location", "interior_constrained")),
        mask_motion=str(data.get("mask_motion", "two_static_two_slow_or_equivalent_random_with_static_prob")),
    )


def _polygon_unit(rng: random.Random, vertices: int | None = None) -> np.ndarray:
    n = vertices or rng.randint(8, 14)
    angles = sorted(rng.random() * 2.0 * math.pi for _ in range(n))
    pts = []
    for theta in angles:
        radius = rng.uniform(0.58, 1.0)
        pts.append([math.cos(theta) * radius, math.sin(theta) * radius])
    return np.asarray(pts, dtype=np.float32)


def _rasterize(points: np.ndarray, width: int, height: int, dilation_iter: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.round(points).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    if dilation_iter > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilation_iter)
    return mask


def _bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _bbox_ratio(mask: np.ndarray) -> list[float]:
    x0, y0, x1, y1 = _bbox(mask)
    h, w = mask.shape[:2]
    return [float((x1 - x0) / w), float((y1 - y0) / h)]


def _bbox_center_ratio(mask: np.ndarray) -> list[float]:
    x0, y0, x1, y1 = _bbox(mask)
    h, w = mask.shape[:2]
    return [float(((x0 + x1) / 2.0) / w), float(((y0 + y1) / 2.0) / h)]


def _bbox_margin_ratio(mask: np.ndarray) -> list[float]:
    x0, y0, x1, y1 = _bbox(mask)
    h, w = mask.shape[:2]
    return [float(x0 / w), float(y0 / h), float((w - x1) / w), float((h - y1) / h)]


def _scale_polygon_to_area(points: np.ndarray, width: int, height: int, target_area: float, dilation_iter: int) -> np.ndarray:
    lo, hi = 1.0, max(width, height)
    best = points
    for _ in range(32):
        scale = (lo + hi) / 2.0
        candidate = points * scale
        mask = _rasterize(candidate + np.asarray([width / 2.0, height / 2.0], dtype=np.float32), width, height, dilation_iter)
        area = float((mask > 0).mean())
        best = candidate
        if area < target_area:
            lo = scale
        else:
            hi = scale
    return best


def _center_bounds(points: np.ndarray, width: int, height: int, margin_ratio: float, pad: int) -> tuple[float, float, float, float]:
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    mx = width * margin_ratio + pad
    my = height * margin_ratio + pad
    cx_min = mx - min_x
    cy_min = my - min_y
    cx_max = width - mx - max_x
    cy_max = height - my - max_y
    if cx_min > cx_max:
        cx_min = cx_max = width / 2.0
    if cy_min > cy_max:
        cy_min = cy_max = height / 2.0
    return cx_min, cy_min, cx_max, cy_max


def _motion_type_for_index(policy: MaskPolicy, mask_index: int, rng: random.Random) -> str:
    if policy.num_masks_per_video == 4 and "two_static_two_slow" in policy.mask_motion:
        return "static" if mask_index < 2 else "slow"
    return "static" if rng.random() < policy.mask_static_prob else "slow"


def generate_mask_sequence(policy: MaskPolicy, sample_id: str, mask_index: int, seed: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    rng = random.Random(seed)
    target_area = rng.uniform(policy.mask_area_min, policy.mask_area_max)
    base = _polygon_unit(rng)
    scaled = _scale_polygon_to_area(base, policy.width, policy.height, target_area, policy.mask_dilation_iter)
    scaled = scaled - scaled.mean(axis=0, keepdims=True)

    pad = max(0, policy.mask_dilation_iter + 2)
    cx_min, cy_min, cx_max, cy_max = _center_bounds(
        scaled,
        policy.width,
        policy.height,
        policy.mask_margin_ratio,
        pad,
    )
    motion_half_x = max(1.0, policy.width * policy.mask_motion_box_ratio / 2.0)
    motion_half_y = max(1.0, policy.height * policy.mask_motion_box_ratio / 2.0)
    motion_cx_min = max(cx_min, policy.width / 2.0 - motion_half_x)
    motion_cx_max = min(cx_max, policy.width / 2.0 + motion_half_x)
    motion_cy_min = max(cy_min, policy.height / 2.0 - motion_half_y)
    motion_cy_max = min(cy_max, policy.height / 2.0 + motion_half_y)
    if motion_cx_min > motion_cx_max:
        motion_cx_min = motion_cx_max = min(max(policy.width / 2.0, cx_min), cx_max)
    if motion_cy_min > motion_cy_max:
        motion_cy_min = motion_cy_max = min(max(policy.height / 2.0, cy_min), cy_max)

    jitter_x = policy.mask_center_jitter_ratio * policy.width
    jitter_y = policy.mask_center_jitter_ratio * policy.height
    cx = min(max(policy.width / 2.0 + rng.uniform(-jitter_x, jitter_x), motion_cx_min), motion_cx_max)
    cy = min(max(policy.height / 2.0 + rng.uniform(-jitter_y, jitter_y), motion_cy_min), motion_cy_max)

    motion_type = _motion_type_for_index(policy, mask_index, rng)
    if motion_type == "static":
        vx = vy = 0.0
    else:
        speed = rng.uniform(policy.mask_speed_min, policy.mask_speed_max)
        theta = rng.uniform(0.0, 2.0 * math.pi)
        vx = math.cos(theta) * speed
        vy = math.sin(theta) * speed

    masks: list[np.ndarray] = []
    frame_bboxes: list[list[int]] = []
    frame_area_ratios: list[float] = []
    frame_meta: list[dict[str, Any]] = []
    initial_velocity = [float(vx), float(vy)]
    for frame_idx in range(policy.num_frames):
        mask = _rasterize(scaled + np.asarray([cx, cy], dtype=np.float32), policy.width, policy.height, policy.mask_dilation_iter)
        masks.append(mask)
        bbox = _bbox(mask)
        area = float((mask > 0).mean())
        frame_bboxes.append(bbox)
        frame_area_ratios.append(area)
        frame_meta.append(
            {
                "frame": frame_idx,
                "bbox": bbox,
                "area_ratio": round(area, 6),
                "bbox_ratio": [round(x, 6) for x in _bbox_ratio(mask)],
                "bbox_center_ratio": [round(x, 6) for x in _bbox_center_ratio(mask)],
                "bbox_margin_ratio": [round(x, 6) for x in _bbox_margin_ratio(mask)],
            }
        )

        nx = cx + vx
        ny = cy + vy
        if nx < motion_cx_min or nx > motion_cx_max:
            vx = -vx
            nx = cx + vx
        if ny < motion_cy_min or ny > motion_cy_max:
            vy = -vy
            ny = cy + vy
        cx = min(max(nx, motion_cx_min), motion_cx_max)
        cy = min(max(ny, motion_cy_min), motion_cy_max)

    all_x0 = min(b[0] for b in frame_bboxes)
    all_y0 = min(b[1] for b in frame_bboxes)
    all_x1 = max(b[2] for b in frame_bboxes)
    all_y1 = max(b[3] for b in frame_bboxes)
    union_bbox = [all_x0, all_y0, all_x1, all_y1]
    meta = {
        "mask_id": f"mask_{mask_index:03d}",
        "sample_id": sample_id,
        "mask_policy": policy.policy_name,
        "seed": seed,
        "area_ratio": float(np.mean(frame_area_ratios)),
        "area_ratio_min": policy.mask_area_min,
        "area_ratio_max": policy.mask_area_max,
        "frame_area_ratio_min": float(np.min(frame_area_ratios)),
        "frame_area_ratio_max": float(np.max(frame_area_ratios)),
        "bbox": union_bbox,
        "bbox_ratio": [float((all_x1 - all_x0) / policy.width), float((all_y1 - all_y0) / policy.height)],
        "bbox_center_ratio": [float(((all_x0 + all_x1) / 2.0) / policy.width), float(((all_y0 + all_y1) / 2.0) / policy.height)],
        "bbox_margin_ratio": [
            float(all_x0 / policy.width),
            float(all_y0 / policy.height),
            float((policy.width - all_x1) / policy.width),
            float((policy.height - all_y1) / policy.height),
        ],
        "motion_type": motion_type,
        "velocity": [round(initial_velocity[0], 6), round(initial_velocity[1], 6)],
        "motion_center_bounds": [
            float(motion_cx_min / policy.width),
            float(motion_cy_min / policy.height),
            float(motion_cx_max / policy.width),
            float(motion_cy_max / policy.height),
        ],
        "motion_box_ratio": policy.mask_motion_box_ratio,
        "static_prob": policy.mask_static_prob,
        "mask_shape": policy.mask_shape,
        "mask_location": policy.mask_location,
        "mask_motion": policy.mask_motion,
        "mask_dilation_iter": policy.mask_dilation_iter,
        "frame_level_bbox": frame_bboxes,
        "frame_level_area_ratio": frame_area_ratios,
        "frames": frame_meta,
    }
    return masks, meta


def save_mask_sequence(masks: Iterable[np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(masks):
        cv2.imwrite(str(output_dir / f"{idx:05d}.png"), mask.astype(np.uint8))


def generate_k_masks(policy: MaskPolicy, sample_id: str, output_root: Path, base_seed: int | None = None) -> list[dict[str, Any]]:
    seed0 = policy.seed if base_seed is None else int(base_seed)
    rows = []
    for mask_index in range(policy.num_masks_per_video):
        seed = seed0 + mask_index * 9973
        masks, meta = generate_mask_sequence(policy, sample_id, mask_index, seed)
        mask_dir = output_root / sample_id / meta["mask_id"] / "mask"
        save_mask_sequence(masks, mask_dir)
        meta["mask_path"] = str(mask_dir)
        rows.append(meta)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate canonical VideoDPO partial masks.")
    parser.add_argument("--policy_config", default=str(DEFAULT_POLICY))
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--sample_id", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--write_metadata_jsonl", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    policy = load_policy(args.policy_config)
    rows = generate_k_masks(policy, args.sample_id, Path(args.output_root), args.seed)
    if args.write_metadata_jsonl:
        write_jsonl(Path(args.write_metadata_jsonl), rows)
    print(json.dumps({"policy": asdict(policy), "num_masks": len(rows), "output_root": args.output_root}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
