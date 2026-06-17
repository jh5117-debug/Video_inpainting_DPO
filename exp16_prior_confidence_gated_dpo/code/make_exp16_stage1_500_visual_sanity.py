#!/usr/bin/env python3
"""Build Exp16 DAVIS10 visual sanity panels from frame-wise eval outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def list_images(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def load_frames(root: Path, limit: int) -> list[np.ndarray]:
    paths = list_images(root)[:limit]
    return [read_rgb(path) for path in paths]


def resize_like(image: np.ndarray, ref: np.ndarray, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    height, width = ref.shape[:2]
    if image.shape[:2] == (height, width):
        return image
    return cv2.resize(image, (width, height), interpolation=interpolation)


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    height, width = out.shape[:2]
    cv2.rectangle(out, (0, 0), (width, 30), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def mask_overlay(gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = resize_like(mask, gt, interpolation=cv2.INTER_NEAREST)
    if mask.ndim == 3:
        mask = mask[..., 0]
    active = mask > 0
    tint = gt.copy()
    tint[..., 0] = np.maximum(tint[..., 0], 235)
    tint[..., 1] = (tint[..., 1] * 0.35).astype(np.uint8)
    tint[..., 2] = (tint[..., 2] * 0.35).astype(np.uint8)
    out = gt.copy()
    out[active] = (0.55 * gt[active] + 0.45 * tint[active]).astype(np.uint8)
    return out


def write_video(path: Path, frames: Iterable[np.ndarray], fps: int = 12) -> None:
    frames = list(frames)
    if not frames:
        raise ValueError(f"no frames for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def read_metrics(root: Path) -> dict[str, dict[str, str]]:
    path = root / "metrics" / "per_video_metrics.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row.get("video", ""): row for row in csv.DictReader(handle)}


def metric_float(row: dict[str, str], key: str) -> float | None:
    try:
        value = row.get(key, "")
        return None if value == "" else float(value)
    except Exception:
        return None


def metric_delta(a: dict[str, str], b: dict[str, str], key: str) -> str:
    av = metric_float(a, key)
    bv = metric_float(b, key)
    if av is None or bv is None:
        return ""
    return f"{av - bv:.6f}"


def build_video(
    video: str,
    video_root: Path,
    mask_root: Path,
    sft_root: Path,
    exp11_root: Path,
    exp16_root: Path,
    output_root: Path,
    video_length: int,
    frame_indices: list[int],
) -> dict[str, str]:
    gt = load_frames(video_root / video, video_length)
    masks = load_frames(mask_root / video, video_length)
    sft = load_frames(sft_root / video / "diffueraser_comp_frames", video_length)
    exp11 = load_frames(exp11_root / video / "diffueraser_comp_frames", video_length)
    exp16 = load_frames(exp16_root / video / "diffueraser_comp_frames", video_length)
    count = min(len(gt), len(masks), len(sft), len(exp11), len(exp16), video_length)
    if count == 0:
        raise RuntimeError(f"empty visual case for {video}")
    gt, masks, sft, exp11, exp16 = gt[:count], masks[:count], sft[:count], exp11[:count], exp16[:count]

    side_dir = output_root / "side_by_side"
    sheet_dir = output_root / "contact_sheets"
    frame_dir = output_root / "frame_by_frame" / video
    side_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    panels: list[np.ndarray] = []
    sheet_panels: list[np.ndarray] = []
    valid_indices = sorted({min(max(index, 0), count - 1) for index in frame_indices})
    for idx, (gt_frame, mask_frame, sft_frame, exp11_frame, exp16_frame) in enumerate(zip(gt, masks, sft, exp11, exp16)):
        sft_frame = resize_like(sft_frame, gt_frame)
        exp11_frame = resize_like(exp11_frame, gt_frame)
        exp16_frame = resize_like(exp16_frame, gt_frame)
        panel = np.concatenate(
            [
                add_label(gt_frame, "GT"),
                add_label(mask_overlay(gt_frame, mask_frame), "mask overlay"),
                add_label(sft_frame, "SFT-48000"),
                add_label(exp11_frame, "Exp11 outer b0.75 S2"),
                add_label(exp16_frame, "Exp16 S1-500"),
            ],
            axis=1,
        )
        panels.append(panel)
        if idx in valid_indices:
            sheet_panels.append(panel)
            cv2.imwrite(str(frame_dir / f"{idx:03d}.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    side_path = side_dir / f"{video}.mp4"
    sheet_path = sheet_dir / f"{video}_frames.jpg"
    write_video(side_path, panels)
    sheet = np.concatenate(sheet_panels, axis=0)
    cv2.imwrite(str(sheet_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    sft_metrics = read_metrics(sft_root).get(video, {})
    exp11_metrics = read_metrics(exp11_root).get(video, {})
    exp16_metrics = read_metrics(exp16_root).get(video, {})
    return {
        "video": video,
        "frames": str(count),
        "side_by_side": str(side_path),
        "contact_sheet": str(sheet_path),
        "frame_dir": str(frame_dir),
        "sft_psnr": sft_metrics.get("whole_video_psnr", ""),
        "exp11_psnr": exp11_metrics.get("whole_video_psnr", ""),
        "exp16_psnr": exp16_metrics.get("whole_video_psnr", ""),
        "exp16_minus_exp11_psnr": metric_delta(exp16_metrics, exp11_metrics, "whole_video_psnr"),
        "exp16_minus_sft_psnr": metric_delta(exp16_metrics, sft_metrics, "whole_video_psnr"),
        "sft_ssim": sft_metrics.get("whole_video_ssim", ""),
        "exp11_ssim": exp11_metrics.get("whole_video_ssim", ""),
        "exp16_ssim": exp16_metrics.get("whole_video_ssim", ""),
        "exp16_minus_exp11_ssim": metric_delta(exp16_metrics, exp11_metrics, "whole_video_ssim"),
        "exp16_strict_mask_psnr": exp16_metrics.get("strict_mask_pixel_psnr", ""),
        "exp11_strict_mask_psnr": exp11_metrics.get("strict_mask_pixel_psnr", ""),
        "exp16_minus_exp11_strict_mask_psnr": metric_delta(exp16_metrics, exp11_metrics, "strict_mask_pixel_psnr"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--mask_root", required=True)
    parser.add_argument("--sft_eval_root", required=True)
    parser.add_argument("--exp11_eval_root", required=True)
    parser.add_argument("--exp16_eval_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--videos", required=True, help="Comma-separated DAVIS video names")
    parser.add_argument("--video_length", type=int, default=24)
    parser.add_argument("--frame_indices", default="0,6,12,18,23")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    videos = [item.strip() for item in args.videos.split(",") if item.strip()]
    frame_indices = [int(item) for item in args.frame_indices.split(",") if item.strip()]

    rows = [
        build_video(
            video,
            Path(args.video_root),
            Path(args.mask_root),
            Path(args.sft_eval_root),
            Path(args.exp11_eval_root),
            Path(args.exp16_eval_root),
            output_root,
            args.video_length,
            frame_indices,
        )
        for video in videos
    ]

    manifest = output_root / "pair_manifest.csv"
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Exp16 Stage1-500 DAVIS10 Visual Sanity",
        "",
        "| Video | SFT PSNR | Exp11 PSNR | Exp16 PSNR | Exp16-Exp11 PSNR | Exp16-Exp11 SSIM | Contact sheet |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {video} | {sft_psnr} | {exp11_psnr} | {exp16_psnr} | {exp16_minus_exp11_psnr} | "
            "{exp16_minus_exp11_ssim} | {contact_sheet} |".format(**row)
        )
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[exp16-visual] wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
