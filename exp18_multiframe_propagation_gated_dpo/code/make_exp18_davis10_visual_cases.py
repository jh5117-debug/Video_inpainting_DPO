#!/usr/bin/env python3
"""Build DAVIS10 visual panels for Exp18 hybrid eval outputs."""

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
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
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


def parse_method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected label=path, got {value!r}")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"Empty method label in {value!r}")
    return label, Path(path)


def metric_delta(rows: dict[str, dict[str, str]], method: str, ref: str, key: str) -> str:
    try:
        value = rows.get(method, {}).get(key, "")
        base = rows.get(ref, {}).get(key, "")
        if value == "" or base == "":
            return ""
        return f"{float(value) - float(base):.6f}"
    except Exception:
        return ""


def build_video(
    video: str,
    video_root: Path,
    mask_root: Path,
    methods: list[tuple[str, Path]],
    output_root: Path,
    video_length: int,
    frame_indices: list[int],
    reference_label: str,
) -> dict[str, str]:
    gt = load_frames(video_root / video, video_length)
    masks = load_frames(mask_root / video, video_length)
    method_frames: list[tuple[str, list[np.ndarray]]] = []
    for label, root in methods:
        method_frames.append((label, load_frames(root / video / "diffueraser_comp_frames", video_length)))

    counts = [len(gt), len(masks), video_length] + [len(frames) for _, frames in method_frames]
    count = min(counts)
    if count == 0:
        raise RuntimeError(f"empty visual case for {video}: counts={counts}")
    gt = gt[:count]
    masks = masks[:count]
    method_frames = [(label, frames[:count]) for label, frames in method_frames]

    side_dir = output_root / "side_by_side"
    sheet_dir = output_root / "contact_sheets"
    frame_dir = output_root / "frame_by_frame" / video
    side_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    panels: list[np.ndarray] = []
    sheet_panels: list[np.ndarray] = []
    valid_indices = sorted({min(max(index, 0), count - 1) for index in frame_indices})
    for idx in range(count):
        gt_frame = gt[idx]
        columns = [add_label(gt_frame, "GT"), add_label(mask_overlay(gt_frame, masks[idx]), "mask overlay")]
        for label, frames in method_frames:
            columns.append(add_label(resize_like(frames[idx], gt_frame), label))
        panel = np.concatenate(columns, axis=1)
        panels.append(panel)
        if idx in valid_indices:
            sheet_panels.append(panel)
            cv2.imwrite(str(frame_dir / f"{idx:03d}.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    side_path = side_dir / f"{video}.mp4"
    sheet_path = sheet_dir / f"{video}_frames.jpg"
    write_video(side_path, panels)
    cv2.imwrite(str(sheet_path), cv2.cvtColor(np.concatenate(sheet_panels, axis=0), cv2.COLOR_RGB2BGR))

    method_metric_rows: dict[str, dict[str, str]] = {}
    for label, root in methods:
        method_metric_rows[label] = read_metrics(root).get(video, {})

    row = {
        "video": video,
        "frames": str(count),
        "side_by_side": str(side_path),
        "contact_sheet": str(sheet_path),
        "frame_dir": str(frame_dir),
    }
    for label, _ in methods:
        metrics = method_metric_rows.get(label, {})
        safe = label.replace(" ", "_")
        row[f"{safe}_psnr"] = metrics.get("whole_video_psnr", "")
        row[f"{safe}_ssim"] = metrics.get("whole_video_ssim", "")
        row[f"{safe}_strict_mask_psnr"] = metrics.get("strict_mask_pixel_psnr", "")
        if label != reference_label:
            row[f"{safe}_minus_{reference_label}_psnr"] = metric_delta(
                method_metric_rows, label, reference_label, "whole_video_psnr"
            )
            row[f"{safe}_minus_{reference_label}_ssim"] = metric_delta(
                method_metric_rows, label, reference_label, "whole_video_ssim"
            )
            row[f"{safe}_minus_{reference_label}_strict_mask_psnr"] = metric_delta(
                method_metric_rows, label, reference_label, "strict_mask_pixel_psnr"
            )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--mask_root", required=True)
    parser.add_argument("--method", action="append", required=True, type=parse_method, help="label=eval_root")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--videos", required=True, help="Comma-separated DAVIS video names")
    parser.add_argument("--video_length", type=int, default=24)
    parser.add_argument("--frame_indices", default="0,6,12,18,23")
    parser.add_argument("--reference_label", default="Exp11")
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
            args.method,
            output_root,
            args.video_length,
            frame_indices,
            args.reference_label,
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
        "# Exp18 DAVIS10 Visual Cases",
        "",
        f"- reference_label: `{args.reference_label}`",
        "",
        "| Video | Contact sheet | Side-by-side |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['video']} | `{row['contact_sheet']}` | `{row['side_by_side']}` |")
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[exp18-visual] wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
