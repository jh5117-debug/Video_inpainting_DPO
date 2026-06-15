#!/usr/bin/env python3
"""Build four-column visual evidence from existing frame-wise eval outputs.

This utility does not run inference and does not recompute metrics. It consumes
saved hard-composited frame-wise eval folders and a case CSV, then writes:

- side-by-side videos
- frame contact sheets
- selected frame panels
- a merged manifest with per-video metric deltas when available

Expected case CSV columns:

dataset,video,video_root,mask_root,base_eval_root,exp_eval_root,reason,category
"""

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
    cv2.rectangle(out, (0, 0), (width, 28), (0, 0, 0), -1)
    cv2.putText(out, text, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
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


def read_metric_rows(root: Path) -> dict[str, dict[str, str]]:
    csv_path = root / "metrics" / "per_video_metrics.csv"
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return {row.get("video", ""): row for row in csv.DictReader(handle)}


def metric_float(row: dict[str, str], key: str) -> float | None:
    try:
        value = row.get(key, "")
        return None if value == "" else float(value)
    except Exception:
        return None


def format_delta(exp_row: dict[str, str], base_row: dict[str, str], key: str) -> str:
    exp_value = metric_float(exp_row, key)
    base_value = metric_float(base_row, key)
    if exp_value is None or base_value is None:
        return ""
    return f"{exp_value - base_value:.6f}"


def build_case(row: dict[str, str], output_root: Path, video_length: int, frame_indices: list[int]) -> dict[str, str]:
    dataset = row["dataset"]
    video = row["video"]
    video_root = Path(row["video_root"])
    mask_root = Path(row["mask_root"])
    base_eval_root = Path(row["base_eval_root"])
    exp_eval_root = Path(row["exp_eval_root"])

    gt = load_frames(video_root / video, video_length)
    masks = load_frames(mask_root / video, video_length)
    base = load_frames(base_eval_root / video / "diffueraser_comp_frames", video_length)
    exp = load_frames(exp_eval_root / video / "diffueraser_comp_frames", video_length)
    count = min(len(gt), len(masks), len(base), len(exp), video_length)
    if count == 0:
        raise RuntimeError(f"empty case: {dataset}/{video}")
    gt, masks, base, exp = gt[:count], masks[:count], base[:count], exp[:count]

    safe_name = f"{dataset}__{video}".replace("/", "_")
    side_dir = output_root / "videos"
    sheet_dir = output_root / "contact_sheets"
    frame_dir = output_root / "frame_by_frame" / safe_name
    side_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    panels: list[np.ndarray] = []
    sheet_panels: list[np.ndarray] = []
    valid_indices = sorted({min(max(index, 0), count - 1) for index in frame_indices})
    for index, (gt_frame, mask_frame, base_frame, exp_frame) in enumerate(zip(gt, masks, base, exp)):
        base_frame = resize_like(base_frame, gt_frame)
        exp_frame = resize_like(exp_frame, gt_frame)
        panel = np.concatenate(
            [
                add_label(gt_frame, "GT"),
                add_label(mask_overlay(gt_frame, mask_frame), "mask overlay"),
                add_label(base_frame, "SFT-48000"),
                add_label(exp_frame, "Exp11 outer b0.75 S2"),
            ],
            axis=1,
        )
        panels.append(panel)
        if index in valid_indices:
            sheet_panels.append(panel)
            cv2.imwrite(str(frame_dir / f"{index:03d}.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    side_path = side_dir / f"{safe_name}.mp4"
    sheet_path = sheet_dir / f"{safe_name}_frames.jpg"
    write_video(side_path, panels)
    sheet = np.concatenate(sheet_panels, axis=0)
    cv2.imwrite(str(sheet_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    base_metrics = read_metric_rows(base_eval_root).get(video, {})
    exp_metrics = read_metric_rows(exp_eval_root).get(video, {})
    out = dict(row)
    out.update(
        {
            "side_by_side": str(side_path),
            "contact_sheet": str(sheet_path),
            "frame_dir": str(frame_dir),
            "base_whole_video_psnr": base_metrics.get("whole_video_psnr", ""),
            "exp_whole_video_psnr": exp_metrics.get("whole_video_psnr", ""),
            "delta_whole_video_psnr": format_delta(exp_metrics, base_metrics, "whole_video_psnr"),
            "base_mask_region_psnr": base_metrics.get("mask_region_psnr", ""),
            "exp_mask_region_psnr": exp_metrics.get("mask_region_psnr", ""),
            "delta_mask_region_psnr": format_delta(exp_metrics, base_metrics, "mask_region_psnr"),
            "base_lpips": base_metrics.get("whole_video_lpips", ""),
            "exp_lpips": exp_metrics.get("whole_video_lpips", ""),
            "delta_lpips": format_delta(exp_metrics, base_metrics, "whole_video_lpips"),
            "base_tc": base_metrics.get("tc", ""),
            "exp_tc": exp_metrics.get("tc", ""),
            "delta_tc": format_delta(exp_metrics, base_metrics, "tc"),
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases_csv", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--video_length", type=int, default=24)
    parser.add_argument("--frame_indices", default="0,6,12,18,23")
    args = parser.parse_args()

    cases_csv = Path(args.cases_csv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    frame_indices = [int(item) for item in args.frame_indices.split(",") if item.strip()]

    with cases_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"no rows in {cases_csv}")

    outputs = [build_case(row, output_root, args.video_length, frame_indices) for row in rows]
    keys: list[str] = []
    for row in outputs:
        for key in row:
            if key not in keys:
                keys.append(key)
    manifest = output_root / "final_20_visual_cases_for_paper.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(outputs)

    lines = [
        "# Final Visual Cases For Paper",
        "",
        "| Dataset | Video | Category | Delta PSNR | Delta Mask PSNR | Delta LPIPS | Reason |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in outputs:
        lines.append(
            "| {dataset} | {video} | {category} | {delta_whole_video_psnr} | "
            "{delta_mask_region_psnr} | {delta_lpips} | {reason} |".format(**row)
        )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[final-cases] wrote {manifest}")


if __name__ == "__main__":
    main()
