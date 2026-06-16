#!/usr/bin/env python3
"""Create two-row OR visual grids from raw method outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(path: Path) -> List[Path]:
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def read_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size, Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def read_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("L").resize(size, Image.NEAREST)
    return (np.asarray(img, dtype=np.uint8) > 0).astype(np.uint8)


def overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = frame.astype(np.float32)
    red = np.zeros_like(out)
    red[..., 0] = 255
    alpha = (mask > 0).astype(np.float32)[..., None] * 0.45
    return np.clip(out * (1 - alpha) + red * alpha, 0, 255).astype(np.uint8)


def label(frame: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, img.width, 30], fill=(0, 0, 0))
    draw.text((6, 5), text, fill=(255, 255, 255), font=font)
    return np.asarray(img, dtype=np.uint8)


def placeholder(size: tuple[int, int], text: str) -> np.ndarray:
    img = Image.new("RGB", size, (35, 35, 35))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, size[1] // 2 - 10), text, fill=(255, 255, 255), font=font)
    return np.asarray(img, dtype=np.uint8)


def write_mp4(frames: Sequence[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def make_grid_for_video(row: dict, methods: List[str], output_root: Path, out_dir: Path, fps: int, max_frames: int, tile_width: int) -> dict:
    name = row["video_name"]
    frame_files = image_files(Path(row["frame_dir"]))
    mask_files = image_files(Path(row["mask_dir"]))
    n = min(len(frame_files), len(mask_files))
    if max_frames > 0:
        n = min(n, max_frames)
    if n == 0:
        return {"video_name": name, "status": "failed", "issue": "no frames"}
    base = read_rgb(frame_files[0])
    scale = min(1.0, float(tile_width) / float(base.shape[1])) if tile_width > 0 else 1.0
    size = (max(1, int(round(base.shape[1] * scale))), max(1, int(round(base.shape[0] * scale))))
    frames = []
    sample_indices = np.linspace(0, n - 1, num=min(6, n), dtype=int).tolist()
    contact_rows = []
    pred_by_method = {method: image_files(output_root / method / "raw_frames" / name) for method in methods}
    for idx in range(n):
        gt = read_rgb(frame_files[idx], size)
        mask = read_mask(mask_files[idx], size)
        method_tiles = []
        for method in methods:
            pred_files = pred_by_method[method]
            if idx < len(pred_files):
                tile = read_rgb(pred_files[idx], size)
            else:
                tile = placeholder(size, "BLOCKED / N.A.")
            method_tiles.append(label(tile, method))
        top = [label(gt, "Input"), label(overlay(gt, mask), "Mask")] + method_tiles[:4]
        bottom = method_tiles[4:]
        while len(bottom) < len(top):
            bottom.append(placeholder(size, ""))
        grid = np.vstack([np.hstack(top), np.hstack(bottom)])
        frames.append(grid)
        if idx in sample_indices:
            contact_rows.append(grid)
    write_mp4(frames, out_dir / "videos" / f"{name}.mp4", fps)
    if contact_rows:
        (out_dir / "contact_sheets").mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.vstack(contact_rows)).save(out_dir / "contact_sheets" / f"{name}.jpg")
    return {"video_name": name, "status": "ok", "issue": "", "video": str(out_dir / "videos" / f"{name}.mp4")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--visual_dir", required=True)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max_frames", type=int, default=24)
    parser.add_argument("--tile_width", type=int, default=320)
    args = parser.parse_args()

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    out_dir = Path(args.visual_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with Path(args.manifest).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    manifest = []
    for row in rows:
        manifest.append(make_grid_for_video(row, methods, Path(args.output_root), out_dir, args.fps, args.max_frames, args.tile_width))
    with (out_dir / "visual_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_name", "status", "issue", "video"])
        writer.writeheader()
        writer.writerows(manifest)
    print(f"[visual] wrote {out_dir}")


if __name__ == "__main__":
    main()
