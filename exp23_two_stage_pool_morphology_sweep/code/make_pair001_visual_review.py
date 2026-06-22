#!/usr/bin/env python3
"""Build Exp23 pair001 DAVIS50 visual contact sheets and review seed table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def list_frames(path: Path) -> List[Path]:
    return sorted([p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if size is not None:
        image = image.resize(size, Image.BICUBIC)
    return image


def mask_overlay(gt: Image.Image, mask_path: Path) -> Image.Image:
    mask = Image.open(mask_path).convert("L").resize(gt.size, Image.NEAREST)
    gt_arr = np.array(gt).astype(np.float32)
    mask_arr = np.array(mask) > 127
    overlay = gt_arr.copy()
    overlay[mask_arr] = 0.55 * overlay[mask_arr] + 0.45 * np.array([255.0, 32.0, 32.0])
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def tile_with_label(image: Image.Image, label: str, width: int, height: int) -> Image.Image:
    tile = Image.new("RGB", (width, height + 18), "white")
    tile.paste(image.resize((width, height), Image.BICUBIC), (0, 18))
    draw = ImageDraw.Draw(tile)
    draw.text((4, 2), label, fill=(0, 0, 0))
    return tile


def make_sheet(
    video: str,
    eval_root: Path,
    gt_root: Path,
    mask_root: Path,
    out_path: Path,
    frame_ids: Sequence[int] = (0, 8, 16, 23),
) -> None:
    labels = [
        ("GT", None),
        ("Mask", None),
        ("SFT", "sft48000_baseline"),
        ("Fresh", "fresh_s2_2000"),
        ("Candidate", "candidate_s2_2000"),
    ]
    gt_frames = list_frames(gt_root / video)
    mask_frames = list_frames(mask_root / video)
    tile_w, tile_h = 216, 120
    sheet = Image.new("RGB", (tile_w * len(labels), (tile_h + 18) * len(frame_ids)), "white")
    for row_idx, frame_idx in enumerate(frame_ids):
        frame_idx = min(frame_idx, len(gt_frames) - 1)
        gt = load_rgb(gt_frames[frame_idx], (432, 240))
        mask_path = mask_frames[min(frame_idx, len(mask_frames) - 1)]
        for col_idx, (title, label) in enumerate(labels):
            if title == "GT":
                image = gt
            elif title == "Mask":
                image = mask_overlay(gt, mask_path)
            else:
                frame_path = eval_root / str(label) / video / "diffueraser_comp_frames" / f"{frame_idx:05d}.png"
                image = load_rgb(frame_path, gt.size) if frame_path.exists() else Image.new("RGB", gt.size, "gray")
            tile = tile_with_label(image, f"{title} f{frame_idx:02d}", tile_w, tile_h)
            sheet.paste(tile, (col_idx * tile_w, row_idx * (tile_h + 18)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def classify(delta_psnr: float, delta_lpips: float) -> str:
    if delta_psnr > 0.05 and delta_lpips <= 0.0005:
        return "candidate_clearly_better_metric_seed"
    if delta_psnr > 0.01 and delta_lpips <= 0.001:
        return "candidate_slightly_better_metric_seed"
    if delta_psnr < -0.05:
        return "fresh_clearly_better_metric_seed"
    if delta_psnr < -0.01:
        return "fresh_slightly_better_metric_seed"
    return "tie_metric_seed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--mask-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--report-prefix", required=True, type=Path)
    args = parser.parse_args()

    fresh_rows = {r["video"]: r for r in read_csv(args.eval_root / "fresh_s2_2000" / "metrics" / "per_video_metrics.csv")}
    cand_rows = {r["video"]: r for r in read_csv(args.eval_root / "candidate_s2_2000" / "metrics" / "per_video_metrics.csv")}
    rows: List[Dict[str, object]] = []
    for video in sorted(set(fresh_rows) & set(cand_rows)):
        fresh = fresh_rows[video]
        cand = cand_rows[video]
        delta_psnr = as_float(cand.get("whole_video_psnr")) - as_float(fresh.get("whole_video_psnr"))
        delta_lpips = as_float(cand.get("whole_video_lpips")) - as_float(fresh.get("whole_video_lpips"))
        delta_boundary = as_float(cand.get("boundary_pixel_psnr")) - as_float(fresh.get("boundary_pixel_psnr"))
        delta_ewarp = as_float(cand.get("ewarp")) - as_float(fresh.get("ewarp"))
        sheet_path = args.out_dir / f"{video}.jpg"
        make_sheet(video, args.eval_root, args.gt_root, args.mask_root, sheet_path)
        rows.append(
            {
                "video": video,
                "delta_psnr": delta_psnr,
                "delta_lpips": delta_lpips,
                "delta_boundary_psnr": delta_boundary,
                "delta_ewarp": delta_ewarp,
                "metric_seed_class": classify(delta_psnr, delta_lpips),
                "visual_review": "pending_manual_review",
                "new_artifact": "pending_manual_review",
                "sheet_path": str(sheet_path),
            }
        )
    write_csv(Path(f"{args.report_prefix}.csv"), rows)
    counts: Dict[str, int] = {}
    for row in rows:
        counts[str(row["metric_seed_class"])] = counts.get(str(row["metric_seed_class"]), 0) + 1
    lines = [
        "# Exp23 Pair001 Visual Review Seed Table",
        "",
        f"eval_root: `{args.eval_root}`",
        f"visual_root: `{args.out_dir}`",
        "",
        "Initial labels below are metric-seeded. Manual review must update the CSV before final decision.",
        "",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"- {key}: {value}")
    Path(f"{args.report_prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
