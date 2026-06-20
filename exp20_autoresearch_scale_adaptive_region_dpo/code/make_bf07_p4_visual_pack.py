#!/usr/bin/env python3
"""Build anonymous shadow-dev visual contact sheets for Exp20 BF07/P4 review."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


BASELINE_LABELS = {
    "SFT": "SFT48000_baseline",
    "Exp11-S1": "Exp11_outer_b075_S1_plus_SFT_S2",
    "Exp11-S2": "Exp11_outer_b075_S2",
}

CANDIDATES = {
    "EQ_P0": "/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp20_autoresearch_scale_adaptive_region_dpo/trials/EQ_P0_1d8cd54758b73251/eval_shadow/EQ_P0_shadow",
    "EQ_P4": "/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp20_autoresearch_scale_adaptive_region_dpo/trials/EQ_P4_edbea07bb785e769/eval_shadow/EQ_P4_shadow",
    "EQ_BF07": "/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp20_autoresearch_scale_adaptive_region_dpo/trials/EQ_BF07_2bc98e58514fb1da/eval_shadow/EQ_BF07_shadow",
    "EQ_AD04": "/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp20_autoresearch_scale_adaptive_region_dpo/trials/EQ_AD04_77a0ed002ad3955d/eval_shadow/EQ_AD04_shadow",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def baseline_dirs(report_csv: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for row in read_csv(report_csv):
        label = row.get("model_label", "")
        for alias, expected in BASELINE_LABELS.items():
            if label == expected:
                out[alias] = Path(row.get("result_dir", ""))
    return out


def list_frames(path: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(p for p in path.iterdir() if p.suffix.lower() in exts)


def read_frame(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size and img.size != size:
        img = img.resize(size, Image.BICUBIC)
    return img


def mask_overlay(gt: Image.Image, mask_path: Path) -> Image.Image:
    mask = Image.open(mask_path).convert("L").resize(gt.size, Image.NEAREST)
    gt_np = np.array(gt).astype(np.float32)
    mask_np = np.array(mask) > 0
    color = np.array([255, 40, 40], dtype=np.float32)
    gt_np[mask_np] = 0.55 * gt_np[mask_np] + 0.45 * color
    return Image.fromarray(np.clip(gt_np, 0, 255).astype(np.uint8))


def method_frame(method_dir: Path, video: str, frame_idx: int, size: tuple[int, int]) -> Image.Image | None:
    frames = list_frames(method_dir / video / "diffueraser_comp_frames")
    if frame_idx >= len(frames):
        return None
    return read_frame(frames[frame_idx], size)


def draw_label(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    pad = 4
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.rectangle((0, 0, bbox[2] + 2 * pad, bbox[3] + 2 * pad), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return out


def paste_grid(rows: list[list[Image.Image]], row_labels: list[str]) -> Image.Image:
    cell_w, cell_h = rows[0][0].size
    label_w = 96
    out = Image.new("RGB", (label_w + cell_w * len(rows[0]), cell_h * len(rows)), (24, 24, 24))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    for r, row in enumerate(rows):
        y = r * cell_h
        draw.text((6, y + 8), row_labels[r], fill=(255, 255, 255), font=font)
        for c, img in enumerate(row):
            out.paste(img, (label_w + c * cell_w, y))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-root", default="/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/dev_boundary_shadow_v1_roots")
    parser.add_argument("--baseline-report", default="reports/exp20_shadow_dev_baselines.csv")
    parser.add_argument("--output-root", default="/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/exp20_bf07_p4_visuals")
    parser.add_argument("--review-csv", default="reports/exp20_bf07_p4_visual_review.csv")
    parser.add_argument("--review-md", default="reports/exp20_bf07_p4_visual_review.md")
    parser.add_argument("--frames", default="0,4,8,12,16,23")
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args()

    shadow = Path(args.shadow_root)
    video_root = shadow / "JPEGImages_432_240"
    mask_root = shadow / "test_masks"
    methods: dict[str, Path] = {}
    methods.update(baseline_dirs(Path(args.baseline_report)))
    methods.update({k: Path(v) for k, v in CANDIDATES.items()})
    missing = [k for k, v in methods.items() if not v.exists()]
    if missing:
        raise FileNotFoundError(f"missing method eval dirs: {missing}")

    rng = random.Random(args.seed)
    aliases = list("ABCDEFG")
    real_methods = list(methods)
    rng.shuffle(aliases)
    alias_map = {method: aliases[i] for i, method in enumerate(real_methods)}

    output = Path(args.output_root)
    sheet_root = output / "anonymous_contact_sheets"
    sheet_root.mkdir(parents=True, exist_ok=True)
    (output / "anonymous_method_mapping_private.json").write_text(json.dumps(alias_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    frame_ids = [int(x) for x in args.frames.split(",") if x.strip()]
    videos = sorted(p.name for p in video_root.iterdir() if p.is_dir())
    review_rows: list[dict[str, Any]] = []
    for video in videos:
        gt_frames = list_frames(video_root / video)
        mask_frames = list_frames(mask_root / video)
        if not gt_frames or not mask_frames:
            continue
        size = read_frame(gt_frames[0]).size
        rows: list[list[Image.Image]] = []
        labels: list[str] = []
        gt_row: list[Image.Image] = []
        mask_row: list[Image.Image] = []
        for idx in frame_ids:
            idx = min(idx, len(gt_frames) - 1)
            gt = read_frame(gt_frames[idx], size)
            gt_row.append(draw_label(gt, f"f{idx:02d}"))
            mask_row.append(mask_overlay(gt, mask_frames[min(idx, len(mask_frames) - 1)]))
        rows.extend([gt_row, mask_row])
        labels.extend(["GT", "mask"])
        for method, method_dir in methods.items():
            row: list[Image.Image] = []
            for idx in frame_ids:
                frame = method_frame(method_dir, video, min(idx, len(gt_frames) - 1), size)
                if frame is None:
                    frame = Image.new("RGB", size, (64, 0, 0))
                row.append(frame)
            rows.append(row)
            labels.append(alias_map[method])
        sheet = paste_grid(rows, labels)
        sheet_path = sheet_root / f"{video}.jpg"
        sheet.save(sheet_path, quality=92)
        review_rows.append(
            {
                "video": video,
                "anonymous_contact_sheet": str(sheet_path),
                "bf07_vs_p4_visual_judgement": "needs_review",
                "p4_vs_exp11_visual_judgement": "needs_review",
                "notes": "",
            }
        )

    review_path = Path(args.review_csv)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "anonymous_contact_sheet", "bf07_vs_p4_visual_judgement", "p4_vs_exp11_visual_judgement", "notes"])
        writer.writeheader()
        writer.writerows(review_rows)

    lines = [
        "# Exp20 BF07/P4 Visual Review",
        "",
        f"- visual_root: `{output}`",
        f"- contact_sheets: `{sheet_root}`",
        "- method names are anonymized in images; private mapping saved separately and excluded from visual titles.",
        "- status: `NEEDS_CODEX_REVIEW`",
        "",
        "| video | contact sheet | BF07 vs P4 | notes |",
        "|---|---|---|---|",
    ]
    for row in review_rows:
        lines.append(f"| {row['video']} | `{row['anonymous_contact_sheet']}` | {row['bf07_vs_p4_visual_judgement']} | {row['notes']} |")
    Path(args.review_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"visual_root": str(output), "videos": len(review_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
