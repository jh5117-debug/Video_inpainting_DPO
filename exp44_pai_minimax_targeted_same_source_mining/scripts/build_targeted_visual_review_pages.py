#!/usr/bin/env python3
"""Build compact visual review pages for Exp44 targeted candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SELECTED_CLASSES = {"SUCCESSFUL_REMOVAL_CANDIDATE", "MEDIUM_HARD_REMOVAL"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--index-csv", required=True)
    parser.add_argument("--page-size", type=int, default=8)
    parser.add_argument("--include-all", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def open_resized(path: str, size: tuple[int, int], fill: tuple[int, int, int]) -> Image.Image:
    if not path or not Path(path).exists():
        return Image.new("RGB", size, fill)
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, fill)
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def row_text(row: dict[str, object]) -> list[str]:
    return [
        f"{row.get('candidate_id', '')}",
        f"group={row.get('scene_group', '')} auto={row.get('auto_classification', '')}",
        (
            f"PSNR full/mask/bound/out="
            f"{safe_float(row.get('full_psnr')):.2f}/"
            f"{safe_float(row.get('mask_psnr')):.2f}/"
            f"{safe_float(row.get('boundary_psnr')):.2f}/"
            f"{safe_float(row.get('outside_psnr')):.2f}"
        ),
        f"temporal={safe_float(row.get('temporal_diff_mae')):.3f} outside_mae={safe_float(row.get('outside_mae')):.3f}",
    ]


def build_page(rows: list[dict[str, object]], page_path: Path, page_number: int) -> None:
    font = ImageFont.load_default()
    card_w = 1420
    card_h = 650
    margin = 24
    page = Image.new("RGB", (card_w + margin * 2, card_h * len(rows) + margin * 2), (245, 245, 245))
    draw = ImageDraw.Draw(page)
    draw.text((margin, 6), f"Exp44 targeted visual relabel page {page_number:03d}", fill=(0, 0, 0), font=font)
    for idx, row in enumerate(rows):
        y = margin + idx * card_h
        draw.rectangle((margin, y, margin + card_w, y + card_h - 10), fill=(255, 255, 255), outline=(40, 40, 40), width=2)
        text_y = y + 10
        for line in row_text(row):
            draw.text((margin + 12, text_y), line[:190], fill=(0, 0, 0), font=font)
            text_y += 16
        review = open_resized(str(row.get("review_sheet", "")), (1024, 256), (230, 230, 230))
        temporal = open_resized(str(row.get("temporal_strip_16", "")), (320, 512), (235, 235, 235))
        page.paste(review, (margin + 12, y + 82))
        page.paste(temporal, (margin + 1060, y + 82))
    page_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(page_path, quality=92)


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.manifest))
    if not args.include_all:
        rows = [row for row in rows if row.get("auto_classification") in SELECTED_CLASSES]
    rows.sort(key=lambda row: (str(row.get("scene_group", "")), str(row.get("sample_id", "")), int(row.get("seed_index", 0) or 0)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    for page_idx in range(0, len(rows), args.page_size):
        page_rows = rows[page_idx : page_idx + args.page_size]
        page_no = page_idx // args.page_size
        page_path = output_dir / f"exp44_targeted_visual_review_page_{page_no:03d}.jpg"
        build_page(page_rows, page_path, page_no)
        for row in page_rows:
            index_rows.append(
                {
                    "candidate_id": row.get("candidate_id", ""),
                    "scene_group": row.get("scene_group", ""),
                    "sample_id": row.get("sample_id", ""),
                    "auto_classification": row.get("auto_classification", ""),
                    "page": str(page_path),
                    "raw_output_mp4": row.get("raw_output_mp4", ""),
                    "side_by_side_mp4": row.get("side_by_side_mp4", ""),
                    "review_sheet": row.get("review_sheet", ""),
                    "temporal_strip_16": row.get("temporal_strip_16", ""),
                }
            )
    index_path = Path(args.index_csv)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0].keys()) if index_rows else ["candidate_id"])
        writer.writeheader()
        writer.writerows(index_rows)
    print(json.dumps({"pages": (len(rows) + args.page_size - 1) // args.page_size, "rows": len(rows), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
