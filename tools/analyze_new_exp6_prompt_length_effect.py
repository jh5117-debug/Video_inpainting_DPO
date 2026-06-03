#!/usr/bin/env python3
"""Prompt-length stratified audit for new Exp6 qual30 side-by-side videos.

This is a qualitative-audit helper. It does not claim that Exp6 is better than
base; it organizes the 30 side-by-side samples by prompt length and creates
contact sheets for human review.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_QUAL_DIR = (
    "/home/nvme01/H20_Video_inpainting_DPO/logs/qual_sbs_30/"
    "exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000_20260601_004753"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qual_dir", default=DEFAULT_QUAL_DIR)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--report", default="")
    parser.add_argument("--frame_index", type=int, default=8)
    parser.add_argument("--thumb_width", type=int, default=520)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--label_csv", default="", help="Optional CSV with prompt_id,human_label,notes.")
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def word_count(prompt: str) -> int:
    words = re.findall(r"[\w']+", prompt, flags=re.UNICODE)
    return len(words)


def absolute_bucket(char_count: int) -> str:
    if char_count <= 40:
        return "short"
    if char_count <= 100:
        return "medium"
    return "long"


def assign_tertiles(rows: List[Dict[str, object]]) -> None:
    ordered = sorted(rows, key=lambda r: (int(r["char_count"]), int(r["prompt_id"])))
    n = len(ordered)
    for rank, row in enumerate(ordered):
        if n <= 1:
            bucket = "medium"
        elif rank < n / 3:
            bucket = "short"
        elif rank < 2 * n / 3:
            bucket = "medium"
        else:
            bucket = "long"
        row["prompt_bucket_by_tertile"] = bucket


def load_labels(path: str) -> Dict[str, Dict[str, str]]:
    if not path:
        return {}
    label_path = Path(path)
    if not label_path.exists():
        return {}
    with label_path.open("r", encoding="utf-8", newline="") as f:
        labels = {}
        for row in csv.DictReader(f):
            labels[str(row.get("prompt_id", ""))] = row
        return labels


def read_video_frame(path: Path, frame_index: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx = min(frame_index, max(total - 1, 0)) if total else frame_index
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def safe_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def make_card(row: Dict[str, object], thumb_width: int, frame_index: int) -> Image.Image:
    video_path = Path(str(row["side_by_side_video_path"]))
    frame = read_video_frame(video_path, frame_index)
    font = safe_font(18)
    small = safe_font(14)
    caption_h = 116
    if frame is None:
        thumb_h = int(thumb_width * 0.35)
        image = Image.new("RGB", (thumb_width, thumb_h), (32, 32, 32))
        draw = ImageDraw.Draw(image)
        draw.text((12, 12), "unreadable video", fill=(255, 120, 120), font=font)
    else:
        img = Image.fromarray(frame)
        ratio = thumb_width / img.width
        thumb_h = max(1, int(img.height * ratio))
        image = img.resize((thumb_width, thumb_h), Image.Resampling.LANCZOS)

    card = Image.new("RGB", (thumb_width, image.height + caption_h), (245, 245, 245))
    card.paste(image, (0, 0))
    draw = ImageDraw.Draw(card)
    prompt = str(row["prompt"])
    title = (
        f"#{int(row['prompt_id']) + 1:02d} chars={row['char_count']} "
        f"words={row['word_count']} abs={row['prompt_bucket_absolute']}"
    )
    draw.rectangle((0, image.height, thumb_width, image.height + caption_h), fill=(245, 245, 245))
    draw.text((8, image.height + 8), title, fill=(20, 20, 20), font=font)
    label = str(row.get("human_label") or "review_pending")
    draw.text((8, image.height + 34), f"label: {label}", fill=(60, 60, 60), font=small)
    y = image.height + 56
    for line in textwrap.wrap(prompt, width=72)[:4]:
        draw.text((8, y), line, fill=(35, 35, 35), font=small)
        y += 17
    return card


def make_contact_sheet(rows: List[Dict[str, object]], out_path: Path, thumb_width: int, cols: int, frame_index: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Image.new("RGB", (thumb_width, 120), (245, 245, 245)).save(out_path)
        return
    cards = [make_card(row, thumb_width, frame_index) for row in rows]
    gutter = 14
    max_h = max(card.height for card in cards)
    rows_n = math.ceil(len(cards) / cols)
    sheet = Image.new(
        "RGB",
        (cols * thumb_width + (cols - 1) * gutter, rows_n * max_h + (rows_n - 1) * gutter),
        (220, 220, 220),
    )
    for idx, card in enumerate(cards):
        x = (idx % cols) * (thumb_width + gutter)
        y = (idx // cols) * (max_h + gutter)
        sheet.paste(card, (x, y))
    sheet.save(out_path, quality=92)


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def bucket_summary(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    for bucket_type in ("prompt_bucket_by_tertile", "prompt_bucket_absolute"):
        grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[str(row[bucket_type])].append(row)
        for bucket in ("short", "medium", "long"):
            group = grouped.get(bucket, [])
            labels = Counter(str(r.get("human_label") or "review_pending") for r in group)
            out.append(
                {
                    "bucket_type": bucket_type,
                    "bucket": bucket,
                    "count": len(group),
                    "char_min": min((int(r["char_count"]) for r in group), default=""),
                    "char_median": float(np.median([int(r["char_count"]) for r in group])) if group else "",
                    "char_max": max((int(r["char_count"]) for r in group), default=""),
                    "base_better": labels.get("base_better", 0),
                    "exp6_better": labels.get("exp6_better", 0),
                    "tie": labels.get("tie", 0),
                    "invalid": labels.get("invalid", 0),
                    "review_pending": labels.get("review_pending", 0),
                }
            )
    return out


def markdown_table(rows: List[Dict[str, object]], cols: List[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def write_report(path: Path, qual_dir: Path, rows: List[Dict[str, object]], summary: List[Dict[str, object]], output_dir: Path) -> None:
    label_counts = Counter(str(r.get("human_label") or "review_pending") for r in rows)
    long_tertile = [r for r in rows if r["prompt_bucket_by_tertile"] == "long"]
    long_abs = [r for r in rows if r["prompt_bucket_absolute"] == "long"]
    all_labeled = label_counts.get("review_pending", 0) == 0
    if all_labeled:
        long_labels = Counter(str(r.get("human_label")) for r in long_tertile)
        long_answer = (
            f"Long-tertile labels: exp6_better={long_labels.get('exp6_better', 0)}, "
            f"base_better={long_labels.get('base_better', 0)}, tie={long_labels.get('tie', 0)}, "
            f"invalid={long_labels.get('invalid', 0)}."
        )
    else:
        long_answer = "Inconclusive until contact sheets are reviewed and labels are filled."

    text = [
        "# New Exp6 Prompt-Length Audit",
        "",
        "status: **hypothesis / qualitative audit**",
        "",
        f"qual_dir: `{qual_dir}`",
        f"output_dir: `{output_dir}`",
        "",
        "Observation under audit: long prompts may look better for new Exp6 than DiffuEraser-base. This report does not treat that as a conclusion unless human labels support it.",
        "",
        "## Outputs",
        "",
        f"- prompt table: `{output_dir / 'prompt_length_table.csv'}`",
        f"- bucket summary: `{output_dir / 'bucket_summary.csv'}`",
        f"- tertile short contact sheet: `{output_dir / 'short_contact_sheet.jpg'}`",
        f"- tertile medium contact sheet: `{output_dir / 'medium_contact_sheet.jpg'}`",
        f"- tertile long contact sheet: `{output_dir / 'long_contact_sheet.jpg'}`",
        "",
        "## Counts",
        "",
        f"- total samples: {len(rows)}",
        f"- long prompt samples by tertile: {len(long_tertile)}",
        f"- long prompt samples by absolute char bucket (>100 chars): {len(long_abs)}",
        f"- label counts: {dict(label_counts)}",
        "",
        "## Bucket Summary",
        "",
        markdown_table(
            summary,
            [
                "bucket_type",
                "bucket",
                "count",
                "char_min",
                "char_median",
                "char_max",
                "base_better",
                "exp6_better",
                "tie",
                "invalid",
                "review_pending",
            ],
        ),
        "",
        "## Required Answers",
        "",
        f"1. Long prompt sample count: tertile={len(long_tertile)}, absolute>100={len(long_abs)}.",
        f"2. Does Exp6 more often beat base on long prompts? {long_answer}",
        "3. Short/medium/long differences: pending human labels unless the table above has no review_pending rows.",
        "4. Support for prompt-aware ablation: not yet; requires visual labels and preferably a larger long-prompt set.",
        "5. Individual-sample risk: high with qual30; do not generalize from this audit alone.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text), encoding="utf-8")


def main() -> int:
    args = parse_args()
    qual_dir = Path(args.qual_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else qual_dir.parents[1] / "analysis" / "new_exp6_prompt_length"
    report = Path(args.report).expanduser().resolve() if args.report else qual_dir.parents[2] / "reports" / "new_exp6_prompt_length_audit.md"
    manifest = qual_dir / "pair_manifest.csv"
    rows_raw = read_rows(manifest)
    labels = load_labels(args.label_csv)

    rows: List[Dict[str, object]] = []
    for row in rows_raw:
        prompt = row.get("prompt", "")
        prompt_id = str(row.get("prompt_id", len(rows)))
        item: Dict[str, object] = {
            "prompt_id": int(prompt_id),
            "prompt": prompt,
            "char_count": len(prompt),
            "word_count": word_count(prompt),
            "prompt_bucket_absolute": absolute_bucket(len(prompt)),
            "base_video_path": row.get("base_video_path", ""),
            "exp_video_path": row.get("exp_video_path", ""),
            "side_by_side_video_path": row.get("side_by_side_video_path", ""),
            "base_weights_dir": row.get("base_weights_dir", ""),
            "exp_weights_dir": row.get("exp_weights_dir", ""),
            "human_label": "review_pending",
            "human_notes": "",
        }
        if prompt_id in labels:
            item["human_label"] = labels[prompt_id].get("human_label", "review_pending") or "review_pending"
            item["human_notes"] = labels[prompt_id].get("notes", "")
        rows.append(item)
    assign_tertiles(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "prompt_length_table.csv", rows)
    for bucket in ("short", "medium", "long"):
        bucket_rows = [row for row in rows if row["prompt_bucket_by_tertile"] == bucket]
        make_contact_sheet(bucket_rows, output_dir / f"{bucket}_contact_sheet.jpg", args.thumb_width, args.cols, args.frame_index)
    summary = bucket_summary(rows)
    write_csv(output_dir / "bucket_summary.csv", summary)
    (output_dir / "bucket_summary.md").write_text(
        markdown_table(
            summary,
            [
                "bucket_type",
                "bucket",
                "count",
                "char_min",
                "char_median",
                "char_max",
                "base_better",
                "exp6_better",
                "tie",
                "invalid",
                "review_pending",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "audit_manifest.json").write_text(
        json.dumps({"qual_dir": str(qual_dir), "report": str(report), "rows": len(rows)}, indent=2),
        encoding="utf-8",
    )
    write_report(report, qual_dir, rows, summary, output_dir)
    print(f"[exp6-prompt-audit] qual_dir={qual_dir}")
    print(f"[exp6-prompt-audit] output={output_dir}")
    print(f"[exp6-prompt-audit] report={report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
