#!/usr/bin/env python3
"""Analyze Exp25 Gate32 OR candidates and classify yield buckets.

This script is intentionally lightweight: it does not generate new candidates
and does not hard-comp outputs. It reads the materialized VOR triplets and the
raw DiffuEraser candidate frames, computes simple paired diagnostics, writes
contact sheets for every sample, and emits a too-close manifest for seed2
supplementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--model", default="diffueraser")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-frames", type=int, default=24)
    p.add_argument("--limit", type=int, default=32)
    p.add_argument("--too-close-mask-psnr", type=float, default=33.0)
    p.add_argument("--trivial-bad-mask-psnr", type=float, default=18.0)
    p.add_argument("--trivial-bad-frame-ratio", type=float, default=0.20)
    return p.parse_args()


def read_jsonl(path: Path, limit: int = 0) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def list_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def load_rgb_sequence(path: Path, limit: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for fp in list_images(path)[:limit]:
        img = Image.open(fp).convert("RGB")
        frames.append(np.asarray(img, dtype=np.uint8))
    return frames


def load_mask_sequence(path: Path, limit: int, size: tuple[int, int] | None = None) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for fp in list_images(path)[:limit]:
        img = Image.open(fp).convert("L")
        if size and img.size != size:
            img = img.resize(size, Image.NEAREST)
        masks.append((np.asarray(img, dtype=np.uint8) > 127).astype(np.float32))
    return masks


def resize_like(frame: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if frame.shape[:2] == ref.shape[:2]:
        return frame
    return cv2.resize(frame, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)


def psnr_from_mse(mse: float) -> float:
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def masked_stats(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    pred = resize_like(pred, target).astype(np.float32)
    target = target.astype(np.float32)
    if mask.shape != target.shape[:2]:
        mask = cv2.resize(mask, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST)
    w = mask.astype(np.float32)[..., None]
    denom = float(w.sum() * 3.0)
    if denom < 1:
        return float("nan"), float("nan")
    diff = (pred - target) * w
    mae = float(np.abs(diff).sum() / denom)
    mse = float((diff * diff).sum() / denom)
    return psnr_from_mse(mse), mae


def full_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    pred = resize_like(pred, target).astype(np.float32)
    target = target.astype(np.float32)
    return psnr_from_mse(float(np.mean((pred - target) ** 2)))


def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = frame.astype(np.float32).copy()
    red = np.zeros_like(out)
    red[..., 0] = 255
    alpha = (mask > 0).astype(np.float32)[..., None] * 0.45
    return np.clip(out * (1 - alpha) + red * alpha, 0, 255).astype(np.uint8)


def add_label(arr: np.ndarray, label: str) -> np.ndarray:
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.rectangle([0, 0, min(img.width, 360), 18], fill=(0, 0, 0))
    draw.text((4, 3), label, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def error_map(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pred = resize_like(pred, target).astype(np.float32)
    target = target.astype(np.float32)
    err = np.abs(pred - target).mean(axis=2)
    if mask.shape != err.shape:
        mask = cv2.resize(mask, (err.shape[1], err.shape[0]), interpolation=cv2.INTER_NEAREST)
    err = np.clip(err * (0.35 + 0.65 * mask), 0, 80) / 80.0
    heat = cv2.applyColorMap((err * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def make_contact_sheet(sample_id: str, winner: list[np.ndarray], cond: list[np.ndarray], masks: list[np.ndarray], pred: list[np.ndarray], out_path: Path) -> None:
    picks = sorted(set([0, 4, 8, 12, 16, min(len(pred), len(winner), len(masks)) - 1]))
    rows: list[np.ndarray] = []
    for idx in picks:
        if idx < 0:
            continue
        gt = winner[idx]
        pr = resize_like(pred[idx], gt)
        mask = masks[idx]
        row = np.concatenate(
            [
                add_label(gt, "V_bg winner"),
                add_label(cond[idx], "V_obj condition"),
                add_label(overlay_mask(gt, mask), "mask"),
                add_label(pr, "OR loser raw"),
                add_label(error_map(pr, gt, mask), "masked error"),
            ],
            axis=1,
        )
        rows.append(row)
    if not rows:
        return
    sheet = np.concatenate(rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out_path, quality=92)


def classify(row: dict, args: argparse.Namespace) -> str:
    if row["status"] != "OK":
        return "trivial-bad"
    if row["black_frame_ratio"] >= args.trivial_bad_frame_ratio:
        return "trivial-bad"
    if row["mask_psnr"] <= args.trivial_bad_mask_psnr:
        return "trivial-bad"
    if row["mask_psnr"] >= args.too_close_mask_psnr or row["masked_mae_to_bg"] <= 4.0:
        return "too-close"
    return "medium-hard"


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.manifest, args.limit)
    results: list[dict] = []
    too_close_rows: list[dict] = []
    for row in rows:
        sid = row["sample_id"]
        winner = load_rgb_sequence(Path(row["winner_video_path"]), args.num_frames)
        cond = load_rgb_sequence(Path(row["condition_video_path"]), args.num_frames)
        candidate_dir = args.candidate_root / args.model / "raw_frames" / sid
        pred = load_rgb_sequence(candidate_dir, args.num_frames)
        masks = load_mask_sequence(Path(row["mask_path"]), args.num_frames, size=winner[0].shape[1::-1] if winner else None)
        item = {
            "sample_id": sid,
            "winner_dir": row.get("winner_video_path", ""),
            "condition_dir": row.get("condition_video_path", ""),
            "mask_dir": row.get("mask_path", ""),
            "candidate_dir": str(candidate_dir),
            "frame_count": len(pred),
            "status": "OK" if len(winner) >= args.num_frames and len(masks) >= args.num_frames and len(pred) >= args.num_frames else "MISSING_FRAMES",
        }
        if item["status"] == "OK":
            full_vals: list[float] = []
            mask_vals: list[float] = []
            mask_mae_vals: list[float] = []
            outside_vals: list[float] = []
            black = 0
            for gt, pr, mask in zip(winner[: args.num_frames], pred[: args.num_frames], masks[: args.num_frames]):
                pr = resize_like(pr, gt)
                full_vals.append(full_psnr(pr, gt))
                m_psnr, m_mae = masked_stats(pr, gt, mask)
                o_psnr, _ = masked_stats(pr, gt, 1.0 - mask)
                mask_vals.append(m_psnr)
                mask_mae_vals.append(m_mae)
                outside_vals.append(o_psnr)
                if float(pr.mean()) < 2.0 or float(pr.std()) < 1.0:
                    black += 1
            temporal = [float(np.mean(np.abs(pred[i].astype(np.float32) - resize_like(pred[i - 1], pred[i]).astype(np.float32)))) for i in range(1, min(len(pred), args.num_frames))]
            item.update(
                {
                    "full_psnr": float(np.nanmean(full_vals)),
                    "mask_psnr": float(np.nanmean(mask_vals)),
                    "outside_psnr": float(np.nanmean(outside_vals)),
                    "masked_mae_to_bg": float(np.nanmean(mask_mae_vals)),
                    "pred_temporal_absdiff": float(np.mean(temporal)) if temporal else 0.0,
                    "black_frame_ratio": float(black / args.num_frames),
                }
            )
            make_contact_sheet(sid, winner, cond, masks, pred, args.output_dir / "contact_sheets" / f"{sid}.jpg")
        else:
            item.update({"full_psnr": float("nan"), "mask_psnr": float("nan"), "outside_psnr": float("nan"), "masked_mae_to_bg": float("nan"), "pred_temporal_absdiff": float("nan"), "black_frame_ratio": 1.0})
        item["yield_bucket"] = classify(item, args)
        if item["yield_bucket"] == "too-close":
            too_close_rows.append(row)
        results.append(item)

    csv_path = args.output_dir / "gate32_yield_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "sample_id",
            "status",
            "yield_bucket",
            "frame_count",
            "full_psnr",
            "mask_psnr",
            "outside_psnr",
            "masked_mae_to_bg",
            "pred_temporal_absdiff",
            "black_frame_ratio",
            "candidate_dir",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in fields} for r in results])
    write_jsonl(args.output_dir / "too_close_seed2_manifest.jsonl", too_close_rows)
    counts = {bucket: sum(1 for r in results if r["yield_bucket"] == bucket) for bucket in ["medium-hard", "too-close", "trivial-bad"]}
    summary = {
        "status": "completed",
        "manifest": str(args.manifest),
        "candidate_root": str(args.candidate_root),
        "num_samples": len(results),
        "counts": counts,
        "csv": str(csv_path),
        "contact_sheets": str(args.output_dir / "contact_sheets"),
        "too_close_manifest": str(args.output_dir / "too_close_seed2_manifest.jsonl"),
    }
    (args.output_dir / "gate32_yield_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# Exp25 Gate32 Yield Review",
        "",
        f"- Samples: {len(results)}",
        f"- Medium-hard: {counts['medium-hard']}",
        f"- Too-close: {counts['too-close']}",
        f"- Trivial-bad: {counts['trivial-bad']}",
        f"- Metrics CSV: `{csv_path}`",
        f"- Contact sheets: `{args.output_dir / 'contact_sheets'}`",
        "",
        "Classification is heuristic and paired with generated contact sheets for full visual review.",
    ]
    (args.output_dir / "gate32_yield_review.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if counts["medium-hard"] + counts["too-close"] + counts["trivial-bad"] == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
