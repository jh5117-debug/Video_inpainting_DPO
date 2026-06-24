#!/usr/bin/env python3
"""Review VideoPainter Probe/Gate outputs with simple metrics and visuals."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--review-dir", type=Path, required=True)
    p.add_argument("--num-frames", type=int, default=49)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--min-valid-psnr", type=float, default=10.0)
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


def load_rgb_dir(path: Path, limit: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for fp in list_images(path)[:limit]:
        out.append(np.asarray(Image.open(fp).convert("RGB"), dtype=np.uint8))
    return out


def load_mask_dir(path: Path, limit: int, size: tuple[int, int] | None = None) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for fp in list_images(path)[:limit]:
        img = Image.open(fp).convert("L")
        if size and img.size != size:
            img = img.resize(size, Image.NEAREST)
        out.append((np.asarray(img, dtype=np.uint8) > 127).astype(np.float32))
    return out


def resize_like(arr: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if arr.shape[:2] == ref.shape[:2]:
        return arr
    return cv2.resize(arr, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)


def psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    pred = resize_like(pred, gt).astype(np.float32)
    gt = gt.astype(np.float32)
    if mask is not None:
        if mask.shape != gt.shape[:2]:
            mask = cv2.resize(mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        w = mask[..., None].astype(np.float32)
        denom = float(w.sum() * 3.0)
        if denom < 1:
            return float("nan")
        mse = float(((pred - gt) ** 2 * w).sum() / denom)
    else:
        mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def add_label(arr: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, min(img.width, 340), 20], fill=(0, 0, 0))
    draw.text((5, 4), text, fill=(255, 255, 255))
    return np.asarray(img)


def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = frame.astype(np.float32).copy()
    red = np.zeros_like(out)
    red[..., 0] = 255
    alpha = (mask > 0).astype(np.float32)[..., None] * 0.45
    return np.clip(out * (1 - alpha) + red * alpha, 0, 255).astype(np.uint8)


def make_sheet(sid: str, gt: list[np.ndarray], masks: list[np.ndarray], pred: list[np.ndarray], out: Path) -> None:
    n = min(len(gt), len(masks), len(pred))
    picks = sorted(set([0, 8, 16, 24, 32, 40, n - 1]))
    rows: list[np.ndarray] = []
    for idx in picks:
        if idx < 0 or idx >= n:
            continue
        pr = resize_like(pred[idx], gt[idx])
        err = np.abs(pr.astype(np.float32) - gt[idx].astype(np.float32)).mean(axis=2)
        err_rgb = cv2.cvtColor(cv2.applyColorMap(np.clip(err, 0, 80).astype(np.uint8) * 3, cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)
        rows.append(
            np.concatenate(
                [
                    add_label(gt[idx], "source/BG"),
                    add_label(overlay_mask(gt[idx], masks[idx]), "mask"),
                    add_label(pr, "VideoPainter"),
                    add_label(err_rgb, "abs error"),
                ],
                axis=1,
            )
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(out, quality=92)


def main() -> int:
    args = parse_args()
    args.review_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.manifest, args.limit)
    metrics: list[dict] = []
    for row in rows:
        sid = row["sample_id"]
        gt = load_rgb_dir(Path(row["frame_dir"]), args.num_frames)
        masks = load_mask_dir(Path(row["mask_dir"]), args.num_frames, size=gt[0].shape[1::-1] if gt else None)
        pred = load_rgb_dir(args.output_dir / "raw_frames" / sid, args.num_frames)
        item = {"sample_id": sid, "frames": len(pred), "status": "OK" if len(gt) == args.num_frames and len(masks) == args.num_frames and len(pred) == args.num_frames else "MISSING_FRAMES"}
        if item["status"] == "OK":
            full = [psnr(p, g) for p, g in zip(pred, gt)]
            mask_vals = [psnr(p, g, m) for p, g, m in zip(pred, gt, masks)]
            temporal = [float(np.mean(np.abs(pred[i].astype(np.float32) - resize_like(pred[i - 1], pred[i]).astype(np.float32)))) for i in range(1, len(pred))]
            black_ratio = float(sum(1 for p in pred if float(p.mean()) < 2.0 or float(p.std()) < 1.0) / len(pred))
            item.update(
                {
                    "psnr": float(np.nanmean(full)),
                    "mask_psnr": float(np.nanmean(mask_vals)),
                    "temporal_absdiff": float(np.mean(temporal)) if temporal else 0.0,
                    "black_frame_ratio": black_ratio,
                }
            )
            make_sheet(sid, gt, masks, pred, args.review_dir / "contact_sheets" / f"{sid}.jpg")
        else:
            item.update({"psnr": float("nan"), "mask_psnr": float("nan"), "temporal_absdiff": float("nan"), "black_frame_ratio": 1.0})
        item["review_status"] = "PASS" if item["status"] == "OK" and item["psnr"] >= args.min_valid_psnr and item["black_frame_ratio"] == 0 else "FAIL"
        metrics.append(item)

    csv_path = args.review_dir / "review_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["sample_id", "status", "review_status", "frames", "psnr", "mask_psnr", "temporal_absdiff", "black_frame_ratio"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: m.get(k, "") for k in fields} for m in metrics])
    passed = all(m["review_status"] == "PASS" for m in metrics) and bool(metrics)
    summary = {
        "status": "passed" if passed else "failed",
        "num_samples": len(metrics),
        "passed": sum(1 for m in metrics if m["review_status"] == "PASS"),
        "failed": sum(1 for m in metrics if m["review_status"] != "PASS"),
        "csv": str(csv_path),
        "contact_sheets": str(args.review_dir / "contact_sheets"),
    }
    (args.review_dir / "review_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# Exp26 VideoPainter Output Review",
        "",
        f"- Status: {summary['status']}",
        f"- Samples: {summary['num_samples']}",
        f"- Passed: {summary['passed']}",
        f"- Failed: {summary['failed']}",
        f"- Metrics: `{csv_path}`",
        f"- Contact sheets: `{args.review_dir / 'contact_sheets'}`",
    ]
    (args.review_dir / "review.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
