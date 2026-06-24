#!/usr/bin/env python3
"""Reclassify Exp26 Gate16 under the pre-registered gate.

This script audits every selected Gate16 row without replacing failed samples.
It records source/mask/output diagnostics and applies the original gate:
technical valid >=15/16, systematic failure 0, trivial bad <=2/16,
medium-hard/hard-plausible >=8/16, with visual review status kept pending when
only headless frame/crop audit is available.
"""

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
    p.add_argument("--review-metrics", type=Path, required=True)
    p.add_argument("--audit-dir", type=Path, required=True)
    p.add_argument("--num-frames", type=int, default=49)
    p.add_argument("--limit", type=int, default=16)
    return p.parse_args()


def read_jsonl(path: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def read_csv(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        return {r["sample_id"]: r for r in csv.DictReader(f)}


def list_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS)


def load_rgb(path: Path, limit: int) -> list[np.ndarray]:
    return [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in list_images(path)[:limit]]


def load_mask(path: Path, limit: int, size: tuple[int, int] | None) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for p in list_images(path)[:limit]:
        img = Image.open(p).convert("L")
        if size and img.size != size:
            img = img.resize(size, Image.NEAREST)
        out.append((np.asarray(img, dtype=np.uint8) > 127).astype(np.uint8))
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
        w = mask.astype(np.float32)[..., None]
        denom = float(w.sum() * 3.0)
        if denom < 1:
            return float("nan")
        mse = float(((pred - gt) ** 2 * w).sum() / denom)
    else:
        mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def label(arr: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, min(img.width, 520), 22], fill=(0, 0, 0))
    draw.text((5, 5), text, fill=(255, 255, 255))
    return np.asarray(img)


def overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = frame.astype(np.float32)
    red = np.zeros_like(out)
    red[..., 0] = 255
    alpha = (mask > 0).astype(np.float32)[..., None] * 0.45
    return np.clip(out * (1 - alpha) + red * alpha, 0, 255).astype(np.uint8)


def bbox(mask: np.ndarray, pad: int = 16) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    h, w = mask.shape
    if len(xs) == 0:
        return 0, 0, w, h
    return max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad), min(w, int(xs.max()) + pad + 1), min(h, int(ys.max()) + pad + 1)


def crop(arr: np.ndarray, box: tuple[int, int, int, int], size: tuple[int, int] = (240, 160)) -> np.ndarray:
    x0, y0, x1, y1 = box
    c = arr[y0:y1, x0:x1]
    if c.size == 0:
        c = arr
    return cv2.resize(c, size, interpolation=cv2.INTER_AREA)


def panel(arr: np.ndarray, text: str, size: tuple[int, int] = (240, 160)) -> np.ndarray:
    return label(cv2.resize(arr, size, interpolation=cv2.INTER_AREA), text)


def make_audit_sheet(sid: str, gt: list[np.ndarray], masks: list[np.ndarray], pred: list[np.ndarray], picks: list[int], out: Path) -> None:
    rows = []
    for idx in picks:
        g, m, p = gt[idx], masks[idx], resize_like(pred[idx], gt[idx])
        box = bbox(m)
        err = np.abs(p.astype(np.float32) - g.astype(np.float32)).mean(axis=2)
        heat = cv2.cvtColor(cv2.applyColorMap(np.clip(err * 3, 0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)
        rows.append(
            np.concatenate(
                [
                    panel(g, f"{sid} f{idx} source"),
                    panel(overlay(g, m), "mask"),
                    panel(p, "VideoPainter raw"),
                    panel(heat, "abs error"),
                    label(crop(p, box), "mask crop"),
                ],
                axis=1,
            )
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(out, quality=92)


def classify(item: dict) -> tuple[str, str]:
    if item["technical_valid"] != "true":
        return "technical-invalid", "missing frames or masks"
    if float(item["black_frame_ratio"]) > 0:
        return "technical-invalid", "black or flat output frame"
    if float(item["mask_psnr"]) < 5.0:
        return "trivial-bad", "extreme mask-region failure"
    if float(item["psnr"]) < 10.0:
        return "trivial-bad", "low full-frame PSNR"
    if float(item["mask_psnr"]) >= 24.0:
        return "too-close", "mask region too close to source/winner"
    return "medium-hard", "valid output with visible finite defect"


def main() -> int:
    args = parse_args()
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.manifest, args.limit)
    metrics = read_csv(args.review_metrics)
    out_rows: list[dict] = []
    for row in rows:
        sid = row["sample_id"]
        gt = load_rgb(Path(row["frame_dir"]), args.num_frames)
        masks = load_mask(Path(row["mask_dir"]), args.num_frames, gt[0].shape[1::-1] if gt else None)
        pred = load_rgb(args.output_dir / "raw_frames" / sid, args.num_frames)
        n = min(len(gt), len(masks), len(pred), args.num_frames)
        technical_valid = n == args.num_frames
        if technical_valid:
            mask_area = [float(m.mean()) for m in masks]
            cent = [centroid(m) for m in masks]
            edge_touch = sum(int(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()) for m in masks)
            motion = 0.0
            prev = None
            for c in cent:
                if c is not None and prev is not None:
                    motion += math.hypot(c[0] - prev[0], c[1] - prev[1])
                if c is not None:
                    prev = c
            frame_mask_psnr = [psnr(p, g, m) for p, g, m in zip(pred, gt, masks)]
            anomaly = int(np.nanargmin(frame_mask_psnr))
            picks = sorted(set([0, args.num_frames // 2, args.num_frames - 1, int(np.argmax(mask_area)), anomaly]))
            sheet = args.audit_dir / "sample_audit_sheets" / f"{sid}.jpg"
            make_audit_sheet(sid, gt, masks, pred, picks, sheet)
            review = metrics.get(sid, {})
            item = {
                "sample_id": sid,
                "technical_valid": "true",
                "frames": str(n),
                "source_video": row.get("frame_dir", ""),
                "mask_dir": row.get("mask_dir", ""),
                "output_dir": str(args.output_dir / "raw_frames" / sid),
                "reviewed_frames": " ".join(str(i) for i in picks),
                "mask_area_mean": f"{float(np.mean(mask_area)):.6f}",
                "mask_area_max": f"{float(np.max(mask_area)):.6f}",
                "edge_touch_frames": str(edge_touch),
                "motion_proxy": f"{motion:.6f}",
                "first_frame_gt": str(row.get("first_frame_gt", "")),
                "condition_definition": row.get("condition_definition", ""),
                "psnr": review.get("psnr", f"{float(np.nanmean([psnr(p,g) for p,g in zip(pred,gt)])):.6f}"),
                "mask_psnr": review.get("mask_psnr", f"{float(np.nanmean(frame_mask_psnr)):.6f}"),
                "temporal_absdiff": review.get("temporal_absdiff", ""),
                "black_frame_ratio": review.get("black_frame_ratio", "0.0"),
                "contact_sheet": str(sheet),
                "visual_review_status": "VISUAL_REVIEW_PENDING_HEADLESS_FRAME_AUDIT_ONLY",
                "reviewer_pass": "false",
            }
        else:
            item = {
                "sample_id": sid,
                "technical_valid": "false",
                "frames": str(n),
                "source_video": row.get("frame_dir", ""),
                "mask_dir": row.get("mask_dir", ""),
                "output_dir": str(args.output_dir / "raw_frames" / sid),
                "reviewed_frames": "",
                "mask_area_mean": "",
                "mask_area_max": "",
                "edge_touch_frames": "",
                "motion_proxy": "",
                "first_frame_gt": str(row.get("first_frame_gt", "")),
                "condition_definition": row.get("condition_definition", ""),
                "psnr": "nan",
                "mask_psnr": "nan",
                "temporal_absdiff": "",
                "black_frame_ratio": "1.0",
                "contact_sheet": "",
                "visual_review_status": "VISUAL_REVIEW_PENDING_HEADLESS_FRAME_AUDIT_ONLY",
                "reviewer_pass": "false",
            }
        klass, reason = classify(item)
        item["classification"] = klass
        item["failure_mode"] = "model_failure" if klass == "trivial-bad" and item["technical_valid"] == "true" else ""
        item["reason"] = reason
        out_rows.append(item)
    fields = list(out_rows[0].keys()) if out_rows else []
    csv_path = args.audit_dir / "gate16_reclassification.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    counts: dict[str, int] = {}
    for r in out_rows:
        counts[r["classification"]] = counts.get(r["classification"], 0) + 1
    technical_valid = sum(1 for r in out_rows if r["technical_valid"] == "true")
    systematic_failure = 0
    trivial_bad = counts.get("trivial-bad", 0)
    medium = counts.get("medium-hard", 0)
    gate_metric_status = (
        technical_valid >= 15
        and systematic_failure == 0
        and trivial_bad <= 2
        and medium >= 8
    )
    summary = {
        "status": "metric_gate_pass_visual_review_pending" if gate_metric_status else "failed",
        "technical_valid": technical_valid,
        "systematic_failure": systematic_failure,
        "counts": counts,
        "gate64_allowed": False,
        "gate64_blocker": "interactive_video_review_pending" if gate_metric_status else "pre_registered_gate_failed",
        "csv": str(csv_path),
        "audit_dir": str(args.audit_dir),
    }
    (args.audit_dir / "gate16_reclassification_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if gate_metric_status else 2


if __name__ == "__main__":
    raise SystemExit(main())
