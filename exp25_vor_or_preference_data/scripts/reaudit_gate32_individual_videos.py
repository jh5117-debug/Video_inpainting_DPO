#!/usr/bin/env python3
"""Per-sample Exp25 Gate32 video reaudit.

This does not generate new losers. It reads the existing Gate32 materialized
triplets and raw DiffuEraser candidates, writes a per-sample mp4 plus sampled
frame/crop evidence, and records a conservative review CSV.  In headless
execution we cannot interactively play videos, so reviewer_pass remains false
and the status is marked VISUAL_REVIEW_PENDING rather than promoted.
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
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="diffueraser")
    p.add_argument("--limit", type=int, default=32)
    p.add_argument("--num-frames", type=int, default=24)
    p.add_argument("--trivial-mask-psnr", type=float, default=18.0)
    p.add_argument("--too-close-mask-psnr", type=float, default=33.0)
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


def mae(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    pred = resize_like(pred, gt).astype(np.float32)
    gt = gt.astype(np.float32)
    if mask.shape != gt.shape[:2]:
        mask = cv2.resize(mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    w = mask.astype(np.float32)[..., None]
    denom = float(w.sum() * 3.0)
    if denom < 1:
        return float("nan")
    return float((np.abs(pred - gt) * w).sum() / denom)


def overlay(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = frame.astype(np.float32)
    tint = np.zeros_like(out)
    tint[..., 0] = color[0]
    tint[..., 1] = color[1]
    tint[..., 2] = color[2]
    alpha = (mask > 0).astype(np.float32)[..., None] * 0.45
    return np.clip(out * (1.0 - alpha) + tint * alpha, 0, 255).astype(np.uint8)


def label(arr: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, min(img.width, 520), 22], fill=(0, 0, 0))
    draw.text((5, 5), text, fill=(255, 255, 255))
    return np.asarray(img)


def bbox_from_mask(mask: np.ndarray, pad: int = 12) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    h, w = mask.shape
    if len(xs) == 0:
        return 0, 0, w, h
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad + 1)
    return x0, y0, x1, y1


def crop_resize(arr: np.ndarray, box: tuple[int, int, int, int], size: tuple[int, int] = (220, 160)) -> np.ndarray:
    x0, y0, x1, y1 = box
    crop = arr[y0:y1, x0:x1]
    if crop.size == 0:
        crop = arr
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)


def panel(arr: np.ndarray, text: str, size: tuple[int, int] = (220, 160)) -> np.ndarray:
    return label(cv2.resize(arr, size, interpolation=cv2.INTER_AREA), text)


def write_mp4(frames: list[np.ndarray], path: Path, fps: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in frames:
        fr = resize_like(fr, frames[0])
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()


def classify(mask_psnr: float, masked_mae: float, black_ratio: float, status: str, args: argparse.Namespace) -> str:
    if status != "OK":
        return "technical-invalid"
    if black_ratio > 0 or mask_psnr <= args.trivial_mask_psnr:
        return "trivial-bad"
    if mask_psnr >= args.too_close_mask_psnr or masked_mae <= 4.0:
        return "too-close"
    return "medium-hard"


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.manifest, args.limit)
    out_rows: list[dict] = []
    for row in rows:
        sid = row["sample_id"]
        winner = load_rgb(Path(row["winner_video_path"]), args.num_frames)
        cond = load_rgb(Path(row["condition_video_path"]), args.num_frames)
        pred_dir = args.candidate_root / args.model / "raw_frames" / sid
        pred = load_rgb(pred_dir, args.num_frames)
        size = winner[0].shape[1::-1] if winner else None
        masks = load_mask(Path(row["mask_path"]), args.num_frames, size)
        n = min(len(winner), len(cond), len(pred), len(masks), args.num_frames)
        status = "OK" if n == args.num_frames else "MISSING_FRAMES"
        video_path = args.output_dir / "mp4" / f"{sid}_{args.model}.mp4"
        write_mp4(pred[:n], video_path)
        if status == "OK":
            mask_area = [float(m.mean()) for m in masks[:n]]
            full_vals = [psnr(p, g) for p, g in zip(pred[:n], winner[:n])]
            mask_vals = [psnr(p, g, m) for p, g, m in zip(pred[:n], winner[:n], masks[:n])]
            outside_vals = [psnr(p, g, 1 - m) for p, g, m in zip(pred[:n], winner[:n], masks[:n])]
            mae_vals = [mae(p, g, m) for p, g, m in zip(pred[:n], winner[:n], masks[:n])]
            frame_err = [mae(p, g, m) for p, g, m in zip(pred[:n], winner[:n], masks[:n])]
            black_ratio = float(sum(1 for p in pred[:n] if float(p.mean()) < 2.0 or float(p.std()) < 1.0) / n)
            picks = sorted(set([0, n // 2, n - 1, int(np.argmax(mask_area)), int(np.argmax(frame_err))]))
            affected = []
            for c, g in zip(cond[:n], winner[:n]):
                diff = np.abs(c.astype(np.float32) - g.astype(np.float32)).mean(axis=2)
                affected.append((diff > max(8.0, float(np.percentile(diff, 90)))).astype(np.uint8))
            crop_rows: list[np.ndarray] = []
            for idx in picks:
                g, c, p, m, a = winner[idx], cond[idx], resize_like(pred[idx], winner[idx]), masks[idx], affected[idx]
                obj_box = bbox_from_mask(m)
                aff_box = bbox_from_mask(np.maximum(m, a))
                outside_mask = (m == 0).astype(np.uint8)
                out_box = bbox_from_mask(outside_mask, pad=0)
                crop_rows.append(
                    np.concatenate(
                        [
                            panel(overlay(g, m), f"{sid} f{idx} mask"),
                            panel(p, "raw loser"),
                            label(crop_resize(p, obj_box), "object/mask crop"),
                            label(crop_resize(p, aff_box), "affected crop"),
                            label(crop_resize(p, out_box), "outside crop"),
                        ],
                        axis=1,
                    )
                )
            sheet = np.concatenate(crop_rows, axis=0)
            sheet_path = args.output_dir / "sample_contact_sheets" / f"{sid}.jpg"
            sheet_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(sheet).save(sheet_path, quality=92)
            classification = classify(float(np.nanmean(mask_vals)), float(np.nanmean(mae_vals)), black_ratio, status, args)
            raw_artifact = "none"
            if black_ratio > 0:
                raw_artifact = "black_or_flat_raw_frame"
            elif float(np.nanmean(mask_vals)) <= args.trivial_mask_psnr:
                raw_artifact = "large_mask_region_mismatch_in_raw_frames"
            reason = f"mask_psnr={float(np.nanmean(mask_vals)):.4f}; masked_mae={float(np.nanmean(mae_vals)):.4f}; outside_psnr={float(np.nanmean(outside_vals)):.4f}; black_ratio={black_ratio:.4f}"
            out_rows.append(
                {
                    "sample_id": sid,
                    "model": args.model,
                    "video_path": str(video_path),
                    "reviewed_frames": " ".join(str(i) for i in picks),
                    "classification": classification,
                    "temporal_artifact": "sampled_frame_only_playback_pending",
                    "spatial_artifact": raw_artifact,
                    "reason": reason,
                    "reviewer_pass": "false",
                    "visual_review_status": "VISUAL_REVIEW_PENDING_HEADLESS_FRAME_AUDIT_ONLY",
                    "source_type": row.get("source_type", ""),
                    "scene_group": row.get("scene_group", ""),
                    "full_psnr": f"{float(np.nanmean(full_vals)):.6f}",
                    "mask_psnr": f"{float(np.nanmean(mask_vals)):.6f}",
                    "outside_psnr": f"{float(np.nanmean(outside_vals)):.6f}",
                    "masked_mae_to_bg": f"{float(np.nanmean(mae_vals)):.6f}",
                    "black_frame_ratio": f"{black_ratio:.6f}",
                    "contact_sheet": str(sheet_path),
                    "candidate_dir": str(pred_dir),
                    "winner_mp4": row.get("winner_mp4", ""),
                    "condition_mp4": row.get("condition_mp4", ""),
                    "mask_mp4": row.get("mask_mp4", ""),
                    "hard_comp": str(row.get("hard_comp", "")),
                    "comp_mode": str(row.get("comp_mode", "")),
                }
            )
        else:
            out_rows.append(
                {
                    "sample_id": sid,
                    "model": args.model,
                    "video_path": str(video_path),
                    "reviewed_frames": "",
                    "classification": "technical-invalid",
                    "temporal_artifact": "missing_frames",
                    "spatial_artifact": "missing_frames",
                    "reason": f"winner={len(winner)} condition={len(cond)} pred={len(pred)} mask={len(masks)}",
                    "reviewer_pass": "false",
                    "visual_review_status": "VISUAL_REVIEW_PENDING_HEADLESS_FRAME_AUDIT_ONLY",
                    "source_type": row.get("source_type", ""),
                    "scene_group": row.get("scene_group", ""),
                    "full_psnr": "",
                    "mask_psnr": "",
                    "outside_psnr": "",
                    "masked_mae_to_bg": "",
                    "black_frame_ratio": "1.0",
                    "contact_sheet": "",
                    "candidate_dir": str(pred_dir),
                    "winner_mp4": row.get("winner_mp4", ""),
                    "condition_mp4": row.get("condition_mp4", ""),
                    "mask_mp4": row.get("mask_mp4", ""),
                    "hard_comp": str(row.get("hard_comp", "")),
                    "comp_mode": str(row.get("comp_mode", "")),
                }
            )
    fields = list(out_rows[0].keys()) if out_rows else []
    csv_path = args.output_dir / "gate32_individual_video_reaudit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    counts: dict[str, int] = {}
    for r in out_rows:
        counts[r["classification"]] = counts.get(r["classification"], 0) + 1
    summary = {
        "status": "completed_frame_audit_visual_review_pending",
        "note": "Headless sampled-frame/crop audit only; mp4 interactive playback is still pending.",
        "counts": counts,
        "csv": str(csv_path),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "gate32_individual_video_reaudit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
