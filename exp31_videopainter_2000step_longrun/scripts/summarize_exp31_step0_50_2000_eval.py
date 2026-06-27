#!/usr/bin/env python3
"""Summarize completed Exp31 Step0/50/2000 VideoPainter evaluations.

This script is read-only with respect to model inference. It consumes the
already generated official outputs and fixed manifests, computes paired metrics,
and writes decision support reports. It deliberately does not mark a scientific
final status by itself because the final decision also requires human visual
review of the evidence sheets and videos.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


STEPS = (0, 50, 2000)
SPLITS = ("search", "shadow")
HIGHER_IS_BETTER = (
    "full_psnr",
    "mask_psnr",
    "boundary_psnr",
    "outside_psnr",
    "full_ssim",
    "mask_ssim",
    "boundary_ssim",
)
LOWER_IS_BETTER = (
    "outside_l1",
    "temporal_absdiff_mean",
    "temporal_absdiff_max",
    "ewarp_proxy",
    "black_frame_ratio",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def list_images(path: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts)


def load_rgb_dir(path: Path, limit: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for fp in list_images(path)[:limit]:
        frames.append(np.asarray(Image.open(fp).convert("RGB"), dtype=np.uint8))
    return frames


def load_mask_dir(path: Path, limit: int, size: tuple[int, int]) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for fp in list_images(path)[:limit]:
        img = Image.open(fp).convert("L")
        if img.size != size:
            img = img.resize(size, Image.NEAREST)
        masks.append((np.asarray(img, dtype=np.uint8) > 127).astype(np.float32))
    return masks


def resize_like(arr: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if arr.shape[:2] == ref.shape[:2]:
        return arr
    return cv2.resize(arr, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)


def psnr_region(pred: np.ndarray, gt: np.ndarray, weight: np.ndarray | None = None) -> float:
    pred = resize_like(pred, gt).astype(np.float32)
    gt = gt.astype(np.float32)
    if weight is None:
        mse = float(np.mean((pred - gt) ** 2))
    else:
        if weight.shape != gt.shape[:2]:
            weight = cv2.resize(weight, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        denom = float(weight.sum() * 3.0)
        if denom < 1.0:
            return float("nan")
        mse = float((((pred - gt) ** 2) * weight[..., None]).sum() / denom)
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def ssim_map_gray(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = resize_like(a, b)
    a_gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    b_gray = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_a = cv2.GaussianBlur(a_gray, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b_gray, (11, 11), 1.5)
    sigma_a = cv2.GaussianBlur(a_gray * a_gray, (11, 11), 1.5) - mu_a * mu_a
    sigma_b = cv2.GaussianBlur(b_gray * b_gray, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(a_gray * b_gray, (11, 11), 1.5) - mu_a * mu_b
    return ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2) + 1e-8
    )


def weighted_mean(values: np.ndarray, weight: np.ndarray | None = None) -> float:
    if weight is None:
        return float(np.mean(values))
    denom = float(weight.sum())
    if denom < 1.0:
        return float("nan")
    return float((values * weight).sum() / denom)


def l1_region(pred: np.ndarray, gt: np.ndarray, weight: np.ndarray | None = None) -> float:
    pred = resize_like(pred, gt)
    err = np.abs(pred.astype(np.float32) - gt.astype(np.float32)).mean(axis=2)
    return weighted_mean(err, weight)


def boundary_mask(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    binary = (mask > 0.5).astype(np.uint8)
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
    dilated = cv2.dilate(binary, kernel)
    eroded = cv2.erode(binary, kernel)
    return ((dilated - eroded) > 0).astype(np.float32)


def temporal_absdiff(frames: list[np.ndarray]) -> tuple[float, float]:
    vals: list[float] = []
    for idx in range(1, len(frames)):
        prev = resize_like(frames[idx - 1], frames[idx])
        vals.append(float(np.abs(frames[idx].astype(np.float32) - prev.astype(np.float32)).mean()))
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.max(vals))


def farneback_ewarp_proxy(pred: list[np.ndarray], gt: list[np.ndarray], sample_every: int = 8) -> float:
    vals: list[float] = []
    for idx in range(1, min(len(pred), len(gt)), sample_every):
        prev_gt = cv2.resize(cv2.cvtColor(gt[idx - 1], cv2.COLOR_RGB2GRAY), (416, 240))
        curr_gt = cv2.resize(cv2.cvtColor(gt[idx], cv2.COLOR_RGB2GRAY), (416, 240))
        flow = cv2.calcOpticalFlowFarneback(prev_gt, curr_gt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prev_pred = cv2.resize(pred[idx - 1], (416, 240), interpolation=cv2.INTER_CUBIC)
        curr_pred = cv2.resize(pred[idx], (416, 240), interpolation=cv2.INTER_CUBIC)
        h, w = curr_gt.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(prev_pred, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        vals.append(float(np.abs(warped.astype(np.float32) - curr_pred.astype(np.float32)).mean()))
    return float(np.mean(vals)) if vals else float("nan")


def mean_float(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def compute_sample(
    *,
    split: str,
    step: int,
    row: dict[str, Any],
    run_root: Path,
    num_frames: int,
) -> dict[str, Any]:
    sid = row["sample_id"]
    gt_dir = Path(row["frame_dir"])
    mask_dir = Path(row["mask_dir"])
    raw_dir = run_root / split / f"step{step}" / "official_generation" / "raw_frames" / sid
    gt = load_rgb_dir(gt_dir, num_frames)
    if not gt:
        return {"split": split, "step": step, "sample_id": sid, "status": "MISSING_GT"}
    masks = load_mask_dir(mask_dir, num_frames, size=gt[0].shape[1::-1])
    raw = load_rgb_dir(raw_dir, num_frames)
    status = "OK" if len(gt) == num_frames and len(masks) == num_frames and len(raw) == num_frames else "MISSING_FRAMES"
    out: dict[str, Any] = {
        "split": split,
        "step": step,
        "sample_id": sid,
        "status": status,
        "frame_count": len(raw),
        "video_path": str(run_root / split / f"step{step}" / "official_generation" / "side_by_side" / f"{sid}.mp4"),
        "evidence_sheet": str(run_root / split / f"step{step}" / f"step{step}_review" / "evidence_sheets" / f"{sid}.jpg"),
        "crop_sheet": str(run_root / split / f"step{step}" / f"step{step}_review" / "crop_sheets" / f"{sid}.jpg"),
        "mask_profile": row.get("mask_profile", ""),
        "area_bucket": row.get("area_bucket", ""),
        "motion_bucket": row.get("motion_bucket", ""),
    }
    if status != "OK":
        return out
    full_psnr: list[float] = []
    mask_psnr: list[float] = []
    boundary_psnr: list[float] = []
    outside_psnr: list[float] = []
    full_ssim: list[float] = []
    mask_ssim: list[float] = []
    boundary_ssim: list[float] = []
    outside_l1: list[float] = []
    black = 0
    for idx in range(num_frames):
        raw_i = resize_like(raw[idx], gt[idx])
        mask = masks[idx]
        if mask.shape != gt[idx].shape[:2]:
            mask = cv2.resize(mask, (gt[idx].shape[1], gt[idx].shape[0]), interpolation=cv2.INTER_NEAREST)
        boundary = boundary_mask(mask)
        outside = (1.0 - np.clip(mask, 0.0, 1.0)).astype(np.float32)
        ssim_map = ssim_map_gray(raw_i, gt[idx])
        full_psnr.append(psnr_region(raw_i, gt[idx]))
        mask_psnr.append(psnr_region(raw_i, gt[idx], mask))
        boundary_psnr.append(psnr_region(raw_i, gt[idx], boundary))
        outside_psnr.append(psnr_region(raw_i, gt[idx], outside))
        full_ssim.append(weighted_mean(ssim_map))
        mask_ssim.append(weighted_mean(ssim_map, mask))
        boundary_ssim.append(weighted_mean(ssim_map, boundary))
        outside_l1.append(l1_region(raw_i, gt[idx], outside))
        if float(raw_i.mean()) < 2.0 or float(raw_i.std()) < 1.0:
            black += 1
    temp_mean, temp_max = temporal_absdiff(raw)
    out.update(
        {
            "full_psnr": mean_float(full_psnr),
            "mask_psnr": mean_float(mask_psnr),
            "boundary_psnr": mean_float(boundary_psnr),
            "outside_psnr": mean_float(outside_psnr),
            "full_ssim": mean_float(full_ssim),
            "mask_ssim": mean_float(mask_ssim),
            "boundary_ssim": mean_float(boundary_ssim),
            "outside_l1": mean_float(outside_l1),
            "temporal_absdiff_mean": temp_mean,
            "temporal_absdiff_max": temp_max,
            "ewarp_proxy": farneback_ewarp_proxy(raw, gt),
            "black_frame_ratio": float(black / max(1, len(raw))),
        }
    )
    return out


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "OK"]
    summary: dict[str, Any] = {"rows": len(rows), "ok_rows": len(ok)}
    for key in HIGHER_IS_BETTER + LOWER_IS_BETTER:
        vals = [float(row[key]) for row in ok if row.get(key) not in ("", None)]
        summary[f"{key}_mean"] = mean_float(vals)
    return summary


def build_pairs(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["split"], int(row["step"]), row["sample_id"]): row for row in metric_rows}
    out: list[dict[str, Any]] = []
    for split in SPLITS:
        sample_ids = sorted({sid for s, _step, sid in by_key if s == split})
        for sample_id in sample_ids:
            cand = by_key.get((split, 2000, sample_id))
            if not cand or cand.get("status") != "OK":
                continue
            for base_step in (0, 50):
                base = by_key.get((split, base_step, sample_id))
                row: dict[str, Any] = {
                    "split": split,
                    "comparison": f"step2000_vs_step{base_step}",
                    "sample_id": sample_id,
                    "candidate_status": cand.get("status", ""),
                    "base_status": base.get("status", "") if base else "MISSING",
                }
                if not base or base.get("status") != "OK":
                    out.append(row)
                    continue
                for key in HIGHER_IS_BETTER + LOWER_IS_BETTER:
                    row[f"{key}_candidate"] = cand.get(key, "")
                    row[f"{key}_base"] = base.get(key, "")
                    row[f"{key}_delta"] = float(cand[key]) - float(base[key])
                full_delta = float(row["full_psnr_delta"])
                mask_delta = float(row["mask_psnr_delta"])
                boundary_delta = float(row["boundary_psnr_delta"])
                ewarp_delta = float(row["ewarp_proxy_delta"])
                row["primary_metric_win"] = bool(
                    full_delta > 0.02
                    or ((mask_delta > 0.02 or boundary_delta > 0.02) and full_delta > -0.02)
                )
                row["temporal_not_worse"] = bool(ewarp_delta <= 0.05)
                out.append(row)
    return out


def aggregate_pairs(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in SPLITS:
        for comparison in ("step2000_vs_step0", "step2000_vs_step50"):
            rows = [row for row in pair_rows if row.get("split") == split and row.get("comparison") == comparison and row.get("base_status") == "OK"]
            item: dict[str, Any] = {"split": split, "comparison": comparison, "rows": len(rows)}
            if not rows:
                out.append(item)
                continue
            item["primary_metric_win_rate"] = float(sum(1 for row in rows if row.get("primary_metric_win") is True) / len(rows))
            item["temporal_not_worse_rate"] = float(sum(1 for row in rows if row.get("temporal_not_worse") is True) / len(rows))
            for key in HIGHER_IS_BETTER + LOWER_IS_BETTER:
                vals = [float(row[f"{key}_delta"]) for row in rows if row.get(f"{key}_delta") not in ("", None)]
                item[f"{key}_delta_mean"] = mean_float(vals)
            out.append(item)
    return out


def metric_status(pair_summary: list[dict[str, Any]]) -> str:
    shadow0 = next((row for row in pair_summary if row["split"] == "shadow" and row["comparison"] == "step2000_vs_step0"), None)
    shadow50 = next((row for row in pair_summary if row["split"] == "shadow" and row["comparison"] == "step2000_vs_step50"), None)
    if not shadow0 or int(shadow0.get("rows", 0)) < 32:
        return "VIDEOPAINTER_2000_METRICS_INCOMPLETE"
    full0 = float(shadow0.get("full_psnr_delta_mean", float("nan")))
    mask0 = float(shadow0.get("mask_psnr_delta_mean", float("nan")))
    boundary0 = float(shadow0.get("boundary_psnr_delta_mean", float("nan")))
    ewarp0 = float(shadow0.get("ewarp_proxy_delta_mean", float("nan")))
    win0 = float(shadow0.get("primary_metric_win_rate", 0.0))
    full50 = float(shadow50.get("full_psnr_delta_mean", float("nan"))) if shadow50 else float("nan")
    if (full0 > 0.02 or ((mask0 > 0.02 or boundary0 > 0.02) and full0 > -0.02)) and ewarp0 <= 0.05 and win0 >= 0.5:
        if not math.isnan(full50) and full50 < -0.02:
            return "VIDEOPAINTER_2000_METRIC_PARETO_MIXED_REVIEW_REQUIRED"
        return "VIDEOPAINTER_2000_METRIC_POSITIVE_REVIEW_REQUIRED"
    if full0 < -0.02 and mask0 <= 0.02 and boundary0 <= 0.02:
        return "VIDEOPAINTER_2000_METRIC_NEGATIVE_REVIEW_REQUIRED"
    return "VIDEOPAINTER_2000_METRIC_PARETO_MIXED_REVIEW_REQUIRED"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--num-frames", type=int, default=49)
    args = parser.parse_args()

    metric_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        manifest = args.run_root / split / "gate64_mask_ready.jsonl"
        rows = read_jsonl(manifest)
        for step in STEPS:
            for row in rows:
                metric_rows.append(compute_sample(split=split, step=step, row=row, run_root=args.run_root, num_frames=args.num_frames))

    fields = [
        "split",
        "step",
        "sample_id",
        "status",
        "frame_count",
        "full_psnr",
        "mask_psnr",
        "boundary_psnr",
        "outside_psnr",
        "full_ssim",
        "mask_ssim",
        "boundary_ssim",
        "outside_l1",
        "temporal_absdiff_mean",
        "temporal_absdiff_max",
        "ewarp_proxy",
        "black_frame_ratio",
        "mask_profile",
        "area_bucket",
        "motion_bucket",
        "video_path",
        "evidence_sheet",
        "crop_sheet",
    ]
    combined_csv = args.reports_dir / "exp31_vp_2000_step0_50_2000_metrics.csv"
    write_csv(combined_csv, metric_rows, fields)
    write_csv(args.reports_dir / "exp31_vp_2000_searchdev_metrics.csv", [row for row in metric_rows if row["split"] == "search"], fields)
    write_csv(args.reports_dir / "exp31_vp_2000_shadowdev_metrics.csv", [row for row in metric_rows if row["split"] == "shadow"], fields)

    pair_rows = build_pairs(metric_rows)
    pair_fields = sorted({key for row in pair_rows for key in row.keys()})
    pair_csv = args.reports_dir / "exp31_vp_2000_step0_50_2000_paired_deltas.csv"
    write_csv(pair_csv, pair_rows, pair_fields)
    pair_summary = aggregate_pairs(pair_rows)
    pair_summary_csv = args.reports_dir / "exp31_vp_2000_step0_50_2000_paired_summary.csv"
    write_csv(pair_summary_csv, pair_summary)

    step_summary: list[dict[str, Any]] = []
    for split in SPLITS:
        for step in STEPS:
            rows = [row for row in metric_rows if row["split"] == split and int(row["step"]) == step]
            step_summary.append({"split": split, "step": step, **summarize_group(rows)})

    status = metric_status(pair_summary)
    final = {
        "status": status,
        "final_scientific_status": "HUMAN_VISUAL_REVIEW_REQUIRED",
        "run_root": str(args.run_root),
        "combined_metrics_csv": str(combined_csv),
        "paired_deltas_csv": str(pair_csv),
        "paired_summary_csv": str(pair_summary_csv),
        "step_summary": step_summary,
        "paired_summary": pair_summary,
        "forbidden_claims": [
            "UNIVERSAL_ADAPTER",
            "FINAL_SOTA",
            "ALL_MODELS_SUPPORTED",
            "TOP_CONFERENCE_NOVELTY_CONFIRMED",
        ],
    }
    json_path = args.reports_dir / "exp31_vp_2000_final_decision.json"
    json_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.reports_dir / "exp31_vp_2000_final_decision.csv", pair_summary)

    md = [
        "# Exp31 VideoPainter 2000-step Metric Summary",
        "",
        f"Metric status: `{status}`",
        "",
        "Final scientific status is not assigned by this script. It requires manual review of all generated videos, evidence sheets, and crop sheets.",
        "",
        f"- run_root: `{args.run_root}`",
        f"- combined_metrics_csv: `{combined_csv}`",
        f"- paired_deltas_csv: `{pair_csv}`",
        f"- paired_summary_csv: `{pair_summary_csv}`",
        "",
        "## Step Summary",
        "",
        "| split | step | rows | ok | full_psnr | mask_psnr | boundary_psnr | outside_psnr | ewarp_proxy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in step_summary:
        md.append(
            f"| {row['split']} | {row['step']} | {row['rows']} | {row['ok_rows']} | "
            f"{row['full_psnr_mean']:.4f} | {row['mask_psnr_mean']:.4f} | "
            f"{row['boundary_psnr_mean']:.4f} | {row['outside_psnr_mean']:.4f} | "
            f"{row['ewarp_proxy_mean']:.4f} |"
        )
    md.extend(
        [
            "",
            "## Paired Summary",
            "",
            "| split | comparison | rows | win_rate | full_psnr_delta | mask_psnr_delta | boundary_psnr_delta | ewarp_delta |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in pair_summary:
        md.append(
            f"| {row['split']} | {row['comparison']} | {row.get('rows', 0)} | "
            f"{float(row.get('primary_metric_win_rate', float('nan'))):.4f} | "
            f"{float(row.get('full_psnr_delta_mean', float('nan'))):.4f} | "
            f"{float(row.get('mask_psnr_delta_mean', float('nan'))):.4f} | "
            f"{float(row.get('boundary_psnr_delta_mean', float('nan'))):.4f} | "
            f"{float(row.get('ewarp_proxy_delta_mean', float('nan'))):.4f} |"
        )
    md.extend(
        [
            "",
            "Forbidden claims remain forbidden: universal adapter, final SOTA, all models supported, and top-conference novelty confirmed.",
        ]
    )
    (args.reports_dir / "exp31_vp_2000_final_decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "json": str(json_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
