#!/usr/bin/env python3
"""Evaluate already generated Exp33 EffectErase VOR-Eval official81 outputs.

This script is intentionally read-only with respect to model inference.  It
loads the fixed held-out manifest and the completed raw outputs, computes
per-video objective signals, and writes visual review evidence for the baseline
summary.
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
from PIL import Image, ImageDraw


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


def video_frames(path: Path, expected: int, size: tuple[int, int]) -> tuple[list[np.ndarray], dict[str, Any]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], {"opens": False, "frames": 0, "width": 0, "height": 0, "fps": 0.0}
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    while len(frames) < expected:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if (rgb.shape[1], rgb.shape[0]) != size:
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_CUBIC)
        frames.append(rgb)
    cap.release()
    return frames, {"opens": True, "frames": len(frames), "width": width, "height": height, "fps": fps}


def mask_frames(path: Path, expected: int, size: tuple[int, int]) -> tuple[list[np.ndarray], dict[str, Any]]:
    frames, stats = video_frames(path, expected, size)
    masks: list[np.ndarray] = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        masks.append((gray > 127).astype(np.float32))
    stats["non_empty_frames"] = int(sum(float(mask.sum()) > 1.0 for mask in masks))
    return masks, stats


def psnr_region(pred: np.ndarray, gt: np.ndarray, weight: np.ndarray | None = None) -> float:
    pred_f = pred.astype(np.float32)
    gt_f = gt.astype(np.float32)
    if weight is None:
        mse = float(np.mean((pred_f - gt_f) ** 2))
    else:
        w = weight.astype(np.float32)
        denom = float(w.sum() * 3.0)
        if denom < 1:
            return float("nan")
        mse = float((((pred_f - gt_f) ** 2) * w[..., None]).sum() / denom)
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def ssim_map_gray(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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
    if denom < 1:
        return float("nan")
    return float((values * weight).sum() / denom)


def l1_region(pred: np.ndarray, gt: np.ndarray, weight: np.ndarray | None = None) -> float:
    err = np.abs(pred.astype(np.float32) - gt.astype(np.float32)).mean(axis=2)
    return weighted_mean(err, weight)


def boundary_mask(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    binary = (mask > 0.5).astype(np.uint8)
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
    dilated = cv2.dilate(binary, kernel)
    eroded = cv2.erode(binary, kernel)
    return ((dilated - eroded) > 0).astype(np.float32)


def temporal_absdiff(frames: list[np.ndarray]) -> tuple[float, float]:
    vals = []
    for idx in range(1, len(frames)):
        vals.append(float(np.abs(frames[idx].astype(np.float32) - frames[idx - 1].astype(np.float32)).mean()))
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.max(vals))


def farneback_ewarp_proxy(pred: list[np.ndarray], gt: list[np.ndarray], sample_every: int = 8) -> float:
    vals = []
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


def try_lpips(device: str):
    try:
        import lpips  # type: ignore
        import torch

        model = lpips.LPIPS(net="alex").to(device)
        model.eval()
        return model, torch
    except Exception as exc:  # pragma: no cover - environment dependent
        return str(exc), None


def lpips_sampled(
    model: Any,
    torch_mod: Any,
    pred: list[np.ndarray],
    gt: list[np.ndarray],
    *,
    device: str,
    frames: int,
    size: int,
) -> tuple[float, str]:
    if torch_mod is None or not pred or not gt:
        return float("nan"), "unavailable"
    n = min(len(pred), len(gt))
    idxs = np.linspace(0, n - 1, num=min(frames, n), dtype=int).tolist()
    vals = []
    with torch_mod.no_grad():
        for idx in idxs:
            a = cv2.resize(pred[idx], (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 127.5 - 1.0
            b = cv2.resize(gt[idx], (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 127.5 - 1.0
            ta = torch_mod.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).to(device)
            tb = torch_mod.from_numpy(b.transpose(2, 0, 1)).unsqueeze(0).to(device)
            vals.append(float(model(ta, tb).item()))
    return float(np.mean(vals)), f"sampled_{len(idxs)}_frames_{size}px"


def add_label(arr: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, min(img.width, 520), 24], fill=(0, 0, 0))
    draw.text((6, 6), text, fill=(255, 255, 255))
    return np.asarray(img)


def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = frame.astype(np.float32).copy()
    red = np.zeros_like(out)
    red[..., 0] = 255
    alpha = (mask > 0.5).astype(np.float32)[..., None] * 0.45
    return np.clip(out * (1 - alpha) + red * alpha, 0, 255).astype(np.uint8)


def error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    err = np.abs(pred.astype(np.float32) - gt.astype(np.float32)).mean(axis=2)
    return cv2.cvtColor(cv2.applyColorMap(np.clip(err * 3.0, 0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)


def crop_mask(arr: np.ndarray, mask: np.ndarray, pad: int = 32) -> np.ndarray:
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return arr
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, arr.shape[1])
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, arr.shape[0])
    return cv2.resize(arr[y0:y1, x0:x1], (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_CUBIC)


def make_review_sheet(
    sample_id: str,
    condition: list[np.ndarray],
    winner: list[np.ndarray],
    mask: list[np.ndarray],
    output: list[np.ndarray],
    out_path: Path,
    crop_path: Path,
) -> str:
    n = min(len(condition), len(winner), len(mask), len(output))
    picks = sorted(set(int(x) for x in np.linspace(0, n - 1, num=min(6, n)))) if n else []
    rows = []
    crop_rows = []
    for idx in picks:
        err = error_map(output[idx], winner[idx])
        overlay = overlay_mask(winner[idx], mask[idx])
        rows.append(
            np.concatenate(
                [
                    add_label(condition[idx], f"condition f{idx}"),
                    add_label(overlay, "winner+mask"),
                    add_label(winner[idx], "winner BG"),
                    add_label(output[idx], "EffectErase raw"),
                    add_label(err, "abs error"),
                ],
                axis=1,
            )
        )
        crop_rows.append(
            np.concatenate(
                [
                    add_label(crop_mask(condition[idx], mask[idx]), f"condition crop f{idx}"),
                    add_label(crop_mask(overlay, mask[idx]), "mask crop"),
                    add_label(crop_mask(winner[idx], mask[idx]), "winner crop"),
                    add_label(crop_mask(output[idx], mask[idx]), "raw crop"),
                    add_label(crop_mask(err, mask[idx]), "error crop"),
                ],
                axis=1,
            )
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        Image.fromarray(np.concatenate(rows, axis=0)).save(out_path, quality=92)
        Image.fromarray(np.concatenate(crop_rows, axis=0)).save(crop_path, quality=92)
    return " ".join(str(x) for x in picks)


def classify(row: dict[str, Any]) -> tuple[str, str]:
    if row["status"] != "OK":
        return "TECHNICAL_INVALID", row["error"]
    if row["black_frame_ratio"] > 0.0:
        return "TECHNICAL_INVALID", "black_or_constant_output_frame"
    if row["full_psnr"] >= 28.0 and row["mask_psnr"] >= 22.0 and row["outside_l1"] <= 4.0:
        return "BASELINE_STRONG", "high full/mask fidelity and low outside damage"
    if row["full_psnr"] >= 24.0 and row["mask_psnr"] >= 17.0 and row["outside_l1"] <= 8.0:
        return "BASELINE_USABLE", "finite local quality with bounded outside damage"
    if row["full_psnr"] < 18.0 or row["mask_psnr"] < 10.0 or row["outside_l1"] > 18.0:
        return "BASELINE_WEAK", "low fidelity or large outside damage"
    return "BASELINE_MIXED", "usable on some signals but not clearly strong"


def evaluate_row(row: dict[str, Any], args: argparse.Namespace, lpips_state: tuple[Any, Any]) -> dict[str, Any]:
    sid = str(row["sample_id"])
    size = (args.width, args.height)
    condition, condition_stats = video_frames(Path(row["condition_path"]), args.num_frames, size)
    winner, winner_stats = video_frames(Path(row["winner_path"]), args.num_frames, size)
    output, output_stats = video_frames(Path(row["output_path"]), args.num_frames, size)
    masks, mask_stats = mask_frames(Path(row["mask_path"]), args.num_frames, size)
    base: dict[str, Any] = {
        "sample_id": sid,
        "source_type": row.get("source_type", ""),
        "scene_group": row.get("scene_group", ""),
        "mask_bucket": row.get("mask_bucket", ""),
        "output_path": row.get("output_path", ""),
        "condition_path": row.get("condition_path", ""),
        "winner_path": row.get("winner_path", ""),
        "mask_path": row.get("mask_path", ""),
        "condition_frames": condition_stats["frames"],
        "winner_frames": winner_stats["frames"],
        "output_frames": output_stats["frames"],
        "mask_frames": mask_stats["frames"],
        "output_resolution": f"{output_stats['width']}x{output_stats['height']}",
        "mask_non_empty_frames": mask_stats.get("non_empty_frames", 0),
    }
    errors = []
    for label, stats in (("condition", condition_stats), ("winner", winner_stats), ("output", output_stats), ("mask", mask_stats)):
        if not stats["opens"]:
            errors.append(f"{label}_does_not_open")
        if stats["frames"] != args.num_frames:
            errors.append(f"{label}_frames_{stats['frames']}_not_{args.num_frames}")
    if errors:
        base.update({"status": "ERROR", "error": ";".join(errors)})
        klass, reason = classify(base)
        base.update({"classification": klass, "classification_reason": reason})
        return base

    full_psnr = []
    full_ssim = []
    mask_psnr = []
    mask_ssim = []
    boundary_psnr = []
    boundary_ssim = []
    outside_psnr = []
    outside_l1 = []
    mask_l1 = []
    residual_to_condition_l1 = []
    for pred, gt, cond, mask in zip(output, winner, condition, masks):
        bmask = boundary_mask(mask)
        outside = 1.0 - (mask > 0.5).astype(np.float32)
        smap = ssim_map_gray(pred, gt)
        full_psnr.append(psnr_region(pred, gt))
        full_ssim.append(weighted_mean(smap))
        mask_psnr.append(psnr_region(pred, gt, mask))
        mask_ssim.append(weighted_mean(smap, mask))
        boundary_psnr.append(psnr_region(pred, gt, bmask))
        boundary_ssim.append(weighted_mean(smap, bmask))
        outside_psnr.append(psnr_region(pred, gt, outside))
        outside_l1.append(l1_region(pred, gt, outside))
        mask_l1.append(l1_region(pred, gt, mask))
        residual_to_condition_l1.append(l1_region(pred, cond, mask))
    pred_tc_mean, pred_tc_max = temporal_absdiff(output)
    gt_tc_mean, _ = temporal_absdiff(winner)
    lpips_model, torch_mod = lpips_state
    lpips_value, lpips_status = lpips_sampled(
        lpips_model,
        torch_mod,
        output,
        winner,
        device=args.lpips_device,
        frames=args.lpips_frames,
        size=args.lpips_size,
    )
    black_ratio = float(sum(1 for frame in output if float(frame.mean()) < 2.0 or float(frame.std()) < 1.0) / len(output))
    ewarp = farneback_ewarp_proxy(output, winner, sample_every=args.ewarp_sample_every)
    review_sheet = args.assets_dir / "review_sheets" / f"{sid}.jpg"
    crop_sheet = args.assets_dir / "crop_sheets" / f"{sid}.jpg"
    reviewed = make_review_sheet(sid, condition, winner, masks, output, review_sheet, crop_sheet)
    base.update(
        {
            "status": "OK",
            "error": "",
            "full_psnr": float(np.nanmean(full_psnr)),
            "full_ssim": float(np.nanmean(full_ssim)),
            "mask_psnr": float(np.nanmean(mask_psnr)),
            "mask_ssim": float(np.nanmean(mask_ssim)),
            "boundary_psnr": float(np.nanmean(boundary_psnr)),
            "boundary_ssim": float(np.nanmean(boundary_ssim)),
            "outside_psnr": float(np.nanmean(outside_psnr)),
            "outside_l1": float(np.nanmean(outside_l1)),
            "mask_l1": float(np.nanmean(mask_l1)),
            "mask_residual_to_condition_l1": float(np.nanmean(residual_to_condition_l1)),
            "lpips": lpips_value,
            "lpips_status": lpips_status,
            "tc_absdiff_mean": pred_tc_mean,
            "tc_absdiff_max": pred_tc_max,
            "tc_absdiff_over_winner": pred_tc_mean - gt_tc_mean,
            "ewarp_proxy": ewarp,
            "black_frame_ratio": black_ratio,
            "reviewed_frames": reviewed,
            "review_sheet": str(review_sheet),
            "crop_sheet": str(crop_sheet),
        }
    )
    klass, reason = classify(base)
    base.update({"classification": klass, "classification_reason": reason})
    return base


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "OK"]
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("classification", "UNKNOWN"))
        counts[key] = counts.get(key, 0) + 1
    metrics = ["full_psnr", "full_ssim", "mask_psnr", "mask_ssim", "boundary_psnr", "outside_l1", "lpips", "tc_absdiff_over_winner", "ewarp_proxy"]
    means = {}
    for metric in metrics:
        vals = [float(r[metric]) for r in ok if r.get(metric, "") != "" and not math.isnan(float(r[metric]))]
        means[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
        means[f"{metric}_median"] = float(np.median(vals)) if vals else float("nan")
    strong_or_usable = counts.get("BASELINE_STRONG", 0) + counts.get("BASELINE_USABLE", 0)
    if len(ok) != len(rows):
        status = "EXP33_EFFECTERASE_BASELINE_TECHNICAL_INVALID"
    elif strong_or_usable >= math.ceil(0.7 * len(rows)) and counts.get("BASELINE_WEAK", 0) == 0:
        status = "EXP33_EFFECTERASE_BASELINE_USABLE"
    elif counts.get("BASELINE_WEAK", 0) > len(rows) * 0.35:
        status = "EXP33_EFFECTERASE_BASELINE_WEAK"
    else:
        status = "EXP33_EFFECTERASE_BASELINE_MIXED"
    return {"status": status, "rows": len(rows), "ok_rows": len(ok), "classification_counts": counts, **means}


def write_reports(args: argparse.Namespace, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = args.reports_dir / "exp33_effecterase_vor_eval_official81_metrics.csv"
    visual_csv = args.reports_dir / "exp33_effecterase_vor_eval_official81_visual_review.csv"
    json_path = args.reports_dir / "exp33_effecterase_vor_eval_official81_metrics_summary.json"
    fields = [
        "sample_id",
        "source_type",
        "scene_group",
        "mask_bucket",
        "status",
        "classification",
        "classification_reason",
        "full_psnr",
        "full_ssim",
        "mask_psnr",
        "mask_ssim",
        "boundary_psnr",
        "boundary_ssim",
        "outside_psnr",
        "outside_l1",
        "mask_l1",
        "mask_residual_to_condition_l1",
        "lpips",
        "lpips_status",
        "tc_absdiff_mean",
        "tc_absdiff_max",
        "tc_absdiff_over_winner",
        "ewarp_proxy",
        "black_frame_ratio",
        "output_frames",
        "output_resolution",
        "mask_non_empty_frames",
        "output_path",
        "review_sheet",
        "crop_sheet",
        "error",
    ]
    write_csv(metrics_csv, rows, fields)
    review_fields = [
        "sample_id",
        "reviewed_frames",
        "classification",
        "classification_reason",
        "mask_psnr",
        "outside_l1",
        "tc_absdiff_over_winner",
        "black_frame_ratio",
        "review_sheet",
        "crop_sheet",
        "output_path",
    ]
    write_csv(visual_csv, rows, review_fields)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_lines = [
        "# Exp33 EffectErase VOR-Eval Official81 Metrics Summary",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"- Manifest: `{args.manifest}`",
        f"- Rows: {summary['rows']}",
        f"- OK rows: {summary['ok_rows']}",
        f"- Metrics CSV: `{metrics_csv}`",
        f"- Visual review CSV: `{visual_csv}`",
        f"- Review sheets: `{args.assets_dir / 'review_sheets'}`",
        f"- Crop sheets: `{args.assets_dir / 'crop_sheets'}`",
        "",
        "## Classification Counts",
        "",
    ]
    for key, value in sorted(summary["classification_counts"].items()):
        md_lines.append(f"- `{key}`: {value}")
    md_lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "",
            f"- full_psnr_mean: {summary['full_psnr_mean']:.4f}",
            f"- mask_psnr_mean: {summary['mask_psnr_mean']:.4f}",
            f"- boundary_psnr_mean: {summary['boundary_psnr_mean']:.4f}",
            f"- outside_l1_mean: {summary['outside_l1_mean']:.4f}",
            f"- lpips_mean: {summary['lpips_mean']:.6f}",
            f"- tc_absdiff_over_winner_mean: {summary['tc_absdiff_over_winner_mean']:.4f}",
            f"- ewarp_proxy_mean: {summary['ewarp_proxy_mean']:.4f}",
            "",
            "LPIPS is computed on an evenly sampled frame subset and resized inputs; it is reported as a baseline signal, not as a training target.",
            "This baseline remains held-out evaluation evidence only and is not used for adapter training or loser mining.",
        ]
    )
    (args.reports_dir / "exp33_effecterase_vor_eval_official81_metrics_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (args.reports_dir / "exp33_effecterase_vor_eval_official81_visual_review.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    final_lines = [
        "# Exp33 EffectErase VOR-Eval Official81 Baseline Report",
        "",
        f"Baseline status: `{summary['status']}`",
        "",
        "EffectErase inference had already completed before this evaluation. This report only evaluates the 43 held-out raw outputs.",
        "No EffectErase adapter, MiniMax job, VideoPainter training, DiffuEraser VOR-OR training, RC-FPO, or universal-adapter claim is launched or made here.",
        "",
        "Scientific use: baseline evidence pack for paper positioning. Promotion claims require metric and per-video visual evidence; this report does not claim final SOTA or top-conference novelty.",
        "",
        f"- Metrics summary: `{args.reports_dir / 'exp33_effecterase_vor_eval_official81_metrics_summary.md'}`",
        f"- Per-video metrics: `{metrics_csv}`",
        f"- Visual review: `{visual_csv}`",
    ]
    (args.reports_dir / "exp33_effecterase_vor_eval_official81_final_report.md").write_text("\n".join(final_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--lpips-device", default="cpu")
    parser.add_argument("--lpips-frames", type=int, default=9)
    parser.add_argument("--lpips-size", type=int, default=256)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--ewarp-sample-every", type=int, default=8)
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    if args.max_rows:
        rows = rows[: args.max_rows]
    if args.skip_lpips:
        lpips_state = ("skipped", None)
    else:
        lpips_state = try_lpips(args.lpips_device)
        if isinstance(lpips_state[0], str):
            print(f"LPIPS unavailable: {lpips_state[0]}")
    out_rows = [evaluate_row(row, args, lpips_state) for row in rows]
    summary = summarize(out_rows)
    write_reports(args, out_rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ok_rows"] == summary["rows"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
