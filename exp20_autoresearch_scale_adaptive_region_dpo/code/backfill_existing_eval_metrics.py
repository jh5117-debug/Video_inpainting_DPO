#!/usr/bin/env python3
"""Backfill optional Exp20 metrics from existing hard-comp frame outputs.

This script intentionally does not run DiffuEraser inference. It reads the
already-saved ``diffueraser_comp_frames`` directories produced by the locked
Exp20 evaluator and recomputes the same frame-wise metric protocol with
optional VFID and TC enabled.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import metrics as metric_backend  # noqa: E402
from exp20_autoresearch_scale_adaptive_region_dpo.code.run_exp20_framewise_protocol_eval import (  # noqa: E402
    boundary_mask,
    finite_mean,
    load_gray_masks,
    load_rgb_frames,
    masked_pixel_psnr,
    resize_frames,
    resize_masks,
)


def parse_label_dir(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=/path/to/eval_dir")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("empty label")
    return label, Path(path)


def list_frame_paths(path: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts)


def read_rgb_frames(path: Path, limit: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for frame_path in list_frame_paths(path)[:limit]:
        bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"failed to read frame: {frame_path}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return frames


def finite_values(values: Iterable[float]) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            val = float(value)
        except Exception:
            continue
        if math.isfinite(val):
            out.append(val)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(label: str, rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {"model_label": label, "rows": len(rows)}
    numeric: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key in {"model_label", "video"}:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool) and key not in numeric:
                numeric.append(key)
    for key in numeric:
        summary[f"{key}_mean"] = finite_mean(row[key] for row in rows if key in row)
    return summary


def compute_one(
    label: str,
    eval_dir: Path,
    video_root: Path,
    mask_root: Path,
    gt_root: Path,
    *,
    video_length: int,
    device: str,
    compute_lpips: bool,
    compute_vfid: bool,
    compute_tc: bool,
    compute_ewarp: bool,
    i3d_model: object | None,
    tc_metric: object | None,
    ewarp_metric: object | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    names = sorted(p.name for p in eval_dir.iterdir() if p.is_dir() and (p / "diffueraser_comp_frames").exists())
    if not names:
        raise FileNotFoundError(f"no diffueraser_comp_frames found under {eval_dir}")

    rows: list[dict[str, object]] = []
    ori_acts: list[np.ndarray] = []
    comp_acts: list[np.ndarray] = []
    for name in names:
        comp_frames = read_rgb_frames(eval_dir / name / "diffueraser_comp_frames", video_length)
        gt_frames = resize_frames(load_rgb_frames(gt_root / name, video_length), comp_frames[0].shape[1], comp_frames[0].shape[0])
        masks = resize_masks(load_gray_masks(mask_root / name, video_length), comp_frames[0].shape[1], comp_frames[0].shape[0])
        n = min(len(comp_frames), len(gt_frames), len(masks), video_length)
        comp_frames = comp_frames[:n]
        gt_frames = gt_frames[:n]
        masks = masks[:n]

        psnr_vals: list[float] = []
        ssim_vals: list[float] = []
        strict_mask_vals: list[float] = []
        boundary_vals: list[float] = []
        lpips_vals: list[float] = []
        for gt, comp, mask in zip(gt_frames, comp_frames, masks):
            psnr_vals.append(metric_backend.compute_psnr(gt, comp))
            ssim_vals.append(metric_backend.compute_ssim(gt, comp))
            strict_mask_vals.append(masked_pixel_psnr(gt, comp, mask))
            boundary_vals.append(masked_pixel_psnr(gt, comp, boundary_mask(mask)))
            if compute_lpips:
                lpips_vals.append(float(metric_backend.LPIPSMetric.compute(gt, comp, device=device)))

        row: dict[str, object] = {
            "model_label": label,
            "video": name,
            "frames": n,
            "whole_video_psnr": finite_mean(psnr_vals),
            "whole_video_ssim": finite_mean(ssim_vals),
            "strict_mask_pixel_psnr": finite_mean(strict_mask_vals),
            "boundary_pixel_psnr": finite_mean(boundary_vals),
        }
        if compute_lpips:
            row["whole_video_lpips"] = finite_mean(lpips_vals)
        if compute_tc and tc_metric is not None:
            row["tc"] = float(tc_metric.compute(comp_frames))
        if compute_ewarp and ewarp_metric is not None:
            masks01 = [(m > 0).astype(np.uint8) for m in masks]
            row["ewarp"] = float(ewarp_metric.compute(comp_frames, masks01=masks01, gt_frames_u8_rgb=gt_frames))
        if compute_vfid and i3d_model is not None:
            ori_act, comp_act = metric_backend.calculate_i3d_activations(
                [Image.fromarray(frame.astype(np.uint8)) for frame in gt_frames],
                [Image.fromarray(frame.astype(np.uint8)) for frame in comp_frames],
                i3d_model,
                device,
            )
            ori_acts.append(ori_act)
            comp_acts.append(comp_act)
        rows.append(row)

    summary = summarize(label, rows)
    if compute_vfid and ori_acts and comp_acts:
        summary["vfid"] = float(metric_backend.calculate_vfid(np.vstack(ori_acts), np.vstack(comp_acts)))
    metrics_dir = eval_dir / "metrics"
    write_csv(metrics_dir / "backfill_per_video_metrics.csv", rows)
    write_csv(metrics_dir / "backfill_summary.csv", [summary])
    (metrics_dir / "backfill_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--label-dir", action="append", type=parse_label_dir, required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--video-length", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-lpips", action="store_true")
    parser.add_argument("--compute-vfid", action="store_true")
    parser.add_argument("--compute-tc", action="store_true")
    parser.add_argument("--compute-ewarp", action="store_true")
    parser.add_argument("--i3d-model-path", default=str(metric_backend.DEFAULT_I3D_MODEL))
    parser.add_argument("--tc-model-path", default="")
    parser.add_argument("--raft-model-path", default=str(metric_backend.DEFAULT_RAFT_MODEL))
    args = parser.parse_args()

    i3d_model = None
    if args.compute_vfid:
        i3d_path = Path(args.i3d_model_path)
        if not i3d_path.exists():
            raise FileNotFoundError(f"I3D model missing: {i3d_path}")
        i3d_model = metric_backend.init_i3d_model(str(i3d_path), args.device)

    tc_metric = None
    if args.compute_tc:
        tc_path = Path(args.tc_model_path) if args.tc_model_path else None
        if tc_path and not (tc_path / "open_clip_pytorch_model.bin").exists():
            raise FileNotFoundError(f"TC model missing: {tc_path / 'open_clip_pytorch_model.bin'}")
        tc_metric = metric_backend.TemporalConsistencyMetric(device=args.device, model_path=str(tc_path) if tc_path else None)

    ewarp_metric = None
    if args.compute_ewarp:
        raft_path = Path(args.raft_model_path)
        if not raft_path.exists():
            raise FileNotFoundError(f"RAFT model missing: {raft_path}")
        ewarp_metric = metric_backend.EwarpMetric(device=args.device, raft_model_path=str(raft_path))

    summaries: list[dict[str, object]] = []
    for label, eval_dir in args.label_dir:
        _, summary = compute_one(
            label,
            eval_dir,
            Path(args.video_root),
            Path(args.mask_root),
            Path(args.gt_root),
            video_length=args.video_length,
            device=args.device,
            compute_lpips=args.compute_lpips,
            compute_vfid=args.compute_vfid,
            compute_tc=args.compute_tc,
            compute_ewarp=args.compute_ewarp,
            i3d_model=i3d_model,
            tc_metric=tc_metric,
            ewarp_metric=ewarp_metric,
        )
        summary["eval_dir"] = str(eval_dir)
        summaries.append(summary)

    write_csv(Path(args.output_csv), summaries)
    metric_cols = [
        ("whole_video_psnr_mean", "PSNR"),
        ("whole_video_ssim_mean", "SSIM"),
        ("whole_video_lpips_mean", "LPIPS"),
        ("vfid", "VFID/FVD"),
        ("tc_mean", "TC"),
        ("ewarp_mean", "Ewarp"),
        ("strict_mask_pixel_psnr_mean", "mask PSNR"),
        ("boundary_pixel_psnr_mean", "boundary PSNR"),
    ]
    lines = [
        "# Exp20 First-Wave Full Metrics Backfill",
        "",
        "- source: existing hard-comp frame outputs; no DiffuEraser re-inference",
        f"- video_root: `{args.video_root}`",
        f"- mask_root: `{args.mask_root}`",
        f"- I3D: `{args.i3d_model_path if args.compute_vfid else 'not requested'}`",
        f"- TC model: `{args.tc_model_path if args.compute_tc else 'not requested'}`",
        "",
        "| Method | " + " | ".join(name for _, name in metric_cols) + " | rows |",
        "|---|" + "|".join(["---:"] * (len(metric_cols) + 1)) + "|",
    ]
    for row in summaries:
        vals: list[str] = []
        for key, _ in metric_cols:
            value = row.get(key, "")
            if value == "":
                vals.append("")
                continue
            try:
                vals.append(f"{float(value):.6f}")
            except Exception:
                vals.append(str(value))
        lines.append(f"| {row['model_label']} | " + " | ".join(vals) + f" | {row.get('rows', '')} |")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(summaries), "output_csv": args.output_csv}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
