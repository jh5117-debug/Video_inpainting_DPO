#!/usr/bin/env python3
"""Calibrate Exp20 adaptive image-space boundary radii from a training manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np


def mask_files(mask_dir: Path, limit: int) -> list[Path]:
    files = sorted(
        p for p in mask_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    return files[:limit]


def clip_stats(mask_dir: Path, num_frames: int) -> dict[str, float | int | str]:
    areas: list[float] = []
    perimeters: list[float] = []
    ap_values: list[float] = []
    sqrt_values: list[float] = []
    empty_frames = 0
    files = mask_files(mask_dir, num_frames)
    for path in files:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        mask = (img > 127).astype(np.uint8)
        area = float(mask.sum())
        if area <= 0:
            empty_frames += 1
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
        perimeter = max(perimeter, 1.0)
        areas.append(area)
        perimeters.append(perimeter)
        ap_values.append(area / perimeter)
        sqrt_values.append(math.sqrt(area / math.pi))
    if not ap_values:
        return {
            "n_frames": len(files),
            "empty_frames": empty_frames,
            "area_median": 0.0,
            "perimeter_median": 0.0,
            "ap_median": 0.0,
            "sqrt_area_median": 0.0,
            "ap_mean": 0.0,
            "sqrt_area_mean": 0.0,
        }
    return {
        "n_frames": len(files),
        "empty_frames": empty_frames,
        "area_median": float(np.median(areas)),
        "perimeter_median": float(np.median(perimeters)),
        "ap_median": float(np.median(ap_values)),
        "sqrt_area_median": float(np.median(sqrt_values)),
        "ap_mean": float(np.mean(ap_values)),
        "sqrt_area_mean": float(np.mean(sqrt_values)),
    }


def distribution(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-csv", default="reports/exp20_adaptive_radius_distribution.csv")
    parser.add_argument("--output-md", default="reports/exp20_adaptive_radius_calibration.md")
    parser.add_argument(
        "--candidate-csv", default="reports/exp20_adaptive_radius_calibration_candidates.csv"
    )
    parser.add_argument("--targets", nargs="+", type=float, default=[12, 16, 20, 24])
    parser.add_argument("--r-min", type=float, default=2.0)
    parser.add_argument("--r-max", type=float, default=48.0)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    start = time.time()
    rows: list[dict[str, object]] = []
    with manifest.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            rec = json.loads(line)
            mask_dir = Path(rec["mask_path"])
            stats = clip_stats(mask_dir, int(rec.get("num_frames", 16)))
            rows.append(
                {
                    "row_index": index,
                    "sample_id": rec.get("sample_id", ""),
                    "mask_id": rec.get("mask_id", ""),
                    "mask_path": str(mask_dir),
                    **stats,
                }
            )

    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    candidate_csv = Path(args.candidate_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "row_index",
        "sample_id",
        "mask_id",
        "mask_path",
        "n_frames",
        "empty_frames",
        "area_median",
        "perimeter_median",
        "ap_median",
        "sqrt_area_median",
        "ap_mean",
        "sqrt_area_mean",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    valid = [r for r in rows if float(r["ap_median"]) > 0 and float(r["sqrt_area_median"]) > 0]
    ap_values = [float(r["ap_median"]) for r in valid]
    sqrt_values = [float(r["sqrt_area_median"]) for r in valid]
    base_stats = {"ap": distribution(ap_values), "sqrt": distribution(sqrt_values)}

    candidates: list[dict[str, object]] = []
    for mode, key, values in [
        ("adaptive_area_perimeter", "ap", ap_values),
        ("adaptive_sqrt_area", "sqrt", sqrt_values),
    ]:
        base = np.asarray(values, dtype=np.float64)
        base_median = float(np.median(base))
        for target in args.targets:
            k = float(target / base_median) if base_median > 0 else 0.0
            radius = np.clip(k * base, args.r_min, args.r_max)
            clamp_min = float((radius <= args.r_min + 1e-6).mean())
            clamp_max = float((radius >= args.r_max - 1e-6).mean())
            radius_median = float(np.median(radius))
            valid_candidate = (
                clamp_min <= 0.30
                and clamp_max <= 0.30
                and abs(radius_median - target) / max(target, 1e-6) <= 0.20
                and float(radius.std()) > 1e-6
            )
            candidates.append(
                {
                    "mode": mode,
                    "target_median": target,
                    "k": k,
                    "radius_mean": float(radius.mean()),
                    "radius_median": radius_median,
                    "radius_std": float(radius.std()),
                    "radius_p10": float(np.percentile(radius, 10)),
                    "radius_p90": float(np.percentile(radius, 90)),
                    "clamp_min_ratio": clamp_min,
                    "clamp_max_ratio": clamp_max,
                    "valid": valid_candidate,
                }
            )

    with candidate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0].keys()))
        writer.writeheader()
        writer.writerows(candidates)

    lines = [
        "# Exp20 Adaptive Radius Calibration",
        "",
        f"- Manifest: `{manifest}`",
        f"- Clips scanned: `{len(rows)}`",
        f"- Valid non-empty clips: `{len(valid)}`",
        f"- Runtime seconds: `{time.time() - start:.1f}`",
        "- Mask convention: `png_255_inpaint_region_0_keep_region`",
        "",
        "## Base Radius Statistics",
        "",
        "| base | mean | median | std | p10 | p25 | p75 | p90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in base_stats.items():
        lines.append(
            f"| {name} | {stats['mean']:.4f} | {stats['median']:.4f} | "
            f"{stats['std']:.4f} | {stats['p10']:.4f} | {stats['p25']:.4f} | "
            f"{stats['p75']:.4f} | {stats['p90']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Calibrated k Candidates",
            "",
            "| mode | target median | k | radius mean | radius median | std | p10 | p90 | clamp min | clamp max | valid |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in candidates:
        lines.append(
            f"| {row['mode']} | {row['target_median']} | {row['k']:.6f} | "
            f"{row['radius_mean']:.3f} | {row['radius_median']:.3f} | "
            f"{row['radius_std']:.3f} | {row['radius_p10']:.3f} | "
            f"{row['radius_p90']:.3f} | {row['clamp_min_ratio']:.3f} | "
            f"{row['clamp_max_ratio']:.3f} | {row['valid']} |"
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "rows": len(rows), "valid": len(valid)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
