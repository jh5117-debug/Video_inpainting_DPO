#!/usr/bin/env python3
"""Audit historical BR mask distribution against Exp26 generated masks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
FIELDS = [
    "dataset",
    "sample_id",
    "mask_id",
    "source_video_id",
    "status",
    "frames",
    "area_mean",
    "area_min",
    "area_max",
    "area_std",
    "bbox_w_mean",
    "bbox_h_mean",
    "edge_touch_ratio",
    "centroid_step_motion_mean",
    "centroid_step_motion_max",
    "first_frame_area",
    "error",
]


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def frame_files(path: Path) -> list[Path]:
    return sorted([p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def stat_mask_dir(path: Path, frames: int) -> dict:
    areas: list[float] = []
    bbox_w: list[float] = []
    bbox_h: list[float] = []
    edge_touch: list[float] = []
    centers_x: list[float] = []
    centers_y: list[float] = []
    for fp in frame_files(path)[:frames]:
        mask = np.array(Image.open(fp).convert("L")) > 8
        h, w = mask.shape
        areas.append(float(mask.mean()))
        if mask.any():
            ys, xs = np.where(mask)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            bbox_w.append(float((x1 - x0 + 1) / w))
            bbox_h.append(float((y1 - y0 + 1) / h))
            edge_touch.append(float(x0 == 0 or y0 == 0 or x1 == w - 1 or y1 == h - 1))
            centers_x.append(float(xs.mean() / w))
            centers_y.append(float(ys.mean() / h))
        else:
            bbox_w.append(0.0)
            bbox_h.append(0.0)
            edge_touch.append(0.0)
            centers_x.append(float("nan"))
            centers_y.append(float("nan"))
    motion: list[float] = []
    for i in range(1, len(centers_x)):
        coords = (centers_x[i], centers_x[i - 1], centers_y[i], centers_y[i - 1])
        if not any(math.isnan(v) for v in coords):
            motion.append(math.hypot(centers_x[i] - centers_x[i - 1], centers_y[i] - centers_y[i - 1]))
    if not areas:
        raise ValueError(f"no mask frames under {path}")
    return {
        "frames": len(areas),
        "area_mean": statistics.fmean(areas),
        "area_min": min(areas),
        "area_max": max(areas),
        "area_std": statistics.pstdev(areas) if len(areas) > 1 else 0.0,
        "bbox_w_mean": statistics.fmean(bbox_w),
        "bbox_h_mean": statistics.fmean(bbox_h),
        "edge_touch_ratio": statistics.fmean(edge_touch),
        "centroid_step_motion_mean": statistics.fmean(motion) if motion else 0.0,
        "centroid_step_motion_max": max(motion) if motion else 0.0,
        "first_frame_area": areas[0],
    }


def quantile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    return vals[min(len(vals) - 1, max(0, round(p * (len(vals) - 1))))]


def summarize(rows: list[dict]) -> dict:
    ok_rows = [r for r in rows if r.get("status") == "OK"]
    out: dict[str, object] = {"ok": len(ok_rows), "failed": len(rows) - len(ok_rows)}
    for key in ["area_mean", "edge_touch_ratio", "centroid_step_motion_mean", "first_frame_area", "bbox_w_mean", "bbox_h_mean"]:
        vals = [float(r.get(key, 0.0)) for r in ok_rows]
        out[key] = {
            "mean": statistics.fmean(vals) if vals else 0.0,
            "min": min(vals) if vals else 0.0,
            "p10": quantile(vals, 0.10),
            "p50": quantile(vals, 0.50),
            "p90": quantile(vals, 0.90),
            "max": max(vals) if vals else 0.0,
        }
    return out


def write_outputs(output_root: Path, rows: list[dict], summary: dict, historical_manifest: Path, probe_manifest: Path, sample_limit: int) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "exp26_br_mask_distribution_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in FIELDS} for row in rows])
    (output_root / "exp26_br_mask_distribution_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Exp26 BR Mask Distribution Audit",
        "",
        f"- historical manifest: `{historical_manifest}`",
        f"- probe manifest: `{probe_manifest}`",
        f"- historical sample limit: {sample_limit}",
        "",
        "## Summary",
    ]
    for name, stats in summary.items():
        lines.append(f"### {name}")
        lines.append(f"- ok={stats['ok']} failed={stats['failed']}")
        for key, val in stats.items():
            if isinstance(val, dict):
                lines.append(
                    f"- {key}: mean={val['mean']:.6f} p10={val['p10']:.6f} "
                    f"p50={val['p50']:.6f} p90={val['p90']:.6f}"
                )
        lines.append("")
    lines += [
        "## Decision",
        "Probe4 masks are ellipse-only and valid for plumbing, but Gate16/Gate64 "
        "should use a mixed BR mask protocol calibrated from historical area, "
        "edge-touch, and motion buckets.",
    ]
    (output_root / "exp26_br_mask_distribution_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--probe-mask-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--historical-limit", type=int, default=512)
    parser.add_argument("--historical-frames", type=int, default=16)
    parser.add_argument("--probe-frames", type=int, default=49)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict] = []
    for idx, row in enumerate(read_jsonl(args.historical_manifest)):
        if args.historical_limit > 0 and idx >= args.historical_limit:
            break
        out = {
            "dataset": f"historical_br_youtubevos_k4_limit{args.historical_limit}",
            "sample_id": row.get("sample_id", ""),
            "mask_id": row.get("mask_id", ""),
            "source_video_id": row.get("source_video_id", ""),
            "error": "",
        }
        try:
            out.update({"status": "OK", **stat_mask_dir(Path(row["mask_path"]), args.historical_frames)})
        except Exception as exc:  # noqa: BLE001
            out.update({"status": "FAILED", "error": repr(exc)})
        rows.append(out)

    probe_rows: list[dict] = []
    for row in read_jsonl(args.probe_mask_manifest):
        out = {
            "dataset": "exp26_probe4_generated_ellipse",
            "sample_id": row.get("sample_id", ""),
            "mask_id": "probe4",
            "source_video_id": row.get("video_id", ""),
            "error": "",
        }
        try:
            out.update({"status": "OK", **stat_mask_dir(Path(row["mask_dir"]), args.probe_frames)})
        except Exception as exc:  # noqa: BLE001
            out.update({"status": "FAILED", "error": repr(exc)})
        probe_rows.append(out)

    summary = {
        f"historical_br_youtubevos_k4_limit{args.historical_limit}": summarize(rows),
        "exp26_probe4_generated_ellipse": summarize(probe_rows),
    }
    write_outputs(args.output_root, rows + probe_rows, summary, args.historical_manifest, args.probe_mask_manifest, args.historical_limit)
    print(json.dumps({"output_root": str(args.output_root), "historical_rows": len(rows), "probe_rows": len(probe_rows), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
