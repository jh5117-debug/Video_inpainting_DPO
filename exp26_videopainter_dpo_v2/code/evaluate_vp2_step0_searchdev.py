#!/usr/bin/env python3
"""Evaluate locked Exp26 VideoPainter search-dev outputs.

This script is intentionally a thin orchestrator. It builds pair manifests for
the already-generated raw and hard-comp outputs, calls the existing
``tools/run_inpainting_metric_eval.py`` backend, and writes Exp26 reports.
It does not regenerate videos and it does not implement metric math.

The default ``--eval-name step0`` preserves the original step0 report paths.
Other checkpoints, such as step1/step10, use the same evaluator by passing a
different eval name and an explicit locked search-dev manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def metric_summary(path: Path) -> dict[str, str]:
    rows = read_csv(path)
    if not rows:
        return {}
    return rows[0]


def first_metric(summary: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = summary.get(key)
        if value not in (None, ""):
            return value
    return ""


def build_pair_rows(mask_manifest: Path, run_root: Path, model_prefix: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    rows = read_jsonl(mask_manifest)
    raw_pairs: list[dict[str, Any]] = []
    comp_pairs: list[dict[str, Any]] = []
    issues: list[str] = []
    for row in rows:
        sid = row["sample_id"]
        gt = Path(row["frame_dir"])
        mask = Path(row["mask_dir"])
        raw = run_root / "official_generation" / "raw_frames" / sid
        comp = run_root / "official_generation" / "comp_frames" / sid
        for name, path in (("gt", gt), ("mask", mask), ("raw", raw), ("comp", comp)):
            if not path.exists():
                issues.append(f"{sid}: missing {name} path {path}")
        base = {
            "sample_id": sid,
            "gt_video_path": str(gt),
            "mask_path": str(mask),
            "source_dataset": row.get("source_dataset", ""),
            "source_sample_id": row.get("source_sample_id", ""),
            "scene_group": row.get("scene_group", ""),
            "mask_profile": row.get("mask_profile", ""),
            "area_bucket": row.get("area_bucket", ""),
            "motion_bucket": row.get("motion_bucket", ""),
        }
        raw_pairs.append({**base, "model_label": f"{model_prefix}_raw", "prediction_video_path": str(raw)})
        comp_pairs.append({**base, "model_label": f"{model_prefix}_comp", "prediction_video_path": str(comp)})
    return raw_pairs, comp_pairs, issues


def run_metrics(
    *,
    project_root: Path,
    pair_manifest: Path,
    output_dir: Path,
    max_frames: int,
    width: int,
    height: int,
    boundary_pixels: int,
    device: str,
    compute_lpips: bool,
    compute_ewarp: bool,
) -> None:
    cmd = [
        sys.executable,
        str(project_root / "tools" / "run_inpainting_metric_eval.py"),
        "--pair_manifest",
        str(pair_manifest),
        "--output_dir",
        str(output_dir),
        "--max_frames",
        str(max_frames),
        "--width",
        str(width),
        "--height",
        str(height),
        "--boundary_pixels",
        str(boundary_pixels),
        "--device",
        device,
        "--strict_missing",
    ]
    if compute_lpips:
        cmd.append("--compute_lpips")
    if compute_ewarp:
        cmd.append("--compute_ewarp")
    subprocess.run(cmd, check=True, cwd=str(project_root))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, default=None)
    parser.add_argument("--mask-manifest", type=Path, default=None)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--boundary-pixels", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-lpips", action="store_true")
    parser.add_argument("--compute-ewarp", action="store_true")
    parser.add_argument("--reuse-existing-metrics", action="store_true")
    parser.add_argument("--eval-name", default="step0")
    parser.add_argument("--model-prefix", default=None)
    parser.add_argument("--report-title", default=None)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    project_root = args.project_root.resolve()
    report_dir = args.report_dir.resolve()
    eval_name = args.eval_name
    model_prefix = args.model_prefix or f"VideoPainter_{eval_name}"
    review_dir = args.review_dir or (run_root / f"{eval_name}_review")
    mask_manifest = args.mask_manifest or (run_root / "gate64_mask_ready.jsonl")
    eval_dir = run_root / f"{eval_name}_metric_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    raw_pairs, comp_pairs, issues = build_pair_rows(mask_manifest, run_root, model_prefix)
    raw_manifest = eval_dir / f"{eval_name}_raw_metric_pairs.jsonl"
    comp_manifest = eval_dir / f"{eval_name}_comp_metric_pairs.jsonl"
    write_jsonl(raw_manifest, raw_pairs)
    write_jsonl(comp_manifest, comp_pairs)
    (eval_dir / "pair_manifest_issues.json").write_text(json.dumps(issues, indent=2) + "\n", encoding="utf-8")
    if issues:
        raise FileNotFoundError("\n".join(issues[:20]))

    raw_summary_path = eval_dir / "raw_metrics" / "metrics" / "summary.csv"
    comp_summary_path = eval_dir / "comp_metrics" / "metrics" / "summary.csv"
    if not (args.reuse_existing_metrics and raw_summary_path.exists()):
        run_metrics(
            project_root=project_root,
            pair_manifest=raw_manifest,
            output_dir=eval_dir / "raw_metrics",
            max_frames=args.num_frames,
            width=args.width,
            height=args.height,
            boundary_pixels=args.boundary_pixels,
            device=args.device,
            compute_lpips=args.compute_lpips,
            compute_ewarp=args.compute_ewarp,
        )
    if not (args.reuse_existing_metrics and comp_summary_path.exists()):
        run_metrics(
            project_root=project_root,
            pair_manifest=comp_manifest,
            output_dir=eval_dir / "comp_metrics",
            max_frames=args.num_frames,
            width=args.width,
            height=args.height,
            boundary_pixels=args.boundary_pixels,
            device=args.device,
            compute_lpips=args.compute_lpips,
            compute_ewarp=args.compute_ewarp,
        )

    raw_summary = metric_summary(raw_summary_path)
    comp_summary = metric_summary(comp_summary_path)
    review_summary_path = review_dir / "gate64_visual_review_summary.json"
    review_csv_path = review_dir / "gate64_visual_review.csv"
    review_summary = json.loads(review_summary_path.read_text(encoding="utf-8")) if review_summary_path.exists() else {}
    review_rows = read_csv(review_csv_path)

    final_rows: list[dict[str, Any]] = []
    for label, summary in (("raw", raw_summary), ("comp", comp_summary)):
        row = {"variant": label, **summary}
        final_rows.append(row)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_stem = f"exp26_vp_{eval_name}_baseline"
    write_csv(report_dir / f"{report_stem}.csv", final_rows)
    if review_csv_path.exists():
        (report_dir / f"exp26_vp_{eval_name}_visual_review.csv").write_text(review_csv_path.read_text(encoding="utf-8"), encoding="utf-8")

    md = [
        f"# {args.report_title or f'Exp26 VideoPainter {eval_name} Search-Dev Evaluation'}",
        "",
        f"- run_root: `{run_root}`",
        f"- mask_manifest: `{mask_manifest}`",
        f"- raw_pair_manifest: `{raw_manifest}`",
        f"- comp_pair_manifest: `{comp_manifest}`",
        f"- review_dir: `{review_dir}`",
        f"- review_status: `{review_summary.get('status', 'missing')}`",
        f"- review_samples: `{review_summary.get('num_samples', len(review_rows))}`",
        "",
        "## Aggregate Metrics",
        "",
        "| variant | PSNR | SSIM | LPIPS | Ewarp | mask PSNR | boundary PSNR | status rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in (("raw", raw_summary), ("comp", comp_summary)):
        md.append(
            "| {label} | {psnr} | {ssim} | {lpips} | {ewarp} | {mask} | {boundary} | {rows} |".format(
                label=label,
                psnr=first_metric(summary, "whole_video_psnr_mean", "psnr_mean"),
                ssim=first_metric(summary, "whole_video_ssim_mean", "ssim_mean"),
                lpips=first_metric(summary, "whole_video_lpips_mean", "lpips_mean"),
                ewarp=first_metric(summary, "ewarp_mask_region_mean", "temporal_diff_delta_vs_gt_mean"),
                mask=first_metric(summary, "strict_mask_pixel_psnr_mean", "mask_region_psnr_mean"),
                boundary=first_metric(summary, "boundary_pixel_psnr_mean", "boundary_psnr_mean"),
                rows=first_metric(summary, "rows", "num_rows", "ok_rows"),
            )
        )
    md.extend(
        [
            "",
            f"This `{eval_name}` report is generated by the same fixed search-dev metric path as step0. It does not authorize later milestones by itself.",
        ]
    )
    (report_dir / f"{report_stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "raw_summary": raw_summary, "comp_summary": comp_summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
