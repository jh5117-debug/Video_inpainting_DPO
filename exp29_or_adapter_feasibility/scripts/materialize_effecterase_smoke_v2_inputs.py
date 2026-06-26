#!/usr/bin/env python3
"""Materialize EffectErase smoke v2 input videos from locked frame dirs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


FRAME_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def frame_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in FRAME_EXTS)


def read_rgb(path: Path, width: int, height: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((width, height), Image.BICUBIC)
    return np.asarray(img)


def read_mask_rgb(path: Path, width: int, height: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((width, height), Image.NEAREST)
    mask = (np.asarray(img) > 10).astype(np.uint8) * 255
    return np.stack([mask, mask, mask], axis=-1)


def write_mp4(path: Path, frames: list[np.ndarray], fps: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter: {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def read_video_stats(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"opens": False, "frames": 0, "width": 0, "height": 0, "mask_non_empty_frames": 0}
    frames = 0
    mask_non_empty = 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        if frame.mean(axis=2).max() > 10:
            mask_non_empty += 1
    cap.release()
    return {
        "opens": True,
        "frames": frames,
        "width": width,
        "height": height,
        "mask_non_empty_frames": mask_non_empty,
    }


def materialize_row(row: dict) -> dict:
    width = int(row.get("width", 832))
    height = int(row.get("height", 480))
    num_frames = int(row.get("num_frames", 17))
    cond_files = frame_files(Path(row["condition_frame_dir"]))
    win_files = frame_files(Path(row["winner_frame_dir"]))
    mask_files = frame_files(Path(row["mask_frame_dir"]))
    errors: list[str] = []
    if len(cond_files) < num_frames:
        errors.append("condition_frame_count_too_low")
    if len(win_files) < num_frames:
        errors.append("winner_frame_count_too_low")
    if len(mask_files) < num_frames:
        errors.append("mask_frame_count_too_low")
    cond_frames = [read_rgb(p, width, height) for p in cond_files[:num_frames]]
    win_frames = [read_rgb(p, width, height) for p in win_files[:num_frames]]
    mask_frames = [read_mask_rgb(p, width, height) for p in mask_files[:num_frames]]
    if not errors:
        write_mp4(Path(row["condition_path"]), cond_frames)
        write_mp4(Path(row["winner_path"]), win_frames)
        write_mp4(Path(row["mask_path"]), mask_frames)
    cond_stats = read_video_stats(Path(row["condition_path"]))
    win_stats = read_video_stats(Path(row["winner_path"]))
    mask_stats = read_video_stats(Path(row["mask_path"]))
    mask_ratios = [(f[..., 0] > 10).mean() for f in mask_frames]
    if cond_stats["frames"] != num_frames:
        errors.append("condition_mp4_frame_count_mismatch")
    if win_stats["frames"] != num_frames:
        errors.append("winner_mp4_frame_count_mismatch")
    if mask_stats["frames"] != num_frames:
        errors.append("mask_mp4_frame_count_mismatch")
    if mask_stats["mask_non_empty_frames"] < 8:
        errors.append("mask_mp4_too_few_non_empty_frames")
    if cond_stats["width"] != width or cond_stats["height"] != height:
        errors.append("condition_resolution_mismatch")
    if win_stats["width"] != width or win_stats["height"] != height:
        errors.append("winner_resolution_mismatch")
    if mask_stats["width"] != width or mask_stats["height"] != height:
        errors.append("mask_resolution_mismatch")
    return {
        "sample_id": row["sample_id"],
        "source_type": row.get("source_type", ""),
        "mask_bucket": row.get("mask_bucket", ""),
        "condition_path": row["condition_path"],
        "winner_path": row["winner_path"],
        "mask_path": row["mask_path"],
        "condition_frames": cond_stats["frames"],
        "winner_frames": win_stats["frames"],
        "mask_frames": mask_stats["frames"],
        "width": width,
        "height": height,
        "mask_non_empty_frames": mask_stats["mask_non_empty_frames"],
        "mask_area_median": float(np.median(mask_ratios)),
        "mask_area_mean": float(np.mean(mask_ratios)),
        "vor_eval": bool(row.get("vor_eval")),
        "eligible_for_training": bool(row.get("eligible_for_training")),
        "status": "EFFECTERASE_SMOKE_V2_INPUTS_READY" if not errors else "EFFECTERASE_SMOKE_V2_INPUTS_BLOCKED",
        "errors": ";".join(dict.fromkeys(errors)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = jsonl(args.manifest)
    results = [materialize_row(row) for row in rows]
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.reports_dir / "exp29_effecterase_smoke_v2_input_materialization.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "status": "EFFECTERASE_SMOKE_V2_INPUTS_READY"
        if all(r["status"] == "EFFECTERASE_SMOKE_V2_INPUTS_READY" for r in results)
        else "EFFECTERASE_SMOKE_V2_INPUTS_BLOCKED",
        "manifest": str(args.manifest),
        "rows": len(results),
        "ready_rows": sum(r["status"] == "EFFECTERASE_SMOKE_V2_INPUTS_READY" for r in results),
        "blocked_rows": [r["sample_id"] for r in results if r["status"] != "EFFECTERASE_SMOKE_V2_INPUTS_READY"],
        "vor_eval_used": any(r["vor_eval"] for r in results),
        "eligible_for_training": any(r["eligible_for_training"] for r in results),
    }
    (args.reports_dir / "exp29_effecterase_smoke_v2_input_materialization.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    table = "\n".join(
        f"| {r['sample_id']} | {r['condition_frames']}/{r['winner_frames']}/{r['mask_frames']} | "
        f"{r['width']}x{r['height']} | {r['mask_non_empty_frames']} | {r['mask_area_mean']:.6f} | "
        f"`{r['status']}` | {r['errors']} |"
        for r in results
    )
    (args.reports_dir / "exp29_effecterase_smoke_v2_input_materialization.md").write_text(
        "\n".join(
            [
                "# Exp29 EffectErase Smoke V2 Input Materialization",
                "",
                f"Status: `{summary['status']}`",
                "",
                f"- Manifest: `{args.manifest}`",
                f"- Rows: {summary['rows']}",
                f"- Ready rows: {summary['ready_rows']}",
                "- Resolution: 832x480",
                "- Frames per stream: 17",
                f"- VOR-Eval use: {summary['vor_eval_used']}",
                f"- Training eligibility: {summary['eligible_for_training']}",
                "",
                "| sample_id | condition/winner/mask frames | resolution | non-empty mask frames | mask area mean | status | errors |",
                "| --- | --- | --- | ---: | ---: | --- | --- |",
                table,
                "",
                "No EffectErase inference was launched by this materialization milestone.",
                "",
            ]
        )
    )


if __name__ == "__main__":
    main()
