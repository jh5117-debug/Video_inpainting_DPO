#!/usr/bin/env python3
"""Materialize EffectErase official 81-frame smoke inputs.

This script consumes the preregistered official-81F diagnostic manifest and
writes exact 81-frame condition/winner/mask MP4s. It does not run EffectErase.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_selected_frames(path: Path, indices: list[int], width: int, height: int, *, mask: bool = False) -> tuple[list[np.ndarray], dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], {"opens": False, "source_width": 0, "source_height": 0, "source_frames": 0}
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[np.ndarray] = []
    want = set(indices)
    max_idx = max(indices)
    idx = 0
    while idx <= max_idx:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            if mask:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_NEAREST)
                arr = ((resized > 10).astype(np.uint8) * 255)
                rgb = np.stack([arr, arr, arr], axis=-1)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_CUBIC)
            frames.append(rgb)
        idx += 1
    cap.release()
    return frames, {
        "opens": True,
        "source_width": source_width,
        "source_height": source_height,
        "source_frames": source_frames,
    }


def write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise RuntimeError(f"Cannot write empty video: {path}")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter: {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def video_stats(path: Path) -> dict:
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


def make_preview(path: Path, sample_id: str, condition: list[np.ndarray], winner: list[np.ndarray], mask: list[np.ndarray], reviewed_indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = len(reviewed_indices)
    thumb_w, thumb_h = 160, 92
    header_h = 32
    canvas = Image.new("RGB", (cols * thumb_w, header_h + 3 * thumb_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 6), f"{sample_id} materialized official81", fill=(0, 0, 0))
    for i, frame_idx in enumerate(reviewed_indices):
        src_idx = max(0, min(len(condition) - 1, frame_idx))
        x = i * thumb_w
        for row, frames in enumerate((condition, winner, mask)):
            img = Image.fromarray(frames[src_idx]).resize((thumb_w, thumb_h))
            if row == 2:
                overlay = Image.blend(Image.fromarray(condition[src_idx]).resize((thumb_w, thumb_h)), img, alpha=0.35)
                img = overlay
            canvas.paste(img, (x, header_h + row * thumb_h))
        draw.text((x + 3, header_h + 2), str(frame_idx), fill=(255, 255, 255))
    canvas.save(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def unique_errors(errors: Iterable[str]) -> str:
    return ";".join(dict.fromkeys(e for e in errors if e))


def materialize(row: dict, preview_dir: Path) -> dict:
    width = int(row.get("width", 832))
    height = int(row.get("height", 480))
    num_frames = int(row.get("num_frames", 81))
    fps = max(1, int(round(float(row.get("fps", 24)))))
    indices = [int(x) for x in row.get("selected_frame_indices", list(range(num_frames)))]
    indices = indices[:num_frames]
    errors: list[str] = []
    condition, cond_source_stats = read_selected_frames(Path(row["condition_source_path"]), indices, width, height, mask=False)
    winner, win_source_stats = read_selected_frames(Path(row["winner_source_path"]), indices, width, height, mask=False)
    mask, mask_source_stats = read_selected_frames(Path(row["mask_source_path"]), indices, width, height, mask=True)
    if len(condition) != num_frames:
        errors.append("condition_selected_frame_count_mismatch")
    if len(winner) != num_frames:
        errors.append("winner_selected_frame_count_mismatch")
    if len(mask) != num_frames:
        errors.append("mask_selected_frame_count_mismatch")
    if not errors:
        write_mp4(Path(row["condition_path"]), condition, fps)
        write_mp4(Path(row["winner_path"]), winner, fps)
        write_mp4(Path(row["mask_path"]), mask, fps)
    cond_out = video_stats(Path(row["condition_path"]))
    win_out = video_stats(Path(row["winner_path"]))
    mask_out = video_stats(Path(row["mask_path"]))
    mask_ratios = [float((f[..., 0] > 10).mean()) for f in mask] if mask else [0.0]
    if cond_out["frames"] != num_frames:
        errors.append("condition_output_frame_count_mismatch")
    if win_out["frames"] != num_frames:
        errors.append("winner_output_frame_count_mismatch")
    if mask_out["frames"] != num_frames:
        errors.append("mask_output_frame_count_mismatch")
    if mask_out["mask_non_empty_frames"] < 40:
        errors.append("mask_too_few_non_empty_frames")
    for label, stats in (("condition", cond_out), ("winner", win_out), ("mask", mask_out)):
        if stats["width"] != width or stats["height"] != height:
            errors.append(f"{label}_resolution_mismatch")
    if bool(row.get("vor_eval")):
        errors.append("vor_eval_row_not_allowed")
    if bool(row.get("eligible_for_training")):
        errors.append("training_eligible_row_not_allowed")
    preview_path = preview_dir / f"{row['selection_rank']:02d}_{row['sample_id']}.jpg"
    if not errors:
        reviewed_indices = [0, 5, 10, 16, 21, 26, 32, 37, 42, 48, 53, 58, 64, 69, 74, 80]
        make_preview(preview_path, row["sample_id"], condition, winner, mask, reviewed_indices)
    return {
        "sample_id": row["sample_id"],
        "source_type": row.get("source_type", ""),
        "scene_group": row.get("scene_group", ""),
        "mask_bucket": row.get("mask_bucket", ""),
        "condition_source_path": row["condition_source_path"],
        "winner_source_path": row["winner_source_path"],
        "mask_source_path": row["mask_source_path"],
        "condition_path": row["condition_path"],
        "winner_path": row["winner_path"],
        "mask_path": row["mask_path"],
        "fps": fps,
        "width": width,
        "height": height,
        "condition_source_frames": cond_source_stats.get("source_frames", 0),
        "winner_source_frames": win_source_stats.get("source_frames", 0),
        "mask_source_frames": mask_source_stats.get("source_frames", 0),
        "condition_output_frames": cond_out["frames"],
        "winner_output_frames": win_out["frames"],
        "mask_output_frames": mask_out["frames"],
        "mask_non_empty_frames": mask_out["mask_non_empty_frames"],
        "mask_area_min": float(np.min(mask_ratios)),
        "mask_area_median": float(np.median(mask_ratios)),
        "mask_area_max": float(np.max(mask_ratios)),
        "preview_sheet": str(preview_path),
        "condition_sha256": sha256_file(Path(row["condition_path"])) if Path(row["condition_path"]).exists() else "",
        "winner_sha256": sha256_file(Path(row["winner_path"])) if Path(row["winner_path"]).exists() else "",
        "mask_sha256": sha256_file(Path(row["mask_path"])) if Path(row["mask_path"]).exists() else "",
        "vor_eval": bool(row.get("vor_eval")),
        "eligible_for_training": bool(row.get("eligible_for_training")),
        "status": "EFFECTERASE_OFFICIAL81_INPUTS_READY" if not errors else "EFFECTERASE_OFFICIAL81_INPUTS_BLOCKED",
        "errors": unique_errors(errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    args = parser.parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.manifest)
    preview_dir = args.reports_dir / "exp29_effecterase_official81_materialized_previews"
    results = [materialize(row, preview_dir) for row in rows]
    ready_rows = sum(r["status"] == "EFFECTERASE_OFFICIAL81_INPUTS_READY" for r in results)
    status = "EFFECTERASE_OFFICIAL81_INPUTS_READY" if ready_rows == len(results) and len(results) >= 6 else "EFFECTERASE_OFFICIAL81_INPUTS_BLOCKED"
    csv_path = args.reports_dir / "exp29_effecterase_official81_input_materialization.csv"
    write_csv(csv_path, results)
    summary = {
        "status": status,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "rows": len(results),
        "ready_rows": ready_rows,
        "blocked_rows": [r["sample_id"] for r in results if r["status"] != "EFFECTERASE_OFFICIAL81_INPUTS_READY"],
        "vor_eval_used": any(r["vor_eval"] for r in results),
        "eligible_for_training": any(r["eligible_for_training"] for r in results),
        "num_frames": 81,
        "resolution": "832x480",
        "preview_dir": str(preview_dir),
    }
    (args.reports_dir / "exp29_effecterase_official81_input_materialization.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    table = "\n".join(
        f"| {r['sample_id']} | {r['condition_output_frames']}/{r['winner_output_frames']}/{r['mask_output_frames']} | "
        f"{r['width']}x{r['height']} | {r['mask_non_empty_frames']} | {r['mask_area_median']:.6f} | "
        f"`{r['status']}` | {r['errors']} |"
        for r in results
    )
    md_lines = [
        "# Exp29 EffectErase Official 81F Input Materialization",
        "",
        f"Status: `{status}`",
        "",
        f"- Manifest: `{args.manifest}`",
        f"- Manifest SHA256: `{summary['manifest_sha256']}`",
        f"- Rows: {summary['rows']}",
        f"- Ready rows: {summary['ready_rows']}",
        f"- Blocked rows: {summary['blocked_rows']}",
        "- Resolution: 832x480",
        "- Frames per stream: 81",
        f"- VOR-Eval use: {summary['vor_eval_used']}",
        f"- Training eligibility: {summary['eligible_for_training']}",
        "",
        "| sample_id | condition/winner/mask frames | resolution | non-empty mask frames | mask area median | status | errors |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
        table,
        "",
        "Materialized preview sheets were generated for input sanity review.",
        "No EffectErase inference was launched by this materialization milestone.",
        "",
    ]
    (args.reports_dir / "exp29_effecterase_official81_input_materialization.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
