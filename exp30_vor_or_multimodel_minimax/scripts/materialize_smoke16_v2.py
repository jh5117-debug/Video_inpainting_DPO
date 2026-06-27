#!/usr/bin/env python3
"""Materialize Exp30 smoke16 triplets into 512px frame directories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extraction-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_member(root: Path, member_path: str) -> Path:
    first = Path(member_path).parts[0]
    if first == "VOR-Train":
        return root / "VOR-Train" / member_path
    if first == "MASK":
        return root / "VOR-Train-MASK" / member_path
    raise ValueError(f"unsupported member path: {member_path}")


def center_crop_resize(frame: np.ndarray, width: int, height: int, interpolation: int) -> np.ndarray:
    h, w = frame.shape[:2]
    target_ratio = width / height
    ratio = w / h
    if ratio > target_ratio:
        new_w = int(round(h * target_ratio))
        x0 = max(0, (w - new_w) // 2)
        frame = frame[:, x0 : x0 + new_w]
    elif ratio < target_ratio:
        new_h = int(round(w / target_ratio))
        y0 = max(0, (h - new_h) // 2)
        frame = frame[y0 : y0 + new_h, :]
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def read_video(path: Path, frames: int, width: int, height: int, *, mask: bool = False) -> tuple[list[np.ndarray], dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out: list[np.ndarray] = []
    while len(out) < frames:
        ok, frame = cap.read()
        if not ok:
            break
        if mask:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = center_crop_resize(frame, width, height, cv2.INTER_NEAREST)
            frame = ((frame > 20).astype(np.uint8) * 255)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = center_crop_resize(frame, width, height, cv2.INTER_LINEAR)
        out.append(frame)
    cap.release()
    if len(out) != frames:
        raise RuntimeError(f"{path} decoded {len(out)} frames, expected {frames}")
    return out, {"source_total_frames": total, "decoded_frames": len(out)}


def write_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR))


def write_mask(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), arr.astype(np.uint8))


def write_mp4(path: Path, frames: list[np.ndarray], fps: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to write {path}")
    for frame in frames:
        if frame.ndim == 2:
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()


def overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = frame.copy()
    m = mask > 20
    out[m] = (0.55 * out[m] + 0.45 * np.array([255, 40, 40])).astype(np.uint8)
    return out


def sample_indices(n: int, count: int = 16) -> list[int]:
    if n <= count:
        return list(range(n))
    return sorted({int(round(i * (n - 1) / (count - 1))) for i in range(count)})


def contact_sheet(frames: list[np.ndarray], labels: list[str], tile_w: int = 256) -> np.ndarray:
    tiles = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for frame, label in zip(frames, labels):
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        h, w = frame.shape[:2]
        tile_h = int(round(h * tile_w / w))
        tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, label, (8, 24), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    rows = []
    for start in range(0, len(tiles), 4):
        chunk = tiles[start : start + 4]
        while len(chunk) < 4:
            chunk.append(np.zeros_like(tiles[0]))
        rows.append(np.concatenate(chunk, axis=1))
    return np.concatenate(rows, axis=0)


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict] = []
    csv_rows: list[dict] = []
    failures: list[dict] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        try:
            condition_mp4 = resolve_member(args.extraction_root, row["condition_member_path"])
            winner_mp4 = resolve_member(args.extraction_root, row["winner_member_path"])
            mask_mp4 = resolve_member(args.extraction_root, row["mask_member_path"])
            condition, condition_info = read_video(condition_mp4, args.frames, args.width, args.height)
            winner, winner_info = read_video(winner_mp4, args.frames, args.width, args.height)
            masks, mask_info = read_video(mask_mp4, args.frames, args.width, args.height, mask=True)
            sample_root = args.output_root / "materialized_sources" / sample_id
            condition_dir = sample_root / "condition_frames"
            winner_dir = sample_root / "winner_frames"
            mask_dir = sample_root / "mask_frames"
            for i, (c, w, m) in enumerate(zip(condition, winner, masks)):
                write_rgb(condition_dir / f"{i:05d}.png", c)
                write_rgb(winner_dir / f"{i:05d}.png", w)
                write_mask(mask_dir / f"{i:05d}.png", m)
            mask_area = [float((m > 20).mean()) for m in masks]
            affected = [float(np.mean(np.abs(c.astype(np.float32) - w.astype(np.float32))[m > 20])) if (m > 20).any() else 0.0 for c, w, m in zip(condition, winner, masks)]
            evidence_dir = args.output_root / "source_evidence" / sample_id
            write_mp4(evidence_dir / "condition.mp4", condition)
            write_mp4(evidence_dir / "winner.mp4", winner)
            write_mp4(evidence_dir / "mask_overlay.mp4", [overlay(c, m) for c, m in zip(condition, masks)])
            side_frames = [np.concatenate([c, overlay(c, m), w], axis=1) for c, w, m in zip(condition, winner, masks)]
            write_mp4(evidence_dir / "source_side_by_side.mp4", side_frames)
            inds = sample_indices(args.frames, 16)
            sheet = contact_sheet([side_frames[i] for i in inds], [f"f{i:03d}" for i in inds], tile_w=384)
            sheet_path = evidence_dir / "source_temporal_strip_16.jpg"
            cv2.imwrite(str(sheet_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
            out = dict(row)
            out.update(
                {
                    "condition_frame_dir": str(condition_dir),
                    "winner_frame_dir": str(winner_dir),
                    "mask_frame_dir": str(mask_dir),
                    "condition_mp4": str(condition_mp4),
                    "winner_mp4": str(winner_mp4),
                    "mask_mp4": str(mask_mp4),
                    "condition_video_path": str(condition_dir),
                    "winner_video_path": str(winner_dir),
                    "mask_path": str(mask_dir),
                    "num_frames": args.frames,
                    "width": args.width,
                    "height": args.height,
                    "mask_area_mean": float(np.mean(mask_area)),
                    "mask_area_max": float(np.max(mask_area)),
                    "affected_mae_mask_mean": float(np.mean(affected)),
                    "source_temporal_strip_16": str(sheet_path),
                    "condition_decode": condition_info,
                    "winner_decode": winner_info,
                    "mask_decode": mask_info,
                }
            )
            out_rows.append(out)
            csv_rows.append(
                {
                    "sample_id": sample_id,
                    "scene_group": row.get("scene_group", ""),
                    "source_type": row.get("source_type", ""),
                    "decoded_frames": args.frames,
                    "width": args.width,
                    "height": args.height,
                    "mask_area_mean": out["mask_area_mean"],
                    "mask_area_max": out["mask_area_max"],
                    "affected_mae_mask_mean": out["affected_mae_mask_mean"],
                    "source_temporal_strip_16": str(sheet_path),
                    "status": "OK",
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"sample_id": sample_id, "error": repr(exc), "status": "FAILED"})

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    failed_path = args.output_manifest.with_suffix(".failed.jsonl")
    with failed_path.open("w", encoding="utf-8") as f:
        for row in failures:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(csv_rows[0].keys()) if csv_rows else ["sample_id", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    summary = {
        "status": "EXP30_SMOKE16_V2_MATERIALIZED" if len(out_rows) == len(rows) else "EXP30_SMOKE16_V2_MATERIALIZATION_PARTIAL",
        "requested_rows": len(rows),
        "materialized_rows": len(out_rows),
        "failed_rows": len(failures),
        "frames": args.frames,
        "width": args.width,
        "height": args.height,
        "source_type_counts": dict(Counter(r.get("source_type") for r in out_rows)),
        "manifest": str(args.output_manifest),
        "manifest_sha256": sha256_file(args.output_manifest) if args.output_manifest.exists() else "",
        "failed_manifest": str(failed_path),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.summary_md.write_text(
        "# Exp30 Smoke16 V2 Materialization\n\n"
        f"Status: `{summary['status']}`\n\n"
        f"- Requested rows: {summary['requested_rows']}\n"
        f"- Materialized rows: {summary['materialized_rows']}\n"
        f"- Failed rows: {summary['failed_rows']}\n"
        f"- Frames per row: {args.frames}\n"
        f"- Resolution: {args.width}x{args.height}\n"
        f"- Output manifest: `{args.output_manifest}`\n"
        f"- Output manifest SHA256: `{summary['manifest_sha256']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
