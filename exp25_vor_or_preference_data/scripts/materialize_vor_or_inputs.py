#!/usr/bin/env python3
"""Materialize VOR OR triplets into frame directories for loser-generation smoke.

The source manifest keeps tar member paths. This script resolves them against a
selective extraction root and decodes condition/winner/mask mp4 files into
frame directories. It never composites outputs and never reads VOR-Eval rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--extraction-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--output-manifest", type=Path, required=True)
    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--prefer-balanced-source", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_member(root: Path, member_path: str) -> Path:
    parts = Path(member_path).parts
    if not parts:
        raise ValueError(f"empty member path: {member_path!r}")
    if parts[0] == "VOR-Train":
        return root / "VOR-Train" / member_path
    if parts[0] == "MASK":
        return root / "VOR-Train-MASK" / member_path
    raise ValueError(f"unsupported member path: {member_path}")


def select_rows(rows: list[dict], limit: int, balanced: bool) -> list[dict]:
    if not balanced:
        return rows[:limit]
    real = [r for r in rows if r.get("source_type") == "REAL"]
    blender = [r for r in rows if r.get("source_type") == "BLENDER"]
    selected: list[dict] = []
    while len(selected) < limit and (real or blender):
        if real and len(selected) < limit:
            selected.append(real.pop(0))
        if blender and len(selected) < limit:
            selected.append(blender.pop(0))
    return selected


def decode_video(src: Path, dst: Path, frames: int, *, mask: bool = False) -> dict:
    dst.mkdir(parents=True, exist_ok=True)
    existing = sorted(dst.glob("*.png"))
    if len(existing) == frames:
        return {"status": "resume_skip", "decoded_frames": len(existing)}
    for old in existing:
        old.unlink()
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {src}")
    decoded = 0
    while decoded < frames:
        ok, frame = cap.read()
        if not ok:
            break
        if mask:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, frame = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
        cv2.imwrite(str(dst / f"{decoded:05d}.png"), frame)
        decoded += 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if decoded != frames:
        raise RuntimeError(f"{src} decoded {decoded} frames, expected {frames}")
    return {"status": "decoded", "decoded_frames": decoded, "source_total_frames": total}


def main() -> int:
    args = parse_args()
    rows = select_rows(read_jsonl(args.manifest), args.limit, args.prefer_balanced_source)
    args.output_root.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict] = []
    failures: list[dict] = []
    for row in rows:
        sample_id = row["sample_id"]
        if row.get("task") != "object_removal" or row.get("hard_comp") is True:
            failures.append({"sample_id": sample_id, "error": "invalid_or_semantics"})
            continue
        condition_mp4 = resolve_member(args.extraction_root, row["condition_member_path"])
        winner_mp4 = resolve_member(args.extraction_root, row["winner_member_path"])
        mask_mp4 = resolve_member(args.extraction_root, row["mask_member_path"])
        try:
            condition_dir = args.output_root / "condition_frames" / sample_id
            winner_dir = args.output_root / "winner_frames" / sample_id
            mask_dir = args.output_root / "mask_frames" / sample_id
            condition_info = decode_video(condition_mp4, condition_dir, args.frames)
            winner_info = decode_video(winner_mp4, winner_dir, args.frames)
            mask_info = decode_video(mask_mp4, mask_dir, args.frames, mask=True)
        except Exception as exc:  # noqa: BLE001 - manifest audit records exact failure.
            failures.append({"sample_id": sample_id, "error": repr(exc)})
            continue
        out = {
            "sample_id": sample_id,
            "split": "gate128_smoke",
            "task": "object_removal",
            "source_type": row.get("source_type", ""),
            "scene_group": row.get("scene_group", ""),
            "condition_video_path": str(condition_dir),
            "winner_video_path": str(winner_dir),
            "mask_path": str(mask_dir),
            "condition_mp4": str(condition_mp4),
            "winner_mp4": str(winner_mp4),
            "mask_mp4": str(mask_mp4),
            "condition_source_role": "V_obj",
            "winner_source_role": "V_bg",
            "mask_source_role": "foreground_object_mask",
            "hard_comp": False,
            "comp_mode": "none",
            "frame_indices": list(range(args.frames)),
            "condition_decode": condition_info,
            "winner_decode": winner_info,
            "mask_decode": mask_info,
        }
        out_rows.append(out)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    failure_path = args.output_manifest.with_suffix(".failed.jsonl")
    with failure_path.open("w", encoding="utf-8") as f:
        for row in failures:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"ok": len(out_rows), "failed": len(failures), "manifest": str(args.output_manifest)}, indent=2))
    return 0 if len(out_rows) == args.limit else 2


if __name__ == "__main__":
    raise SystemExit(main())
