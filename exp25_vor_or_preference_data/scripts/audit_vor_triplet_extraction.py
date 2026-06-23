#!/usr/bin/env python3
"""Audit extracted VOR triplets with ffprobe, frame alignment, and visuals."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FIELDS = [
    "sample_id",
    "scene_group",
    "source_type",
    "fg_bg_path",
    "bg_path",
    "mask_path",
    "fg_bg_frames",
    "bg_frames",
    "mask_frames",
    "fg_bg_size",
    "bg_size",
    "mask_size",
    "aligned_frames",
    "aligned_size",
    "mask_area_mean",
    "masked_absdiff_mean",
    "probe_backend",
    "probe_note",
    "status",
    "reason",
    "contact_sheet",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--audit-jsonl", type=Path, required=True)
    p.add_argument("--extracted-root", type=Path, required=True)
    p.add_argument("--report-csv", type=Path, default=Path("reports/vor_triplet_audit64_semantic.csv"))
    p.add_argument("--report-md", type=Path, default=Path("reports/vor_triplet_audit64_semantic.md"))
    p.add_argument("--visual-dir", type=Path, default=Path("reports/vor_triplet_audit64_visuals"))
    p.add_argument("--max-visuals", type=int, default=64)
    return p.parse_args()


def ffprobe(path: Path) -> Dict[str, object]:
    ffprobe_bin = os.environ.get("FFPROBE_BIN", "ffprobe")
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,nb_read_frames,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        fallback = cv2_probe(path)
        if not fallback.get("error"):
            fallback["backend"] = "opencv_fallback"
            fallback["note"] = proc.stderr.strip() or f"ffprobe_exit_{proc.returncode}"
            return fallback
        return {"error": proc.stderr.strip() or f"ffprobe_exit_{proc.returncode}", "backend": "ffprobe", "note": ""}
    obj = json.loads(proc.stdout or "{}")
    streams = obj.get("streams") or []
    if not streams:
        return {"error": "no_video_stream"}
    s = streams[0]
    frames = s.get("nb_read_frames") or s.get("nb_frames") or "0"
    try:
        frames_i = int(frames)
    except Exception:
        frames_i = 0
    return {
        "width": int(s.get("width") or 0),
        "height": int(s.get("height") or 0),
        "frames": frames_i,
        "duration": s.get("duration") or "",
        "error": "",
        "backend": "ffprobe",
        "note": "",
    }


def cv2_probe(path: Path) -> Dict[str, object]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": "opencv_open_failed", "backend": "opencv", "note": ""}
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if frames <= 0 or width <= 0 or height <= 0:
        return {"error": "opencv_invalid_metadata", "backend": "opencv", "note": ""}
    return {"width": width, "height": height, "frames": frames, "duration": "", "error": "", "backend": "opencv", "note": ""}


def role_path(root: Path, member_path: str, group: str) -> Path:
    return root / group / member_path


def read_frame(path: Path, frame_index: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def add_label(arr: np.ndarray, label: str) -> np.ndarray:
    img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, img.width, 30], fill=(0, 0, 0))
    draw.text((6, 4), label, fill=(255, 255, 255), font=font)
    return np.asarray(img, dtype=np.uint8)


def make_contact(sample_id: str, fg: Path, bg: Path, mask: Path, frames: int, out_path: Path) -> tuple[float, float]:
    picks = sorted(set([0, max(0, frames // 2), max(0, frames - 1)]))
    rows: List[np.ndarray] = []
    mask_areas: List[float] = []
    diffs: List[float] = []
    for idx in picks:
        fg_f = read_frame(fg, idx)
        bg_f = read_frame(bg, idx)
        mask_f = read_frame(mask, idx)
        if fg_f is None or bg_f is None or mask_f is None:
            continue
        if bg_f.shape[:2] != fg_f.shape[:2]:
            bg_f = cv2.resize(bg_f, (fg_f.shape[1], fg_f.shape[0]), interpolation=cv2.INTER_CUBIC)
        if mask_f.shape[:2] != fg_f.shape[:2]:
            mask_f = cv2.resize(mask_f, (fg_f.shape[1], fg_f.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_gray = mask_f[..., 0] > 127
        mask_areas.append(float(mask_gray.mean()))
        if mask_gray.any():
            diffs.append(float(np.abs(fg_f.astype(np.float32) - bg_f.astype(np.float32))[mask_gray].mean()))
        overlay = fg_f.copy()
        overlay[mask_gray] = (0.55 * overlay[mask_gray] + np.array([255, 0, 0]) * 0.45).astype(np.uint8)
        row = np.concatenate(
            [
                add_label(fg_f, f"{sample_id} FG_BG f{idx}"),
                add_label(bg_f, "BG / winner"),
                add_label(mask_f, "MASK"),
                add_label(overlay, "mask overlay"),
            ],
            axis=1,
        )
        rows.append(row)
    if rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.concatenate(rows, axis=0)).save(out_path)
    return (float(np.mean(mask_areas)) if mask_areas else 0.0, float(np.mean(diffs)) if diffs else 0.0)


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in args.audit_jsonl.read_text().splitlines() if line.strip()]
    out_rows = []
    for idx, row in enumerate(rows):
        sample_id = row["sample_id"]
        fg = role_path(args.extracted_root, row["condition_member_path"], "VOR-Train")
        bg = role_path(args.extracted_root, row["winner_member_path"], "VOR-Train")
        mask = role_path(args.extracted_root, row["mask_member_path"], "VOR-Train-MASK")
        fg_meta = ffprobe(fg)
        bg_meta = ffprobe(bg)
        mask_meta = ffprobe(mask)
        status = "OK"
        reason = ""
        for role, path, meta in [("FG_BG", fg, fg_meta), ("BG", bg, bg_meta), ("MASK", mask, mask_meta)]:
            if not path.exists():
                status, reason = "FAIL", f"{role}_missing"
            elif meta.get("error"):
                status, reason = "FAIL", f"{role}_ffprobe:{meta.get('error')}"
        aligned_frames = len({fg_meta.get("frames"), bg_meta.get("frames"), mask_meta.get("frames")}) == 1
        aligned_size = len({(fg_meta.get("width"), fg_meta.get("height")), (bg_meta.get("width"), bg_meta.get("height")), (mask_meta.get("width"), mask_meta.get("height"))}) == 1
        if status == "OK" and (not aligned_frames or not aligned_size):
            status = "FAIL"
            reason = "frame_or_size_mismatch"
        contact = args.visual_dir / f"{idx:03d}_{sample_id}.jpg"
        mask_area = 0.0
        masked_diff = 0.0
        if status == "OK" and idx < args.max_visuals:
            mask_area, masked_diff = make_contact(sample_id, fg, bg, mask, int(fg_meta.get("frames") or 0), contact)
        out_rows.append(
            {
                "sample_id": sample_id,
                "scene_group": row.get("scene_group", ""),
                "source_type": row.get("source_type", ""),
                "fg_bg_path": str(fg),
                "bg_path": str(bg),
                "mask_path": str(mask),
                "fg_bg_frames": fg_meta.get("frames", 0),
                "bg_frames": bg_meta.get("frames", 0),
                "mask_frames": mask_meta.get("frames", 0),
                "fg_bg_size": f"{fg_meta.get('width', 0)}x{fg_meta.get('height', 0)}",
                "bg_size": f"{bg_meta.get('width', 0)}x{bg_meta.get('height', 0)}",
                "mask_size": f"{mask_meta.get('width', 0)}x{mask_meta.get('height', 0)}",
                "aligned_frames": aligned_frames,
                "aligned_size": aligned_size,
                "mask_area_mean": mask_area,
                "masked_absdiff_mean": masked_diff,
                "probe_backend": ",".join(sorted({str(fg_meta.get("backend", "")), str(bg_meta.get("backend", "")), str(mask_meta.get("backend", ""))})),
                "probe_note": " | ".join(str(m.get("note", "")) for m in [fg_meta, bg_meta, mask_meta] if m.get("note")),
                "status": status,
                "reason": reason,
                "contact_sheet": str(contact if contact.exists() else ""),
            }
        )
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)
    ok = sum(1 for r in out_rows if r["status"] == "OK")
    fallback = sum(1 for r in out_rows if "opencv_fallback" in r["probe_backend"])
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(
        "# VOR Triplet Audit64 Semantic Report\n\n"
        f"- samples: {len(out_rows)}\n"
        f"- ok: {ok}\n"
        f"- failed: {len(out_rows) - ok}\n"
        f"- opencv_fallback_samples: {fallback}\n"
        f"- visual_dir: `{args.visual_dir}`\n"
        f"- report_csv: `{args.report_csv}`\n"
        "- checks: ffprobe decode, frame-count alignment, resolution alignment, mask-area, masked FG_BG/BG difference, contact sheets.\n",
        encoding="utf-8",
    )
    print(json.dumps({"samples": len(out_rows), "ok": ok, "failed": len(out_rows) - ok}, indent=2))
    return 0 if ok == len(out_rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
