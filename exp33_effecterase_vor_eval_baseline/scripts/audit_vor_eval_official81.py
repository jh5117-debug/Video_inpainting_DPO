#!/usr/bin/env python3
"""Audit held-out VOR-Eval rows for official EffectErase 81-frame inference.

This script is Exp33-specific. Unlike the Exp29 diagnostic smoke scripts, VOR-
Eval rows are allowed here, but only as held-out baseline inputs. The generated
manifest is never eligible for training, loser mining, adapter tuning, or
checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_type(sample_id: str) -> str:
    if sample_id.startswith("REAL_"):
        return "REAL"
    if sample_id.startswith("BLENDER_"):
        return "BLENDER"
    return "UNKNOWN"


def scene_group(sample_id: str) -> str:
    parts = sample_id.split("_")
    if sample_id.startswith("REAL_") and len(parts) >= 3:
        return "_".join(parts[:3])
    if sample_id.startswith("BLENDER_") and len(parts) >= 2:
        return "_".join(parts[:2])
    return sample_id


def role_from_member(member_path: str) -> str:
    parts = member_path.replace("\\", "/").split("/")
    if "FG_BG" in parts:
        return "condition"
    if "BG" in parts:
        return "winner"
    if "MASK" in parts:
        return "mask"
    return ""


def sample_id_from_member(member_path: str) -> str:
    return Path(member_path).stem


def strip_vor_eval_prefix(member_path: str) -> str:
    normalized = member_path.replace("\\", "/")
    prefix = "VOR-Eval/"
    if normalized.startswith(prefix):
        return normalized[len(prefix) :]
    return normalized


def build_triplets(member_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_sample: dict[str, dict[str, str]] = {}
    for row in member_rows:
        if row.get("type") != "file":
            continue
        member = row.get("member_path", "")
        if not member.lower().endswith((".mp4", ".mov", ".avi")):
            continue
        role = role_from_member(member)
        if not role:
            continue
        sid = sample_id_from_member(member)
        by_sample.setdefault(sid, {})[role] = strip_vor_eval_prefix(member)
    out = []
    for sid in sorted(by_sample):
        roles = by_sample[sid]
        if all(key in roles for key in ("condition", "winner", "mask")):
            out.append(
                {
                    "sample_id": sid,
                    "condition_member_path": roles["condition"],
                    "winner_member_path": roles["winner"],
                    "mask_member_path": roles["mask"],
                }
            )
    return out


def video_props(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"opens": False, "frames": 0, "width": 0, "height": 0, "fps": 0.0}
    props = {
        "opens": True,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
    }
    cap.release()
    return props


def read_next_rgb(cap: cv2.VideoCapture) -> np.ndarray | None:
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def mask_bucket(area: float) -> str:
    if area < 0.035:
        return "small"
    if area < 0.10:
        return "medium"
    return "large"


def make_preview(row: dict[str, Any], output_path: Path) -> None:
    cond = cv2.VideoCapture(str(row["condition_source_path"]))
    winner = cv2.VideoCapture(str(row["winner_source_path"]))
    mask_cap = cv2.VideoCapture(str(row["mask_source_path"]))
    frame_count = int(row["selected_frame_count"])
    indices = np.linspace(0, frame_count - 1, 16, dtype=int).tolist()
    tile_w, tile_h = 128, 72
    header_h = 34
    canvas = np.full((header_h + 3 * tile_h, tile_w * len(indices), 3), 255, dtype=np.uint8)
    cv2.putText(
        canvas,
        f"{row['sample_id']} VOR-Eval {row['mask_bucket']} med={row['mask_area_median']:.4f}",
        (4, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    for col, index in enumerate(indices):
        for cap in (cond, winner, mask_cap):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        c = read_next_rgb(cond)
        w = read_next_rgb(winner)
        m = read_next_rgb(mask_cap)
        if c is None or w is None or m is None:
            continue
        mask = m.mean(axis=2) > 10
        c_small = cv2.resize(c, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        w_small = cv2.resize(w, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        m_small = cv2.resize(mask.astype(np.uint8), (tile_w, tile_h), interpolation=cv2.INTER_NEAREST) > 0
        overlay = c_small.copy()
        overlay[m_small] = (0.55 * overlay[m_small] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
        x = col * tile_w
        y = header_h
        canvas[y : y + tile_h, x : x + tile_w] = c_small
        canvas[y + tile_h : y + 2 * tile_h, x : x + tile_w] = w_small
        canvas[y + 2 * tile_h : y + 3 * tile_h, x : x + tile_w] = overlay
        cv2.putText(canvas, str(index), (x + 3, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cond.release()
    winner.release()
    mask_cap.release()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def audit_triplet(row: dict[str, str], extracted_root: Path, num_frames: int) -> dict[str, Any]:
    sid = row["sample_id"]
    condition = extracted_root / row["condition_member_path"]
    winner = extracted_root / row["winner_member_path"]
    mask_path = extracted_root / row["mask_member_path"]
    c_props = video_props(condition)
    w_props = video_props(winner)
    m_props = video_props(mask_path)
    errors: list[str] = []
    for label, props in (("condition", c_props), ("winner", w_props), ("mask", m_props)):
        if not props["opens"]:
            errors.append(f"{label}_decode_failed")
        if props["frames"] < num_frames:
            errors.append(f"{label}_frame_count_lt_{num_frames}")
    if c_props["width"] != w_props["width"] or c_props["height"] != w_props["height"]:
        errors.append("condition_winner_resolution_mismatch")
    if c_props["width"] <= 0 or c_props["height"] <= 0:
        errors.append("invalid_resolution")

    areas: list[float] = []
    masked_diffs: list[float] = []
    outside_diffs: list[float] = []
    if not errors:
        c_cap = cv2.VideoCapture(str(condition))
        w_cap = cv2.VideoCapture(str(winner))
        m_cap = cv2.VideoCapture(str(mask_path))
        for _ in range(num_frames):
            c = read_next_rgb(c_cap)
            w = read_next_rgb(w_cap)
            m = read_next_rgb(m_cap)
            if c is None or w is None or m is None:
                errors.append("frame_decode_failed")
                break
            mask = m.mean(axis=2) > 10
            if mask.shape != c.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), (c.shape[1], c.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            areas.append(float(mask.mean()))
            diff = np.abs(c.astype(np.float32) - w.astype(np.float32)).mean(axis=2)
            if mask.any():
                masked_diffs.append(float(diff[mask].mean()))
            if (~mask).any():
                outside_diffs.append(float(diff[~mask].mean()))
        c_cap.release()
        w_cap.release()
        m_cap.release()

    non_empty = sum(1 for area in areas if area > 0.0)
    median = float(np.median(areas)) if areas else 0.0
    amin = float(np.min(areas)) if areas else 0.0
    amax = float(np.max(areas)) if areas else 0.0
    if non_empty < 40:
        errors.append("non_empty_mask_frames_lt_40")
    if not (0.001 <= median <= 0.60):
        errors.append("median_mask_area_out_of_range")
    if amax >= 0.95:
        errors.append("all_or_nearly_all_white_mask")
    if amax <= 0.0:
        errors.append("all_black_mask")
    masked_diff = float(np.mean(masked_diffs)) if masked_diffs else 0.0
    outside_diff = float(np.mean(outside_diffs)) if outside_diffs else 0.0
    if masked_diff <= 1.0:
        errors.append("condition_winner_too_similar_inside_mask")

    return {
        "sample_id": sid,
        "scene_group": scene_group(sid),
        "source_type": source_type(sid),
        "condition_source_path": str(condition),
        "winner_source_path": str(winner),
        "mask_source_path": str(mask_path),
        "condition_member_path": row["condition_member_path"],
        "winner_member_path": row["winner_member_path"],
        "mask_member_path": row["mask_member_path"],
        "condition_frame_count": c_props["frames"],
        "winner_frame_count": w_props["frames"],
        "mask_frame_count": m_props["frames"],
        "resolution": f"{c_props['width']}x{c_props['height']}",
        "fps": c_props["fps"],
        "selected_start_frame": 0,
        "selected_end_frame": num_frames - 1,
        "selected_frame_count": num_frames,
        "non_empty_mask_frames": non_empty,
        "mask_area_min": amin,
        "mask_area_median": median,
        "mask_area_max": amax,
        "mask_bucket": mask_bucket(median),
        "masked_absdiff_mean": masked_diff,
        "outside_absdiff_mean": outside_diff,
        "vor_eval": True,
        "eligible_for_training": False,
        "accepted": not errors,
        "reject_reason": ";".join(dict.fromkeys(errors)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member-index", type=Path, required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rejected-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=81)
    args = parser.parse_args()

    triplets = build_triplets(read_csv(args.member_index))
    audit_rows = [audit_triplet(row, args.extracted_root, args.num_frames) for row in triplets]
    ready_rows = [row for row in audit_rows if row["accepted"]]
    rejected_rows = [row for row in audit_rows if not row["accepted"]]

    manifest_rows: list[dict[str, Any]] = []
    preview_dir = args.reports_dir / "exp33_effecterase_vor_eval_official81_previews"
    for rank, row in enumerate(ready_rows):
        sid = row["sample_id"]
        preview = preview_dir / f"{rank:02d}_{sid}.jpg"
        make_preview(row, preview)
        new_row = dict(row)
        new_row.update(
            {
                "selection_rank": rank,
                "source_role": "held_out_vor_eval_baseline",
                "scientific_role": "held_out_baseline_only_not_training",
                "num_frames": args.num_frames,
                "height": 480,
                "width": 832,
                "seed": 2025,
                "num_inference_steps": 50,
                "cfg": 1.0,
                "frame_interval": 1,
                "selected_frame_indices": list(range(args.num_frames)),
                "raw_output_primary": True,
                "diagnostic_comp_optional": True,
                "preview_sheet": str(preview),
                "condition_path": str(args.output_root / "inputs" / sid / "condition_81f.mp4"),
                "winner_path": str(args.output_root / "inputs" / sid / "winner_81f.mp4"),
                "mask_path": str(args.output_root / "inputs" / sid / "mask_81f.mp4"),
                "output_path": str(args.output_root / "outputs" / sid / "raw_output.mp4"),
                "diagnostic_comp_path": str(args.output_root / "outputs" / sid / "diagnostic_comp.mp4"),
            }
        )
        manifest_rows.append(new_row)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.manifest, manifest_rows)
    write_jsonl(args.rejected_manifest, rejected_rows)
    write_csv(args.reports_dir / "exp33_effecterase_vor_eval_official81_compatibility.csv", audit_rows)

    status = (
        "EXP33_VOREVAL_OFFICIAL81_COMPATIBILITY_PASSED"
        if len(manifest_rows) == 43 and not rejected_rows
        else "EXP33_VOREVAL_OFFICIAL81_COMPATIBILITY_PARTIAL"
        if manifest_rows
        else "EXP33_VOREVAL_OFFICIAL81_COMPATIBILITY_FAILED"
    )
    summary = {
        "status": status,
        "member_index": str(args.member_index),
        "member_index_sha256": sha256_file(args.member_index),
        "extracted_root": str(args.extracted_root),
        "triplets": len(triplets),
        "ready_rows": len(manifest_rows),
        "rejected_rows": len(rejected_rows),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest) if args.manifest.exists() else "",
        "rejected_manifest": str(args.rejected_manifest),
        "rejected_manifest_sha256": sha256_file(args.rejected_manifest) if args.rejected_manifest.exists() else "",
        "source_type_counts": dict(Counter(row["source_type"] for row in manifest_rows)),
        "mask_bucket_counts": dict(Counter(row["mask_bucket"] for row in manifest_rows)),
        "vor_eval_used": True,
        "eligible_for_training": False,
        "adapter_training": "forbidden",
        "inference_started": False,
    }
    (args.reports_dir / "exp33_effecterase_vor_eval_official81_compatibility.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    table = "\n".join(
        f"| {row['sample_id']} | {row['condition_frame_count']}/{row['winner_frame_count']}/{row['mask_frame_count']} | "
        f"{row['resolution']} | {row['non_empty_mask_frames']} | {row['mask_area_median']:.6f} | "
        f"`{'READY' if row['accepted'] else 'REJECTED'}` | {row['reject_reason']} |"
        for row in audit_rows
    )
    md = [
        "# Exp33 EffectErase VOR-Eval Official81 Compatibility",
        "",
        f"Status: `{status}`",
        "",
        f"- VOR-Eval triplets audited: {len(triplets)}",
        f"- Ready rows: {len(manifest_rows)}",
        f"- Rejected rows: {len(rejected_rows)}",
        f"- Manifest: `{args.manifest}`",
        f"- Manifest SHA256: `{summary['manifest_sha256']}`",
        f"- Preview dir: `{preview_dir}`",
        "- Role: held-out EffectErase baseline only.",
        "- Training/adaptation/loser mining/checkpoint selection: forbidden.",
        "",
        "| sample_id | condition/winner/mask frames | resolution | non-empty mask frames | mask area median | status | reason |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
        table,
        "",
        "No EffectErase inference was launched by this compatibility audit.",
        "",
    ]
    (args.reports_dir / "exp33_effecterase_vor_eval_official81_compatibility.md").write_text(
        "\n".join(md), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
