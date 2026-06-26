#!/usr/bin/env python3
"""Preregister EffectErase official 81-frame diagnostic smoke rows.

The script uses already completed Exp25 exact selective-extraction caches,
then verifies each selected row against the full VOR metadata index. It does
not modify Exp25, does not use VOR-Eval, and does not launch model inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def infer_source_type(sample_id: str) -> str:
    if sample_id.startswith("REAL_"):
        return "REAL"
    if sample_id.startswith("BLENDER_"):
        return "BLENDER"
    return "UNKNOWN"


def infer_scene_group(sample_id: str) -> str:
    parts = sample_id.split("_")
    if sample_id.startswith("REAL_") and len(parts) >= 3:
        return "_".join(parts[:3])
    if sample_id.startswith("BLENDER_") and len(parts) >= 2:
        return "_".join(parts[:2])
    return sample_id


def mask_bucket(area: float) -> str:
    if area < 0.035:
        return "small"
    if area < 0.10:
        return "medium"
    return "large"


def video_props(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"opens": False, "frames": 0, "width": 0, "height": 0, "fps": 0.0}
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return {"opens": True, "frames": frames, "width": width, "height": height, "fps": fps}


def read_frame(cap: cv2.VideoCapture, index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def read_next_rgb(cap: cv2.VideoCapture) -> np.ndarray | None:
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id", ""))
        if sample_id:
            metadata[sample_id] = row
    return metadata


def load_extractions(paths: list[Path]) -> dict[str, dict[str, str]]:
    by_sample: dict[str, dict[str, str]] = defaultdict(dict)
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "OK":
                    continue
                sample_id = row["sample_id"]
                member = row["member_path"]
                if "/FG_BG/" in member:
                    by_sample[sample_id]["condition_source_path"] = row["output_path"]
                elif "/BG/" in member:
                    by_sample[sample_id]["winner_source_path"] = row["output_path"]
                elif member.startswith("MASK/") or "/MASK/" in member:
                    by_sample[sample_id]["mask_source_path"] = row["output_path"]
    return {sample_id: row for sample_id, row in by_sample.items() if len(row) == 3}


def audit_sample(sample_id: str, paths: dict[str, str], metadata: dict[str, Any], num_frames: int) -> dict[str, Any]:
    condition_path = Path(paths["condition_source_path"])
    winner_path = Path(paths["winner_source_path"])
    mask_path = Path(paths["mask_source_path"])
    c_props = video_props(condition_path)
    w_props = video_props(winner_path)
    m_props = video_props(mask_path)
    reject: list[str] = []
    for name, props in (("condition", c_props), ("winner", w_props), ("mask", m_props)):
        if not props["opens"]:
            reject.append(f"{name}_decode_failed")
        if props["frames"] < num_frames:
            reject.append(f"{name}_frame_count_lt_{num_frames}")
    if c_props["width"] != w_props["width"] or c_props["height"] != w_props["height"]:
        reject.append("condition_winner_resolution_mismatch")
    if c_props["width"] <= 0 or c_props["height"] <= 0:
        reject.append("invalid_resolution")

    areas: list[float] = []
    masked_diffs: list[float] = []
    outside_diffs: list[float] = []
    if not reject:
        c_cap = cv2.VideoCapture(str(condition_path))
        w_cap = cv2.VideoCapture(str(winner_path))
        m_cap = cv2.VideoCapture(str(mask_path))
        for _index in range(num_frames):
            c = read_next_rgb(c_cap)
            w = read_next_rgb(w_cap)
            m_rgb = read_next_rgb(m_cap)
            if c is None or w is None or m_rgb is None:
                reject.append("frame_decode_failed")
                break
            mask = m_rgb.mean(axis=2) > 10
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
    total = float(np.sum(areas)) if areas else 0.0
    masked_diff = float(np.mean(masked_diffs)) if masked_diffs else 0.0
    outside_diff = float(np.mean(outside_diffs)) if outside_diffs else 0.0

    if non_empty < 40:
        reject.append("non_empty_mask_frames_lt_40")
    if not (0.001 <= median <= 0.60):
        reject.append("median_mask_area_out_of_range")
    if amax >= 0.95:
        reject.append("all_or_nearly_all_white_mask")
    if amax <= 0.0:
        reject.append("all_black_mask")
    if masked_diff <= 1.0:
        reject.append("condition_winner_too_similar_inside_mask")
    if "VOR-Eval" in " ".join(str(metadata.get(k, "")) for k in metadata):
        reject.append("vor_eval_path")

    return {
        "sample_id": sample_id,
        "scene_group": metadata.get("scene_group") or infer_scene_group(sample_id),
        "source_type": infer_source_type(sample_id),
        "effect_type": metadata.get("effect_type", "unknown"),
        "selected_start_frame": 0,
        "selected_end_frame": num_frames - 1,
        "selected_frame_count": num_frames,
        "non_empty_mask_frames": non_empty,
        "mask_area_total": total,
        "mask_area_median": median,
        "mask_area_min": amin,
        "mask_area_max": amax,
        "mask_bucket": mask_bucket(median),
        "condition_frame_count": c_props["frames"],
        "winner_frame_count": w_props["frames"],
        "mask_frame_count": m_props["frames"],
        "resolution": f"{c_props['width']}x{c_props['height']}",
        "fps": c_props["fps"],
        "masked_absdiff_mean": masked_diff,
        "outside_absdiff_mean": outside_diff,
        "condition_source_path": str(condition_path),
        "winner_source_path": str(winner_path),
        "mask_source_path": str(mask_path),
        "condition_member_path": metadata.get("condition_member_path", ""),
        "winner_member_path": metadata.get("winner_member_path", ""),
        "mask_member_path": metadata.get("mask_member_path", ""),
        "accepted": not reject,
        "reject_reason": ";".join(dict.fromkeys(reject)),
    }


def select_rows(audit_rows: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    accepted = [row for row in audit_rows if row["accepted"]]
    selected: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    # Balance source type first, then mask bucket. Unknown effect type is kept explicit.
    strata = [
        ("REAL", "small"),
        ("BLENDER", "small"),
        ("REAL", "medium"),
        ("BLENDER", "medium"),
        ("REAL", "large"),
        ("BLENDER", "large"),
    ]
    while len(selected) < target:
        advanced = False
        for source_type, bucket in strata:
            candidates = sorted(
                [
                    row
                    for row in accepted
                    if row["source_type"] == source_type
                    and row["mask_bucket"] == bucket
                    and row["scene_group"] not in seen_groups
                ],
                key=lambda row: (row["scene_group"], row["sample_id"]),
            )
            if candidates and len(selected) < target:
                picked = candidates[0]
                selected.append(picked)
                seen_groups.add(picked["scene_group"])
                advanced = True
        if not advanced:
            remaining = sorted(
                [row for row in accepted if row["scene_group"] not in seen_groups],
                key=lambda row: (row["source_type"], row["mask_bucket"], row["scene_group"], row["sample_id"]),
            )
            if not remaining:
                break
            picked = remaining[0]
            selected.append(picked)
            seen_groups.add(picked["scene_group"])
    return selected[:target]


def balanced_sample_ids(extracted: dict[str, dict[str, str]], max_candidates: int) -> list[str]:
    by_type: dict[str, list[str]] = {"REAL": [], "BLENDER": [], "UNKNOWN": []}
    for sample_id in sorted(extracted):
        by_type[infer_source_type(sample_id)].append(sample_id)
    ordered: list[str] = []
    while len(ordered) < max_candidates:
        advanced = False
        for source_type in ("REAL", "BLENDER", "UNKNOWN"):
            if by_type[source_type]:
                ordered.append(by_type[source_type].pop(0))
                advanced = True
                if len(ordered) >= max_candidates:
                    break
        if not advanced:
            break
    return ordered


def resize_rgb(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def make_preview(row: dict[str, Any], out_path: Path) -> None:
    cond = cv2.VideoCapture(row["condition_source_path"])
    winner = cv2.VideoCapture(row["winner_source_path"])
    mask_cap = cv2.VideoCapture(row["mask_source_path"])
    indices = np.linspace(0, int(row["selected_frame_count"]) - 1, 16, dtype=int).tolist()
    tile_w, tile_h = 128, 72
    header_h = 34
    canvas = np.full((header_h + tile_h * 3, tile_w * len(indices), 3), 255, dtype=np.uint8)
    cv2.putText(
        canvas,
        f"{row['sample_id']} {row['source_type']} {row['mask_bucket']} mask_med={row['mask_area_median']:.4f}",
        (4, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    for col, idx in enumerate(indices):
        c = read_frame(cond, idx)
        w = read_frame(winner, idx)
        m = read_frame(mask_cap, idx)
        if c is None or w is None or m is None:
            continue
        mask = m.mean(axis=2) > 10
        c_small = resize_rgb(c, tile_w, tile_h)
        w_small = resize_rgb(w, tile_w, tile_h)
        m_small = cv2.resize(mask.astype(np.uint8), (tile_w, tile_h), interpolation=cv2.INTER_NEAREST) > 0
        overlay = c_small.copy()
        overlay[m_small] = (0.55 * overlay[m_small] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
        x = col * tile_w
        y = header_h
        canvas[y : y + tile_h, x : x + tile_w] = c_small
        canvas[y + tile_h : y + 2 * tile_h, x : x + tile_w] = w_small
        canvas[y + 2 * tile_h : y + 3 * tile_h, x : x + tile_w] = overlay
        cv2.putText(canvas, str(idx), (x + 3, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cond.release()
    winner.release()
    mask_cap.release()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-index", type=Path, required=True)
    parser.add_argument("--extraction-csv", type=Path, action="append", required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rejected-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-rows", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--max-candidates", type=int, default=96)
    args = parser.parse_args()

    metadata = load_metadata(args.metadata_index)
    extracted = load_extractions(args.extraction_csv)
    audit_rows: list[dict[str, Any]] = []
    for sample_id in balanced_sample_ids(extracted, args.max_candidates):
        paths = extracted[sample_id]
        meta = metadata.get(sample_id, {"sample_id": sample_id, "scene_group": infer_scene_group(sample_id)})
        audit_rows.append(audit_sample(sample_id, paths, meta, args.num_frames))
    selected = select_rows(audit_rows, args.target_rows)
    selected_ids = {row["sample_id"] for row in selected}

    manifest_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    preview_dir = args.reports_dir / "exp29_effecterase_official81_previews"
    for rank, row in enumerate(selected):
        sample_id = row["sample_id"]
        preview_path = preview_dir / f"{rank:02d}_{sample_id}.jpg"
        make_preview(row, preview_path)
        new_row = dict(row)
        new_row.update(
            {
                "selection_rank": rank,
                "source_role": "vor_diagnostic_only",
                "vor_eval": False,
                "eligible_for_training": False,
                "scientific_role": "diagnostic_only_vor_confounded",
                "num_frames": args.num_frames,
                "height": 480,
                "width": 832,
                "seed": 2025,
                "num_inference_steps": 50,
                "cfg": 1.0,
                "frame_interval": 1,
                "raw_output_primary": True,
                "diagnostic_comp_optional": True,
                "selected_frame_indices": list(range(args.num_frames)),
                "preview_sheet": str(preview_path),
                "condition_path": str(args.output_root / "inputs" / sample_id / "condition_81f.mp4"),
                "winner_path": str(args.output_root / "inputs" / sample_id / "winner_81f.mp4"),
                "mask_path": str(args.output_root / "inputs" / sample_id / "mask_81f.mp4"),
                "output_path": str(args.output_root / "outputs" / sample_id / "raw_output.mp4"),
                "diagnostic_comp_path": str(args.output_root / "outputs" / sample_id / "diagnostic_comp.mp4"),
            }
        )
        manifest_rows.append(new_row)
        preview_rows.append(
            {
                "sample_id": sample_id,
                "preview_sheet": str(preview_path),
                "review_method": "codex_preview_sheet_pending_open",
                "reviewer_pass": "PENDING",
            }
        )

    rejected_rows = [
        {**row, "rejected_from_manifest": row["sample_id"] not in selected_ids}
        for row in audit_rows
        if row["sample_id"] not in selected_ids
    ]
    write_jsonl(args.manifest, manifest_rows)
    write_jsonl(args.rejected_manifest, rejected_rows)
    write_csv(args.reports_dir / "exp29_effecterase_official81_source_audit.csv", audit_rows)
    write_csv(args.reports_dir / "exp29_effecterase_official81_rejected_rows.csv", rejected_rows)
    write_csv(args.reports_dir / "exp29_effecterase_official81_preview_review.csv", preview_rows)

    manifest_sha = sha256_file(args.manifest)
    rejected_sha = sha256_file(args.rejected_manifest)
    status = "EFFECTERASE_OFFICIAL81_PREREGISTERED" if len(manifest_rows) >= 6 else "EFFECTERASE_OFFICIAL81_BLOCKED_NO_VALID_ROWS"
    summary = {
        "status": status,
        "metadata_index": str(args.metadata_index),
        "metadata_sha256": sha256_file(args.metadata_index),
        "extraction_csvs": [str(path) for path in args.extraction_csv],
        "candidate_count": len(audit_rows),
        "max_candidates": args.max_candidates,
        "accepted_candidate_count": sum(bool(row["accepted"]) for row in audit_rows),
        "selected_count": len(manifest_rows),
        "rejected_count": len(rejected_rows),
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha,
        "rejected_manifest": str(args.rejected_manifest),
        "rejected_sha256": rejected_sha,
        "source_type_counts": Counter(row["source_type"] for row in manifest_rows),
        "mask_bucket_counts": Counter(row["mask_bucket"] for row in manifest_rows),
        "vor_eval_used": False,
        "eligible_for_training": False,
    }
    (args.reports_dir / "exp29_effecterase_official81_preregistration.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Exp29 EffectErase Official 81F Source Audit",
        "",
        f"Status: `{status}`",
        "",
        "## Inputs",
        "",
        f"- Metadata index: `{args.metadata_index}`",
        f"- Metadata SHA256: `{summary['metadata_sha256']}`",
        f"- Extraction CSVs: `{[str(path) for path in args.extraction_csv]}`",
        "- VOR-Eval use: false",
        "- Training eligibility: false",
        "",
        "## Counts",
        "",
        f"- Candidate triplets audited from exact extraction caches: {len(audit_rows)}",
        f"- Accepted by 81F/frame/mask rules: {summary['accepted_candidate_count']}",
        f"- Manifest rows locked: {len(manifest_rows)}",
        f"- Rejected rows recorded: {len(rejected_rows)}",
        f"- Manifest SHA256: `{manifest_sha}`",
        f"- Rejected manifest SHA256: `{rejected_sha}`",
        f"- Source type counts: `{dict(summary['source_type_counts'])}`",
        f"- Mask bucket counts: `{dict(summary['mask_bucket_counts'])}`",
        "",
        "## Protocol",
        "",
        "- Selected frames: exact consecutive frame indices 0-80.",
        "- Frame count requirement: condition/winner/mask each have at least 81 decoded frames.",
        "- Mask requirement: at least 40/81 non-empty frames and median mask area in [0.001, 0.60].",
        "- Primary output remains raw EffectErase output; diagnostic comp is optional later.",
        "- Rows are diagnostic-only VOR-confounded rows and are not eligible for training.",
        "",
        "## Preview Review",
        "",
        "Preview sheets were generated for each locked row under:",
        "",
        f"`{preview_dir}`",
        "",
        "They contain condition, winner, and mask-overlay strips across 16 sampled frames.",
        "Codex visual opening is recorded separately in `exp29_effecterase_official81_preview_review.csv`.",
        "",
    ]
    (args.reports_dir / "exp29_effecterase_official81_source_audit.md").write_text("\n".join(lines), encoding="utf-8")
    (args.reports_dir / "exp29_effecterase_official81_preregistration.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
