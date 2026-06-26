#!/usr/bin/env python3
"""Pre-register Exp26 external 49F validation sources and masks.

This script intentionally stops before any VideoPainter checkpoint is loaded.
It creates exact-49-frame materialized source directories, deterministic mixed
BR masks, and locked manifests for fixed Step0-vs-Step50 validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from generate_vp2_moving_br_masks import moving_mask_sequence, sha256_file  # noqa: E402


MASK_PROFILES = [
    "irregular_freeform",
    "object_like_polygon",
    "soft_blob",
    "edge_touch_freeform",
    "ellipse_circle_subset",
    "thin_structure_freeform",
]
AREA_CYCLE = ["small", "medium", "medium", "large", "small", "medium", "large", "medium"]
DEFORM_CYCLE = ["slow", "moderate", "slow", "moderate"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_payload(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def parse_frame_paths(row: dict[str, Any]) -> list[Path]:
    value = row.get("selected_frame_paths") or row.get("frame_paths")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError(f"{row.get('sample_id')}: missing selected_frame_paths")
    paths = [Path(str(v)) for v in value]
    if len(paths) != 49:
        raise ValueError(f"{row.get('sample_id')}: expected exactly 49 selected frames, got {len(paths)}")
    return paths


def image_size(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode frame: {path}")
    return int(img.shape[0]), int(img.shape[1])


def safe_link_or_copy(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and os.readlink(dst) == str(src):
            return
        if dst.is_file() and sha256_file(dst) == sha256_file(src):
            return
        raise FileExistsError(f"existing materialized frame does not match source: {dst}")
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def materialize_frames(row: dict[str, Any], frame_root: Path) -> tuple[Path, list[str], list[str], int, int]:
    sid = str(row["sample_id"])
    source_paths = parse_frame_paths(row)
    height, width = image_size(source_paths[0])
    out_dir = frame_root / sid / "frames"
    hashes: list[str] = []
    out_paths: list[str] = []
    for idx, src in enumerate(source_paths):
        h, w = image_size(src)
        if (h, w) != (height, width):
            raise ValueError(f"{sid}: frame {src} shape {(h, w)} != first shape {(height, width)}")
        dst = out_dir / f"{idx:05d}{src.suffix.lower() or '.png'}"
        safe_link_or_copy(src, dst)
        hashes.append(sha256_file(src))
        out_paths.append(str(dst))
    if len(set(hashes)) != 49:
        raise ValueError(f"{sid}: selected frames are not hash-unique")
    return out_dir, out_paths, hashes, height, width


def choose_mask_protocol(row: dict[str, Any], idx: int) -> dict[str, Any]:
    profile = MASK_PROFILES[idx % len(MASK_PROFILES)]
    if idx % 13 == 0:
        profile = "edge_touch_freeform"
    area_bucket = AREA_CYCLE[idx % len(AREA_CYCLE)]
    source_motion = str(row.get("motion_bucket") or "medium")
    motion_bucket = source_motion if source_motion in {"low", "medium", "high"} else ["low", "medium", "high"][idx % 3]
    deform = DEFORM_CYCLE[idx % len(DEFORM_CYCLE)]
    return {
        "mask_profile": profile,
        "area_bucket": area_bucket,
        "motion_bucket": motion_bucket,
        "deformation_bucket": deform,
        "edge_touch_target": bool(profile == "edge_touch_freeform" or idx % 11 == 0),
    }


def generate_masks(
    *,
    row: dict[str, Any],
    idx: int,
    mask_root: Path,
    height: int,
    width: int,
    seed: int,
    first_frame_gt: bool,
) -> tuple[Path, list[str], dict[str, Any]]:
    sid = str(row["sample_id"])
    protocol = choose_mask_protocol(row, idx)
    masks, meta = moving_mask_sequence(
        sample_id=sid,
        num_frames=49,
        height=height,
        width=width,
        seed=seed,
        first_frame_gt=first_frame_gt,
        mask_profile=protocol["mask_profile"],
        area_bucket=protocol["area_bucket"],
        motion_bucket=protocol["motion_bucket"],
        deformation_bucket=protocol["deformation_bucket"],
        edge_touch_target=protocol["edge_touch_target"],
    )
    out_dir = mask_root / sid / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes: list[str] = []
    for frame_idx, mask in enumerate(masks):
        dst = out_dir / f"{frame_idx:05d}.png"
        if dst.exists():
            existing = cv2.imread(str(dst), cv2.IMREAD_GRAYSCALE)
            if existing is None or existing.shape != mask.shape or not (existing == mask).all():
                raise FileExistsError(f"existing mask does not match deterministic regenerated mask: {dst}")
        else:
            cv2.imwrite(str(dst), mask)
        hashes.append(sha256_file(dst))
    return out_dir, hashes, {**protocol, **meta}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fields} for row in rows])


def make_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]], status_rows: list[dict[str, Any]]) -> None:
    mask_counts = Counter(str(row.get("mask_profile", "")) for row in rows)
    area_counts = Counter(str(row.get("area_bucket", "")) for row in rows)
    motion_counts = Counter(str(row.get("motion_bucket", "")) for row in rows)
    source_counts = Counter(str(row.get("source_dataset", "")) for row in rows)
    lines = [
        "# Exp26 External Validation Preregistration",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This milestone locks the external source rows, exact 49-frame materialization, mixed masks, prompt/seed rule, and raw/comp semantics before any Step0/Step50 output exists.",
        "",
        "## Identity",
        "",
        f"- source manifest: `{summary['source_manifest']}`",
        f"- source SHA256: `{summary['source_sha256']}`",
        f"- preregistered manifest: `{summary['preregistered_manifest']}`",
        f"- preregistered SHA256: `{summary['preregistered_sha256']}`",
        f"- mask manifest: `{summary['mask_manifest']}`",
        f"- mask manifest SHA256: `{summary['mask_manifest_sha256']}`",
        f"- rows: `{summary['rows']}`",
        "",
        "## Locked Inference Protocol",
        "",
        f"- primary comparison: `{summary['primary_comparison']}`",
        f"- secondary checkpoints: `{', '.join(summary['secondary_checkpoints'])}`",
        f"- first-frame GT: `{summary['first_frame_gt']}`",
        f"- formal frames: `{summary['num_frames']}`",
        f"- inference seed: `{summary['inference_seed']}`",
        f"- mask seed: `{summary['mask_seed']}`",
        f"- inference resolution: `{summary['inference_width']}x{summary['inference_height']}`",
        f"- steps/guidance/dtype: `{summary['num_inference_steps']}` / `{summary['guidance_scale']}` / `{summary['dtype']}`",
        "",
        "## Distribution",
        "",
        f"- source datasets: `{dict(sorted(source_counts.items()))}`",
        f"- mask profiles: `{dict(sorted(mask_counts.items()))}`",
        f"- area buckets: `{dict(sorted(area_counts.items()))}`",
        f"- motion buckets: `{dict(sorted(motion_counts.items()))}`",
        "",
        "## Hard Constraints",
        "",
        "- No model outputs were generated during preregistration.",
        "- Shadow-dev/search-dev/primary32 were not changed.",
        "- Step10/Step30 remain trajectory-only and cannot replace Step50.",
        "- External validation cannot be used for tuning, source replacement, seed replacement, or checkpoint selection.",
        "- Comp keeps winner outside the mask only; primary local metrics use frame1-48.",
        "",
        "## Row Status",
        "",
    ]
    for status in status_rows:
        lines.append(
            f"- `{status['sample_id']}`: {status['status']}, frames={status['frames']}, "
            f"mask_profile={status.get('mask_profile')}, area={status.get('area_bucket')}, motion={status.get('motion_bucket')}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Exp26 external validation preregistration")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--status-csv", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--mask-seed", type=int, default=20260623)
    parser.add_argument("--inference-seed", type=int, default=20260619)
    parser.add_argument("--first-frame-gt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--dtype", default="bf16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_rows = read_jsonl(args.source_manifest)
    if len(source_rows) < 16:
        raise ValueError(f"external validation requires at least 16 rows, got {len(source_rows)}")
    frame_root = args.run_root / "materialized_49f"
    mask_root = args.run_root / "masks"
    rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(source_rows):
        sid = str(row["sample_id"])
        frame_dir, frame_paths, frame_hashes, native_h, native_w = materialize_frames(row, frame_root)
        mask_dir, mask_hashes, mask_meta = generate_masks(
            row=row,
            idx=idx,
            mask_root=mask_root,
            height=native_h,
            width=native_w,
            seed=args.mask_seed,
            first_frame_gt=args.first_frame_gt,
        )
        config_payload = {
            "sample_id": sid,
            "frame_hashes": frame_hashes,
            "mask_hashes": mask_hashes,
            "mask_meta": mask_meta,
            "first_frame_gt": args.first_frame_gt,
            "inference_seed": args.inference_seed,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "dtype": args.dtype,
        }
        config_hash = sha256_payload(config_payload)
        out_row = dict(row)
        out_row.update(
            {
                "status": "EXTERNAL_VALIDATION_PREREGISTERED",
                "split": "external_validation_preregistered",
                "frame_dir": str(frame_dir),
                "frame_paths": frame_paths,
                "frame_hashes": frame_hashes,
                "mask_dir": str(mask_dir),
                "mask_hashes": mask_hashes,
                "mask_generation": "exp26_postconfirmation_external_mixed_br_mask_v1",
                "mask_generator_seed": args.mask_seed,
                "mask_profile": mask_meta["mask_profile"],
                "area_bucket": mask_meta["area_bucket"],
                "motion_bucket": mask_meta["motion_bucket"],
                "deformation_bucket": mask_meta["deformation_bucket"],
                "edge_touch_target": mask_meta["edge_touch_target"],
                "first_frame_gt": args.first_frame_gt,
                "num_frames": 49,
                "native_height": native_h,
                "native_width": native_w,
                "inference_height": args.height,
                "inference_width": args.width,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "dtype": args.dtype,
                "inference_seed": args.inference_seed,
                "prompt": "",
                "prompt_protocol": "empty_prompt_matches_vp2_gate64_shadowdev_protocol",
                "condition_definition": "winner * (1 - generated_moving_br_mask)",
                "raw_comp_definition": "raw=model output; comp=raw inside mask and winner outside mask",
                "primary_metric_frame_range": "frames_1_48_no_first_frame",
                "config_hash": config_hash,
            }
        )
        rows.append(out_row)
        mask_rows.append(
            {
                "sample_id": sid,
                "mask_dir": str(mask_dir),
                "mask_hashes": mask_hashes,
                "mask_meta": mask_meta,
                "first_frame_gt": args.first_frame_gt,
                "num_frames": 49,
                "config_hash": config_hash,
            }
        )
        status_rows.append(
            {
                "sample_id": sid,
                "status": "OK",
                "frames": 49,
                "masks": len(mask_hashes),
                "native_height": native_h,
                "native_width": native_w,
                "mask_profile": mask_meta["mask_profile"],
                "area_bucket": mask_meta["area_bucket"],
                "motion_bucket": mask_meta["motion_bucket"],
                "deformation_bucket": mask_meta["deformation_bucket"],
                "edge_touch_target": mask_meta["edge_touch_target"],
                "first_frame_mask_hash": mask_hashes[0],
                "first_frame_gt_zero_mask": mask_meta["area_min"] == 0.0,
            }
        )
    write_jsonl(args.output_manifest, rows)
    write_jsonl(args.mask_manifest, mask_rows)
    write_csv(args.status_csv, status_rows)
    summary = {
        "status": "EXP26_EXTERNAL_VALIDATION_PREREGISTERED",
        "rows": len(rows),
        "source_manifest": str(args.source_manifest),
        "source_sha256": sha256_text(args.source_manifest),
        "preregistered_manifest": str(args.output_manifest),
        "preregistered_sha256": sha256_text(args.output_manifest),
        "mask_manifest": str(args.mask_manifest),
        "mask_manifest_sha256": sha256_text(args.mask_manifest),
        "run_root": str(args.run_root),
        "primary_comparison": "Step50 - Step0",
        "secondary_checkpoints": ["Step10", "Step30"],
        "first_frame_gt": args.first_frame_gt,
        "num_frames": 49,
        "mask_seed": args.mask_seed,
        "inference_seed": args.inference_seed,
        "inference_height": args.height,
        "inference_width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "dtype": args.dtype,
        "no_model_outputs_generated": True,
        "status_csv": str(args.status_csv),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_report(args.report_md, summary, rows, status_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
