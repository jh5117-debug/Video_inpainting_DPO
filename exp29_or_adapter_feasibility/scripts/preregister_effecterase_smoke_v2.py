#!/usr/bin/env python3
"""Build EffectErase smoke v2 preregistration from non-empty-mask rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


OLD_EMPTY_SAMPLE = "REAL_ENV249_00103_004_04"
FRAME_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class CandidateAudit:
    sample_id: str
    scene_group: str
    source_type: str
    mask_bucket: str
    non_empty_mask_frames: int
    mask_area_total: float
    mask_area_median: float
    mask_area_min: float
    mask_area_max: float
    condition_frame_count: int
    winner_frame_count: int
    mask_frame_count: int
    resolution: str
    masked_absdiff_mean: float
    accepted: bool
    reject_reason: str


def jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in FRAME_EXTS)


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    return arr > 10


def audit_candidate(row: dict) -> CandidateAudit:
    cond_dir = Path(row["condition_frame_dir"])
    win_dir = Path(row["winner_frame_dir"])
    mask_dir = Path(row["mask_frame_dir"])
    cond = frame_files(cond_dir) if cond_dir.exists() else []
    win = frame_files(win_dir) if win_dir.exists() else []
    masks = frame_files(mask_dir) if mask_dir.exists() else []
    counts = (len(cond), len(win), len(masks))
    reject: list[str] = []
    if counts != (17, 17, 17):
        reject.append(f"frame_count_{counts[0]}_{counts[1]}_{counts[2]}")
    areas: list[float] = []
    masked_diffs: list[float] = []
    resolution = "unknown"
    if len(cond) >= 1:
        c0 = read_rgb(cond[0])
        resolution = f"{c0.shape[1]}x{c0.shape[0]}"
    for c_path, w_path, m_path in zip(cond[:17], win[:17], masks[:17]):
        c = read_rgb(c_path)
        w = read_rgb(w_path)
        m = read_mask(m_path)
        if c.shape[:2] != w.shape[:2]:
            reject.append("condition_winner_resolution_mismatch")
            continue
        if m.shape != c.shape[:2]:
            m_img = Image.fromarray((m.astype(np.uint8) * 255))
            m = np.asarray(m_img.resize((c.shape[1], c.shape[0]), Image.NEAREST)) > 10
        area = float(m.mean())
        areas.append(area)
        if m.any():
            masked_diffs.append(float(np.abs(c.astype(np.float32) - w.astype(np.float32))[m].mean()))
    non_empty = sum(1 for a in areas if a > 0)
    if non_empty < 8:
        reject.append("too_few_non_empty_mask_frames")
    median = float(np.median(areas)) if areas else 0.0
    amin = float(np.min(areas)) if areas else 0.0
    amax = float(np.max(areas)) if areas else 0.0
    total = float(np.sum(areas)) if areas else 0.0
    if median < 0.001:
        reject.append("median_mask_area_too_small")
    if median > 0.60:
        reject.append("median_mask_area_too_large")
    if amax >= 0.95:
        reject.append("all_or_nearly_all_white_mask")
    if amax <= 0:
        reject.append("all_black_mask")
    diff = float(np.mean(masked_diffs)) if masked_diffs else 0.0
    if diff <= 1.0:
        reject.append("condition_winner_too_similar_inside_mask")
    return CandidateAudit(
        sample_id=row["sample_id"],
        scene_group=row.get("scene_group", row["sample_id"]),
        source_type=row.get("source_type", "unknown"),
        mask_bucket=row.get("mask_bucket", "unknown"),
        non_empty_mask_frames=non_empty,
        mask_area_total=total,
        mask_area_median=median,
        mask_area_min=amin,
        mask_area_max=amax,
        condition_frame_count=counts[0],
        winner_frame_count=counts[1],
        mask_frame_count=counts[2],
        resolution=resolution,
        masked_absdiff_mean=diff,
        accepted=not reject,
        reject_reason=";".join(dict.fromkeys(reject)),
    )


def make_preview(row: dict, out_path: Path) -> None:
    cond = frame_files(Path(row["condition_frame_dir"]))
    win = frame_files(Path(row["winner_frame_dir"]))
    masks = frame_files(Path(row["mask_frame_dir"]))
    idxs = np.linspace(0, 16, 6, dtype=int).tolist()
    thumb_w, thumb_h = 192, 112
    canvas = Image.new("RGB", (thumb_w * len(idxs), thumb_h * 3 + 36), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), f"{row['sample_id']} | {row.get('source_type')} | {row.get('mask_bucket')}", fill=(0, 0, 0))
    y0 = 24
    for col, idx in enumerate(idxs):
        c = Image.open(cond[idx]).convert("RGB").resize((thumb_w, thumb_h))
        w = Image.open(win[idx]).convert("RGB").resize((thumb_w, thumb_h))
        m = Image.open(masks[idx]).convert("L").resize((thumb_w, thumb_h), Image.NEAREST)
        overlay = c.copy()
        red = Image.new("RGB", overlay.size, (255, 0, 0))
        overlay = Image.composite(red, overlay, m.point(lambda v: 96 if v > 10 else 0))
        x = col * thumb_w
        canvas.paste(c, (x, y0))
        canvas.paste(w, (x, y0 + thumb_h))
        canvas.paste(overlay, (x, y0 + thumb_h * 2))
        draw.text((x + 4, y0 + 2), f"f{idx}", fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def with_v2_paths(row: dict, output_root: Path) -> dict:
    sample = row["sample_id"]
    new = dict(row)
    new.update(
        {
            "source_role": "vor_diagnostic_only",
            "vor_eval": False,
            "eligible_for_training": False,
            "scientific_role": "diagnostic_only_vor_confounded",
            "num_frames": 17,
            "height": 480,
            "width": 832,
            "seed": 2025,
            "num_inference_steps": 50,
            "cfg": 1.0,
            "frame_interval": 1,
            "condition_path": str(output_root / "inputs" / sample / "fg_bg.mp4"),
            "mask_path": str(output_root / "inputs" / sample / "mask.mp4"),
            "winner_path": str(output_root / "inputs" / sample / "bg.mp4"),
            "output_path": str(output_root / "outputs" / sample / "raw_output.mp4"),
            "diagnostic_comp_path": str(output_root / "outputs" / sample / "diagnostic_comp.mp4"),
        }
    )
    return new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--source32-manifest", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--rejected-manifest", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    old_rows = jsonl(args.old_manifest)
    source_rows = jsonl(args.source32_manifest)
    old_by_id = {row["sample_id"]: row for row in old_rows}
    retained = [row for row in old_rows if row["sample_id"] != OLD_EMPTY_SAMPLE]
    old_ids = set(old_by_id)

    audits = {row["sample_id"]: audit_candidate(row) for row in source_rows}
    preferred = [
        row
        for row in source_rows
        if row["sample_id"] not in old_ids
        and row.get("source_type") == "REAL"
        and row.get("mask_bucket") == "small"
        and audits[row["sample_id"]].accepted
    ]
    fallback = [
        row
        for row in source_rows
        if row["sample_id"] not in old_ids and audits[row["sample_id"]].accepted
    ]
    replacement = preferred[0] if preferred else (fallback[0] if fallback else None)
    rejected_rows: list[dict] = [
        {
            **old_by_id[OLD_EMPTY_SAMPLE],
            "rejected_from_v2": True,
            "reject_reason": "locked materialized mask empty across all 17 frames",
        }
    ]

    if replacement is None:
        accepted_rows = [with_v2_paths(row, args.output_root) for row in retained]
        status = "EFFECTERASE_SMOKE_V2_BLOCKED_NO_VALID_REPLACEMENT"
    else:
        accepted_rows = [with_v2_paths(row, args.output_root) for row in retained + [replacement]]
        status = "EFFECTERASE_SMOKE_V2_PREREGISTERED"

    for row in accepted_rows:
        make_preview(row, args.reports_dir / "exp29_effecterase_smoke_v2_previews" / f"{row['sample_id']}.jpg")

    write_jsonl(args.out_manifest, accepted_rows)
    write_jsonl(args.rejected_manifest, rejected_rows)

    audit_csv = args.reports_dir / "exp29_effecterase_smoke_v2_input_audit.csv"
    with audit_csv.open("w", newline="") as f:
        fieldnames = list(CandidateAudit.__annotations__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in source_rows:
            writer.writerow(audits[row["sample_id"]].__dict__)

    rejected_csv = args.reports_dir / "exp29_effecterase_smoke_v2_rejected_rows.csv"
    with rejected_csv.open("w", newline="") as f:
        fieldnames = ["sample_id", "scene_group", "source_type", "mask_bucket", "reject_reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rejected_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    preview_csv = args.reports_dir / "exp29_effecterase_smoke_v2_preview_review.csv"
    with preview_csv.open("w", newline="") as f:
        fieldnames = [
            "sample_id",
            "source_type",
            "mask_bucket",
            "preview_path",
            "reviewed_frames",
            "classification",
            "reviewer_pass",
            "reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in accepted_rows:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "source_type": row.get("source_type", ""),
                    "mask_bucket": row.get("mask_bucket", ""),
                    "preview_path": str(args.reports_dir / "exp29_effecterase_smoke_v2_previews" / f"{row['sample_id']}.jpg"),
                    "reviewed_frames": "0,3,6,9,12,16",
                    "classification": "PREVIEW_PENDING_CODEX_VIEW_IMAGE",
                    "reviewer_pass": "pending",
                    "reason": "preview generated; main agent must inspect before final lock",
                }
            )

    summary = {
        "status": status,
        "old_manifest": str(args.old_manifest),
        "old_manifest_sha256": sha256_file(args.old_manifest),
        "new_manifest": str(args.out_manifest),
        "new_manifest_sha256": sha256_file(args.out_manifest),
        "rejected_manifest": str(args.rejected_manifest),
        "rejected_manifest_sha256": sha256_file(args.rejected_manifest),
        "accepted_rows": len(accepted_rows),
        "replacement_row": replacement["sample_id"] if replacement else None,
        "replacement_rule": "prefer REAL small from source32, fallback to first accepted non-old row",
        "real_blender_counts": {
            k: sum(1 for row in accepted_rows if row.get("source_type") == k) for k in ["REAL", "BLENDER"]
        },
        "mask_bucket_counts": {
            k: sum(1 for row in accepted_rows if row.get("mask_bucket") == k) for k in ["small", "medium", "large"]
        },
        "vor_eval_used": any(bool(row.get("vor_eval")) for row in accepted_rows),
        "eligible_for_training": any(bool(row.get("eligible_for_training")) for row in accepted_rows),
    }
    (args.reports_dir / "exp29_effecterase_smoke_v2_preregistration.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    md = args.reports_dir / "exp29_effecterase_smoke_v2_input_audit.md"
    rows_table = "\n".join(
        f"| {row['sample_id']} | {row.get('source_type')} | {row.get('mask_bucket')} | "
        f"{audits[row['sample_id']].non_empty_mask_frames}/17 | "
        f"{audits[row['sample_id']].mask_area_median:.6f} | "
        f"{audits[row['sample_id']].masked_absdiff_mean:.3f} |"
        for row in accepted_rows
    )
    md.write_text(
        "\n".join(
            [
                "# Exp29 EffectErase Smoke V2 Input Audit",
                "",
                f"Status: `{status}`",
                "",
                f"Old manifest SHA256: `{summary['old_manifest_sha256']}`",
                f"New manifest SHA256: `{summary['new_manifest_sha256']}`",
                f"Rejected old row: `{OLD_EMPTY_SAMPLE}` because its smoke materialized mask is empty across all 17 frames.",
                f"Replacement row: `{summary['replacement_row']}`",
                "",
                "## Accepted Rows",
                "",
                "| sample_id | source_type | mask_bucket | non-empty mask frames | median mask area | masked absdiff mean |",
                "| --- | --- | --- | ---: | ---: | ---: |",
                rows_table,
                "",
                "All accepted rows are tagged `diagnostic_only_vor_confounded`, `eligible_for_training=false`, and `vor_eval=false`.",
                "",
                "Preview sheets were generated under `reports/exp29_effecterase_smoke_v2_previews/` and must be inspected before inference.",
                "",
            ]
        )
    )
    prereg_md = args.reports_dir / "exp29_effecterase_smoke_v2_preregistration.md"
    prereg_md.write_text(
        "\n".join(
            [
                "# Exp29 EffectErase Smoke V2 Pre-Registration",
                "",
                f"Status: `{status}`",
                "",
                "This v2 manifest replaces only the prior empty-mask blocker row. The old v1 manifest is preserved.",
                "",
                f"- Old manifest: `{args.old_manifest}`",
                f"- Old manifest SHA256: `{summary['old_manifest_sha256']}`",
                f"- New manifest: `{args.out_manifest}`",
                f"- New manifest SHA256: `{summary['new_manifest_sha256']}`",
                f"- Rejected manifest: `{args.rejected_manifest}`",
                f"- Rejected manifest SHA256: `{summary['rejected_manifest_sha256']}`",
                f"- Replacement row: `{summary['replacement_row']}`",
                f"- REAL / BLENDER counts: `{summary['real_blender_counts']}`",
                f"- Mask bucket counts: `{summary['mask_bucket_counts']}`",
                f"- VOR-Eval used: `{summary['vor_eval_used']}`",
                f"- Eligible for training: `{summary['eligible_for_training']}`",
                "",
                "Fixed protocol remains: 17 frames, 832x480, seed 2025, CFG 1.0, 50 steps, raw output primary, diagnostic comp optional.",
                "",
                "No EffectErase inference, adapter training, scientific-positive claim, or universal-adapter claim is made by this preregistration.",
                "",
            ]
        )
    )


if __name__ == "__main__":
    main()
