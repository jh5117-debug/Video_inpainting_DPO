#!/usr/bin/env python3
"""Prepare Exp46 runner manifests from Exp45 H20 pseudo-success manifests."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import imageio
from PIL import Image

SPLITS = ("train", "search", "shadow")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def frame_count(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    return len([p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])


def decode_first(path: Path) -> bool:
    files = sorted([p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not files:
        return False
    with Image.open(files[0]) as im:
        im.verify()
    return True


def extract_mp4(mp4_path: Path, out_dir: Path, frames: int = 17) -> tuple[int, str]:
    if out_dir.exists() and frame_count(out_dir) >= frames and decode_first(out_dir):
        return frame_count(out_dir), "reused"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(str(mp4_path))
    written = 0
    try:
        for idx, frame in enumerate(reader):
            if idx >= frames:
                break
            Image.fromarray(frame).save(out_dir / f"{idx:05d}.png")
            written += 1
    finally:
        reader.close()
    if written < frames:
        return written, "short"
    decode_first(out_dir)
    return written, "extracted"


def main() -> int:
    repo = Path.cwd()
    source_dir = repo / "manifests"
    frame_root = Path("/home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/minimax/exp46_pseudosuccess_target_frames")
    reports = repo / "reports"
    all_report_rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    failures = 0
    for split in SPLITS:
        rows = read_jsonl(source_dir / f"exp45_h20_pseudosuccess_{split}.jsonl")
        out_rows: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            row_id = str(row.get("row_id") or f"{split}_{idx:04d}")
            mp4_path = Path(str(row["target_path"]))
            out_dir = frame_root / split / row_id
            try:
                count, status = extract_mp4(mp4_path, out_dir)
                ok = count >= 17 and decode_first(out_dir)
            except Exception as exc:  # noqa: BLE001
                count, status, ok = 0, f"ERROR:{type(exc).__name__}:{exc}", False
            if not ok:
                failures += 1
            runner_row = {
                "sample_id": row_id,
                "row_id": row_id,
                "source_group": row.get("source_group", ""),
                "source_id": row.get("source_id", ""),
                "split": split,
                "condition_path": row["condition_path"],
                "mask_path": row["mask_path"],
                "winner_path": str(out_dir),
                "loser_path": row.get("gt_background_path", row.get("condition_path", "")),
                "pseudo_success_mp4": row["target_path"],
                "pseudo_success_label": row.get("pseudo_success_label", ""),
                "target_type": row.get("target_type", ""),
                "width": 512,
                "height": 512,
                "num_frames": 17,
                "hard_comp_used": False,
                "vor_eval_used": False,
                "training_run": False,
                "optimizer_step": False,
                "preferred_first_h20_experiment": True,
            }
            out_rows.append(runner_row)
            all_report_rows.append({
                "split": split,
                "row_id": row_id,
                "source_group": row.get("source_group", ""),
                "mp4_path": str(mp4_path),
                "runner_winner_path": str(out_dir),
                "frame_count": count,
                "status": status,
                "ok": ok,
            })
        split_counts[split] = len(out_rows)
        write_jsonl(source_dir / f"exp46_runner_pseudosuccess_{split}.jsonl", out_rows)
    reports.mkdir(exist_ok=True)
    csv_path = reports / "exp46_pseudosuccess_runner_manifest_prep.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_report_rows)
    summary = {
        "status": "EXP46_RUNNER_MANIFEST_READY" if failures == 0 else "EXP46_RUNNER_MANIFEST_BLOCKED",
        "frame_root": str(frame_root),
        "split_counts": split_counts,
        "rows": len(all_report_rows),
        "failures": failures,
        "training_run": False,
        "optimizer_step": False,
    }
    (reports / "exp46_pseudosuccess_runner_manifest_prep_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    md = (
        "# Exp46 Pseudo-Success Runner Manifest Prep\n\n"
        f"Status: {summary['status']}\n\n"
        f"Extracted/reused target frames under `{frame_root}`.\n\n"
        f"Split counts: `{split_counts.get('train')}/{split_counts.get('search')}/{split_counts.get('shadow')}`.\n\n"
        f"Failures: `{failures}`. No training or optimizer step occurred.\n"
    )
    (reports / "exp46_pseudosuccess_runner_manifest_prep.md").write_text(md)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
