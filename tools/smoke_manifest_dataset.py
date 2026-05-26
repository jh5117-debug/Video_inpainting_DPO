#!/usr/bin/env python3
"""Smoke-test generated-loser manifest dataset loading."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

for env_name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(env_name, "1")

import torch
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.dpo.dataset.generated_loser_manifest_dataset import (
    GeneratedLoserManifestDataset,
    list_image_frames,
)


class DummyTokenizer:
    model_max_length = 77

    def __call__(self, caption, max_length, padding, truncation, return_tensors):
        _ = (caption, padding, truncation, return_tensors)
        return SimpleNamespace(input_ids=torch.zeros((1, max_length), dtype=torch.long))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def comp_outside_diff_hint(row: dict[str, Any]) -> dict[str, Any]:
    win = list_image_frames(row.get("win_video_path", ""))
    comp = list_image_frames(row.get("comp_loser_video_path") or row.get("final_loser_video_path", ""))
    mask = list_image_frames(row.get("mask_path", ""))
    outside_mean_abs_diff = None
    if win and comp and mask:
        win_img = Image.open(win[0]).convert("RGB")
        comp_img = Image.open(comp[0]).convert("RGB").resize(win_img.size, Image.BILINEAR)
        mask_img = Image.open(mask[0]).convert("L").resize(win_img.size, Image.NEAREST)
        outside = np.asarray(mask_img) <= 127
        if outside.any():
            win_arr = np.asarray(win_img, dtype=np.int16)
            comp_arr = np.asarray(comp_img, dtype=np.int16)
            outside_mean_abs_diff = float(np.abs(win_arr[outside] - comp_arr[outside]).mean())
    return {
        "win_frames": len(win),
        "comp_frames": len(comp),
        "mask_frames": len(mask),
        "can_check": bool(win and comp and mask),
        "outside_mean_abs_diff_first_frame": outside_mean_abs_diff,
    }


def build_dataset_args(manifest: Path, train_mask_mode: str, mask_from_manifest: bool, args: argparse.Namespace):
    return SimpleNamespace(
        preference_manifest=str(manifest),
        train_mask_mode=train_mask_mode,
        mask_from_manifest=mask_from_manifest,
        loss_region_mode="full",
        nframes=args.nframes,
        train_height=args.height,
        train_width=args.width,
        resolution=args.width,
        videodpo_full_mask_value=args.full_mask_value,
        proportion_empty_prompts=0.0,
        max_resample_attempts=8,
    )


def check_dataset(manifest: Path, label: str, train_mask_mode: str, mask_from_manifest: bool, args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(manifest)
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(rows)), min(args.sample_size, len(rows)))
    dataset = GeneratedLoserManifestDataset(
        build_dataset_args(manifest, train_mask_mode, mask_from_manifest, args),
        DummyTokenizer(),
    )
    samples = []
    issues = []
    for idx in indices:
        sample = dataset[idx]
        item = {
            "index": idx,
            "sample_id": sample.get("sample_id"),
            "mask_id": sample.get("mask_id"),
            "prompt": sample.get("prompt"),
            "win_shape": list(sample["pixel_values_pos"].shape),
            "loser_shape": list(sample["pixel_values_neg"].shape),
            "conditioning_shape": list(sample["conditioning_pixel_values"].shape),
            "mask_shape": list(sample["masks"].shape),
            "mask_min": float(sample["masks"].min().item()),
            "mask_max": float(sample["masks"].max().item()),
            "comp_hint": comp_outside_diff_hint(sample["manifest_row"]),
        }
        expected_video_shape = [args.nframes, 3, args.height, args.width]
        expected_mask_shape = [args.nframes, 1, args.height, args.width]
        if item["win_shape"] != expected_video_shape:
            issues.append(f"{label}:{idx} win_shape={item['win_shape']}")
        if item["loser_shape"] != expected_video_shape:
            issues.append(f"{label}:{idx} loser_shape={item['loser_shape']}")
        if item["conditioning_shape"] != expected_video_shape:
            issues.append(f"{label}:{idx} conditioning_shape={item['conditioning_shape']}")
        if item["mask_shape"] != expected_mask_shape:
            issues.append(f"{label}:{idx} mask_shape={item['mask_shape']}")
        if train_mask_mode == "full" and (item["mask_min"] != args.full_mask_value or item["mask_max"] != args.full_mask_value):
            issues.append(f"{label}:{idx} expected full mask value {args.full_mask_value}, got {item['mask_min']}..{item['mask_max']}")
        if train_mask_mode == "partial" and item["mask_min"] == item["mask_max"]:
            issues.append(f"{label}:{idx} expected non-constant partial mask, got {item['mask_min']}..{item['mask_max']}")
        samples.append(item)
    return {
        "label": label,
        "manifest": str(manifest),
        "rows": len(rows),
        "train_mask_mode": train_mask_mode,
        "mask_from_manifest": mask_from_manifest,
        "batch_keys": [
            "pixel_values_pos",
            "pixel_values_neg",
            "conditioning_pixel_values",
            "masks",
            "input_ids",
        ],
        "samples": samples,
        "issues": issues,
    }


def write_report(path: Path, results: list[dict[str, Any]]) -> None:
    lines = ["# Manifest Dataset Smoke Report", ""]
    for result in results:
        lines.extend([
            f"## {result['label']}",
            "",
            f"- manifest: `{result['manifest']}`",
            f"- rows: `{result['rows']}`",
            f"- train_mask_mode: `{result['train_mask_mode']}`",
            f"- mask_from_manifest: `{result['mask_from_manifest']}`",
            f"- issues: `{len(result['issues'])}`",
            "",
            "```json",
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = "/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4"
    parser.add_argument("--d2_root", default=default_root)
    parser.add_argument("--comp_manifest", default="")
    parser.add_argument("--nocomp_manifest", default="")
    parser.add_argument("--sample_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--nframes", type=int, default=16)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--full_mask_value", type=float, default=0.0)
    parser.add_argument("--report", default="reports/manifest_dataset_smoke_report.md")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    d2_root = Path(args.d2_root)
    comp = Path(args.comp_manifest) if args.comp_manifest else d2_root / "manifests" / "selected_primary_comp.repaired.jsonl"
    nocomp = Path(args.nocomp_manifest) if args.nocomp_manifest else d2_root / "manifests" / "selected_primary_nocomp.repaired.jsonl"
    checks = [
        (comp, "exp5_comp_full", "full", False),
        (nocomp, "exp6_nocomp_full", "full", False),
        (comp, "exp7_comp_partial", "partial", True),
        (comp, "exp8_comp_partial_region_dataset_only", "partial", True),
    ]
    results = []
    for manifest, label, train_mask_mode, mask_from_manifest in checks:
        results.append(check_dataset(manifest, label, train_mask_mode, mask_from_manifest, args))
    write_report(Path(args.report), results)
    issue_count = sum(len(result["issues"]) for result in results)
    print(f"[manifest-smoke] report={args.report}")
    for result in results:
        print(
            f"[manifest-smoke] {result['label']} rows={result['rows']} "
            f"mode={result['train_mask_mode']} issues={len(result['issues'])}"
        )
    print(f"[manifest-smoke] total_issues={issue_count}")
    return 1 if issue_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
