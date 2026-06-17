#!/usr/bin/env python3
"""Compute Exp16 prior-confidence statistics for a cached prior manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp16_loss import (  # noqa: E402
    boundary_outer_from_hole,
    compute_prior_confidence_from_gt_error,
)
from training.dpo.dataset.generated_loser_manifest_dataset import list_image_frames  # noqa: E402


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_rgb_stack(path: str, nframes: int, size: tuple[int, int]) -> torch.Tensor:
    frames = list_image_frames(path)
    if len(frames) < nframes:
        raise ValueError(f"expected at least {nframes} RGB frames under {path}, found {len(frames)}")
    out = []
    for frame_path in frames[:nframes]:
        img = Image.open(frame_path).convert("RGB").resize(size, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        out.append(torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0)
    return torch.stack(out).unsqueeze(0)


def load_hole_mask_stack(path: str, nframes: int, size: tuple[int, int]) -> torch.Tensor:
    frames = list_image_frames(path)
    if len(frames) < nframes:
        raise ValueError(f"expected at least {nframes} mask frames under {path}, found {len(frames)}")
    out = []
    for frame_path in frames[:nframes]:
        img = Image.open(frame_path).convert("L").resize(size, Image.NEAREST)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
        out.append(torch.from_numpy(arr).unsqueeze(0))
    return torch.stack(out).unsqueeze(0)


def finite_mean(values: list[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_md", default="reports/exp16_prior_confidence_limit100_audit.md")
    parser.add_argument("--output_csv", default="exp16_prior_confidence_gated_dpo/manifests/prior_confidence_limit100.csv")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--nframes", type=int, default=16)
    parser.add_argument("--width", type=int, default=432)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--alpha", type=float, default=5.0)
    args = parser.parse_args()

    rows_out: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    size = (args.width, args.height)
    for idx, row in enumerate(read_jsonl(Path(args.manifest).expanduser())):
        if args.limit and idx >= args.limit:
            break
        sample_id = str(row.get("sample_id") or idx)
        prior = row.get("propainter_prior_frame_dir") or row.get("prior_frame_dir")
        gt = row.get("gt_frame_dir") or row.get("win_video_path")
        mask = row.get("mask_frame_dir") or row.get("mask_path")
        try:
            prior_rgb = load_rgb_stack(str(prior), args.nframes, size)
            gt_rgb = load_rgb_stack(str(gt), args.nframes, size)
            hole = load_hole_mask_stack(str(mask), args.nframes, size)
            conf, stats = compute_prior_confidence_from_gt_error(prior_rgb, gt_rgb, hole, args.alpha)
            boundary = boundary_outer_from_hole(hole.reshape(args.nframes, 1, args.height, args.width))
            hole4 = hole.reshape(args.nframes, 1, args.height, args.width)
            conf4 = conf.reshape(args.nframes, 1, args.height, args.width)
            reliable = hole4 * conf4
            generate = hole4 * (1.0 - conf4)
            hole_sum = hole4.sum().clamp(min=1e-6)
            conf_inside = conf4[hole4 > 0.5] if (hole4 > 0.5).any() else conf4.reshape(-1)
            reliable_weight_mass = float(reliable.sum() / hole_sum)
            generate_weight_mass = float(generate.sum() / hole_sum)
            out = {
                "sample_id": sample_id,
                **stats,
                "reliable_area_ratio": float((reliable > 1e-4).float().mean()),
                "generate_area_ratio": float((generate > 1e-4).float().mean()),
                "prior_conf_mean_inside_mask": float(conf_inside.float().mean()),
                "prior_conf_std_inside_mask": float(conf_inside.float().std(unbiased=False)),
                "prior_conf_p10_inside_mask": float(torch.quantile(conf_inside.float(), 0.10)),
                "prior_conf_p50_inside_mask": float(torch.quantile(conf_inside.float(), 0.50)),
                "prior_conf_p90_inside_mask": float(torch.quantile(conf_inside.float(), 0.90)),
                "reliable_weight_mass": reliable_weight_mass,
                "generate_weight_mass": generate_weight_mass,
                "reliable_generate_mass_sum": reliable_weight_mass + generate_weight_mass,
                "boundary_area_ratio": float((boundary > 0.5).float().mean()),
                "confidence_mode": "gt_error",
                "confidence_alpha": args.alpha,
            }
            rows_out.append(out)
        except Exception as exc:  # noqa: BLE001
            errors.append({"sample_id": sample_id, "error": repr(exc)})

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows_out:
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)

    summary_keys = [
        "prior_conf_mean",
        "prior_conf_p10",
        "prior_conf_p50",
        "prior_conf_p90",
        "prior_conf_mean_inside_mask",
        "prior_conf_std_inside_mask",
        "prior_conf_p10_inside_mask",
        "prior_conf_p50_inside_mask",
        "prior_conf_p90_inside_mask",
        "reliable_area_ratio",
        "generate_area_ratio",
        "reliable_weight_mass",
        "generate_weight_mass",
        "reliable_generate_mass_sum",
        "mask_area_ratio",
        "boundary_area_ratio",
    ]
    lines = [
        "# Exp16 Prior Confidence Limit100 Audit",
        "",
        f"manifest: `{args.manifest}`",
        f"rows_ok: {len(rows_out)}",
        f"rows_failed: {len(errors)}",
        f"confidence_mode: `gt_error`",
        f"confidence_alpha: {args.alpha}",
        "",
        "| metric | mean |",
        "|---|---:|",
    ]
    for key in summary_keys:
        lines.append(f"| {key} | {finite_mean([r[key] for r in rows_out if key in r]):.6f} |")
    if errors:
        lines.extend(["", "## Failed Cases", ""])
        for err in errors[:20]:
            lines.append(f"- `{err['sample_id']}`: {err['error']}")
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if rows_out else 2


if __name__ == "__main__":
    raise SystemExit(main())
