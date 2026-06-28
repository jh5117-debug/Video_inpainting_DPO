#!/usr/bin/env python3
"""Run Exp40 MiniMax Step0 baseline inference on locked splits.

This is inference only. It reuses the Exp30 MiniMax `run_pipeline` path so the
baseline matches the previous MiniMax adapter gates and does not touch the
shared metric code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--split-manifest", action="append", required=True, help="NAME=PATH; may be repeated")
    parser.add_argument("--label", default="exp40_step0")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reports-root", required=True)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--num-inference-steps", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--heartbeat", default="")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def heartbeat(path: Path | None, message: str) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{time.time():.0f}\t{message}\n", encoding="utf-8")


def parse_split_manifest(values: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"split manifest must be NAME=PATH, got {value!r}")
        split, path = value.split("=", 1)
        split = split.strip()
        if not split:
            raise ValueError(f"empty split name in {value!r}")
        parsed.append((split, Path(path).resolve()))
    return parsed


def torch_dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "float16" else torch.bfloat16


def mean_metric(rows: list[dict[str, object]], key: str) -> float | None:
    vals: list[float] = []
    for row in rows:
        val = row.get(key)
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            vals.append(float(val))
    return sum(vals) / len(vals) if vals else None


def aggregate_by_split(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    splits = sorted({str(row["split"]) for row in rows})
    metrics = (
        "full_psnr",
        "mask_psnr",
        "boundary_psnr",
        "outside_psnr",
        "outside_mae",
        "temporal_diff_mae",
    )
    for split in splits:
        split_rows = [row for row in rows if row["split"] == split]
        out[split] = {"rows": len(split_rows)}
        for metric in metrics:
            out[split][metric] = mean_metric(split_rows, metric)
    return out


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.repo_dir).resolve()))
    sys.path.insert(0, str(Path(args.project_root).resolve()))

    from diffusers.models import AutoencoderKLWan  # noqa: WPS433
    from diffusers.schedulers import UniPCMultistepScheduler  # noqa: WPS433
    from transformer_minimax_remover import Transformer3DModel  # noqa: WPS433
    from exp30_vor_or_multimodel_minimax.scripts.run_minimax_gate64_adapter_gate_v3 import (  # noqa: WPS433
        BatchCache,
        run_pipeline,
    )

    output_root = Path(args.output_root).resolve()
    reports_root = Path(args.reports_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    hb = Path(args.heartbeat) if args.heartbeat else output_root / f"{args.label}.heartbeat"

    split_rows = [(split, read_jsonl(path), path) for split, path in parse_split_manifest(args.split_manifest)]
    model_dir = Path(args.model_dir).resolve()
    for child in ("vae", "transformer", "scheduler"):
        if not (model_dir / child).exists():
            raise FileNotFoundError(f"missing MiniMax model component: {model_dir / child}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch_dtype(args.dtype)
    torch.manual_seed(args.seed)
    start = time.time()

    heartbeat(hb, "load_step0")
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=dtype).to(device).eval()
    transformer = Transformer3DModel.from_pretrained(model_dir / "transformer", torch_dtype=dtype).to(device).eval()
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    latents_std = (1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)).to(device, dtype)
    cache = BatchCache(vae, latents_mean, latents_std, device, dtype)

    metric_rows: list[dict[str, object]] = []
    visual_rows: list[dict[str, object]] = []
    for split, rows, manifest_path in split_rows:
        for idx, row in enumerate(rows, 1):
            sample_id = str(row["sample_id"])
            heartbeat(hb, f"{args.label}:{split}:{idx}/{len(rows)}:{sample_id}")
            row_root = output_root / args.label / split / sample_id / "step0"
            metrics = run_pipeline(
                transformer,
                vae,
                UniPCMultistepScheduler,
                model_dir,
                cache,
                row,
                row_root,
                args.seed,
                args.num_inference_steps,
                args.iterations,
            )
            metric_rows.append(
                {
                    "label": args.label,
                    "split": split,
                    "split_manifest": str(manifest_path),
                    "sample_id": sample_id,
                    "scene_group": row.get("scene_group", row.get("source_group", "")),
                    "source_type": row.get("source_type", ""),
                    "classification": row.get("classification", ""),
                    "profile": row.get("profile", ""),
                    "condition_path": row.get("condition_path", ""),
                    "winner_path": row.get("winner_path", ""),
                    "loser_path": row.get("loser_path", ""),
                    "mask_path": row.get("mask_path", ""),
                    "raw_step0": metrics["raw_output_mp4"],
                    "side_by_side": metrics["side_by_side_mp4"],
                    "temporal_strip_16": metrics["temporal_strip_16"],
                    "review_sheet": metrics["review_sheet"],
                    "frames_dir": metrics["frames_dir"],
                    "full_psnr": metrics.get("full_psnr", ""),
                    "mask_psnr": metrics.get("mask_psnr", ""),
                    "boundary_psnr": metrics.get("boundary_psnr", ""),
                    "outside_psnr": metrics.get("outside_psnr", ""),
                    "outside_mae": metrics.get("outside_mae", ""),
                    "temporal_diff_mae": metrics.get("temporal_diff_mae", ""),
                    "raw_primary": True,
                    "diagnostic_comp_used": False,
                }
            )
            visual_rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "checkpoint": "step0_official_minimax",
                    "condition_path": row.get("condition_path", ""),
                    "winner_path": row.get("winner_path", ""),
                    "loser_path": row.get("loser_path", ""),
                    "mask_path": row.get("mask_path", ""),
                    "raw_step0": metrics["raw_output_mp4"],
                    "side_by_side": metrics["side_by_side_mp4"],
                    "temporal_strip_16": metrics["temporal_strip_16"],
                    "review_sheet": metrics["review_sheet"],
                    "frames_reviewed": "PENDING_CODEX_REVIEW",
                    "classification": "PENDING_CODEX_VISUAL_REVIEW",
                    "reason": "",
                }
            )

    summary = {
        "status": "MINIMAX_STEP0_BASELINE_PENDING_CODEX_REVIEW",
        "label": args.label,
        "model_dir": str(model_dir),
        "device": str(device),
        "dtype": args.dtype,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "iterations": args.iterations,
        "split_counts": {split: len(rows) for split, rows, _ in split_rows},
        "aggregate_by_split": aggregate_by_split(metric_rows),
        "output_root": str(output_root / args.label),
        "runtime_seconds": time.time() - start,
        "raw_output_primary": True,
        "diagnostic_comp_used": False,
        "training_launched": False,
    }
    write_csv(reports_root / f"{args.label}_metrics.csv", metric_rows)
    write_csv(reports_root / f"{args.label}_visual_review.csv", visual_rows)
    write_json(reports_root / f"{args.label}_summary.json", summary)
    md = [
        "# Exp40 MiniMax Step0 Baseline",
        "",
        f"Status: `{summary['status']}`",
        "",
        f"Label: `{args.label}`",
        f"Rows: `{summary['split_counts']}`",
        f"Model dir: `{model_dir}`",
        f"Raw output primary: `{summary['raw_output_primary']}`",
        f"Diagnostic comp used: `{summary['diagnostic_comp_used']}`",
        "",
        "Aggregate by split:",
    ]
    for split, agg in summary["aggregate_by_split"].items():
        md.append(f"- `{split}`: `{agg}`")
    (reports_root / f"{args.label}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    heartbeat(hb, "done")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
