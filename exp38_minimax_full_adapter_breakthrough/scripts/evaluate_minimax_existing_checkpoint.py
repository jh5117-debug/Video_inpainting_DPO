#!/usr/bin/env python3
"""Evaluate an existing MiniMax checkpoint on train/heldout manifests.

This script performs inference only. It never trains or modifies the input
checkpoint. It is used by Exp38 to answer whether prior MiniMax recipes improve
training rows before running any new rescue recipe.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--heldout-manifest", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-format", choices=["hf", "model_state"], default="hf")
    parser.add_argument("--scope", choices=["S0", "S1"], default="S0")
    parser.add_argument("--label", required=True)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--train-rows", type=int, default=32)
    parser.add_argument("--heldout-rows", type=int, default=16)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reports-root", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=6)
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


def mean_metric(rows: list[dict[str, object]], key: str) -> float | None:
    vals: list[float] = []
    for row in rows:
        val = row.get(key)
        if isinstance(val, (int, float)):
            vals.append(float(val))
    return sum(vals) / len(vals) if vals else None


def load_model_state_checkpoint(transformer_cls, model_dir: Path, checkpoint_dir: Path, dtype: torch.dtype, device: torch.device, scope: str, seed: int):
    from exp36_minimax_objective_rescue.scripts.run_minimax_winner_sft_positive_control import (  # noqa: WPS433
        load_transformer_for_scope,
    )

    return load_transformer_for_scope(transformer_cls, model_dir, dtype, device, scope, seed, checkpoint_dir).eval()


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
    from exp37_minimax_localdpo_badnoise_rescue.scripts.run_exp37_minimax_localdpo_badnoise_10step import (  # noqa: WPS433
        comparison_strip,
    )

    output_root = Path(args.output_root).resolve()
    reports_root = Path(args.reports_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    hb = Path(args.heartbeat) if args.heartbeat else output_root / f"{args.label}.heartbeat"

    train_rows = read_jsonl(Path(args.train_manifest))[: args.train_rows]
    heldout_rows = read_jsonl(Path(args.heldout_manifest))[: args.heldout_rows]
    split_rows = [("train", train_rows), ("heldout", heldout_rows)]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    model_dir = Path(args.model_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()

    heartbeat(hb, "load_models")
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=dtype).to(device).eval()
    base = Transformer3DModel.from_pretrained(model_dir / "transformer", torch_dtype=dtype).to(device).eval()
    if args.checkpoint_format == "hf":
        step = Transformer3DModel.from_pretrained(checkpoint_dir, torch_dtype=dtype).to(device).eval()
    else:
        step = load_model_state_checkpoint(Transformer3DModel, model_dir, checkpoint_dir, dtype, device, args.scope, args.seed)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    latents_std = (1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)).to(device, dtype)
    cache = BatchCache(vae, latents_mean, latents_std, device, dtype)

    metric_rows: list[dict[str, object]] = []
    visual_rows: list[dict[str, object]] = []
    start = time.time()
    for split, rows in split_rows:
        for idx, row in enumerate(rows, 1):
            sample_id = str(row["sample_id"])
            heartbeat(hb, f"{args.label}:{split}:{idx}/{len(rows)}:{sample_id}")
            row_root = output_root / args.label / split / sample_id
            step0_metrics = run_pipeline(
                base,
                vae,
                UniPCMultistepScheduler,
                model_dir,
                cache,
                row,
                row_root / "step0",
                args.seed,
                args.num_inference_steps,
                args.iterations,
            )
            step_metrics = run_pipeline(
                step,
                vae,
                UniPCMultistepScheduler,
                model_dir,
                cache,
                row,
                row_root / "step",
                args.seed,
                args.num_inference_steps,
                args.iterations,
            )
            strip = row_root / "step0_vs_step_strip_16.jpg"
            diff_stats = comparison_strip(Path(step0_metrics["frames_dir"]), Path(step_metrics["frames_dir"]), strip)
            metric_row: dict[str, object] = {
                "label": args.label,
                "split": split,
                "sample_id": sample_id,
                "source_group": row.get("source_group", ""),
                "source_type": row.get("source_type", ""),
                "classification": row.get("classification_final", row.get("classification", "")),
                "model": row.get("model", row.get("candidate_source", "")),
                "step0_frames": step0_metrics["frames_dir"],
                "step_frames": step_metrics["frames_dir"],
                "step0_raw_output_mp4": step0_metrics["raw_output_mp4"],
                "step_raw_output_mp4": step_metrics["raw_output_mp4"],
                "step0_temporal_strip_16": step0_metrics["temporal_strip_16"],
                "step_temporal_strip_16": step_metrics["temporal_strip_16"],
                "comparison_strip": str(strip),
                **diff_stats,
            }
            for key in ("full_psnr", "mask_psnr", "boundary_psnr", "outside_psnr", "outside_mae", "temporal_diff_mae"):
                metric_row[f"step0_{key}"] = step0_metrics.get(key, "")
                metric_row[f"step_{key}"] = step_metrics.get(key, "")
                if isinstance(step0_metrics.get(key), float) and isinstance(step_metrics.get(key), float):
                    metric_row[f"delta_{key}"] = float(step_metrics[key]) - float(step0_metrics[key])
            metric_rows.append(metric_row)
            visual_rows.append(
                {
                    "label": args.label,
                    "split": split,
                    "sample_id": sample_id,
                    "source_group": row.get("source_group", ""),
                    "comparison_strip": str(strip),
                    "step0_output": step0_metrics["raw_output_mp4"],
                    "step_output": step_metrics["raw_output_mp4"],
                    "frames_reviewed": "PENDING_CODEX_REVIEW",
                    "pixel_diff_mean": diff_stats["pixel_diff_mean_review_frames"],
                    "classification": "PENDING_CODEX_VISUAL_REVIEW",
                    "reason": "",
                }
            )

    train_metric_rows = [r for r in metric_rows if r["split"] == "train"]
    heldout_metric_rows = [r for r in metric_rows if r["split"] == "heldout"]
    summary = {
        "status": "PENDING_CODEX_VISUAL_REVIEW",
        "label": args.label,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_format": args.checkpoint_format,
        "scope": args.scope,
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
        "train_mean_delta_full_psnr": mean_metric(train_metric_rows, "delta_full_psnr"),
        "train_mean_delta_mask_psnr": mean_metric(train_metric_rows, "delta_mask_psnr"),
        "train_mean_delta_boundary_psnr": mean_metric(train_metric_rows, "delta_boundary_psnr"),
        "train_mean_delta_outside_psnr": mean_metric(train_metric_rows, "delta_outside_psnr"),
        "heldout_mean_delta_full_psnr": mean_metric(heldout_metric_rows, "delta_full_psnr"),
        "heldout_mean_delta_mask_psnr": mean_metric(heldout_metric_rows, "delta_mask_psnr"),
        "heldout_mean_delta_boundary_psnr": mean_metric(heldout_metric_rows, "delta_boundary_psnr"),
        "heldout_mean_delta_outside_psnr": mean_metric(heldout_metric_rows, "delta_outside_psnr"),
        "output_root": str(output_root / args.label),
        "runtime_seconds": time.time() - start,
    }
    write_csv(reports_root / f"{args.label}_metrics.csv", metric_rows)
    write_csv(reports_root / f"{args.label}_visual_review.csv", visual_rows)
    write_json(reports_root / f"{args.label}_summary.json", summary)
    heartbeat(hb, "done")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

