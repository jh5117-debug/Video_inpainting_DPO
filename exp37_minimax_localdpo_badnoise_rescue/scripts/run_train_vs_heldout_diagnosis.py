#!/usr/bin/env python3
"""Evaluate an Exp36 MiniMax checkpoint on train16 and heldout16."""

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
    parser.add_argument("--scope", default="S1")
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--train-rows", type=int, default=16)
    parser.add_argument("--heldout-rows", type=int, default=16)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reports-root", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--heartbeat", default="")
    return parser.parse_args()


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
                seen.add(key)
                keys.append(key)
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


def mean_delta(rows: list[dict[str, object]], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.repo_dir).resolve()))
    sys.path.insert(0, str(Path(args.project_root).resolve()))

    from diffusers.models import AutoencoderKLWan  # noqa: WPS433
    from diffusers.schedulers import UniPCMultistepScheduler  # noqa: WPS433
    from transformer_minimax_remover import Transformer3DModel  # noqa: WPS433
    from exp30_vor_or_multimodel_minimax.scripts.run_minimax_gate64_adapter_gate_v3 import (  # noqa: WPS433
        BatchCache,
        read_jsonl,
        run_pipeline,
    )
    from exp36_minimax_objective_rescue.scripts.run_minimax_winner_sft_positive_control import (  # noqa: WPS433
        comparison_strip,
        load_transformer_for_scope,
    )

    output_root = Path(args.output_root)
    reports_root = Path(args.reports_root)
    output_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    hb = Path(args.heartbeat) if args.heartbeat else output_root / "train_vs_heldout.heartbeat"

    train_rows = read_jsonl(Path(args.train_manifest))[: args.train_rows]
    heldout_rows = read_jsonl(Path(args.heldout_manifest))[: args.heldout_rows]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    model_dir = Path(args.model_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    heartbeat(hb, "load_models")
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=dtype).to(device).eval()
    base = load_transformer_for_scope(
        Transformer3DModel,
        model_dir,
        dtype,
        device,
        args.scope,
        args.seed,
        None,
    ).eval()
    step10 = load_transformer_for_scope(
        Transformer3DModel,
        model_dir,
        dtype,
        device,
        args.scope,
        args.seed,
        checkpoint_dir,
    ).eval()
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    latents_std = (1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)).to(device, dtype)
    cache = BatchCache(vae, latents_mean, latents_std, device, dtype)

    metric_rows: list[dict[str, object]] = []
    visual_rows: list[dict[str, object]] = []
    split_rows = [("train16", train_rows), ("heldout16", heldout_rows)]
    for split, rows in split_rows:
        for idx, row in enumerate(rows, 1):
            sample_id = str(row["sample_id"])
            heartbeat(hb, f"{split}:{idx}/{len(rows)}:{sample_id}")
            row_root = output_root / split / sample_id
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
            step10_metrics = run_pipeline(
                step10,
                vae,
                UniPCMultistepScheduler,
                model_dir,
                cache,
                row,
                row_root / "step10",
                args.seed,
                args.num_inference_steps,
                args.iterations,
            )
            strip = row_root / "step0_vs_step10_strip_16.jpg"
            comparison_strip(Path(step0_metrics["frames_dir"]), Path(step10_metrics["frames_dir"]), strip)
            metric_row: dict[str, object] = {
                "split": split,
                "sample_id": sample_id,
                "source_group": row.get("source_group", ""),
                "source_type": row.get("source_type", ""),
                "classification_final": row.get("classification_final", ""),
                "model": row.get("model", ""),
                "step0_frames": step0_metrics["frames_dir"],
                "step10_frames": step10_metrics["frames_dir"],
                "step0_raw_output_mp4": step0_metrics["raw_output_mp4"],
                "step10_raw_output_mp4": step10_metrics["raw_output_mp4"],
                "comparison_strip": str(strip),
            }
            for key in ("full_psnr", "mask_psnr", "boundary_psnr", "outside_psnr", "outside_mae", "temporal_diff_mae"):
                metric_row[f"step0_{key}"] = step0_metrics.get(key, "")
                metric_row[f"step10_{key}"] = step10_metrics.get(key, "")
                if isinstance(step0_metrics.get(key), float) and isinstance(step10_metrics.get(key), float):
                    metric_row[f"delta_{key}"] = float(step10_metrics[key]) - float(step0_metrics[key])
            metric_rows.append(metric_row)
            visual_rows.append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "source_group": row.get("source_group", ""),
                    "comparison_strip": str(strip),
                    "step0_output": step0_metrics["raw_output_mp4"],
                    "step10_output": step10_metrics["raw_output_mp4"],
                    "frames_reviewed": "0,mid,last,16-strip",
                    "classification": "PENDING_CODEX_VISUAL_REVIEW",
                    "reason": "",
                }
            )

    summary = {
        "status": "PENDING_CODEX_VISUAL_REVIEW",
        "scope": args.scope,
        "checkpoint_dir": str(checkpoint_dir),
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
        "train_mean_delta_mask_psnr": mean_delta([r for r in metric_rows if r["split"] == "train16"], "delta_mask_psnr"),
        "train_mean_delta_boundary_psnr": mean_delta([r for r in metric_rows if r["split"] == "train16"], "delta_boundary_psnr"),
        "heldout_mean_delta_mask_psnr": mean_delta([r for r in metric_rows if r["split"] == "heldout16"], "delta_mask_psnr"),
        "heldout_mean_delta_boundary_psnr": mean_delta([r for r in metric_rows if r["split"] == "heldout16"], "delta_boundary_psnr"),
        "output_root": str(output_root),
    }
    write_csv(reports_root / "exp37_minimax_train_vs_heldout_metrics.csv", metric_rows)
    write_csv(reports_root / "exp37_minimax_train_vs_heldout_visual_review.csv", visual_rows)
    write_json(reports_root / "exp37_minimax_train_vs_heldout_summary.json", summary)
    md = [
        "# Exp37 MiniMax Train-vs-Heldout Diagnosis",
        "",
        "Status: `PENDING_CODEX_VISUAL_REVIEW`",
        "",
        f"- Scope: `{args.scope}`",
        f"- Checkpoint: `{checkpoint_dir}`",
        f"- Train rows: `{len(train_rows)}`",
        f"- Heldout rows: `{len(heldout_rows)}`",
        f"- Train mask/boundary mean deltas: `{summary['train_mean_delta_mask_psnr']}` / `{summary['train_mean_delta_boundary_psnr']}`",
        f"- Heldout mask/boundary mean deltas: `{summary['heldout_mean_delta_mask_psnr']}` / `{summary['heldout_mean_delta_boundary_psnr']}`",
        "",
        "Codex visual review is required before assigning MINIMAX_GENERALIZATION_FAILURE, MINIMAX_OBJECTIVE_SIGNAL_TOO_WEAK, or MINIMAX_RESCUE_PROMISING.",
    ]
    (reports_root / "exp37_minimax_train_vs_heldout_diagnosis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    heartbeat(hb, "done")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
