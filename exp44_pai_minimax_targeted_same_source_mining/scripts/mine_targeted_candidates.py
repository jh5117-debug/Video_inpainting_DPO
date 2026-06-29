#!/usr/bin/env python3
"""Run Exp44 targeted MiniMax same-source mining with official protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch

from exp42_pai_minimax_successful_removal_badnoise.scripts.mine_successful_removal_candidates import (
    auto_classify,
    heartbeat,
    metric_summary,
    read_jsonl,
    sha256_file,
    torch_dtype,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--target-manifest", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reports-root", required=True)
    parser.add_argument("--repo-manifest-root", required=True)
    parser.add_argument("--heartbeat", default="")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def finite_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def seed_rows(rows: list[dict[str, object]]) -> list[tuple[int, dict[str, object], int, int]]:
    out = []
    for row_idx, row in enumerate(rows):
        seeds = row.get("exp44_seeds", [])
        if not isinstance(seeds, list):
            raise ValueError(f"row {row_idx} has invalid exp44_seeds")
        for seed_idx, seed in enumerate(seeds):
            out.append((row_idx, row, seed_idx, int(seed)))
    return out


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("invalid shard settings")

    sys.path.insert(0, str(Path(args.repo_dir).resolve()))
    sys.path.insert(0, str(Path(args.project_root).resolve()))

    from diffusers.models import AutoencoderKLWan  # noqa: WPS433
    from diffusers.schedulers import UniPCMultistepScheduler  # noqa: WPS433
    from transformer_minimax_remover import Transformer3DModel  # noqa: WPS433
    from exp30_vor_or_multimodel_minimax.scripts.run_minimax_gate64_adapter_gate_v3 import (  # noqa: WPS433
        BatchCache,
        run_pipeline,
    )

    target_manifest = Path(args.target_manifest).resolve()
    rows = read_jsonl(target_manifest)
    all_seed_rows = seed_rows(rows)
    assigned = [item for idx, item in enumerate(all_seed_rows) if idx % args.num_shards == args.shard_index]
    if not assigned:
        raise RuntimeError("no assigned seed rows")

    output_root = Path(args.output_root).resolve()
    reports_root = Path(args.reports_root).resolve()
    manifest_root = Path(args.repo_manifest_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    hb = Path(args.heartbeat).resolve() if args.heartbeat else output_root / f"worker{args.shard_index}.heartbeat"

    model_dir = Path(args.model_dir).resolve()
    for child in ("vae", "transformer", "scheduler"):
        if not (model_dir / child).exists():
            raise FileNotFoundError(f"missing MiniMax model component: {model_dir / child}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch_dtype(args.dtype)
    torch.manual_seed(min(seed for _, _, _, seed in assigned))

    heartbeat(hb, "loading_model")
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=dtype).to(device).eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    transformer = Transformer3DModel.from_pretrained(model_dir / "transformer", torch_dtype=dtype).to(device).eval()
    for param in transformer.parameters():
        param.requires_grad_(False)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    latents_std = (1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)).to(device, dtype)
    cache = BatchCache(vae, latents_mean, latents_std, device, dtype)

    all_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    status_jsonl = output_root / f"worker{args.shard_index}_status.jsonl"
    start_time = time.time()
    for count, (row_idx, row, seed_idx, seed) in enumerate(assigned, 1):
        sample_id = str(row["sample_id"])
        scene_group = str(row.get("scene_group", ""))
        candidate_id = f"{sample_id}__exp44seed{seed_idx}_{seed}"
        heartbeat(hb, f"{count}/{len(assigned)}:{candidate_id}")
        candidate_root = output_root / "candidates" / scene_group / sample_id / f"seed{seed_idx}_{seed}"
        metrics_path = candidate_root / "metrics.json"
        try:
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            else:
                metrics = run_pipeline(
                    transformer,
                    vae,
                    UniPCMultistepScheduler,
                    model_dir,
                    cache,
                    row,
                    candidate_root,
                    seed,
                    args.num_inference_steps,
                    args.iterations,
                )
                write_json(metrics_path, metrics)
            auto_class, reason = auto_classify(metrics)
            out_row = {
                "candidate_id": candidate_id,
                "sample_id": sample_id,
                "scene_group": scene_group,
                "source_type": row.get("source_type", ""),
                "source_manifest_path": row.get("exp44_from_exp42_source_manifest_path", row.get("source_manifest_path", "")),
                "source_row_index": row_idx,
                "seed_index": seed_idx,
                "seed": seed,
                "exp44_bucket": row.get("exp44_bucket", ""),
                "exp44_goal": row.get("exp44_goal", ""),
                "exp44_priority": row.get("exp44_priority", ""),
                "condition_path": row.get("condition_path", ""),
                "winner_path": row.get("winner_path", ""),
                "mask_path": row.get("mask_path", ""),
                "raw_output_mp4": metrics.get("raw_output_mp4", ""),
                "side_by_side_mp4": metrics.get("side_by_side_mp4", ""),
                "temporal_strip_16": metrics.get("temporal_strip_16", ""),
                "review_sheet": metrics.get("review_sheet", ""),
                "frames_dir": metrics.get("frames_dir", ""),
                "full_psnr": metrics.get("full_psnr", ""),
                "mask_psnr": metrics.get("mask_psnr", ""),
                "boundary_psnr": metrics.get("boundary_psnr", ""),
                "outside_psnr": metrics.get("outside_psnr", ""),
                "outside_mae": metrics.get("outside_mae", ""),
                "temporal_diff_mae": metrics.get("temporal_diff_mae", ""),
                "auto_classification": auto_class,
                "auto_reason": reason,
                "codex_visual_review": "PENDING",
                "raw_output_primary": True,
                "diagnostic_comp_used": False,
                "vor_eval_used": False,
                "num_inference_steps": args.num_inference_steps,
                "iterations": args.iterations,
                "dtype": args.dtype,
                "worker_shard": args.shard_index,
                "num_shards": args.num_shards,
            }
            all_rows.append(out_row)
            with status_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(out_row, sort_keys=True) + "\n")
        except Exception as exc:  # noqa: BLE001
            failed = {
                "candidate_id": candidate_id,
                "sample_id": sample_id,
                "scene_group": scene_group,
                "seed": seed,
                "auto_classification": "TECHNICAL_INVALID",
                "error": repr(exc),
                "worker_shard": args.shard_index,
            }
            failed_rows.append(failed)
            all_rows.append(failed)
            with status_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failed, sort_keys=True) + "\n")

    successes = [row for row in all_rows if row.get("auto_classification") == "SUCCESSFUL_REMOVAL_CANDIDATE"]
    failures = [row for row in all_rows if row.get("auto_classification") == "MEDIUM_HARD_REMOVAL"]

    suffix = f"worker{args.shard_index}" if args.num_shards > 1 else ""
    all_manifest = manifest_root / f"exp44_targeted_candidates_all{('_' + suffix) if suffix else ''}.jsonl"
    success_manifest = manifest_root / f"exp44_targeted_success_auto{('_' + suffix) if suffix else ''}.jsonl"
    failure_manifest = manifest_root / f"exp44_targeted_failure_auto{('_' + suffix) if suffix else ''}.jsonl"
    write_jsonl(all_manifest, all_rows)
    write_jsonl(success_manifest, successes)
    write_jsonl(failure_manifest, failures)
    write_csv(reports_root / f"exp44_targeted_mining_metrics{('_' + suffix) if suffix else ''}.csv", all_rows)

    summary = {
        "status": "MINIMAX_TARGETED_MINING_COMPLETED" if not failed_rows else "MINIMAX_TARGETED_MINING_PARTIAL",
        "runtime_seconds": round(time.time() - start_time, 3),
        "device": str(device),
        "dtype": args.dtype,
        "target_manifest": str(target_manifest),
        "target_manifest_sha256": sha256_file(target_manifest),
        "total_target_candidates": len(all_seed_rows),
        "assigned_candidates": len(assigned),
        "num_candidates": len(all_rows),
        "num_successful_candidates": len(successes),
        "num_failure_candidates": len(failures),
        "num_failed_candidates": len(failed_rows),
        "worker_shard": args.shard_index,
        "num_shards": args.num_shards,
        "protocol": {
            "scheduler": "UniPCMultistepScheduler",
            "num_inference_steps": args.num_inference_steps,
            "iterations": args.iterations,
            "cfg": "none",
            "raw_output_primary": True,
        },
        "metrics": metric_summary(all_rows),
        "all_manifest": str(all_manifest),
        "success_manifest": str(success_manifest),
        "failure_manifest": str(failure_manifest),
    }
    write_json(reports_root / f"exp44_targeted_mining_summary{('_' + suffix) if suffix else ''}.json", summary)
    report = [
        "# Exp44 Targeted MiniMax Same-Source Mining",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This worker ran official MiniMax raw inference only. It did not train,",
        "did not use VOR-Eval, did not use hard comp, and did not modify shared",
        "metrics or the MiniMax official repository.",
        "",
        "## Protocol",
        "",
        f"- Assigned candidates: `{len(assigned)}`",
        f"- Successful candidates: `{len(successes)}`",
        f"- Medium-hard failure candidates: `{len(failures)}`",
        f"- Technical invalid / failed: `{len(failed_rows)}`",
        "- Scheduler: `UniPCMultistepScheduler`",
        f"- num_inference_steps: `{args.num_inference_steps}`",
        f"- iterations: `{args.iterations}`",
        f"- dtype: `{args.dtype}`",
        "- raw output primary: `true`",
        "- VOR-Eval used: `false`",
        "",
        "Automatic labels are provisional; Milestone C must perform strict visual relabeling.",
        "",
        f"NAS evidence root: `{output_root}`",
        "",
    ]
    (reports_root / f"exp44_targeted_mining{('_' + suffix) if suffix else ''}.md").write_text("\n".join(report), encoding="utf-8")
    heartbeat(hb, f"done:{summary['status']}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
