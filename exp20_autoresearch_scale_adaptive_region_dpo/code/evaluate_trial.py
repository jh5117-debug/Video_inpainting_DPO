#!/usr/bin/env python3
"""Evaluate one Exp20 Stage1 trial on the locked dev split.

This wrapper intentionally delegates metric computation to the existing
frame-wise evaluator. It does not implement metrics itself.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp20_autoresearch_scale_adaptive_region_dpo.code.search_controller import RESULT_FIELDS  # noqa: E402


def run(cmd: list[str], *, env: dict[str, str] | None = None, log_path: Path | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True) if log_path else None
    if log_path:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
    else:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def read_summary(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    return row


def append_result(results_path: Path, row: dict[str, Any]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    exists = results_path.exists() and results_path.stat().st_size > 0
    lock_path = results_path.with_suffix(results_path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with results_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t", extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in RESULT_FIELDS})
        fcntl.flock(lock, fcntl.LOCK_UN)


def strict_weight_dir(path: Path) -> None:
    required = [
        path / "unet_main" / "config.json",
        path / "brushnet" / "config.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"candidate checkpoint is not a DiffuEraser exported weight dir: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-config", required=True)
    parser.add_argument("--trial-dir", required=True)
    parser.add_argument("--results-tsv", default="exp20_autoresearch_scale_adaptive_region_dpo/results.tsv")
    parser.add_argument("--candidate-weights", default="")
    parser.add_argument("--dev-root", required=True)
    parser.add_argument("--sft-stage2-weights", default="/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000")
    parser.add_argument("--base-model", default="/mnt/nas/hj/weights/stable-diffusion-v1-5")
    parser.add_argument("--vae", default="/mnt/nas/hj/weights/sd-vae-ft-mse")
    parser.add_argument("--propainter", default="/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter")
    parser.add_argument("--pcm", default="/mnt/nas/hj/weights/PCM_Weights")
    parser.add_argument("--python-bin", default="/mnt/nas/hj/conda_envs/diffueraser/bin/python")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--video-length", type=int, default=24)
    parser.add_argument("--compute-lpips", action="store_true")
    parser.add_argument("--compute-ewarp", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.trial_config)
    cfg = json.loads(cfg_path.read_text())
    trial_dir = Path(args.trial_dir)
    candidate = Path(args.candidate_weights) if args.candidate_weights else trial_dir / "last_weights"
    strict_weight_dir(candidate)

    eval_root = trial_dir / "eval_dev"
    hybrid_root = trial_dir / "eval_hybrid_s1_sft_s2"
    hybrid_weights = hybrid_root / "last_weights"
    if not hybrid_weights.exists():
        run(
            [
                args.python_bin,
                "tools/build_diffueraser_dpoS1_sftS2_hybrid.py",
                "--dpo_stage1_weights",
                str(candidate),
                "--sft_stage2_weights",
                args.sft_stage2_weights,
                "--output_dir",
                str(hybrid_root),
                "--report_path",
                str(trial_dir / "hybrid_key_merge_report.md"),
                "--strict",
                "false",
            ],
            log_path=trial_dir / "build_hybrid.log",
        )
    strict_weight_dir(hybrid_weights)

    dev_root = Path(args.dev_root)
    save_path = eval_root / cfg["trial_id"]
    cmd = [
        args.python_bin,
        "exp20_autoresearch_scale_adaptive_region_dpo/code/run_exp20_framewise_protocol_eval.py",
        "--video_root",
        str(dev_root / "JPEGImages_432_240"),
        "--mask_root",
        str(dev_root / "test_masks"),
        "--gt_root",
        str(dev_root / "JPEGImages_432_240"),
        "--save_path",
        str(save_path),
        "--inference_seed",
        "20260619",
        "--label",
        cfg["trial_id"],
        "--diffueraser_path",
        str(hybrid_weights),
        "--base_model_path",
        args.base_model,
        "--vae_path",
        args.vae,
        "--propainter_model_dir",
        args.propainter,
        "--pcm_weights_path",
        args.pcm,
        "--input_size",
        "432x240",
        "--video_length",
        str(args.video_length),
        "--num_inference_steps",
        "6",
        "--use_pcm",
        "false",
        "--mask_dilation_iter",
        "0",
        "--save_videos",
        "--save_comp_frames",
    ]
    if args.compute_lpips:
        cmd.append("--compute_lpips")
    if args.compute_ewarp:
        cmd.extend(["--compute_ewarp", "--raft_model_path", str(Path(args.propainter) / "raft-things.pth")])
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    run(cmd, env=env, log_path=trial_dir / "eval_dev.log")

    summary = read_summary(save_path / "metrics" / "summary.csv")
    diag_path = trial_dir / "dpo_diagnostics.csv"
    loser_ratio = ""
    max_grad = ""
    if diag_path.exists():
        with diag_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            loser_ratio = rows[-1].get("loser_dominant_ratio", "")
            grad_values = []
            for row in rows:
                try:
                    grad_values.append(float(row.get("grad_norm", "")))
                except Exception:
                    pass
            max_grad = max(grad_values) if grad_values else ""

    result = {
        "trial_id": cfg["trial_id"],
        "parent_id": cfg.get("parent_id", ""),
        "config_hash": cfg["config_hash"],
        "branch_commit": cfg.get("branch_commit", ""),
        "radius_mode": cfg["radius_mode"],
        "radius_value": cfg.get("radius_value", ""),
        "adaptive_k": cfg.get("adaptive_k", ""),
        "boundary_contribution": cfg.get("boundary_contribution", ""),
        "aggregation": cfg.get("aggregation", ""),
        "seed": cfg.get("seed", ""),
        "gpu_ids": args.gpu_id,
        "world_size": 1,
        "effective_batch": 4,
        "train_seconds": cfg.get("train_minutes", 0) * 60,
        "dev_psnr": summary.get("whole_video_psnr_mean", ""),
        "dev_ssim": summary.get("whole_video_ssim_mean", ""),
        "dev_lpips": summary.get("whole_video_lpips_mean", ""),
        "dev_vfid_or_fvd": summary.get("vfid", ""),
        "dev_tc": summary.get("tc_mean", ""),
        "dev_ewarp": summary.get("ewarp_mean", ""),
        "dev_mask_psnr": summary.get("strict_mask_pixel_psnr_mean", ""),
        "dev_boundary_psnr": summary.get("boundary_pixel_psnr_mean", ""),
        "loser_dominant_ratio": loser_ratio,
        "max_grad_norm": max_grad,
        "status": "evaluating_complete",
        "keep_reason": "pending_controller_gate",
        "description": cfg.get("description", ""),
        "checkpoint_path": str(candidate),
        "log_path": str(trial_dir / "eval_dev.log"),
    }
    append_result(Path(args.results_tsv), result)
    (trial_dir / "eval_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
