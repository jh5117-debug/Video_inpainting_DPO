#!/usr/bin/env python3
"""Run one real Exp20 Stage1 trial and evaluate it on the locked dev split."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def write_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp20_autoresearch_scale_adaptive_region_dpo")
    parser.add_argument("--log-root", default="/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp20")
    parser.add_argument("--dev-root", default="/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/dev_boundary_search_v1_e385cc27_roots")
    parser.add_argument("--results-tsv", default="exp20_autoresearch_scale_adaptive_region_dpo/results.tsv")
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--main-process-port", default="29630")
    parser.add_argument("--retry-on-crash", type=int, default=1)
    parser.add_argument("--python-bin", default="/mnt/nas/hj/conda_envs/diffueraser/bin/python")
    parser.add_argument("--accelerate-bin", default="/mnt/nas/hj/conda_envs/diffueraser/bin/accelerate")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())
    config_hash = cfg["config_hash"]
    trial_id = cfg["trial_id"]
    trial_dir = Path(args.output_root) / "trials" / f"{trial_id}_{config_hash}"
    log_dir = Path(args.log_root) / "trials" / f"{trial_id}_{config_hash}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "trial_config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")

    train_seconds = int(float(cfg.get("train_minutes", 30)) * 60)
    checkpointing_steps = 999999
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["BOUNDARY_MODE"] = "outer"
    env["LINGBOT_PROCESS_NAME"] = f"exp20-{trial_id}"
    env["PROCESS_TITLE"] = f"exp20-{trial_id}"

    train_cmd = [
        args.accelerate_bin,
        "launch",
        "--num_processes",
        "1",
        "--mixed_precision",
        "bf16",
        "--main_process_port",
        str(args.main_process_port),
        "exp20_autoresearch_scale_adaptive_region_dpo/code/train_exp20_stage1.py",
        "--base_model_name_or_path",
        "/mnt/nas/hj/weights/stable-diffusion-v1-5",
        "--vae_path",
        "/mnt/nas/hj/weights/sd-vae-ft-mse",
        "--ref_model_path",
        "/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000",
        "--policy_init_path",
        "/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000",
        "--dpo_data_root",
        "/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/exp09_10_11_youtubevos_gtwin_d3comp_pai",
        "--dpo_dataset_type",
        "generated_loser_manifest",
        "--preference_manifest",
        "/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/exp09_10_11_youtubevos_gtwin_d3comp_pai/manifests/selected_primary_comp.gtwin.pai_paths.jsonl",
        "--train_mask_mode",
        "partial",
        "--mask_from_manifest",
        "true",
        "--loss_region_mode",
        "region",
        "--gap_normalization",
        "log_ratio",
        "--gap_eps",
        "1e-6",
        "--lose_gap_clip_tau",
        "1.0",
        "--mask_region_weight",
        str(cfg.get("mask_contribution", 1.0)),
        "--boundary_region_weight",
        str(cfg.get("boundary_contribution", 0.75)),
        "--outside_region_weight",
        str(cfg.get("outside_contribution", 0.05)),
        "--radius_mode",
        str(cfg["radius_mode"]),
        "--radius_value",
        str(cfg.get("radius_value", 0.0)),
        "--adaptive_k",
        str(cfg.get("adaptive_k", 1.0)),
        "--aggregation",
        str(cfg.get("aggregation", "legacy_global_weighted_mean")),
        "--boundary_contribution",
        str(cfg.get("boundary_contribution", 0.75)),
        "--mask_contribution",
        str(cfg.get("mask_contribution", 1.0)),
        "--outside_contribution",
        str(cfg.get("outside_contribution", 0.05)),
        "--distance_cache_dir",
        str(trial_dir / "distance_cache"),
        "--max_train_seconds",
        str(train_seconds),
        "--legacy_exact",
        "true" if cfg["radius_mode"] == "legacy_latent_exact" else "false",
        "--trial_config",
        str(cfg_path),
        "--trial_id",
        trial_id,
        "--output_dir",
        str(trial_dir),
        "--logging_dir",
        "logs-dpo-stage1",
        "--val_data_dir",
        "/mnt/workspace/hj/nas_hj/data/external/davis_432_240",
        "--resolution",
        "512",
        "--train_height",
        "320",
        "--train_width",
        "512",
        "--nframes",
        "16",
        "--train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "4",
        "--dataloader_num_workers",
        "0",
        "--learning_rate",
        "1e-6",
        "--lr_scheduler",
        "constant",
        "--lr_warmup_steps",
        "500",
        "--max_train_steps",
        "999999",
        "--checkpointing_steps",
        str(checkpointing_steps),
        "--checkpoints_total_limit",
        "3",
        "--validation_steps",
        "999999",
        "--logging_steps",
        "10",
        "--val_num_inference_steps",
        "6",
        "--val_mask_dilation_iter",
        "0",
        "--mixed_precision",
        "bf16",
        "--vae_dtype",
        "fp32",
        "--policy_dtype",
        "auto",
        "--ref_dtype",
        "bf16",
        "--text_dtype",
        "bf16",
        "--beta_dpo",
        "10",
        "--sft_reg_weight",
        "0.0",
        "--lose_gap_weight",
        "0.25",
        "--winner_abs_reg_weight",
        "0.05",
        "--winner_gap_reg_weight",
        "1.0",
        "--winner_gap_reg_margin",
        "0.0",
        "--winner_gap_reg_mode",
        "relu",
        "--dpo_diag_log_every",
        "10",
        "--dpo_diag_save_csv",
        "true",
        "--dpo_diag_save_wandb",
        "false",
        "--report_to",
        "none",
        "--seed",
        str(cfg.get("seed", 20260619)),
        "--split_pos_neg_forward",
        "--set_grads_to_none",
    ]

    attempts = max(1, int(args.retry_on_crash) + 1)
    train_rc = 1
    for attempt in range(1, attempts + 1):
        write_status(trial_dir / "trial_status.json", {"status": "running", "attempt": attempt, "gpu_id": args.gpu_id})
        train_rc = run_logged(train_cmd, log_dir / f"train_attempt{attempt}.log", env)
        if train_rc == 0:
            break
        time.sleep(5)
    if train_rc != 0:
        write_status(trial_dir / "trial_status.json", {"status": "crash", "returncode": train_rc})
        return train_rc

    required = [trial_dir / "last_weights" / "unet_main" / "config.json", trial_dir / "last_weights" / "brushnet" / "config.json"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        write_status(trial_dir / "trial_status.json", {"status": "blocked", "missing": missing})
        return 4

    eval_cmd = [
        args.python_bin,
        "exp20_autoresearch_scale_adaptive_region_dpo/code/evaluate_trial.py",
        "--trial-config",
        str(cfg_path),
        "--trial-dir",
        str(trial_dir),
        "--results-tsv",
        args.results_tsv,
        "--dev-root",
        args.dev_root,
        "--gpu-id",
        args.gpu_id,
        "--video-length",
        "24",
        "--compute-lpips",
        "--compute-ewarp",
    ]
    write_status(trial_dir / "trial_status.json", {"status": "evaluating", "gpu_id": args.gpu_id})
    eval_rc = run_logged(eval_cmd, log_dir / "evaluate.log", env)
    if eval_rc != 0:
        write_status(trial_dir / "trial_status.json", {"status": "blocked", "stage": "evaluate", "returncode": eval_rc})
        return eval_rc
    write_status(trial_dir / "trial_status.json", {"status": "evaluated", "gpu_id": args.gpu_id})
    print(json.dumps({"status": "evaluated", "trial_dir": str(trial_dir), "log_dir": str(log_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
