#!/usr/bin/env python3
"""Run the first Exp23 paired Stage1/Stage2 sweep item on PAI."""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = PROJECT_ROOT / "exp23_two_stage_pool_morphology_sweep"
REG_ROOT = PROJECT_ROOT / "experiment_registry" / "exp23_two_stage_pool_morphology_sweep"
OUTPUT_ROOT = Path("/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp23_two_stage_pool_morphology_sweep")
LOG_ROOT = Path("/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp23")
EVAL_ROOT = Path("/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp23_two_stage_pool_morphology_sweep")


@dataclass(frozen=True)
class RegionConfig:
    name: str
    legacy_exact: bool
    pool_grid_scale: int
    inner_pool_steps: int
    outer_pool_steps: int
    inner_weight: float
    outer_weight: float
    boundary_region_weight: float


@dataclass(frozen=True)
class RunConfig:
    pair_id: str
    seed: int = 20260619
    stage1_steps: int = 2000
    stage2_steps: int = 2000
    checkpointing_steps: int = 500


def choose_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n[exp23-runner] start={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write("[exp23-runner] cmd=" + " ".join(cmd) + "\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )
        rc = int(proc.wait())
        log.write(f"\n[exp23-runner] end={time.strftime('%Y-%m-%d %H:%M:%S')} returncode={rc}\n")
    return rc


def base_args(out_dir: Path, run: RunConfig, region: RegionConfig, stage: int, stage1_weights: Path | None = None) -> list[str]:
    args = [
        "--base_model_name_or_path", "/mnt/nas/hj/weights/stable-diffusion-v1-5",
        "--vae_path", "/mnt/nas/hj/weights/sd-vae-ft-mse",
        "--dpo_data_root", "/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/exp09_10_11_youtubevos_gtwin_d3comp_pai",
        "--dpo_dataset_type", "generated_loser_manifest",
        "--preference_manifest", "/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/exp09_10_11_youtubevos_gtwin_d3comp_pai/manifests/selected_primary_comp.gtwin.pai_paths.jsonl",
        "--train_mask_mode", "partial",
        "--mask_from_manifest", "true",
        "--loss_region_mode", "region",
        "--gap_normalization", "log_ratio",
        "--gap_eps", "1e-6",
        "--lose_gap_clip_tau", "1.0",
        "--mask_region_weight", "1.0",
        "--boundary_region_weight", str(region.boundary_region_weight),
        "--outside_region_weight", "0.05",
        "--pool_grid_scale", str(region.pool_grid_scale),
        "--inner_pool_steps", str(region.inner_pool_steps),
        "--outer_pool_steps", str(region.outer_pool_steps),
        "--inner_weight", str(region.inner_weight),
        "--outer_weight", str(region.outer_weight),
        "--legacy_exact", "true" if region.legacy_exact else "false",
        "--aggregation", "legacy_global_weighted_mean",
        "--output_dir", str(out_dir),
        "--logging_dir", f"logs-dpo-stage{stage}",
        "--val_data_dir", "/mnt/workspace/hj/nas_hj/data/external/davis_432_240",
        "--resolution", "512",
        "--train_height", "320",
        "--train_width", "512",
        "--nframes", "16",
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--dataloader_num_workers", "0",
        "--learning_rate", "1e-6",
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "500",
        "--checkpointing_steps", str(run.checkpointing_steps),
        "--checkpoints_total_limit", "3",
        "--validation_steps", "999999",
        "--logging_steps", "10",
        "--val_num_inference_steps", "6",
        "--val_mask_dilation_iter", "0",
        "--mixed_precision", "bf16",
        "--vae_dtype", "fp32",
        "--policy_dtype", "auto",
        "--ref_dtype", "bf16",
        "--text_dtype", "bf16",
        "--beta_dpo", "10",
        "--sft_reg_weight", "0.0",
        "--lose_gap_weight", "0.25",
        "--winner_abs_reg_weight", "0.05",
        "--winner_gap_reg_weight", "1.0",
        "--winner_gap_reg_margin", "0.0",
        "--winner_gap_reg_mode", "relu",
        "--dpo_diag_log_every", "10",
        "--dpo_diag_save_csv", "true",
        "--dpo_diag_save_wandb", "false",
        "--report_to", "none",
        "--seed", str(run.seed),
        "--split_pos_neg_forward",
        "--set_grads_to_none",
    ]
    if stage == 1:
        args += [
            "--ref_model_path", "/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000",
            "--policy_init_path", "/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000",
            "--max_train_steps", str(run.stage1_steps),
        ]
    else:
        if stage1_weights is None:
            raise ValueError("stage2 requires stage1 weights")
        args += [
            "--pretrained_dpo_stage1", str(stage1_weights),
            "--baseline_unet_path", "/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000",
            "--ref_model_path", "/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000",
            "--max_train_steps", str(run.stage2_steps),
        ]
    return args


def torchrun(phy_python: str, nproc: int, script: str, args: list[str]) -> list[str]:
    return [
        phy_python, "-m", "torch.distributed.run",
        "--nproc_per_node", str(nproc),
        "--master_port", str(choose_free_port()),
        script,
        *args,
    ]


def make_env(gpus: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": gpus,
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        "PROCESS_TITLE": "Phy",
        "SETPROCTITLE": "Phy",
        "LINGBOT_PROCESS_NAME": "Phy",
        "WANDB_SILENT": "true",
        "WANDB_QUIET": "true",
        "WANDB_CONSOLE": "off",
    })
    return env


def run_model(run: RunConfig, region: RegionConfig, phy_python: str, gpus: str, nproc: int) -> int:
    model_root = OUTPUT_ROOT / "pairs" / run.pair_id / region.name
    log_dir = LOG_ROOT / "pairs" / run.pair_id / region.name
    stage1_dir = model_root / "stage1"
    stage2_dir = model_root / "stage2"
    write_json(model_root / "region_config.json", asdict(region))
    env = make_env(gpus)

    stage1_last = stage1_dir / "last_weights"
    if not (stage1_last / "unet_main" / "config.json").exists():
        rc = run_logged(
            torchrun(
                phy_python,
                nproc,
                "exp23_two_stage_pool_morphology_sweep/code/train_exp23_stage1.py",
                base_args(stage1_dir, run, region, stage=1),
            ),
            log_dir / "stage1.log",
            env,
        )
        if rc != 0:
            return rc

    stage2_last = stage2_dir / "last_weights"
    if not (stage2_last / "unet_main" / "config.json").exists():
        rc = run_logged(
            torchrun(
                phy_python,
                nproc,
                "exp23_two_stage_pool_morphology_sweep/code/train_exp23_stage2.py",
                base_args(stage2_dir, run, region, stage=2, stage1_weights=stage1_last),
            ),
            log_dir / "stage2.log",
            env,
        )
        if rc != 0:
            return rc
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-id", default="phaseA_scale1_pair001_outer2")
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--phy-python", default=os.environ.get("PHY_PYTHON", "/mnt/nas/hj/conda_envs/diffueraser/bin/Phy"))
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    run = RunConfig(pair_id=args.pair_id)
    regions = [
        RegionConfig("fresh_exp11_outer_b075", True, 1, 0, 1, 0.0, 0.75, 0.75),
        RegionConfig("candidate_scale1_outer2_b075", False, 1, 0, 2, 0.0, 0.75, 0.75),
    ]
    state = {
        "status": "RUNNING",
        "pair_id": run.pair_id,
        "gpus": args.gpus,
        "nproc_per_node": args.nproc_per_node,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(REG_ROOT / "queue_state.json", state)
    write_json(EXP_ROOT / "queue_state.json", state)

    for index, region in enumerate(regions, start=1):
        row = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pair_id": run.pair_id,
            "model": region.name,
            "stage": "start",
            "queue_index": index,
            "gpus": args.gpus,
        }
        append_csv(REG_ROOT / "process_registry.csv", row)
        state.update({"current_model": region.name, "queue_index": index})
        write_json(REG_ROOT / "queue_state.json", state)
        rc = run_model(run, region, args.phy_python, args.gpus, args.nproc_per_node)
        row["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        row["stage"] = "done" if rc == 0 else "failed"
        append_csv(REG_ROOT / "process_registry.csv", row)
        if rc != 0:
            state.update({"status": "FAILED", "failed_model": region.name, "returncode": rc})
            write_json(REG_ROOT / "queue_state.json", state)
            write_json(EXP_ROOT / "queue_state.json", state)
            return rc

    state.update({"status": "STAGE1_STAGE2_PAIR_COMPLETED", "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    write_json(REG_ROOT / "queue_state.json", state)
    write_json(EXP_ROOT / "queue_state.json", state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
