#!/usr/bin/env python3
"""Run Exp28 fresh-control vs inner-boundary pairs on PAI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = PROJECT_ROOT / "exp28_fine_inner_boundary_sweep"
REG_ROOT = PROJECT_ROOT / "experiment_registry" / "exp28_fine_inner_boundary_sweep"
OUTPUT_ROOT = Path("/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp28_fine_inner_boundary_sweep")
LOG_ROOT = Path("/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp28_fine_inner_boundary_sweep")
EVAL_ROOT = LOG_ROOT / "paired_davis50_eval"

REFERENCE_PER_DEVICE_BATCH = 1
REFERENCE_GRAD_ACCUM = 1
REFERENCE_WORLD_SIZE = 4
REFERENCE_EFFECTIVE_BATCH = REFERENCE_PER_DEVICE_BATCH * REFERENCE_GRAD_ACCUM * REFERENCE_WORLD_SIZE


@dataclass(frozen=True)
class RegionConfig:
    geometry_mode: str
    inner_radius_px: int
    mask_core_weight: float = 1.0
    inner_weight: float = 0.75
    outer_weight: float = 0.75
    outside_weight: float = 0.05
    boundary_mode: str = "outer"
    legacy_outer_pool_steps: int = 1


@dataclass(frozen=True)
class ModelPlan:
    name: str
    stage1_region: RegionConfig
    stage2_region: RegionConfig


@dataclass(frozen=True)
class RunConfig:
    pair_id: str
    seed: int = 20260625
    stage1_steps: int = 2000
    stage2_steps: int = 2000
    checkpointing_steps: int = 500


PAIR_SPECS = {
    "A": ("pairA_inner2_cli4", "fresh_control_A", "inner2_candidate", 2),
    "B": ("pairB_inner4_cli4", "fresh_control_B", "inner4_candidate", 4),
    "C": ("pairC_inner8_cli4", "fresh_control_C", "inner8_candidate", 8),
}

QUEUE_STATES = {
    "queued",
    "training_control_s1",
    "training_control_s2",
    "training_candidate_s1",
    "training_candidate_s2",
    "evaluating",
    "completed",
    "failed",
}


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
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()), lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def append_queue_event(pair_id: str, state: str, payload: dict[str, object] | None = None) -> None:
    if state not in QUEUE_STATES:
        raise ValueError(f"unknown queue state: {state}")
    row: dict[str, object] = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pair_id": pair_id,
        "state": state,
    }
    if payload:
        row.update(payload)
    append_jsonl(REG_ROOT / "queue_events.jsonl", row)
    append_jsonl(EXP_ROOT / "queue_events.jsonl", row)


def stable_config_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def adjusted_grad_accum(world_size: int, per_device_batch: int = REFERENCE_PER_DEVICE_BATCH) -> int:
    denom = int(world_size) * int(per_device_batch)
    if denom <= 0 or REFERENCE_EFFECTIVE_BATCH % denom != 0:
        raise ValueError(
            f"cannot match reference effective batch {REFERENCE_EFFECTIVE_BATCH} "
            f"with world_size={world_size}, per_device_batch={per_device_batch}"
        )
    return REFERENCE_EFFECTIVE_BATCH // denom


def pair_config_hash(run: RunConfig, plans: list[ModelPlan], world_size: int) -> str:
    payload = {
        "seed": run.seed,
        "stage1_steps": run.stage1_steps,
        "stage2_steps": run.stage2_steps,
        "checkpointing_steps": run.checkpointing_steps,
        "world_size": int(world_size),
        "per_device_batch": REFERENCE_PER_DEVICE_BATCH,
        "gradient_accumulation_steps": adjusted_grad_accum(world_size),
        "effective_global_batch": REFERENCE_EFFECTIVE_BATCH,
        "models": [
            {
                "name": plan.name,
                "stage1_region": asdict(plan.stage1_region),
                "stage2_region": asdict(plan.stage2_region),
            }
            for plan in plans
        ],
        "dataset_manifest": "/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/exp09_10_11_youtubevos_gtwin_d3comp_pai/manifests/selected_primary_comp.gtwin.pai_paths.jsonl",
        "sft_init": "/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000",
        "frozen_reference": "/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000",
        "beta_dpo": 10,
        "lose_gap_weight": 0.25,
        "winner_abs_reg_weight": 0.05,
        "winner_gap_reg_weight": 1.0,
    }
    return stable_config_hash(payload)


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-", 1)[1])
        except Exception:
            continue
        checkpoints.append((step, path))
    return sorted(checkpoints)[-1][1] if checkpoints else None


def resume_args(output_dir: Path) -> list[str]:
    if (output_dir / "last_weights" / "unet_main" / "config.json").exists():
        return []
    return ["--resume_from_checkpoint", "latest"] if latest_checkpoint(output_dir) else []


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n[exp28-runner] start={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write("[exp28-runner] cmd=" + " ".join(cmd) + "\n")
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
        log.write(f"\n[exp28-runner] end={time.strftime('%Y-%m-%d %H:%M:%S')} returncode={rc}\n")
    return rc


def base_args(
    out_dir: Path,
    run: RunConfig,
    region: RegionConfig,
    stage: int,
    *,
    world_size: int = 2,
    stage1_weights: Path | None = None,
) -> list[str]:
    grad_accum = adjusted_grad_accum(world_size)
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
        "--mask_region_weight", str(region.mask_core_weight),
        "--boundary_region_weight", str(region.outer_weight),
        "--outside_region_weight", str(region.outside_weight),
        "--boundary_mode", str(region.boundary_mode),
        "--pool_grid_scale", "1",
        "--inner_pool_steps", "0",
        "--outer_pool_steps", str(region.legacy_outer_pool_steps),
        "--inner_weight", str(region.inner_weight),
        "--outer_weight", str(region.outer_weight),
        "--legacy_exact", "false",
        "--exp28_geometry_mode", str(region.geometry_mode),
        "--inner_radius_px", str(region.inner_radius_px),
        "--aggregation", "legacy_global_weighted_mean",
        "--output_dir", str(out_dir),
        "--logging_dir", f"logs-dpo-stage{stage}",
        "--val_data_dir", "/mnt/workspace/hj/nas_hj/data/external/davis_432_240",
        "--resolution", "512",
        "--train_height", "320",
        "--train_width", "512",
        "--nframes", "16",
        "--train_batch_size", str(REFERENCE_PER_DEVICE_BATCH),
        "--gradient_accumulation_steps", str(grad_accum),
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
    args += resume_args(out_dir)
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
        "PROCESS_TITLE": "Exp28CLI4",
        "SETPROCTITLE": "Exp28CLI4",
        "LINGBOT_PROCESS_NAME": "Exp28CLI4",
        "WANDB_SILENT": "true",
        "WANDB_QUIET": "true",
        "WANDB_CONSOLE": "off",
    })
    return env


def run_model(run: RunConfig, plan: ModelPlan, phy_python: str, gpus: str, nproc: int, queue_role: str) -> int:
    model_root = OUTPUT_ROOT / "pairs" / run.pair_id / plan.name
    log_dir = LOG_ROOT / "pairs" / run.pair_id / plan.name
    stage1_dir = model_root / "stage1"
    stage2_dir = model_root / "stage2"
    write_json(
        model_root / "region_config.json",
        {
            "model": plan.name,
            "stage1_region": asdict(plan.stage1_region),
            "stage2_region": asdict(plan.stage2_region),
            "world_size": nproc,
            "gradient_accumulation_steps": adjusted_grad_accum(nproc),
            "effective_global_batch": REFERENCE_EFFECTIVE_BATCH,
        },
    )
    env = make_env(gpus)

    stage1_last = stage1_dir / "last_weights"
    if not (stage1_last / "unet_main" / "config.json").exists():
        append_queue_event(run.pair_id, f"training_{queue_role}_s1", {"model": plan.name})
        rc = run_logged(
            torchrun(
                phy_python,
                nproc,
                "exp28_fine_inner_boundary_sweep/code/train_exp28_stage1.py",
                base_args(stage1_dir, run, plan.stage1_region, stage=1, world_size=nproc),
            ),
            log_dir / "stage1.log",
            env,
        )
        if rc != 0:
            return rc

    stage2_last = stage2_dir / "last_weights"
    if not (stage2_last / "unet_main" / "config.json").exists():
        append_queue_event(run.pair_id, f"training_{queue_role}_s2", {"model": plan.name})
        rc = run_logged(
            torchrun(
                phy_python,
                nproc,
                "exp28_fine_inner_boundary_sweep/code/train_exp28_stage2.py",
                base_args(stage2_dir, run, plan.stage2_region, stage=2, world_size=nproc, stage1_weights=stage1_last),
            ),
            log_dir / "stage2.log",
            env,
        )
        if rc != 0:
            return rc
    return 0


def build_auto_eval_command(pair_id: str, control_model: str, candidate_model: str) -> list[str]:
    return [
        "bash",
        "exp28_fine_inner_boundary_sweep/scripts/eval_exp28_pair_davis50_pai.sh",
        pair_id,
        control_model,
        candidate_model,
    ]


def run_pair_eval(run: RunConfig, plans: list[ModelPlan], env: dict[str, str]) -> int:
    append_queue_event(run.pair_id, "evaluating", {"script": "eval_exp28_pair_davis50_pai.sh"})
    eval_env = dict(env)
    eval_env["PAIR_ID"] = run.pair_id
    eval_env["ROOT"] = str(PROJECT_ROOT)
    eval_env["EVAL_ROOT"] = str(EVAL_ROOT / run.pair_id)
    eval_env.setdefault("EVAL_GPU", env.get("CUDA_VISIBLE_DEVICES", "1").split(",")[0])
    eval_env.setdefault("COMPUTE_VFID", "1")
    eval_env.setdefault("COMPUTE_TC", "1")
    eval_env.setdefault("COMPUTE_EWARP", "1")
    return run_logged(
        build_auto_eval_command(run.pair_id, plans[0].name, plans[1].name),
        LOG_ROOT / "pairs" / run.pair_id / "paired_davis50_eval.log",
        eval_env,
    )


def make_pair_plans(pair: str) -> tuple[RunConfig, list[ModelPlan]]:
    pair_id, control_name, candidate_name, radius = PAIR_SPECS[pair]
    control = RegionConfig("legacy_outer_one_ring", 0)
    candidate = RegionConfig("inner_boundary_px", radius)
    return RunConfig(pair_id=pair_id), [
        ModelPlan(control_name, control, control),
        ModelPlan(candidate_name, candidate, candidate),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=sorted(PAIR_SPECS), default="A")
    parser.add_argument("--pair-id", default="", help="Override the canonical pair id only for controlled resume/debug.")
    parser.add_argument("--gpus", default="3,4")
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--phy-python", default=os.environ.get("PHY_PYTHON", "/mnt/nas/hj/conda_envs/diffueraser/bin/Phy"))
    parser.add_argument("--auto-eval", action="store_true", help="Run paired DAVIS50 after both models finish.")
    parser.add_argument("--no-auto-eval", action="store_false", dest="auto_eval")
    parser.set_defaults(auto_eval=True)
    args = parser.parse_args()

    if args.nproc_per_node != 2:
        raise ValueError("Exp28 CLI4 pairs must use world_size=2")
    adjusted_grad_accum(args.nproc_per_node)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    run, plans = make_pair_plans(args.pair)
    if args.pair_id:
        run = RunConfig(pair_id=args.pair_id, seed=run.seed, stage1_steps=run.stage1_steps, stage2_steps=run.stage2_steps, checkpointing_steps=run.checkpointing_steps)
    config_hash = pair_config_hash(run, plans, args.nproc_per_node)
    state = {
        "status": "queued",
        "pair": args.pair,
        "pair_id": run.pair_id,
        "config_hash": config_hash,
        "gpus": args.gpus,
        "nproc_per_node": args.nproc_per_node,
        "gradient_accumulation_steps": adjusted_grad_accum(args.nproc_per_node),
        "effective_global_batch": REFERENCE_EFFECTIVE_BATCH,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    append_queue_event(run.pair_id, "queued", {"config_hash": config_hash})
    write_json(REG_ROOT / f"queue_state_{run.pair_id}.json", state)
    write_json(EXP_ROOT / f"queue_state_{run.pair_id}.json", state)
    write_json(
        OUTPUT_ROOT / "pairs" / run.pair_id / "pair_config.json",
        {
            "run": asdict(run),
            "models": [asdict(p) for p in plans],
            "config_hash": config_hash,
            "world_size": args.nproc_per_node,
            "gradient_accumulation_steps": adjusted_grad_accum(args.nproc_per_node),
            "effective_global_batch": REFERENCE_EFFECTIVE_BATCH,
        },
    )

    for index, plan in enumerate(plans, start=1):
        row = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pair_id": run.pair_id,
            "model": plan.name,
            "stage": "start",
            "queue_index": index,
            "gpus": args.gpus,
            "config_hash": config_hash,
        }
        append_csv(REG_ROOT / "process_registry.csv", row)
        state.update({"status": "running", "current_model": plan.name, "queue_index": index})
        write_json(REG_ROOT / f"queue_state_{run.pair_id}.json", state)
        queue_role = "control" if index == 1 else "candidate"
        rc = run_model(run, plan, args.phy_python, args.gpus, args.nproc_per_node, queue_role=queue_role)
        row["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        row["stage"] = "done" if rc == 0 else "failed"
        append_csv(REG_ROOT / "process_registry.csv", row)
        if rc != 0:
            append_queue_event(run.pair_id, "failed", {"model": plan.name, "returncode": rc})
            state.update({"status": "failed", "failed_model": plan.name, "returncode": rc})
            write_json(REG_ROOT / f"queue_state_{run.pair_id}.json", state)
            write_json(EXP_ROOT / f"queue_state_{run.pair_id}.json", state)
            return rc

    if args.auto_eval:
        rc = run_pair_eval(run, plans, make_env(args.gpus))
        if rc != 0:
            append_queue_event(run.pair_id, "failed", {"stage": "eval", "returncode": rc})
            state.update({"status": "failed", "failed_stage": "eval", "returncode": rc})
            write_json(REG_ROOT / f"queue_state_{run.pair_id}.json", state)
            write_json(EXP_ROOT / f"queue_state_{run.pair_id}.json", state)
            return rc

    append_queue_event(run.pair_id, "completed", {"config_hash": config_hash})
    state.update({"status": "completed", "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    write_json(REG_ROOT / f"queue_state_{run.pair_id}.json", state)
    write_json(EXP_ROOT / f"queue_state_{run.pair_id}.json", state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
