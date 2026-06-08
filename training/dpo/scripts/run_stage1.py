#!/usr/bin/env python
# coding=utf-8
"""
run_dpo_stage1.py — DPO Stage 1 训练入口

自动检测项目根目录（DPO_finetune/scripts/ 的祖父目录），无需硬编码路径。
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.common.experiment import (
    first_existing,
    latest_dir,
    prepare_experiment_dir,
    resolve_output_dir,
)


def get_project_root():
    """自动检测项目根目录。"""
    return str(Path(__file__).resolve().parents[3])


def build_cmd(project_root, args):
    weights_dir = args.weights_dir or os.path.join(project_root, "weights")
    dpo_data_root = args.dpo_data_root or first_existing(
        os.path.join(project_root, "data", "external", "DPO_Finetune_data"),
        os.path.join(project_root, "data", "DPO_Finetune_data"),
    )
    latest_sft_stage2 = latest_dir(
        project_root, "sft", "stage2", args.experiments_dir
    ) / "converted_weights"
    ref_model_path = args.ref_model_path or first_existing(
        latest_sft_stage2,
        os.path.join(weights_dir, "diffuEraser", "converted_weights_step48000"),
        os.path.join(weights_dir, "diffuEraser", "converted_weights_step34000"),
        os.path.join(project_root, "finetune-stage2", "converted_weights_step48000"),
        os.path.join(project_root, "finetune-stage2", "converted_weights_step34000"),
    )
    eval_dir = args.val_data_dir or first_existing(
        os.path.join(project_root, "data_val"),
        os.path.join(project_root, "data", "external", "davis_432_240"),
    )
    output_dir = resolve_output_dir(
        project_root,
        "dpo",
        "stage1",
        explicit_output_dir=args.output_dir,
        experiments_dir=args.experiments_dir,
        run_name=args.run_name,
        run_version=args.run_version,
    )

    cmd = [
        "accelerate", "launch",
        "--num_processes", str(args.num_gpus),
        "--mixed_precision", args.mixed_precision,
    ]
    if args.main_process_port is not None:
        cmd.extend(["--main_process_port", str(args.main_process_port)])
    train_entrypoint = os.environ.get(
        "DPO_STAGE1_ENTRYPOINT",
        "training/dpo/lingbot-worldmodel-stage1.py",
    )
    cmd.extend([
        train_entrypoint,
        "--base_model_name_or_path", os.path.join(weights_dir, "stable-diffusion-v1-5"),
        "--vae_path", os.path.join(weights_dir, "sd-vae-ft-mse"),
        "--ref_model_path", ref_model_path,
        "--dpo_data_root", dpo_data_root,
        "--dpo_dataset_type", args.dpo_dataset_type,
        "--preference_manifest", args.preference_manifest,
        "--train_mask_mode", args.train_mask_mode,
        "--mask_from_manifest", str(args.mask_from_manifest).lower(),
        "--loss_region_mode", args.loss_region_mode,
        "--gap_normalization", args.gap_normalization,
        "--gap_eps", str(args.gap_eps),
        "--lose_gap_clip_tau", str(args.lose_gap_clip_tau),
        "--mask_region_weight", str(args.mask_region_weight),
        "--boundary_region_weight", str(args.boundary_region_weight),
        "--outside_region_weight", str(args.outside_region_weight),
        "--dpo_gap_trace_csv", str(args.dpo_gap_trace_csv),
        "--dpo_gap_samples_jsonl_gz", str(args.dpo_gap_samples_jsonl_gz),
        "--enable_dpo_diag", str(args.enable_dpo_diag).lower(),
        "--dpo_diag_log_every", str(args.dpo_diag_log_every),
        "--dpo_diag_save_csv", str(args.dpo_diag_save_csv).lower(),
        "--dpo_diag_save_wandb", str(args.dpo_diag_save_wandb).lower(),
        "--output_dir", str(output_dir),
        "--logging_dir", "logs-dpo-stage1",
        "--val_data_dir", eval_dir,
        "--resolution", str(args.resolution),
        "--nframes", str(args.nframes),
        "--train_batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--dataloader_num_workers", str(args.num_workers),
        "--learning_rate", str(args.learning_rate),
        "--lr_scheduler", args.lr_scheduler,
        "--lr_warmup_steps", str(args.lr_warmup_steps),
        "--checkpointing_steps", str(args.checkpointing_steps),
        "--validation_steps", str(args.validation_steps),
        "--logging_steps", str(args.logging_steps),
        "--val_num_inference_steps", str(args.val_num_inference_steps),
        "--val_mask_dilation_iter", str(args.val_mask_dilation_iter),
        "--vae_dtype", args.vae_dtype,
        "--policy_dtype", args.policy_dtype,
        "--ref_dtype", args.ref_dtype,
        "--text_dtype", args.text_dtype,
        "--beta_dpo", str(args.beta_dpo),
        "--sft_reg_weight", str(args.sft_reg_weight),
        "--lose_gap_weight", str(args.lose_gap_weight),
        "--winner_abs_reg_weight", str(args.winner_abs_reg_weight),
        "--winner_gap_reg_weight", str(args.winner_gap_reg_weight),
        "--winner_gap_reg_margin", str(args.winner_gap_reg_margin),
        "--winner_gap_reg_mode", str(args.winner_gap_reg_mode),
        "--davis_oversample", str(args.davis_oversample),
        "--videodpo_frame_stride", str(args.videodpo_frame_stride),
        "--videodpo_clip_length", str(args.videodpo_clip_length),
        "--videodpo_full_mask_value", str(args.videodpo_full_mask_value),
        "--seed", str(args.seed),
        "--report_to", args.report_to,
        "--tracker_project_name", args.wandb_project,
        "--set_grads_to_none",
        "--resume_from_checkpoint", "latest",
    ])
    if args.max_train_steps is not None:
        cmd.extend(["--max_train_steps", str(args.max_train_steps)])
    if args.num_train_epochs is not None:
        cmd.extend(["--num_train_epochs", str(args.num_train_epochs)])
    if args.train_height is not None:
        cmd.extend(["--train_height", str(args.train_height)])
    if args.train_width is not None:
        cmd.extend(["--train_width", str(args.train_width)])

    if args.enable_xformers:
        cmd.append("--enable_xformers_memory_efficient_attention")
    if args.allow_tf32:
        cmd.append("--allow_tf32")
    if not args.disable_gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if args.checkpoints_total_limit:
        cmd.extend(["--checkpoints_total_limit", str(args.checkpoints_total_limit)])
    if args.wandb_entity:
        cmd.extend(["--wandb_entity", args.wandb_entity])
    if args.chunk_aligned:
        cmd.append("--chunk_aligned")
    if args.split_pos_neg_forward:
        cmd.append("--split_pos_neg_forward")
    if args.debug_first_batch_stages:
        cmd.append("--debug_first_batch_stages")
    if args.disable_dpo_diagnostics:
        cmd.append("--disable_dpo_diagnostics")
    if args.use_8bit_adam:
        cmd.append("--use_8bit_adam")

    prepare_experiment_dir(
        output_dir,
        root=project_root,
        family="dpo",
        stage="stage1",
        command=cmd,
        inputs={
            "dpo_data_root": dpo_data_root,
            "dpo_dataset_type": args.dpo_dataset_type,
            "preference_manifest": args.preference_manifest,
            "ref_model_path": ref_model_path,
            "val_data_dir": eval_dir,
            "weights_dir": weights_dir,
        },
        params={
            "nframes": args.nframes,
            "resolution": args.resolution,
            "train_height": args.train_height,
            "train_width": args.train_width,
            "max_train_steps": args.max_train_steps,
            "num_train_epochs": args.num_train_epochs,
            "logging_steps": args.logging_steps,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "beta_dpo": args.beta_dpo,
            "sft_reg_weight": args.sft_reg_weight,
            "lose_gap_weight": args.lose_gap_weight,
            "winner_abs_reg_weight": args.winner_abs_reg_weight,
            "winner_gap_reg_weight": args.winner_gap_reg_weight,
            "winner_gap_reg_margin": args.winner_gap_reg_margin,
            "winner_gap_reg_mode": args.winner_gap_reg_mode,
            "videodpo_frame_stride": args.videodpo_frame_stride,
            "videodpo_full_mask_value": args.videodpo_full_mask_value,
            "train_mask_mode": args.train_mask_mode,
            "mask_from_manifest": args.mask_from_manifest,
            "loss_region_mode": args.loss_region_mode,
            "gap_normalization": args.gap_normalization,
            "gap_eps": args.gap_eps,
            "lose_gap_clip_tau": args.lose_gap_clip_tau,
            "mask_region_weight": args.mask_region_weight,
            "boundary_region_weight": args.boundary_region_weight,
            "outside_region_weight": args.outside_region_weight,
            "dpo_gap_trace_csv": args.dpo_gap_trace_csv,
            "dpo_gap_samples_jsonl_gz": args.dpo_gap_samples_jsonl_gz,
            "enable_dpo_diag": args.enable_dpo_diag,
            "dpo_diag_log_every": args.dpo_diag_log_every,
            "dpo_diag_save_csv": args.dpo_diag_save_csv,
            "dpo_diag_save_wandb": args.dpo_diag_save_wandb,
        },
    )

    return cmd, dpo_data_root, ref_model_path, str(output_dir)


def get_output_dir(project_root, args):
    return str(resolve_output_dir(
        project_root,
        "dpo",
        "stage1",
        explicit_output_dir=args.output_dir,
        experiments_dir=args.experiments_dir,
        run_name=args.run_name,
        run_version=args.run_version,
    ))


def find_slurm_log_path(project_root):
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    candidate_dirs = [
        Path(project_root) / "logs",
        Path(project_root) / "log",
        Path(project_root),
    ]

    if slurm_job_id:
        for log_dir in candidate_dirs:
            exact_path = log_dir / f"dpo-stage1-{slurm_job_id}.out"
            if exact_path.exists():
                return exact_path

    candidates = []
    for log_dir in candidate_dirs:
        if not log_dir.exists():
            continue
        candidates.extend(log_dir.glob("dpo-stage1-*.out"))

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def upload_full_crash_log_to_wandb(project_root, output_dir, returncode):
    run_info_path = Path(output_dir) / "wandb_run_info.json"
    if not run_info_path.exists():
        print(f"[launcher] W&B run info not found at {run_info_path}, skip crash log upload.", file=sys.stderr)
        return

    try:
        import wandb
    except Exception as e:
        print(f"[launcher] Failed to import wandb for crash upload: {e}", file=sys.stderr)
        return

    try:
        with open(run_info_path) as f:
            run_info = json.load(f)

        run = wandb.init(
            project=run_info["project"],
            entity=run_info.get("entity"),
            id=run_info["id"],
            resume="allow",
            name=run_info.get("name"),
        )

        slurm_log_path = find_slurm_log_path(project_root)
        if slurm_log_path is None or not slurm_log_path.exists():
            wandb.alert(
                title="DPO Stage 1 Launcher Failed",
                text=f"Launcher exited with code {returncode}, but no Slurm log file was found.",
                level=wandb.AlertLevel.ERROR,
            )
            wandb.finish(exit_code=returncode)
            return

        full_text = slurm_log_path.read_text(errors="replace")
        tail_chars = 3500
        full_log_name = "slurm_full_crash_output.out"
        full_log_copy = Path(run.dir) / full_log_name
        shutil.copyfile(slurm_log_path, full_log_copy)
        wandb.save(str(full_log_copy), policy="now")

        summary_name = "launcher_crash_summary.txt"
        summary_copy = Path(run.dir) / summary_name
        with open(summary_copy, "w") as f:
            f.write(f"returncode: {returncode}\n")
            f.write(f"slurm_log_path: {slurm_log_path}\n")
            f.write(f"wandb_run_url: {run_info.get('url')}\n")
            f.write("\n===== LOG TAIL =====\n")
            f.write(full_text[-tail_chars:])
            f.write("\n")
        wandb.save(str(summary_copy), policy="now")

        if getattr(wandb, "run", None) is not None:
            wandb.run.summary["launcher_returncode"] = returncode
            wandb.run.summary["slurm_full_log_uploaded"] = True
            wandb.run.summary["slurm_log_path"] = str(slurm_log_path)
            wandb.run.summary["slurm_full_log_file"] = full_log_name

        wandb.alert(
            title="DPO Stage 1 Launcher Failed",
            text=(
                f"Launcher exited with code {returncode}.\n"
                f"Full Slurm log uploaded as `{full_log_name}`.\n\n"
                f"Tail of the full log:\n```\n{full_text[-tail_chars:]}\n```"
            ),
            level=wandb.AlertLevel.ERROR,
        )
        wandb.finish(exit_code=returncode)
    except Exception:
        print("[launcher] Failed to upload full crash log to W&B.", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)


def run(args=None):
    if args is None:
        args = parse_args()
    project_root = get_project_root()
    cmd, dpo_data_root, ref_model_path, output_dir = build_cmd(project_root, args)
    args.resolved_output_dir = output_dir

    print("=" * 60)
    print("  DiffuEraser DPO Stage 1 Training")
    print("=" * 60)
    print(f"  Project Root:    {project_root}")
    print(f"  DPO Data Root:   {dpo_data_root}")
    print(f"  Dataset Type:    {args.dpo_dataset_type}")
    print(f"  Ref Model:       {ref_model_path}")
    print(f"  GPUs:            {args.num_gpus}")
    print(f"  Resolution:      {args.resolution}")
    if args.train_height is not None or args.train_width is not None:
        print(f"  Train Size:      {args.train_height or args.resolution}x{args.train_width or args.resolution}")
    print(f"  Max Steps:       {args.max_train_steps}")
    print(f"  Max Epochs:      {args.num_train_epochs}")
    print(f"  Logging Steps:   {args.logging_steps}")
    print(f"  Beta DPO:        {args.beta_dpo}")
    print(f"  SFT Reg Weight:  {args.sft_reg_weight}")
    print(f"  Lose Gap Weight: {args.lose_gap_weight}")
    print(f"  Winner Abs Reg:  {args.winner_abs_reg_weight}")
    print(f"  Winner Gap Reg:  {args.winner_gap_reg_weight}")
    print(f"  Winner Gap Mgn:  {args.winner_gap_reg_margin}")
    print(f"  LR:              {args.learning_rate}")
    print(f"  Num Workers:     {args.num_workers}")
    print(f"  Mixed Precision: {args.mixed_precision}")
    print(f"  VAE dtype:       {args.vae_dtype}")
    print(f"  Policy dtype:    {args.policy_dtype}")
    print(f"  Ref dtype:       {args.ref_dtype}")
    print(f"  Text dtype:      {args.text_dtype}")
    print(f"  Report To:       {args.report_to}")
    print(f"  Main Port:       {args.main_process_port}")
    print(f"  XFormers:        {args.enable_xformers}")
    print(f"  TF32:            {args.allow_tf32}")
    print(f"  Grad Ckpt:       {not args.disable_gradient_checkpointing}")
    print(f"  Split Pos/Neg:   {args.split_pos_neg_forward}")
    print(f"  8bit Adam:       {args.use_8bit_adam}")
    print(f"  Debug Stages:    {args.debug_first_batch_stages}")
    print("=" * 60)
    print(f"\n  Command:\n  {' '.join(cmd[:6])} \\\n    " + " \\\n    ".join(cmd[6:]))
    print()

    result = subprocess.run(cmd, cwd=project_root)
    if result.returncode != 0 and args.report_to.lower() == "wandb":
        upload_full_crash_log_to_wandb(project_root, output_dir, result.returncode)
    return result.returncode


def parse_args():
    parser = argparse.ArgumentParser(description="DPO Stage 1 Training Entry")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--weights_dir", type=str, default=None)
    parser.add_argument("--dpo_data_root", type=str, default=None)
    parser.add_argument("--dpo_dataset_type", type=str, default="diffueraser_inpainting",
                        choices=["diffueraser_inpainting", "videodpo_fullmask", "generated_loser_manifest"])
    parser.add_argument("--preference_manifest", type=str, default="")
    parser.add_argument("--train_mask_mode", type=str, default="full", choices=["full", "partial"])
    parser.add_argument("--mask_from_manifest", type=str, default="false")
    parser.add_argument("--loss_region_mode", type=str, default="full", choices=["full", "region"])
    parser.add_argument("--gap_normalization", type=str, default="raw", choices=["raw", "log_ratio"])
    parser.add_argument("--gap_eps", type=float, default=1e-6)
    parser.add_argument("--lose_gap_clip_tau", type=str, default="")
    parser.add_argument("--mask_region_weight", type=float, default=1.0)
    parser.add_argument("--boundary_region_weight", type=float, default=0.5)
    parser.add_argument("--outside_region_weight", type=float, default=0.05)
    parser.add_argument("--dpo_gap_trace_csv", type=str, default="")
    parser.add_argument("--dpo_gap_samples_jsonl_gz", type=str, default="")
    parser.add_argument("--enable_dpo_diag", type=str, default="true")
    parser.add_argument("--dpo_diag_log_every", type=int, default=10)
    parser.add_argument("--dpo_diag_save_csv", type=str, default="true")
    parser.add_argument("--dpo_diag_save_wandb", type=str, default="true")
    parser.add_argument("--ref_model_path", type=str, default=None)
    parser.add_argument("--val_data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--experiments_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="stage1")
    parser.add_argument("--run_version", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--checkpointing_steps", type=int, default=2000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--validation_steps", type=int, default=2000)
    parser.add_argument("--logging_steps", type=int, default=300,
                        help="Print detailed DPO diagnostics every N optimizer steps.")
    parser.add_argument("--val_num_inference_steps", type=int, default=6)
    parser.add_argument("--val_mask_dilation_iter", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_height", type=int, default=None)
    parser.add_argument("--train_width", type=int, default=None)
    parser.add_argument("--nframes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--vae_dtype", type=str, default="auto", choices=["auto", "fp32"])
    parser.add_argument("--policy_dtype", type=str, default="auto", choices=["auto", "fp32"])
    parser.add_argument("--ref_dtype", type=str, default="auto", choices=["auto", "fp32", "bf16", "fp16"])
    parser.add_argument("--text_dtype", type=str, default="auto", choices=["auto", "fp32", "bf16", "fp16"])
    parser.add_argument("--main_process_port", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="wandb",
                        help="Accelerate tracker backend. Use 'none'/'off' to disable W&B/tracker logging.")
    parser.add_argument("--wandb_project", type=str, default="DPO_Diffueraser")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--beta_dpo", type=float, default=500.0)
    parser.add_argument("--sft_reg_weight", type=float, default=0.0)
    parser.add_argument("--lose_gap_weight", type=float, default=1.0)
    parser.add_argument("--winner_abs_reg_weight", type=float, default=0.0)
    parser.add_argument("--winner_gap_reg_weight", type=float, default=0.0)
    parser.add_argument("--winner_gap_reg_margin", type=float, default=0.0)
    parser.add_argument("--winner_gap_reg_mode", type=str, default="relu", choices=["relu"])
    parser.add_argument("--davis_oversample", type=int, default=10)
    parser.add_argument("--videodpo_frame_stride", type=int, default=1)
    parser.add_argument("--videodpo_clip_length", type=float, default=1.0)
    parser.add_argument("--videodpo_full_mask_value", type=float, default=0.0)
    parser.add_argument("--chunk_aligned", action="store_true")
    parser.add_argument("--enable_xformers", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--disable_gradient_checkpointing", action="store_true")
    parser.add_argument("--split_pos_neg_forward", action="store_true")
    parser.add_argument("--debug_first_batch_stages", action="store_true")
    parser.add_argument("--disable_dpo_diagnostics", action="store_true")
    parser.add_argument("--use_8bit_adam", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(run())
