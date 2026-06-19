#!/usr/bin/env python3
"""Recoverable Exp20 queue worker for first-wave trials."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_gpu_allocation(num_gpus: int, max_memory_mib: int, lock: bool) -> dict[str, object]:
    cmd = [
        sys.executable,
        "exp20_autoresearch_scale_adaptive_region_dpo/code/gpu_allocator.py",
        "--num-gpus",
        str(num_gpus),
        "--max-memory-mib",
        str(max_memory_mib),
        "--exclude",
        "7",
        "--samples",
        "6",
        "--interval-seconds",
        "10",
    ]
    if lock:
        cmd.append("--lock")
    out = subprocess.check_output(cmd, cwd=PROJECT_ROOT, text=True)
    return json.loads(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", default="exp20_autoresearch_scale_adaptive_region_dpo")
    parser.add_argument("--queue-dir", default="")
    parser.add_argument("--max-trials", type=int, default=1)
    parser.add_argument("--gpus-per-trial", type=int, default=1)
    parser.add_argument("--max-memory-mib", type=int, default=1024)
    parser.add_argument("--dev-root", default="/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/dev_boundary_search_v1_e385cc27_roots")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    queue_dir = Path(args.queue_dir) if args.queue_dir else exp_dir / "queue"
    running_dir = exp_dir / "running"
    done_dir = exp_dir / "done"
    crash_dir = exp_dir / "crash"
    for path in [running_dir, done_dir, crash_dir]:
        path.mkdir(parents=True, exist_ok=True)

    queue = sorted(queue_dir.glob("*.json"))
    if not queue:
        print(json.dumps({"status": "EMPTY_QUEUE"}))
        return 0

    completed = 0
    for queue_file in queue:
        if completed >= args.max_trials:
            break
        cfg_path = Path(queue_file.read_text().strip())
        if not cfg_path.exists():
            (crash_dir / queue_file.name).write_text(f"missing config path {cfg_path}\n")
            continue
        allocation = load_gpu_allocation(args.gpus_per_trial, args.max_memory_mib, lock=False)
        if allocation.get("status") != "GPU_AVAILABLE":
            print(json.dumps({"status": "NO_GPU_AVAILABLE", "allocation": allocation}, indent=2))
            return 75
        gpu = str(allocation["selected"][0])
        marker = running_dir / queue_file.name
        if marker.exists():
            continue
        marker.write_text(str(cfg_path) + "\n")
        cmd = [
            sys.executable,
            "exp20_autoresearch_scale_adaptive_region_dpo/code/trial_runner.py",
            "--config",
            str(cfg_path),
            "--gpu-id",
            gpu,
            "--dev-root",
            args.dev_root,
            "--main-process-port",
            str(29640 + completed),
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        rc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env).returncode
        if rc == 0:
            done_dir.joinpath(queue_file.name).write_text(str(cfg_path) + "\n")
        else:
            crash_dir.joinpath(queue_file.name).write_text(f"{cfg_path}\nreturncode={rc}\n")
            return rc
        completed += 1
    print(json.dumps({"status": "WORKER_DONE", "completed": completed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
