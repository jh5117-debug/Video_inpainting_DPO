#!/usr/bin/env python3
"""Run official EffectErase smoke v2 inference rows sequentially."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def heartbeat(runtime_dir: Path, status: str, sample: str) -> None:
    write_text(
        runtime_dir / "heartbeat",
        "\n".join(
            [
                f"time={time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
                f"pid={os.getpid()}",
                f"pgid={os.getpgid(0)}",
                "gpu=GPU0",
                f"status={status}",
                f"sample={sample}",
                "",
            ]
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()

    rows = jsonl(args.manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)
    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    write_text(args.runtime_dir / "pid", str(os.getpid()) + "\n")
    write_text(args.runtime_dir / "pgid", str(os.getpgid(0)) + "\n")
    resolved = {
        "branch": "research/exp29-minimax-effecterase-adapter-feasibility-20260626",
        "commit": args.commit,
        "gpu": args.gpu,
        "manifest": str(args.manifest),
        "repo": str(args.repo),
        "python": str(args.python),
        "asset_root": str(args.asset_root),
        "output_root": str(args.output_root),
        "num_frames": 17,
        "height": 480,
        "width": 832,
        "seed": 2025,
        "cfg": 1.0,
        "num_inference_steps": 50,
        "raw_output_primary": True,
        "diagnostic_comp_optional": True,
        "vor_eval_used": False,
        "eligible_for_training": False,
    }
    write_text(args.runtime_dir / "resolved_config.json", json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    commands_path = args.output_root / "commands.jsonl"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(args.repo) + os.pathsep + env.get("PYTHONPATH", "")

    lora = args.asset_root / "FudanCVL/EffectErase/EffectErase.ckpt"
    wan = args.asset_root / "Wan-AI/Wan2.1-Fun-1.3B-InP"
    commands = []
    for row in rows:
        out = Path(row["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(args.python),
            "examples/remove_wan/infer_remove_wan.py",
            "--fg_bg_path",
            row["condition_path"],
            "--mask_path",
            row["mask_path"],
            "--output_path",
            row["output_path"],
            "--text_encoder_path",
            str(wan / "models_t5_umt5-xxl-enc-bf16.pth"),
            "--vae_path",
            str(wan / "Wan2.1_VAE.pth"),
            "--dit_path",
            str(wan / "diffusion_pytorch_model.safetensors"),
            "--image_encoder_path",
            str(wan / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
            "--pretrained_lora_path",
            str(lora),
            "--num_frames",
            "17",
            "--height",
            "480",
            "--width",
            "832",
            "--seed",
            "2025",
            "--cfg",
            "1.0",
            "--num_inference_steps",
            "50",
        ]
        commands.append({"sample_id": row["sample_id"], "cmd": cmd})
    with commands_path.open("w") as f:
        for item in commands:
            f.write(json.dumps(item) + "\n")

    exit_code = 0
    status_rows = []
    for item in commands:
        sample = item["sample_id"]
        heartbeat(args.runtime_dir, "running", sample)
        log_path = args.output_root / "logs" / f"{sample}.log"
        cmd_path = args.output_root / "logs" / f"{sample}.command.json"
        write_text(cmd_path, json.dumps(item, indent=2) + "\n")
        start = time.time()
        with log_path.open("w") as log:
            proc = subprocess.run(item["cmd"], cwd=args.repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.time() - start
        write_text(args.output_root / "logs" / f"{sample}.exitcode", str(proc.returncode) + "\n")
        status_rows.append(
            {
                "sample_id": sample,
                "exit_code": proc.returncode,
                "elapsed_sec": elapsed,
                "output_path": next(row["output_path"] for row in rows if row["sample_id"] == sample),
                "log_path": str(log_path),
            }
        )
        write_text(args.output_root / "per_row_status.jsonl", "".join(json.dumps(r) + "\n" for r in status_rows))
        if proc.returncode != 0:
            exit_code = proc.returncode
            break
    heartbeat(args.runtime_dir, "finished", f"rc={exit_code}")
    write_text(args.runtime_dir / "exitcode", str(exit_code) + "\n")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
