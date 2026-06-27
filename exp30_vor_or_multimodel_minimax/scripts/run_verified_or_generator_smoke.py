#!/usr/bin/env python3
"""Run small Exp30 VOR OR loser-generation smoke for verified wrappers."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import hashlib
from pathlib import Path


PROMPT = "remove the masked object and restore the background"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["diffueraser", "propainter"], required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--num-frames", type=int, default=17)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=288)
    p.add_argument("--base-model-path", type=Path, default=Path("/mnt/workspace/hj/nas_hj/weights/stable-diffusion-v1-5"))
    p.add_argument("--vae-path", type=Path, default=Path("/mnt/workspace/hj/nas_hj/weights/sd-vae-ft-mse"))
    p.add_argument("--diffueraser-path", type=Path, default=Path("/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000"))
    p.add_argument("--propainter-model-dir", type=Path, default=Path("/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter"))
    p.add_argument("--pcm-weights-path", type=Path, default=Path("/mnt/workspace/hj/nas_hj/weights/PCM_Weights/sd15/pcm_sd15_smallcfg_2step_converted.safetensors"))
    p.add_argument("--pcm-mode", choices=["official_pcm2", "none"], default="official_pcm2")
    p.add_argument("--prior-mode", choices=["propainter"], default="propainter")
    p.add_argument("--no-pcm-steps", type=int, default=6)
    p.add_argument("--no-pcm-guidance", type=float, default=0.0)
    p.add_argument("--mask-dilation-iter", type=int, default=8)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def symlink_dir(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst, target_is_directory=True)


def run_command(cmd: list[str], log_path: Path, cwd: Path, env: dict) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[cmd] " + " ".join(cmd) + "\n")
        log.write("[env] CUDA_VISIBLE_DEVICES=" + env.get("CUDA_VISIBLE_DEVICES", "") + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode, time.time() - start


def output_count(path: Path) -> int:
    return len(list(path.glob("*.png"))) if path.exists() else 0


def generator_id(args: argparse.Namespace) -> str:
    payload = {
        "model": args.model,
        "pcm_mode": args.pcm_mode if args.model == "diffueraser" else None,
        "prior_mode": args.prior_mode if args.model == "diffueraser" else None,
        "no_pcm_steps": args.no_pcm_steps if args.model == "diffueraser" and args.pcm_mode == "none" else None,
        "no_pcm_guidance": args.no_pcm_guidance if args.model == "diffueraser" and args.pcm_mode == "none" else None,
        "width": args.width,
        "height": args.height,
        "num_frames": args.num_frames,
        "mask_dilation_iter": getattr(args, "mask_dilation_iter", 8),
        "seed": getattr(args, "seed", None),
        "diffueraser_path": str(args.diffueraser_path),
        "propainter_model_dir": str(args.propainter_model_dir),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    if args.model == "diffueraser":
        return f"diffueraser_or_{args.pcm_mode}_{args.prior_mode}_{digest}"
    if args.model == "propainter":
        return f"propainter_official_{digest}"
    return f"{args.model}_{digest}"


def diffueraser_cmd(args: argparse.Namespace, row: dict, out_dir: Path, work_dir: Path) -> list[str]:
    sample = row["sample_id"]
    video_root = work_dir / "batch" / "video_root"
    mask_root = work_dir / "batch" / "mask_root"
    symlink_dir(Path(row["condition_video_path"]), video_root / sample)
    symlink_dir(Path(row["mask_path"]), mask_root / sample)
    cmd = [
        args.python,
        str(args.project_root / "exp30_vor_or_multimodel_minimax" / "scripts" / "infer_diffueraser_or_exp30.py"),
        "--video_root",
        str(video_root),
        "--mask_root",
        str(mask_root),
        "--output_dir",
        str(out_dir),
        "--work_dir",
        str(work_dir),
        "--project_root",
        str(args.project_root),
        "--base_model_path",
        str(args.base_model_path),
        "--vae_path",
        str(args.vae_path),
        "--diffueraser_path",
        str(args.diffueraser_path),
        "--propainter_model_dir",
        str(args.propainter_model_dir),
        "--pcm_weights_path",
        str(args.pcm_weights_path),
        "--pcm_mode",
        args.pcm_mode,
        "--prior_mode",
        args.prior_mode,
        "--no_pcm_steps",
        str(args.no_pcm_steps),
        "--no_pcm_guidance",
        str(args.no_pcm_guidance),
        "--identity_out",
        str(work_dir / "generator_identity.json"),
        "--prompt",
        PROMPT,
        "--num_frames",
        str(args.num_frames),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--mask_dilation_iter",
        str(args.mask_dilation_iter),
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    return cmd


def propainter_cmd(args: argparse.Namespace, row: dict, out_dir: Path) -> list[str]:
    return [
        args.python,
        str(args.project_root / "DPO_finetune" / "infer_propainter_candidate.py"),
        "--video_dir",
        row["condition_video_path"],
        "--mask_dir",
        row["mask_path"],
        "--output_dir",
        str(out_dir),
        "--model_dir",
        str(args.propainter_model_dir),
        "--num_frames",
        str(args.num_frames),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--mask_dilation",
        str(args.mask_dilation_iter),
    ]


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.manifest)[: args.limit]
    gen_id = generator_id(args)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    results = []
    for row in rows:
        sample = row["sample_id"]
        out_dir = args.output_root / args.model / "raw_frames" / sample
        log_path = args.output_root / args.model / "logs" / f"{sample}.log"
        work_dir = args.output_root / args.model / "work" / sample
        if output_count(out_dir) >= args.num_frames:
            results.append({"sample_id": sample, "status": "resume_skip", "frames": output_count(out_dir), "output_dir": str(out_dir)})
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = diffueraser_cmd(args, row, out_dir, work_dir) if args.model == "diffueraser" else propainter_cmd(args, row, out_dir)
        rc, elapsed = run_command(cmd, log_path, args.project_root, env)
        frames = output_count(out_dir)
        results.append(
            {
                "sample_id": sample,
                "model": args.model,
                "generator_id": gen_id,
                "pcm_mode": args.pcm_mode if args.model == "diffueraser" else "",
                "prior_mode": args.prior_mode if args.model == "diffueraser" else "",
                "returncode": rc,
                "elapsed_seconds": elapsed,
                "frames": frames,
                "status": "OK" if rc == 0 and frames >= args.num_frames else "FAILED",
                "output_dir": str(out_dir),
                "log_path": str(log_path),
                "hard_comp": False,
                "comp_mode": "none",
            }
        )
    summary = {
        "model": args.model,
        "generator_id": gen_id,
        "ok": sum(1 for r in results if r["status"] in {"OK", "resume_skip"}),
        "failed": sum(1 for r in results if r["status"] == "FAILED"),
        "rows": results,
    }
    (args.output_root / f"{args.model}_smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
