#!/usr/bin/env python3
"""Run DAVIS50 object-removal inference for runnable frozen baselines.

This launcher is intentionally conservative:
- it never composites outputs for metrics;
- it runs full DAVIS50 for methods whose local runtime is ready;
- it writes BLOCKED status for methods without a verified OR runtime instead of
  fabricating outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROMPT = "remove the masked object and restore the background"


def image_files(path: Path) -> List[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def ensure_batch_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, dst, target_is_directory=True)


def run_command(cmd: List[str], log_path: Path, cwd: Path, env: dict) -> Tuple[bool, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[cmd] " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode == 0:
        return True, ""
    tail = ""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-40:])
    except Exception:
        tail = f"return code {proc.returncode}"
    return False, tail


def expected_frames(row: dict, max_frames: int) -> int:
    try:
        n = int(row.get("num_frames") or 0)
    except ValueError:
        n = 0
    if max_frames > 0 and n > 0:
        n = min(n, max_frames)
    return n


def enough_frames(out_dir: Path, row: dict, max_frames: int) -> bool:
    n = expected_frames(row, max_frames)
    files = image_files(out_dir) if out_dir.is_dir() else []
    return n > 0 and len(files) >= n


def run_propainter(args: argparse.Namespace, row: dict, env: dict) -> Tuple[bool, str]:
    name = row["video_name"]
    out_dir = args.output_root / "propainter" / "raw_frames" / name
    if args.resume and enough_frames(out_dir, row, args.max_frames):
        return True, "resume_skip"
    cmd = [
        args.python,
        str(args.project_root / "DPO_finetune" / "infer_propainter_candidate.py"),
        "--video_dir",
        row["frame_dir"],
        "--mask_dir",
        row["mask_dir"],
        "--output_dir",
        str(out_dir),
        "--model_dir",
        str(args.propainter_model_dir),
        "--num_frames",
        str(expected_frames(row, args.max_frames) or -1),
        "--width",
        str(args.propainter_width),
        "--height",
        str(args.propainter_height),
        "--mask_dilation",
        str(args.propainter_mask_dilation),
    ]
    log_path = args.output_root / "propainter" / "logs" / f"{name}.log"
    return run_command(cmd, log_path, args.project_root, env)


def run_cococo(args: argparse.Namespace, row: dict, env: dict) -> Tuple[bool, str]:
    name = row["video_name"]
    out_dir = args.output_root / "cococo" / "raw_frames" / name
    if args.resume and enough_frames(out_dir, row, args.max_frames):
        return True, "resume_skip"
    cmd = [
        args.python,
        str(args.project_root / "DPO_finetune" / "infer_cococo_candidate.py"),
        "--repo_dir",
        str(args.cococo_repo),
        "--video_dir",
        row["frame_dir"],
        "--mask_dir",
        row["mask_dir"],
        "--output_dir",
        str(out_dir),
        "--work_dir",
        str(args.output_root / "cococo" / "work" / name),
        "--model_path",
        str(args.cococo_weight / "cococo"),
        "--pretrain_model_path",
        str(args.cococo_weight / "stable-diffusion-v1-5-inpainting"),
        "--prompt",
        PROMPT,
        "--num_frames",
        str(expected_frames(row, args.max_frames) or -1),
        "--width",
        str(args.diffusion_width),
        "--height",
        str(args.diffusion_height),
    ]
    log_path = args.output_root / "cococo" / "logs" / f"{name}.log"
    return run_command(cmd, log_path, args.project_root, env)


def run_diffueraser(args: argparse.Namespace, row: dict, env: dict, method: str, weights: Path) -> Tuple[bool, str]:
    name = row["video_name"]
    out_dir = args.output_root / method / "raw_frames" / name
    if args.resume and enough_frames(out_dir, row, args.max_frames):
        return True, "resume_skip"
    batch_root = args.output_root / method / "work" / name / "batch"
    video_root = batch_root / "video_root"
    mask_root = batch_root / "mask_root"
    ensure_batch_symlink(Path(row["frame_dir"]), video_root / name)
    ensure_batch_symlink(Path(row["mask_dir"]), mask_root / name)
    cmd = [
        args.python,
        str(args.project_root / "exp15_or_benchmark_davis50" / "code" / "infer_diffueraser_or_exp15.py"),
        "--video_root",
        str(video_root),
        "--mask_root",
        str(mask_root),
        "--output_dir",
        str(out_dir),
        "--work_dir",
        str(args.output_root / method / "work" / name),
        "--project_root",
        str(args.project_root),
        "--base_model_path",
        str(args.base_model_path),
        "--vae_path",
        str(args.vae_path),
        "--diffueraser_path",
        str(weights),
        "--propainter_model_dir",
        str(args.propainter_model_dir),
        "--pcm_weights_path",
        str(args.pcm_weights_path),
        "--prompt",
        PROMPT,
        "--num_frames",
        str(expected_frames(row, args.max_frames) or -1),
        "--width",
        str(args.diffusion_width),
        "--height",
        str(args.diffusion_height),
        "--mask_dilation_iter",
        str(args.diffueraser_mask_dilation_iter),
    ]
    log_path = args.output_root / method / "logs" / f"{name}.log"
    return run_command(cmd, log_path, args.project_root, env)


def run_minimax(args: argparse.Namespace, row: dict, env: dict) -> Tuple[bool, str]:
    name = row["video_name"]
    out_dir = args.output_root / "minimax_remover" / "raw_frames" / name
    if args.resume and enough_frames(out_dir, row, args.max_frames):
        return True, "resume_skip"
    cmd = [
        str(args.minimax_python),
        str(args.project_root / "DPO_finetune" / "infer_minimax_candidate.py"),
        "--repo_dir",
        str(args.minimax_repo),
        "--video_dir",
        row["frame_dir"],
        "--mask_dir",
        row["mask_dir"],
        "--output_dir",
        str(out_dir),
        "--model_dir",
        str(args.minimax_weight),
        "--num_frames",
        str(expected_frames(row, args.max_frames) or -1),
        "--width",
        str(args.minimax_width),
        "--height",
        str(args.minimax_height),
    ]
    log_path = args.output_root / "minimax_remover" / "logs" / f"{name}.log"
    return run_command(cmd, log_path, args.project_root, env)


def validate_ready(args: argparse.Namespace) -> Dict[str, Tuple[str, str]]:
    status: Dict[str, Tuple[str, str]] = {}

    checks = {
        "propainter": args.propainter_model_dir / "raft-things.pth",
        "diffueraser_sft48000": args.sft48000_weights / "brushnet" / "config.json",
        "ours_exp11_outer_b075_s2": args.ours_weights / "brushnet" / "config.json",
    }
    for method, path in checks.items():
        status[method] = ("READY_TO_RUN", "") if path.exists() else ("BLOCKED_NO_WEIGHT", f"missing {path}")

    cococo_required = [
        args.cococo_repo / "valid_code_release.py",
        args.cococo_weight / "cococo" / "model_0.pth",
        args.cococo_weight / "stable-diffusion-v1-5-inpainting" / "vae" / "config.json",
        args.cococo_weight / "stable-diffusion-v1-5-inpainting" / "unet" / "config.json",
        args.cococo_weight / "stable-diffusion-v1-5-inpainting" / "tokenizer",
        args.cococo_weight / "stable-diffusion-v1-5-inpainting" / "text_encoder" / "config.json",
        args.cococo_weight / "stable-diffusion-v1-5-inpainting" / "scheduler" / "scheduler_config.json",
    ]
    cococo_missing = [str(p) for p in cococo_required if not p.exists()]
    if cococo_missing:
        status["cococo"] = (
            "BLOCKED_NO_WEIGHT",
            "COCOCO repo/checkpoints exist, but SD inpainting dependency is incomplete: " + "; ".join(cococo_missing[:4]),
        )
    else:
        status["cococo"] = ("READY_TO_RUN", "")

    if args.minimax_python.exists() and args.minimax_repo.exists() and (args.minimax_weight / "transformer").exists():
        status["minimax_remover"] = ("READY_TO_RUN", "")
    else:
        status["minimax_remover"] = (
            "BLOCKED_IMPORT_ERROR",
            "MiniMax requires isolated env with newer diffusers; current diffueraser env lacks AutoencoderKLWan/FP32LayerNorm.",
        )

    status["videopainter"] = (
        "BLOCKED_NO_OR_WRAPPER",
        "VideoPainter BR eval exists, but no verified DAVIS2017 foreground-mask OR thin wrapper is available in Exp15.",
    )
    status["floed"] = ("BLOCKED_NO_REPO", "No verified PAI repo+weights+OR wrapper.")
    status["vace"] = ("BLOCKED_NO_REPO", "No verified PAI repo+weights+OR wrapper.")
    status["videocomposer"] = ("BLOCKED_NO_REPO", "No verified PAI repo+weights+OR wrapper.")
    return status


def write_runtime_status(args: argparse.Namespace, status: Dict[str, Tuple[str, str]]) -> None:
    rows = []
    info = {
        "propainter": ("ProPainter", str(args.project_root / "DPO_finetune" / "infer_propainter_candidate.py"), str(args.propainter_model_dir), "DiffuEraser env"),
        "videocomposer": ("VideoComposer / VideoComp", "not verified", "", ""),
        "cococo": ("CoCoCo", str(args.project_root / "DPO_finetune" / "infer_cococo_candidate.py"), str(args.cococo_weight), "DiffuEraser env"),
        "floed": ("FloED", "not verified", "", ""),
        "diffueraser_sft48000": ("DiffuEraser SFT-48000", str(args.project_root / "DPO_finetune" / "infer_diffueraser_candidate.py"), str(args.sft48000_weights), "DiffuEraser env"),
        "videopainter": ("VideoPainter", "not verified for OR", str(args.videopainter_ckpt), ""),
        "vace": ("VACE", "not verified", "", ""),
        "ours_exp11_outer_b075_s2": ("Ours Exp11 outer b0.75 S2", str(args.project_root / "DPO_finetune" / "infer_diffueraser_candidate.py"), str(args.ours_weights), "DiffuEraser env"),
        "minimax_remover": ("MiniMax-Remover", str(args.project_root / "DPO_finetune" / "infer_minimax_candidate.py"), str(args.minimax_weight), str(args.minimax_python)),
    }
    for method in args.methods:
        method_status, reason = status.get(method, ("BLOCKED_NO_OR_WRAPPER", "not configured"))
        title, entry, weight, env = info.get(method, (method, "", "", ""))
        rows.append(
            {
                "method": title,
                "method_key": method,
                "repo_path": entry,
                "weight_path": weight,
                "env": env,
                "inference_entry": entry,
                "supports_or": "true" if method_status == "READY_TO_RUN" else "unknown",
                "supports_mask": "true" if method_status == "READY_TO_RUN" else "unknown",
                "supports_prompt": "true" if method in {"cococo", "diffueraser_sft48000", "ours_exp11_outer_b075_s2", "videopainter", "videocomposer", "vace"} else "false",
                "official_setting_found": "partial" if method_status == "READY_TO_RUN" else "false",
                "status": method_status,
                "blocked_reason": reason,
            }
        )
    csv_path = args.project_root / "reports" / "exp15_or_method_runtime_status.csv"
    md_path = args.project_root / "reports" / "exp15_or_method_runtime_status.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    md = [
        "# Exp15 OR Method Runtime Status",
        "",
        "| Method | Status | Entry | Weight | Blocked reason |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        md.append(f"| {row['method']} | {row['status']} | `{row['inference_entry']}` | `{row['weight_path']}` | {row['blocked_reason']} |")
    md.append("")
    md.append("Only `READY_TO_RUN` methods are executed. Blocked methods are carried into visual grids as explicit placeholders.")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")


def write_failed_cases(args: argparse.Namespace, method: str, rows: List[dict]) -> None:
    failed_path = args.output_root / method / "failed_cases.csv"
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["video_name", "status", "issue"]
    with failed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--project_root", default=".", type=Path)
    parser.add_argument("--methods", default="propainter,videocomposer,cococo,floed,diffueraser_sft48000,videopainter,vace,ours_exp11_outer_b075_s2,minimax_remover")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check_only", action="store_true", help="Only write runtime status reports, do not run inference.")
    parser.add_argument("--propainter_model_dir", default="/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter", type=Path)
    parser.add_argument("--propainter_width", type=int, default=512)
    parser.add_argument("--propainter_height", type=int, default=288)
    parser.add_argument("--propainter_mask_dilation", type=int, default=8)
    parser.add_argument("--cococo_repo", default="/mnt/nas/hj/official_repos/COCOCO_9ebe984", type=Path)
    parser.add_argument("--cococo_weight", default="/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/COCOCO_weight", type=Path)
    parser.add_argument("--base_model_path", default="/mnt/nas/hj/weights/stable-diffusion-v1-5", type=Path)
    parser.add_argument("--vae_path", default="/mnt/nas/hj/weights/sd-vae-ft-mse", type=Path)
    parser.add_argument("--pcm_weights_path", default="/mnt/nas/hj/weights/PCM_Weights", type=Path)
    parser.add_argument("--sft48000_weights", default="/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000", type=Path)
    parser.add_argument("--ours_weights", default="/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s2_2000_davis_pai/last_weights", type=Path)
    parser.add_argument("--diffusion_width", type=int, default=512)
    parser.add_argument("--diffusion_height", type=int, default=288)
    parser.add_argument("--diffueraser_mask_dilation_iter", type=int, default=8)
    parser.add_argument("--minimax_repo", default="/mnt/nas/hj/official_repos/MiniMax-Remover_28e12b4", type=Path)
    parser.add_argument("--minimax_weight", default="/mnt/nas/hj/H20_Video_inpainting_DPO/weights/minimax_remover/current", type=Path)
    parser.add_argument("--minimax_python", default="/mnt/nas/hj/conda_envs/minimax_remover/bin/python", type=Path)
    parser.add_argument("--minimax_width", type=int, default=832)
    parser.add_argument("--minimax_height", type=int, default=480)
    parser.add_argument("--videopainter_ckpt", default="/mnt/nas/hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt", type=Path)
    args = parser.parse_args()

    args.project_root = args.project_root.resolve()
    args.output_root = args.output_root.resolve()
    args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    args.output_root.mkdir(parents=True, exist_ok=True)

    with Path(args.manifest).open("r", encoding="utf-8", newline="") as handle:
        videos = list(csv.DictReader(handle))
    if len(videos) != 50:
        raise RuntimeError(f"Exp15 DAVIS50 requires 50 videos, got {len(videos)} from {args.manifest}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env.setdefault("PYTHONUNBUFFERED", "1")
    status = validate_ready(args)
    write_runtime_status(args, status)
    if args.check_only:
        print("[exp15-or] runtime status written; check_only=true")
        return

    runners: Dict[str, Callable[[argparse.Namespace, dict, dict], Tuple[bool, str]]] = {
        "propainter": lambda a, r, e: run_propainter(a, r, e),
        "cococo": lambda a, r, e: run_cococo(a, r, e),
        "diffueraser_sft48000": lambda a, r, e: run_diffueraser(a, r, e, "diffueraser_sft48000", a.sft48000_weights),
        "ours_exp11_outer_b075_s2": lambda a, r, e: run_diffueraser(a, r, e, "ours_exp11_outer_b075_s2", a.ours_weights),
        "minimax_remover": lambda a, r, e: run_minimax(a, r, e),
    }
    run_summary = []
    for method in args.methods:
        method_status, reason = status.get(method, ("BLOCKED_NO_OR_WRAPPER", "not configured"))
        if method_status != "READY_TO_RUN" or method not in runners:
            write_failed_cases(args, method, [{"video_name": row["video_name"], "status": method_status, "issue": reason} for row in videos])
            run_summary.append({"method": method, "status": method_status, "success": 0, "failed": len(videos), "reason": reason})
            continue
        failures = []
        successes = 0
        for row in videos:
            ok, issue = runners[method](args, row, env)
            if ok:
                successes += 1
            else:
                failures.append({"video_name": row["video_name"], "status": "FAILED", "issue": issue})
        write_failed_cases(args, method, failures)
        run_summary.append(
            {
                "method": method,
                "status": "OK" if not failures else "PARTIAL",
                "success": successes,
                "failed": len(failures),
                "reason": "" if not failures else "see failed_cases.csv",
            }
        )

    summary_path = args.output_root / "reports" / "inference_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(f"[exp15-or] wrote {summary_path}")


if __name__ == "__main__":
    main()
