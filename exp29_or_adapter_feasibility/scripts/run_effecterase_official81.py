#!/usr/bin/env python3
"""Validate or run official EffectErase 81-frame diagnostic smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import cv2


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def video_stats(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"opens": False, "frames": 0, "width": 0, "height": 0, "mask_non_empty_frames": 0}
    frames = 0
    non_empty = 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        if frame.mean(axis=2).max() > 10:
            non_empty += 1
    cap.release()
    return {
        "opens": True,
        "frames": frames,
        "width": width,
        "height": height,
        "mask_non_empty_frames": non_empty,
    }


def heartbeat(runtime_dir: Path, status: str, sample: str, gpu: str) -> None:
    write_text(
        runtime_dir / "heartbeat",
        "\n".join(
            [
                f"time={time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
                f"pid={os.getpid()}",
                f"pgid={os.getpgid(0)}",
                f"gpu=GPU{gpu}",
                f"status={status}",
                f"sample={sample}",
                "",
            ]
        ),
    )


def asset_records(asset_root: Path) -> list[dict]:
    wan = asset_root / "Wan-AI/Wan2.1-Fun-1.3B-InP"
    assets = [
        ("lora", asset_root / "FudanCVL/EffectErase/EffectErase.ckpt"),
        ("text_encoder", wan / "models_t5_umt5-xxl-enc-bf16.pth"),
        ("vae", wan / "Wan2.1_VAE.pth"),
        ("dit", wan / "diffusion_pytorch_model.safetensors"),
        ("image_encoder", wan / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ]
    records = []
    for name, path in assets:
        records.append(
            {
                "name": name,
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
                "sha256": sha256_file(path) if path.exists() else "",
            }
        )
    return records


def build_command(args: argparse.Namespace, row: dict) -> list[str]:
    wan = args.asset_root / "Wan-AI/Wan2.1-Fun-1.3B-InP"
    lora = args.asset_root / "FudanCVL/EffectErase/EffectErase.ckpt"
    return [
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
        "81",
        "--height",
        "480",
        "--width",
        "832",
        "--seed",
        str(row.get("seed", 2025)),
        "--cfg",
        str(row.get("cfg", 1.0)),
        "--num_inference_steps",
        str(row.get("num_inference_steps", 50)),
    ]


def validate_inputs(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        cond = video_stats(Path(row["condition_path"]))
        mask = video_stats(Path(row["mask_path"]))
        winner = video_stats(Path(row["winner_path"]))
        errors: list[str] = []
        for label, stats in (("condition", cond), ("winner", winner), ("mask", mask)):
            if not stats["opens"]:
                errors.append(f"{label}_does_not_open")
            if stats["frames"] != 81:
                errors.append(f"{label}_frames_not_81")
            if stats["width"] != 832 or stats["height"] != 480:
                errors.append(f"{label}_resolution_not_832x480")
        if mask["mask_non_empty_frames"] < 40:
            errors.append("mask_too_few_non_empty_frames")
        if bool(row.get("vor_eval")):
            errors.append("vor_eval_row_not_allowed")
        if bool(row.get("eligible_for_training")):
            errors.append("training_eligible_row_not_allowed")
        out.append(
            {
                "sample_id": row["sample_id"],
                "condition_frames": cond["frames"],
                "winner_frames": winner["frames"],
                "mask_frames": mask["frames"],
                "condition_resolution": f"{cond['width']}x{cond['height']}",
                "winner_resolution": f"{winner['width']}x{winner['height']}",
                "mask_resolution": f"{mask['width']}x{mask['height']}",
                "mask_non_empty_frames": mask["mask_non_empty_frames"],
                "vor_eval": bool(row.get("vor_eval")),
                "eligible_for_training": bool(row.get("eligible_for_training")),
                "status": "READY" if not errors else "BLOCKED",
                "errors": ";".join(dict.fromkeys(errors)),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    write_text(args.runtime_dir / "pid", str(os.getpid()) + "\n")
    write_text(args.runtime_dir / "pgid", str(os.getpgid(0)) + "\n")
    asset_info = asset_records(args.asset_root)
    input_rows = validate_inputs(rows)
    input_ready = all(r["status"] == "READY" for r in input_rows)
    assets_ready = all(r["exists"] for r in asset_info)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    help_proc = subprocess.run(
        [str(args.python), "examples/remove_wan/infer_remove_wan.py", "--help"],
        cwd=args.repo,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    help_text = help_proc.stdout
    commands = [{"sample_id": row["sample_id"], "cmd": build_command(args, row)} for row in rows]
    write_text(args.output_root / "commands.jsonl", "".join(json.dumps(c) + "\n" for c in commands))
    write_csv(args.reports_dir / "exp29_effecterase_official81_command_validation_inputs.csv", input_rows)
    resolved = {
        "branch": "research/exp29-minimax-effecterase-adapter-feasibility-20260626",
        "commit": args.commit,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "repo": str(args.repo),
        "python": str(args.python),
        "asset_root": str(args.asset_root),
        "output_root": str(args.output_root),
        "runtime_dir": str(args.runtime_dir),
        "gpu": args.gpu,
        "num_frames": 81,
        "height": 480,
        "width": 832,
        "seed": 2025,
        "cfg": 1.0,
        "num_inference_steps": 50,
        "raw_output_primary": True,
        "diagnostic_comp_optional": True,
        "vor_eval_used": any(r["vor_eval"] for r in input_rows),
        "eligible_for_training": any(r["eligible_for_training"] for r in input_rows),
        "asset_records": asset_info,
        "assets_ready": assets_ready,
        "input_ready": input_ready,
        "official_help_returncode": help_proc.returncode,
        "supports_num_frames": "--num_frames" in help_text,
        "supports_cfg": "--cfg" in help_text,
        "supports_steps": "--num_inference_steps" in help_text,
        "supports_seed": "--seed" in help_text,
        "dry_run_only": args.dry_run_only,
        "commands_path": str(args.output_root / "commands.jsonl"),
    }
    status = "EFFECTERASE_OFFICIAL81_COMMAND_READY" if (
        assets_ready
        and input_ready
        and help_proc.returncode == 0
        and resolved["supports_num_frames"]
        and not resolved["vor_eval_used"]
        and not resolved["eligible_for_training"]
    ) else "EFFECTERASE_OFFICIAL81_COMMAND_BLOCKED"
    resolved["status"] = status
    write_text(args.runtime_dir / "resolved_config.json", json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    write_text(args.reports_dir / "exp29_effecterase_official81_command_validation.json", json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    command_example = " ".join(commands[0]["cmd"]) if commands else ""
    md_lines = [
        "# Exp29 EffectErase Official 81F Command Validation",
        "",
        f"Status: `{status}`",
        "",
        f"- Manifest: `{args.manifest}`",
        f"- Manifest SHA256: `{resolved['manifest_sha256']}`",
        f"- Rows validated: {len(rows)}",
        f"- Repo: `{args.repo}`",
        f"- Python: `{args.python}`",
        f"- Asset root: `{args.asset_root}`",
        f"- Assets ready: {assets_ready}",
        f"- Inputs ready: {input_ready}",
        f"- Official help return code: {help_proc.returncode}",
        f"- Supports `--num_frames`: {resolved['supports_num_frames']}",
        f"- VOR-Eval use: {resolved['vor_eval_used']}",
        f"- Training eligibility: {resolved['eligible_for_training']}",
        f"- Dry-run only: {args.dry_run_only}",
        "",
        "## Constructed Command Example",
        "",
        "```bash",
        f"cd {args.repo} && CUDA_VISIBLE_DEVICES=<RIGHT_GPU> {command_example}",
        "```",
        "",
        "No full EffectErase inference was launched by command validation.",
        "",
    ]
    write_text(args.reports_dir / "exp29_effecterase_official81_command_validation.md", "\n".join(md_lines))
    print(json.dumps(resolved, indent=2, sort_keys=True))
    if args.dry_run_only:
        heartbeat(args.runtime_dir, "dry_run_finished", f"status={status}", args.gpu)
        raise SystemExit(0 if status == "EFFECTERASE_OFFICIAL81_COMMAND_READY" else 2)
    if status != "EFFECTERASE_OFFICIAL81_COMMAND_READY":
        heartbeat(args.runtime_dir, "blocked", status, args.gpu)
        raise SystemExit(2)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    status_rows = []
    exit_code = 0
    for item in commands:
        sample = item["sample_id"]
        heartbeat(args.runtime_dir, "running", sample, args.gpu)
        log_path = args.output_root / "logs" / f"{sample}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(item["cmd"], cwd=args.repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.time() - start
        row = {
            "sample_id": sample,
            "exit_code": proc.returncode,
            "elapsed_sec": elapsed,
            "output_path": next(r["output_path"] for r in rows if r["sample_id"] == sample),
            "log_path": str(log_path),
        }
        status_rows.append(row)
        write_text(args.output_root / "per_row_status.jsonl", "".join(json.dumps(r) + "\n" for r in status_rows))
        if proc.returncode != 0:
            exit_code = proc.returncode
            break
    heartbeat(args.runtime_dir, "finished", f"rc={exit_code}", args.gpu)
    write_text(args.runtime_dir / "exitcode", str(exit_code) + "\n")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
