#!/usr/bin/env python3
"""Validate or run Exp33 held-out VOR-Eval EffectErase official81 baseline."""

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


READY_STATUS = "EXP33_VOREVAL_OFFICIAL81_COMMAND_READY"
BLOCKED_STATUS = "EXP33_VOREVAL_OFFICIAL81_COMMAND_BLOCKED"
RUNNING_STATUS = "EXP33_VOREVAL_EFFECTERASE_INFERENCE_RUNNING"
COMPLETED_STATUS = "EXP33_VOREVAL_EFFECTERASE_INFERENCE_COMPLETED"
FAILED_STATUS = "EXP33_VOREVAL_EFFECTERASE_INFERENCE_FAILED"


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def validate_exp33_row(row: dict, output_root: Path) -> list[str]:
    errors: list[str] = []
    if not bool(row.get("vor_eval")):
        errors.append("row_not_marked_vor_eval")
    if bool(row.get("eligible_for_training")):
        errors.append("training_eligible_row_not_allowed")
    if row.get("source_role") != "held_out_vor_eval_baseline":
        errors.append("source_role_not_held_out_vor_eval_baseline")
    if row.get("scientific_role") != "held_out_baseline_only_not_training":
        errors.append("scientific_role_not_baseline_only")
    if row.get("raw_output_primary") is not True:
        errors.append("raw_output_not_primary")
    output_path = Path(row.get("output_path", ""))
    try:
        output_path.relative_to(output_root)
    except ValueError:
        errors.append("output_path_outside_output_root")
    return errors


def validate_inputs(rows: list[dict], output_root: Path, expected_rows: int) -> list[dict]:
    out = []
    seen_sample_ids: set[str] = set()
    for row in rows:
        cond = video_stats(Path(row["condition_path"]))
        mask = video_stats(Path(row["mask_path"]))
        winner = video_stats(Path(row["winner_path"]))
        errors = validate_exp33_row(row, output_root)
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            errors.append("missing_sample_id")
        if sample_id in seen_sample_ids:
            errors.append("duplicate_sample_id")
        seen_sample_ids.add(sample_id)
        for label, stats in (("condition", cond), ("winner", winner), ("mask", mask)):
            if not stats["opens"]:
                errors.append(f"{label}_does_not_open")
            if stats["frames"] != 81:
                errors.append(f"{label}_frames_not_81")
            if stats["width"] != 832 or stats["height"] != 480:
                errors.append(f"{label}_resolution_not_832x480")
        if mask["mask_non_empty_frames"] < 40:
            errors.append("mask_too_few_non_empty_frames")
        if int(row.get("num_frames", 0)) != 81:
            errors.append("row_num_frames_not_81")
        if int(row.get("height", 0)) != 480 or int(row.get("width", 0)) != 832:
            errors.append("row_resolution_not_832x480")
        out.append(
            {
                "sample_id": sample_id,
                "condition_frames": cond["frames"],
                "winner_frames": winner["frames"],
                "mask_frames": mask["frames"],
                "condition_resolution": f"{cond['width']}x{cond['height']}",
                "winner_resolution": f"{winner['width']}x{winner['height']}",
                "mask_resolution": f"{mask['width']}x{mask['height']}",
                "mask_non_empty_frames": mask["mask_non_empty_frames"],
                "vor_eval": bool(row.get("vor_eval")),
                "eligible_for_training": bool(row.get("eligible_for_training")),
                "source_role": row.get("source_role", ""),
                "scientific_role": row.get("scientific_role", ""),
                "status": "READY" if not errors else "BLOCKED",
                "errors": ";".join(dict.fromkeys(errors)),
            }
        )
    if len(rows) != expected_rows:
        out.append(
            {
                "sample_id": "__manifest__",
                "condition_frames": 0,
                "winner_frames": 0,
                "mask_frames": 0,
                "condition_resolution": "",
                "winner_resolution": "",
                "mask_resolution": "",
                "mask_non_empty_frames": 0,
                "vor_eval": False,
                "eligible_for_training": False,
                "source_role": "",
                "scientific_role": "",
                "status": "BLOCKED",
                "errors": f"row_count_{len(rows)}_not_{expected_rows}",
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_command_reports(args: argparse.Namespace, resolved: dict, input_rows: list[dict], commands: list[dict]) -> None:
    write_text(args.output_root / "commands.jsonl", "".join(json.dumps(row) + "\n" for row in commands))
    write_csv(args.reports_dir / "exp33_effecterase_vor_eval_official81_command_validation_inputs.csv", input_rows)
    write_text(
        args.runtime_dir / "resolved_config.json",
        json.dumps(resolved, indent=2, sort_keys=True) + "\n",
    )
    write_text(
        args.reports_dir / "exp33_effecterase_vor_eval_official81_command_validation.json",
        json.dumps(resolved, indent=2, sort_keys=True) + "\n",
    )
    command_example = " ".join(commands[0]["cmd"]) if commands else ""
    md_lines = [
        "# Exp33 EffectErase VOR-Eval Official81 Command Validation",
        "",
        f"Status: `{resolved['status']}`",
        "",
        f"- Branch: `{resolved['branch']}`",
        f"- Commit: `{resolved['commit']}`",
        f"- Manifest: `{args.manifest}`",
        f"- Manifest SHA256: `{resolved['manifest_sha256']}`",
        f"- Rows validated: {resolved['rows']}",
        f"- Expected rows: {args.expected_rows}",
        f"- Repo: `{args.repo}`",
        f"- Python: `{args.python}`",
        f"- Asset root: `{args.asset_root}`",
        f"- Output root: `{args.output_root}`",
        f"- Runtime dir: `{args.runtime_dir}`",
        f"- Assets ready: {resolved['assets_ready']}",
        f"- Inputs ready: {resolved['input_ready']}",
        f"- Official help return code: {resolved['official_help_returncode']}",
        f"- Supports `--num_frames`: {resolved['supports_num_frames']}",
        f"- VOR-Eval rows required: {resolved['all_vor_eval_rows']}",
        f"- Training eligible rows present: {resolved['eligible_for_training']}",
        f"- Adapter/training launched: false",
        f"- Dry-run only: {args.dry_run_only}",
        "",
        "## Constructed Command Example",
        "",
        "```bash",
        f"cd {args.repo} && CUDA_VISIBLE_DEVICES=<NON_RESERVED_GPU> {command_example}",
        "```",
        "",
        "This runner is restricted to held-out VOR-Eval EffectErase raw baseline inference.",
        "It does not launch adapter training and refuses training-eligible rows.",
        "",
    ]
    write_text(args.reports_dir / "exp33_effecterase_vor_eval_official81_command_validation.md", "\n".join(md_lines))


def output_stats(path: Path) -> dict:
    stats = video_stats(path)
    return {
        "output_opens": stats["opens"],
        "output_frames": stats["frames"],
        "output_resolution": f"{stats['width']}x{stats['height']}",
        "output_bytes": path.stat().st_size if path.exists() else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--expected-rows", type=int, default=43)
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
    input_rows = validate_inputs(rows, args.output_root, args.expected_rows if args.max_rows == 0 else len(rows))
    input_ready = all(row["status"] == "READY" for row in input_rows)
    assets_ready = all(row["exists"] for row in asset_info)
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
    all_vor_eval_rows = all(bool(row.get("vor_eval")) for row in rows)
    eligible_for_training = any(bool(row.get("eligible_for_training")) for row in rows)
    resolved = {
        "branch": args.branch,
        "commit": args.commit,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "rows": len(rows),
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
        "all_vor_eval_rows": all_vor_eval_rows,
        "eligible_for_training": eligible_for_training,
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
        "adapter_training_allowed": False,
    }
    status = (
        READY_STATUS
        if assets_ready
        and input_ready
        and help_proc.returncode == 0
        and resolved["supports_num_frames"]
        and all_vor_eval_rows
        and not eligible_for_training
        else BLOCKED_STATUS
    )
    resolved["status"] = status
    write_command_reports(args, resolved, input_rows, commands)
    print(json.dumps(resolved, indent=2, sort_keys=True))
    if args.dry_run_only:
        heartbeat(args.runtime_dir, "dry_run_finished", f"status={status}", args.gpu)
        return 0 if status == READY_STATUS else 2
    if status != READY_STATUS:
        heartbeat(args.runtime_dir, "blocked", status, args.gpu)
        return 2

    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    write_text(args.runtime_dir / "status", RUNNING_STATUS + "\n")
    status_rows: list[dict] = []
    exit_code = 0
    for item in commands:
        sample_id = item["sample_id"]
        heartbeat(args.runtime_dir, RUNNING_STATUS, sample_id, args.gpu)
        log_path = args.output_root / "logs" / f"{sample_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        with log_path.open("w", encoding="utf-8") as log_handle:
            proc = subprocess.run(item["cmd"], cwd=args.repo, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
        elapsed = time.time() - start
        output_path = Path(next(row["output_path"] for row in rows if row["sample_id"] == sample_id))
        row = {
            "sample_id": sample_id,
            "exit_code": proc.returncode,
            "elapsed_sec": elapsed,
            "output_path": str(output_path),
            "log_path": str(log_path),
            **output_stats(output_path),
        }
        status_rows.append(row)
        write_text(args.output_root / "per_row_status.jsonl", "".join(json.dumps(r) + "\n" for r in status_rows))
        write_csv(args.reports_dir / "exp33_effecterase_vor_eval_official81_inference_status.csv", status_rows)
        if proc.returncode != 0:
            exit_code = proc.returncode
            break

    final_status = COMPLETED_STATUS if exit_code == 0 and len(status_rows) == len(commands) else FAILED_STATUS
    summary = {
        "status": final_status,
        "exit_code": exit_code,
        "completed_rows": len(status_rows),
        "expected_rows": len(commands),
        "failed_rows": [row["sample_id"] for row in status_rows if row["exit_code"] != 0],
        "output_root": str(args.output_root),
        "runtime_dir": str(args.runtime_dir),
        "gpu": args.gpu,
    }
    write_text(args.runtime_dir / "status", final_status + "\n")
    write_text(args.runtime_dir / "exitcode", str(exit_code) + "\n")
    write_text(
        args.reports_dir / "exp33_effecterase_vor_eval_official81_inference_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    heartbeat(args.runtime_dir, final_status, f"rc={exit_code}", args.gpu)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
