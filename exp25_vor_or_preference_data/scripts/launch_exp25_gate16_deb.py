#!/usr/bin/env python3
"""Run the Exp25 fixed DE-B Gate16 pipeline on PAI."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


LOG_ROOT = Path("/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_vor_or_preference_data/cli4")
EXP_ROOT = Path("/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp25_vor_or_preference_data/cli4")
STACK_ID = "DE-B_sft_raw6_d8_propainter"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--run-id", default="")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-frames", type=int, default=24)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=288)
    p.add_argument("--sft-diffueraser-path", type=Path, default=Path("/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000"))
    p.add_argument("--base-model-path", type=Path, default=Path("/mnt/nas/hj/weights/stable-diffusion-v1-5"))
    p.add_argument("--vae-path", type=Path, default=Path("/mnt/nas/hj/weights/sd-vae-ft-mse"))
    p.add_argument("--propainter-model-dir", type=Path, default=Path("/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter"))
    p.add_argument("--pcm-weights-path", type=Path, default=Path("/mnt/nas/hj/weights/PCM_Weights/sd15/pcm_sd15_smallcfg_2step_converted.safetensors"))
    p.add_argument("--archive-dir", type=Path, default=Path("/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/downloads/fa09dc61128ca0418a4a13364d97a08018ea9cc7"))
    p.add_argument("--compute-lpips", action="store_true")
    p.add_argument("--compute-ewarp", action="store_true")
    return p.parse_args()


def run_logged(name: str, cmd: list[str], cwd: Path, log_path: Path, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n[{name}] start={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"[{name}] cmd=" + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\n[{name}] end={time.strftime('%Y-%m-%d %H:%M:%S')} returncode={proc.returncode}\n")
        return int(proc.returncode)


def write_stack_status(run_root: Path, stack_root: Path, log_path: Path, summary: dict) -> None:
    row = {
        "stack_id": STACK_ID,
        "status": "OK" if int(summary.get("failed", 1)) == 0 else "FAILED",
        "runnable": True,
        "block_reason": "",
        "returncode": 0 if int(summary.get("failed", 1)) == 0 else 2,
        "ok": summary.get("ok", ""),
        "failed": summary.get("failed", ""),
        "elapsed_seconds": "",
        "pcm_mode": "none",
        "prior_mode": "propainter",
        "no_pcm_steps": 6,
        "no_pcm_guidance": 0.0,
        "mask_dilation_iter": 8,
        "diffueraser_path": "",
        "stack_root": str(stack_root),
        "log_path": str(log_path),
        "description": "Exp25 Gate16 fixed DE-B stack: SFT raw6, d8, ProPainter prior, no PCM.",
    }
    fields = list(row.keys())
    with (run_root / "stack_status.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    (run_root / "stack_status.json").write_text(json.dumps([row], indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_id = args.run_id or "gate16_deb_" + time.strftime("%Y%m%d_%H%M%S")
    run_root = LOG_ROOT / run_id
    exp_root = EXP_ROOT / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    exp_root.mkdir(parents=True, exist_ok=True)

    manifest = run_root / "gate16_deb_manifest.jsonl"
    selection_json = run_root / "gate16_deb_selection_audit.json"
    selection_csv = run_root / "gate16_deb_selection_audit.csv"
    selection_md = run_root / "gate16_deb_selection_audit.md"
    rc = run_logged(
        "select_gate16_deb_sources",
        [
            args.python,
            str(args.project_root / "exp25_vor_or_preference_data" / "scripts" / "select_gate16_deb_sources.py"),
            "--output-manifest", str(manifest),
            "--audit-json", str(selection_json),
            "--audit-csv", str(selection_csv),
            "--audit-md", str(selection_md),
        ],
        args.project_root,
        run_root / "select_gate16_deb_sources.log",
    )
    if rc != 0:
        return rc

    extraction_root = exp_root / "extracted"
    rc = run_logged(
        "safe_extract_gate16_deb",
        [
            args.python,
            str(args.project_root / "exp25_vor_or_preference_data" / "scripts" / "safe_extract_vor_subset.py"),
            "--archive-dir", str(args.archive_dir),
            "--groups", "VOR-Train", "VOR-Train-MASK",
            "--triplet-jsonl", str(manifest),
            "--output-root", str(extraction_root),
            "--manifest-csv", str(run_root / "gate16_deb_extraction.csv"),
            "--state-json", str(run_root / "gate16_deb_extraction_state.json"),
        ],
        args.project_root,
        run_root / "safe_extract_gate16_deb.log",
    )
    if rc != 0:
        return rc

    materialized_root = exp_root / "materialized"
    materialized_manifest = run_root / "gate16_deb_materialized.jsonl"
    rc = run_logged(
        "materialize_gate16_deb",
        [
            args.python,
            str(args.project_root / "exp25_vor_or_preference_data" / "scripts" / "materialize_vor_or_inputs.py"),
            "--manifest", str(manifest),
            "--extraction-root", str(extraction_root),
            "--output-root", str(materialized_root),
            "--output-manifest", str(materialized_manifest),
            "--limit", "16",
            "--frames", str(args.num_frames),
        ],
        args.project_root,
        run_root / "materialize_gate16_deb.log",
    )
    if rc != 0:
        return rc

    stack_root = exp_root / "diffueraser_deb"
    gen_log = run_root / "diffueraser_deb_generation.log"
    env = dict(os.environ)
    rc = run_logged(
        "diffueraser_deb_generation",
        [
            args.python,
            str(args.project_root / "exp25_vor_or_preference_data" / "scripts" / "run_vor_or_model_smoke.py"),
            "--model", "diffueraser",
            "--manifest", str(materialized_manifest),
            "--project-root", str(args.project_root),
            "--output-root", str(stack_root),
            "--python", args.python,
            "--limit", "16",
            "--num-frames", str(args.num_frames),
            "--width", str(args.width),
            "--height", str(args.height),
            "--base-model-path", str(args.base_model_path),
            "--vae-path", str(args.vae_path),
            "--diffueraser-path", str(args.sft_diffueraser_path),
            "--propainter-model-dir", str(args.propainter_model_dir),
            "--pcm-weights-path", str(args.pcm_weights_path),
            "--pcm-mode", "none",
            "--prior-mode", "propainter",
            "--no-pcm-steps", "6",
            "--no-pcm-guidance", "0.0",
            "--mask-dilation-iter", "8",
            "--seed", "20260625",
        ],
        args.project_root,
        gen_log,
        env=env,
    )
    summary_path = stack_root / "diffueraser_smoke_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"ok": 0, "failed": 16}
    write_stack_status(run_root, stack_root, gen_log, summary)
    if rc != 0:
        return rc

    report_dir = run_root / "reports"
    review_cmd = [
        args.python,
        str(args.project_root / "exp25_vor_or_preference_data" / "scripts" / "review_diffueraser_root_cause_matrix.py"),
        "--run-root", str(run_root),
        "--source-manifest", str(materialized_manifest),
        "--project-root", str(args.project_root),
        "--report-dir", str(report_dir),
        "--num-frames", str(args.num_frames),
        "--width", str(args.width),
        "--height", str(args.height),
        "--device", args.device,
    ]
    if args.compute_lpips:
        review_cmd.append("--compute-lpips")
    if args.compute_ewarp:
        review_cmd.append("--compute-ewarp")
    rc = run_logged("review_gate16_deb", review_cmd, args.project_root, run_root / "review_gate16_deb.log")
    final = {
        "run_id": run_id,
        "run_root": str(run_root),
        "exp_root": str(exp_root),
        "manifest": str(manifest),
        "materialized_manifest": str(materialized_manifest),
        "stack_id": STACK_ID,
        "generation_summary": summary,
        "review_report_dir": str(report_dir),
        "status": "REVIEW_COMPLETE" if rc == 0 else "REVIEW_FAILED",
    }
    (run_root / "gate16_deb_final_state.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
