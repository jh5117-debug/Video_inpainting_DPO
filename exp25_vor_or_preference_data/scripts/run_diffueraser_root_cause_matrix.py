#!/usr/bin/env python3
"""Run the fixed Exp25 DiffuEraser OR root-cause generation matrix.

This launcher is intentionally conservative:
- it only runs stacks whose configuration can be represented by the current
  verified Exp25 DiffuEraser wrapper;
- it records unavailable official-core / no-prior variants as BLOCKED instead
  of substituting a different generator;
- it is resumable through the underlying per-stack smoke runner.

Metrics and visual classification are separate gates. This file only creates
the raw stack outputs and a stack-status manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class StackSpec:
    stack_id: str
    description: str
    runnable: bool
    block_reason: str
    diffueraser_path: str
    pcm_mode: str
    prior_mode: str
    no_pcm_steps: int
    no_pcm_guidance: float
    mask_dilation_iter: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--num-frames", type=int, default=24)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=288)
    p.add_argument("--sft-diffueraser-path", type=Path, default=Path("/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000"))
    p.add_argument("--official-core-path", type=Path, default=None)
    p.add_argument("--base-model-path", type=Path, default=Path("/mnt/nas/hj/weights/stable-diffusion-v1-5"))
    p.add_argument("--vae-path", type=Path, default=Path("/mnt/nas/hj/weights/sd-vae-ft-mse"))
    p.add_argument("--propainter-model-dir", type=Path, default=Path("/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter"))
    p.add_argument("--pcm-weights-path", type=Path, default=Path("/mnt/nas/hj/weights/PCM_Weights/sd15/pcm_sd15_smallcfg_2step_converted.safetensors"))
    p.add_argument("--seed", type=int, default=20260625)
    return p.parse_args()


def is_diffueraser_root(path: Path | None) -> bool:
    return bool(path and (path / "brushnet" / "config.json").is_file() and (path / "unet_main" / "config.json").is_file())


def build_stacks(args: argparse.Namespace) -> list[StackSpec]:
    official_ok = is_diffueraser_root(args.official_core_path)
    official_path = str(args.official_core_path) if args.official_core_path else ""
    official_block = "" if official_ok else "official DiffuEraser core checkpoint path not provided or not strict-load identifiable"
    return [
        StackSpec(
            stack_id="DE-A_sft_canonical_raw6_d0_propainter",
            description="SFT-48000 current canonical raw6: UniPC 6 steps, guidance 0, no PCM, ProPainter prior, mask dilation 0",
            runnable=True,
            block_reason="",
            diffueraser_path=str(args.sft_diffueraser_path),
            pcm_mode="none",
            prior_mode="propainter",
            no_pcm_steps=6,
            no_pcm_guidance=0.0,
            mask_dilation_iter=0,
        ),
        StackSpec(
            stack_id="DE-B_sft_raw6_d8_propainter",
            description="SFT-48000 raw6 with OR-style mask dilation 8, no PCM, ProPainter prior",
            runnable=True,
            block_reason="",
            diffueraser_path=str(args.sft_diffueraser_path),
            pcm_mode="none",
            prior_mode="propainter",
            no_pcm_steps=6,
            no_pcm_guidance=0.0,
            mask_dilation_iter=8,
        ),
        StackSpec(
            stack_id="DE-C_sft_official_pcm2_d8_propainter",
            description="SFT-48000 official/native OR PCM2 path when the active environment can load PCM LoRA",
            runnable=True,
            block_reason="",
            diffueraser_path=str(args.sft_diffueraser_path),
            pcm_mode="official_pcm2",
            prior_mode="propainter",
            no_pcm_steps=6,
            no_pcm_guidance=0.0,
            mask_dilation_iter=8,
        ),
        StackSpec(
            stack_id="DE-D_official_core_official_pcm2_d8_propainter",
            description="Official DiffuEraser core with official/native PCM2 OR stack",
            runnable=official_ok,
            block_reason=official_block,
            diffueraser_path=official_path,
            pcm_mode="official_pcm2",
            prior_mode="propainter",
            no_pcm_steps=6,
            no_pcm_guidance=0.0,
            mask_dilation_iter=8,
        ),
        StackSpec(
            stack_id="DE-E_official_core_canonical_raw6_d0_propainter",
            description="Official DiffuEraser core with project canonical raw6/d0 diagnostic",
            runnable=official_ok,
            block_reason=official_block,
            diffueraser_path=official_path,
            pcm_mode="none",
            prior_mode="propainter",
            no_pcm_steps=6,
            no_pcm_guidance=0.0,
            mask_dilation_iter=0,
        ),
        StackSpec(
            stack_id="DE-F_sft_native_high_quality_no_prior",
            description="SFT-48000 native/high-quality sampler without ProPainter prior",
            runnable=False,
            block_reason="current verified Exp25 wrapper only exposes prior_mode=propainter; no official no-prior OR definition is pinned",
            diffueraser_path=str(args.sft_diffueraser_path),
            pcm_mode="none",
            prior_mode="propainter",
            no_pcm_steps=6,
            no_pcm_guidance=0.0,
            mask_dilation_iter=0,
        ),
        StackSpec(
            stack_id="DE-G_official_pcm2_alias",
            description="Official PCM2 diagnostic is represented by DE-C/DE-D depending on core checkpoint identity",
            runnable=False,
            block_reason="deduplicated: DE-C covers SFT+PCM2; DE-D covers official-core+PCM2 if official core exists",
            diffueraser_path="",
            pcm_mode="official_pcm2",
            prior_mode="propainter",
            no_pcm_steps=6,
            no_pcm_guidance=0.0,
            mask_dilation_iter=8,
        ),
    ]


def run_stack(args: argparse.Namespace, spec: StackSpec) -> dict:
    stack_root = args.output_root / "stacks" / spec.stack_id
    stack_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cmd = [
        args.python,
        str(args.project_root / "exp25_vor_or_preference_data" / "scripts" / "run_vor_or_model_smoke.py"),
        "--model",
        "diffueraser",
        "--manifest",
        str(args.manifest),
        "--project-root",
        str(args.project_root),
        "--output-root",
        str(stack_root),
        "--python",
        args.python,
        "--limit",
        str(args.limit),
        "--num-frames",
        str(args.num_frames),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--base-model-path",
        str(args.base_model_path),
        "--vae-path",
        str(args.vae_path),
        "--diffueraser-path",
        spec.diffueraser_path,
        "--propainter-model-dir",
        str(args.propainter_model_dir),
        "--pcm-weights-path",
        str(args.pcm_weights_path),
        "--pcm-mode",
        spec.pcm_mode,
        "--prior-mode",
        spec.prior_mode,
        "--no-pcm-steps",
        str(spec.no_pcm_steps),
        "--no-pcm-guidance",
        str(spec.no_pcm_guidance),
        "--mask-dilation-iter",
        str(spec.mask_dilation_iter),
        "--seed",
        str(args.seed),
    ]
    log_path = args.output_root / "logs" / f"{spec.stack_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[cmd] " + " ".join(cmd) + "\n")
        log.write("[env] CUDA_VISIBLE_DEVICES=" + os.environ.get("CUDA_VISIBLE_DEVICES", "") + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(args.project_root), stdout=log, stderr=subprocess.STDOUT)
    summary_path = stack_root / "diffueraser_smoke_summary.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        **asdict(spec),
        "status": "OK" if proc.returncode == 0 and summary.get("failed", 1) == 0 else "FAILED",
        "returncode": proc.returncode,
        "elapsed_seconds": time.time() - started,
        "stack_root": str(stack_root),
        "log_path": str(log_path),
        "ok": summary.get("ok"),
        "failed": summary.get("failed"),
    }


def write_status(output_root: Path, rows: list[dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "stack_status.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (output_root / "stack_status.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "stack_id",
            "status",
            "runnable",
            "block_reason",
            "returncode",
            "ok",
            "failed",
            "elapsed_seconds",
            "pcm_mode",
            "prior_mode",
            "no_pcm_steps",
            "no_pcm_guidance",
            "mask_dilation_iter",
            "diffueraser_path",
            "stack_root",
            "log_path",
            "description",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    stacks = build_stacks(args)
    rows: list[dict] = []
    for spec in stacks:
        if not spec.runnable:
            rows.append({**asdict(spec), "status": "BLOCKED", "returncode": "", "ok": "", "failed": "", "elapsed_seconds": 0.0})
            write_status(args.output_root, rows)
            continue
        result = run_stack(args, spec)
        rows.append(result)
        write_status(args.output_root, rows)
    ok = [r for r in rows if r.get("status") == "OK"]
    failed = [r for r in rows if r.get("status") == "FAILED"]
    print(json.dumps({"ok": len(ok), "failed": len(failed), "total": len(rows), "output_root": str(args.output_root)}, indent=2))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
