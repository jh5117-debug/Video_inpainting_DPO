#!/usr/bin/env python3
"""Exp25-only DiffuEraser OR wrapper with explicit PCM identity.

The shared OR file `diffueraser/diffueraser_OR.py` still tries to load the
Stable Diffusion safety checker. The local SD1.5 folder on PAI does not contain
that component, while the non-OR DiffuEraser path already disables it.

To keep the shared code untouched, this wrapper builds a tiny temporary project
overlay for each video:
- copied `inference/run_OR.py` so its REPO_ROOT points to the overlay;
- symlinked repo folders/files for the rest;
- patched overlay-only `diffueraser/diffueraser_OR.py`.

Unlike the legacy Exp15 wrapper, this file requires a declared PCM mode:

- `official_pcm2`: official DiffuEraser 2-step PCM LoRA acceleration.
- `none`: no PCM weights are read; UniPC scheduler remains active and a locked
  no-PCM step count is used.

The no-PCM path is not a silent fallback. It is a separate generator identity.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import cv2

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def image_files(path: Path) -> List[Path]:
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def first_sequence_name(video_root: Path, mask_root: Path) -> str:
    for child in sorted(video_root.iterdir()):
        if child.is_dir() and (mask_root / child.name).is_dir() and image_files(child):
            return child.name
    raise RuntimeError(f"no matching frame/mask sequence found under {video_root} and {mask_root}")


def pad_sequence_tail(src_dir: Path, dst_dir: Path, target_len: int) -> int:
    files = image_files(src_dir)
    if not files:
        raise RuntimeError(f"no frames found under {src_dir}")
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(files):
        shutil.copy2(src, dst_dir / f"{idx:05d}{src.suffix.lower()}")
    if len(files) >= target_len:
        return len(files)
    last = files[-1]
    for idx in range(len(files), target_len):
        shutil.copy2(last, dst_dir / f"{idx:05d}{last.suffix.lower()}")
    return target_len


def mp4_to_frames(mp4_path: Path, output_dir: Path, limit: int) -> int:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open DiffuEraser mp4: {mp4_path}")
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if limit > 0 and count >= limit:
            break
        cv2.imwrite(str(output_dir / f"{count:05d}.png"), frame)
        count += 1
    cap.release()
    if count == 0:
        raise RuntimeError(f"no frames decoded from DiffuEraser mp4: {mp4_path}")
    return count


def is_diffueraser_checkpoint_root(path: Path) -> bool:
    return (path / "brushnet" / "config.json").exists() and (path / "unet_main" / "config.json").exists()


def resolve_diffueraser_path(path: Path) -> Path:
    path = path.resolve()
    if is_diffueraser_checkpoint_root(path):
        return path
    candidates = []
    for child in path.iterdir() if path.exists() else []:
        if child.is_dir() and is_diffueraser_checkpoint_root(child):
            score = 100 if child.name.startswith("converted_weights_step") else 10
            candidates.append((score, child))
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1].resolve()
    raise FileNotFoundError(f"DiffuEraser checkpoint root not found: {path}")


def patch_diffueraser_or(src_text: str, pcm_mode: str) -> str:
    old = """brushnet=self.brushnet
        ).to(self.device, torch.float16)"""
    new = """brushnet=self.brushnet,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        ).to(self.device, torch.float16)"""
    if old not in src_text:
        if "requires_safety_checker=False" in src_text:
            patched = src_text
        else:
            raise RuntimeError("could not patch diffueraser_OR safety checker call")
    else:
        patched = src_text.replace(old, new)
    if pcm_mode == "official_pcm2":
        return patched
    if pcm_mode != "none":
        raise ValueError(f"unsupported pcm_mode: {pcm_mode}")

    pcm_block = """        ## use PCM
        self.ckpt = ckpt
        PCM_ckpts = checkpoints[ckpt][0].format(mode)
        self.guidance_scale = checkpoints[ckpt][2]
        if loaded != (ckpt + mode):
            # MODIFIED: use pcm_weights_path parameter instead of hardcoded path
            self.pipeline.load_lora_weights(
                pcm_weights_path, weight_name=PCM_ckpts, subfolder=mode
            )
            loaded = ckpt + mode

            if ckpt == "LCM-Like LoRA":
                self.pipeline.scheduler = LCMScheduler()
            else:
                self.pipeline.scheduler = TCDScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    timestep_spacing="trailing",
                )
        self.num_inference_steps = checkpoints[ckpt][1]
"""
    no_pcm_block = """        ## Exp25 explicit no-PCM mode.
        ## This is a separate generator identity, not a fallback.
        self.ckpt = "none"
        self.guidance_scale = float(os.environ.get("EXP25_NO_PCM_GUIDANCE", "0.0"))
        self.num_inference_steps = int(os.environ.get("EXP25_NO_PCM_STEPS", "6"))
        loaded = "none"
"""
    if pcm_block not in patched:
        if 'self.ckpt = "none"' in patched:
            return patched
        raise RuntimeError("could not patch diffueraser_OR PCM block for explicit no-PCM mode")
    return patched.replace(pcm_block, no_pcm_block)


def ensure_overlay(project_root: Path, overlay_root: Path, pcm_mode: str) -> Path:
    if overlay_root.exists():
        shutil.rmtree(overlay_root)
    overlay_root.mkdir(parents=True, exist_ok=True)
    skip = {".git", "logs", "reports", "exp15_or_benchmark_davis50"}
    for child in project_root.iterdir():
        if child.name in skip:
            continue
        target = overlay_root / child.name
        os.symlink(child, target, target_is_directory=child.is_dir())

    inference_dir = overlay_root / "inference"
    if inference_dir.exists() or inference_dir.is_symlink():
        inference_dir.unlink() if inference_dir.is_symlink() else shutil.rmtree(inference_dir)
    inference_dir.mkdir()
    for child in (project_root / "inference").iterdir():
        target = inference_dir / child.name
        if child.name == "run_OR.py":
            shutil.copy2(child, target)
        else:
            os.symlink(child, target, target_is_directory=child.is_dir())

    diff_dir = overlay_root / "diffueraser"
    if diff_dir.exists() or diff_dir.is_symlink():
        diff_dir.unlink() if diff_dir.is_symlink() else shutil.rmtree(diff_dir)
    diff_dir.mkdir()
    for child in (project_root / "diffueraser").iterdir():
        target = diff_dir / child.name
        if child.name == "diffueraser_OR.py":
            patched = patch_diffueraser_or(child.read_text(encoding="utf-8"), pcm_mode=pcm_mode)
            target.write_text(patched, encoding="utf-8")
        else:
            os.symlink(child, target, target_is_directory=child.is_dir())
    return overlay_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--mask_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--diffueraser_path", required=True)
    parser.add_argument("--propainter_model_dir", required=True)
    parser.add_argument("--pcm_weights_path", required=True)
    parser.add_argument("--pcm_mode", choices=["official_pcm2", "none"], required=True)
    parser.add_argument("--prior_mode", choices=["propainter"], default="propainter")
    parser.add_argument("--no_pcm_steps", type=int, default=6)
    parser.add_argument("--no_pcm_guidance", type=float, default=0.0)
    parser.add_argument("--identity_out", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative_prompt", default="worst quality. bad quality.")
    parser.add_argument("--text_guidance_scale", type=float, default=2.0)
    parser.add_argument("--num_frames", type=int, default=-1)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--mask_dilation_iter", type=int, default=8)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    video_root = Path(args.video_root).resolve()
    mask_root = Path(args.mask_root).resolve()
    work_dir = Path(args.work_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    diffueraser_path = resolve_diffueraser_path(Path(args.diffueraser_path))
    sequence_name = first_sequence_name(video_root, mask_root)
    effective_num_frames = args.num_frames
    run_video_root = video_root
    run_mask_root = mask_root

    if 0 < args.num_frames < 23:
        padded_root = work_dir / "padded_inputs"
        run_video_root = padded_root / "videos"
        run_mask_root = padded_root / "masks"
        pad_sequence_tail(video_root / sequence_name, run_video_root / sequence_name, 23)
        pad_sequence_tail(mask_root / sequence_name, run_mask_root / sequence_name, 23)
        effective_num_frames = 23

    overlay_root = ensure_overlay(project_root, work_dir / "patched_project_root", args.pcm_mode)
    run_dir = work_dir / "run_or"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(overlay_root / "inference" / "run_OR.py"),
        "--dataset",
        "custom",
        "--video_root",
        str(run_video_root),
        "--mask_root",
        str(run_mask_root),
        "--save_path",
        str(run_dir),
        "--video_length",
        str(effective_num_frames),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--mask_dilation_iter",
        str(args.mask_dilation_iter),
        "--base_model_path",
        str(Path(args.base_model_path).resolve()),
        "--vae_path",
        str(Path(args.vae_path).resolve()),
        "--diffueraser_path",
        str(diffueraser_path),
        "--propainter_model_dir",
        str(Path(args.propainter_model_dir).resolve()),
        "--pcm_weights_path",
        str(Path(args.pcm_weights_path).resolve()),
        "--summary_out",
        "summary.json",
    ]
    env = os.environ.copy()
    if args.pcm_mode == "none":
        env["EXP25_NO_PCM_STEPS"] = str(args.no_pcm_steps)
        env["EXP25_NO_PCM_GUIDANCE"] = str(args.no_pcm_guidance)
    if args.prompt.strip():
        cmd.extend(
            [
                "--use_text",
                f"--prompt={args.prompt.strip()}",
                f"--n_prompt={args.negative_prompt}",
                "--text_guidance_scale",
                str(args.text_guidance_scale),
            ]
        )

    print("[exp25-diffueraser] run:", " ".join(cmd), flush=True)
    identity = {
        "wrapper": "exp25_vor_or_preference_data/scripts/infer_diffueraser_or_exp25.py",
        "pcm_mode": args.pcm_mode,
        "prior_mode": args.prior_mode,
        "no_pcm_steps": args.no_pcm_steps if args.pcm_mode == "none" else None,
        "no_pcm_guidance": args.no_pcm_guidance if args.pcm_mode == "none" else None,
        "pcm_weights_path": str(Path(args.pcm_weights_path).resolve()) if args.pcm_mode != "none" else None,
        "diffueraser_path": str(diffueraser_path),
        "base_model_path": str(Path(args.base_model_path).resolve()),
        "vae_path": str(Path(args.vae_path).resolve()),
        "propainter_model_dir": str(Path(args.propainter_model_dir).resolve()),
        "raw_no_comp": True,
        "fallback": False,
        "mask_dilation_iter": args.mask_dilation_iter,
        "width": args.width,
        "height": args.height,
        "num_frames": args.num_frames,
    }
    if args.identity_out:
        identity_path = Path(args.identity_out)
        identity_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[exp25-diffueraser] identity:", identity, flush=True)
    subprocess.run(cmd, cwd=str(overlay_root), check=True, env=env)
    pred_mp4 = run_dir / sequence_name / "diffueraser.mp4"
    saved = mp4_to_frames(pred_mp4, output_dir, args.num_frames)
    print(f"[exp25-diffueraser] saved {saved} frames to {output_dir}")


if __name__ == "__main__":
    main()
