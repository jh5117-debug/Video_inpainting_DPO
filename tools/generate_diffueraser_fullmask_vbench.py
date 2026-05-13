#!/usr/bin/env python
"""Generate VBench-standard videos with DiffuEraser full-mask conditioning."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL, UniPCMultistepScheduler
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffueraser.pipeline_diffueraser_stage1 import StableDiffusionDiffuEraserPipelineStageOne
from libs.brushnet_CA import BrushNetModel
from libs.unet_2d_condition import UNet2DConditionModel
from libs.unet_motion_model import UNetMotionModel
from training.dpo.train_stage1 import import_model_class_from_model_name_or_path, resolve_torch_dtype


def read_prompts(path: Path, limit: int = 0) -> list[str]:
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            prompt = line.strip()
            if prompt:
                prompts.append(prompt)
            if limit and len(prompts) >= limit:
                break
    if not prompts:
        raise RuntimeError(f"No prompts found in {path}")
    return prompts


def extract_2d_from_motion(motion_unet, base_model_path: str, revision=None, variant=None):
    unet_2d = UNet2DConditionModel.from_pretrained(
        base_model_path, subfolder="unet", revision=revision, variant=variant
    )
    unet_2d.conv_in.load_state_dict(motion_unet.conv_in.state_dict())
    unet_2d.time_proj.load_state_dict(motion_unet.time_proj.state_dict())
    unet_2d.time_embedding.load_state_dict(motion_unet.time_embedding.state_dict())

    for i, down_block in enumerate(motion_unet.down_blocks):
        unet_2d.down_blocks[i].resnets.load_state_dict(down_block.resnets.state_dict())
        if hasattr(unet_2d.down_blocks[i], "attentions") and hasattr(down_block, "attentions"):
            unet_2d.down_blocks[i].attentions.load_state_dict(down_block.attentions.state_dict())
        if unet_2d.down_blocks[i].downsamplers and down_block.downsamplers:
            unet_2d.down_blocks[i].downsamplers.load_state_dict(down_block.downsamplers.state_dict())

    for i, up_block in enumerate(motion_unet.up_blocks):
        unet_2d.up_blocks[i].resnets.load_state_dict(up_block.resnets.state_dict())
        if hasattr(unet_2d.up_blocks[i], "attentions") and hasattr(up_block, "attentions"):
            unet_2d.up_blocks[i].attentions.load_state_dict(up_block.attentions.state_dict())
        if unet_2d.up_blocks[i].upsamplers and up_block.upsamplers:
            unet_2d.up_blocks[i].upsamplers.load_state_dict(up_block.upsamplers.state_dict())

    unet_2d.mid_block.resnets.load_state_dict(motion_unet.mid_block.resnets.state_dict())
    unet_2d.mid_block.attentions.load_state_dict(motion_unet.mid_block.attentions.state_dict())
    if motion_unet.conv_norm_out is not None:
        unet_2d.conv_norm_out.load_state_dict(motion_unet.conv_norm_out.state_dict())
    if hasattr(motion_unet, "conv_act") and motion_unet.conv_act is not None:
        unet_2d.conv_act.load_state_dict(motion_unet.conv_act.state_dict())
    unet_2d.conv_out.load_state_dict(motion_unet.conv_out.state_dict())
    return unet_2d


def load_unet(weights_path: Path, base_model_path: str, revision=None, variant=None):
    config_path = weights_path / "unet_main" / "config.json"
    is_motion_model = False
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            is_motion_model = json.load(f).get("_class_name") == "UNetMotionModel"
    if is_motion_model:
        motion_unet = UNetMotionModel.from_pretrained(str(weights_path), subfolder="unet_main")
        unet = extract_2d_from_motion(motion_unet, base_model_path, revision=revision, variant=variant)
        del motion_unet
        return unet
    return UNet2DConditionModel.from_pretrained(str(weights_path), subfolder="unet_main")


def save_video(frames, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in frames]
    imageio.mimsave(path, arrays, fps=fps)


def build_pipeline(args, device: torch.device):
    weight_dtype = resolve_torch_dtype(args.torch_dtype, torch.float32)
    vae_dtype = resolve_torch_dtype(args.vae_dtype, weight_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False
    )
    text_encoder_cls = import_model_class_from_model_name_or_path(args.base_model_name_or_path, args.revision)
    text_encoder = text_encoder_cls.from_pretrained(
        args.base_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.vae_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = load_unet(args.weights_path, args.base_model_name_or_path, args.revision, args.variant)
    brushnet = BrushNetModel.from_pretrained(str(args.weights_path), subfolder="brushnet")

    pipeline = StableDiffusionDiffuEraserPipelineStageOne.from_pretrained(
        args.base_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        brushnet=brushnet,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.vae.to(device=device, dtype=vae_dtype)
    pipeline.set_progress_bar_config(disable=not args.show_progress)
    return pipeline, weight_dtype


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", required=True)
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--weights_path", required=True, type=Path)
    parser.add_argument("--prompts_file", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--samples_per_prompt", type=int, default=5)
    parser.add_argument("--prompt_limit", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=12.0)
    parser.add_argument("--seed_base", type=int, default=20230211)
    parser.add_argument("--full_mask_value", type=float, default=0.0)
    parser.add_argument("--torch_dtype", choices=["auto", "fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--vae_dtype", choices=["auto", "fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--show_progress", action="store_true")
    args = parser.parse_args()

    if not (args.weights_path / "unet_main").is_dir() or not (args.weights_path / "brushnet").is_dir():
        raise FileNotFoundError(
            f"--weights_path must contain unet_main/ and brushnet/: {args.weights_path}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompts = read_prompts(args.prompts_file, args.prompt_limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline, weight_dtype = build_pipeline(args, device)
    blank_rgb = Image.new("RGB", (args.width, args.height), (0, 0, 0))
    mask_pixel = int(max(0.0, min(1.0, args.full_mask_value)) * 255)
    full_mask = Image.new("L", (args.width, args.height), mask_pixel)
    images = [blank_rgb.copy() for _ in range(args.frames)]
    masks = [full_mask.copy() for _ in range(args.frames)]

    autocast_dtype = None
    if device.type == "cuda" and weight_dtype in {torch.float16, torch.bfloat16}:
        autocast_dtype = weight_dtype

    for sample_idx in range(args.samples_per_prompt):
        for prompt_idx, prompt in enumerate(prompts):
            out_path = args.output_dir / f"{prompt}-{sample_idx}.mp4"
            if args.skip_existing and out_path.exists():
                print(f"[fullmask-vbench-gen] skip existing {out_path}")
                continue
            seed = args.seed_base + sample_idx
            generator = torch.Generator(device=device).manual_seed(seed)
            print(
                f"[fullmask-vbench-gen] sample={sample_idx} "
                f"prompt={prompt_idx + 1}/{len(prompts)} seed={seed}: {prompt}"
            )
            with torch.no_grad():
                if autocast_dtype is None:
                    frames = pipeline(
                        num_frames=args.frames,
                        prompt=prompt,
                        images=images,
                        masks=masks,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=generator,
                    ).frames
                else:
                    with torch.autocast("cuda", dtype=autocast_dtype):
                        frames = pipeline(
                            num_frames=args.frames,
                            prompt=prompt,
                            images=images,
                            masks=masks,
                            height=args.height,
                            width=args.width,
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=args.guidance_scale,
                            generator=generator,
                        ).frames
            save_video(frames, out_path, args.fps)
            print(f"[fullmask-vbench-gen] saved {out_path}")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"[fullmask-vbench-gen] done: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
