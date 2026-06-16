#!/usr/bin/env python3
"""Isolated VideoPainter DPO adapter trainer.

This file intentionally lives under exp14_adapter_videopainter/code/ so it does
not modify upstream VideoPainter or shared DiffuEraser DPO training code.

The trainer adapts the current best Exp11 outer b0.75 objective to
VideoPainter's CogVideoX branch training loop:

    m_w, m_l, m_w_ref, m_l_ref are region-weighted latent denoising losses
    computed for policy and frozen-reference branches at the same timestep/noise.

The implementation is deliberately conservative. It does not fall back to
upstream VideoPainter training if the policy/reference DPO path cannot be
constructed.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
import os
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


DIAG_COLUMNS = [
    "step",
    "loss",
    "dpo_loss",
    "implicit_acc",
    "m_w",
    "m_l",
    "m_w_ref",
    "m_l_ref",
    "raw_win_gap",
    "raw_lose_gap",
    "norm_win_gap",
    "norm_lose_gap",
    "norm_lose_gap_clipped",
    "winner_abs_reg",
    "winner_gap_reg",
    "mse_w_over_ref_mse_w",
    "mse_l_over_ref_mse_l",
    "loser_dominant_ratio",
    "grad_norm",
    "lr",
    "mask_area_ratio",
    "boundary_area_ratio",
    "outside_area_ratio",
    "region_weight_sum",
    "boundary_mode",
    "mask_weight",
    "boundary_weight",
    "outside_weight",
]


@dataclass
class VideoPainterBatch:
    winner: torch.Tensor
    loser: torch.Tensor
    conditioning: torch.Tensor
    mask: torch.Tensor
    prompts: List[str]
    sample_ids: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("VideoPainter direct Diff-DPO adapter trainer")

    parser.add_argument("--videopainter_root", required=True)
    parser.add_argument("--pretrained_model_name_or_path", default=None)
    parser.add_argument("--policy_checkpoint", required=True)
    parser.add_argument("--reference_checkpoint", required=True)
    parser.add_argument("--pair_manifest", required=True)
    parser.add_argument("--youtubevos_root", default=None)
    parser.add_argument("--davis_root", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dpo_diag_csv", required=True)

    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--dpo_diag_log_every", type=int, default=10)
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--limit_train_samples", type=int, default=0)

    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--first_frame_gt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt_max_sequence_length", type=int, default=226)
    parser.add_argument("--branch_layer_num", type=int, default=2)
    parser.add_argument("--enable_slicing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_tiling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask_add", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wo_text", action="store_true")
    parser.add_argument("--add_first", action="store_true")
    parser.add_argument("--noised_image_dropout", type=float, default=0.0)

    parser.add_argument("--beta_dpo", type=float, default=10.0)
    parser.add_argument("--lose_gap_weight", type=float, default=0.25)
    parser.add_argument("--lose_gap_clip_tau", type=float, default=1.0)
    parser.add_argument("--winner_abs_reg_weight", type=float, default=0.05)
    parser.add_argument("--winner_gap_reg_weight", type=float, default=1.0)
    parser.add_argument("--winner_gap_reg_margin", type=float, default=0.0)
    parser.add_argument("--gap_eps", type=float, default=1e-6)
    parser.add_argument("--boundary_mode", choices=["outer"], default="outer")
    parser.add_argument("--mask_weight", type=float, default=1.0)
    parser.add_argument("--boundary_weight", type=float, default=0.75)
    parser.add_argument("--outside_weight", type=float, default=0.05)

    return parser.parse_args()


def setup_videopainter_imports(videopainter_root: str):
    root = Path(videopainter_root).resolve()
    train_dir = root / "train"
    diffusers_src = root / "diffusers" / "src"
    for p in (str(diffusers_src), str(train_dir), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # VideoPainter vendors a Diffusers snapshot that still imports this
    # Transformers constant. Newer Transformers releases removed the re-export.
    try:
        import transformers.utils as transformers_utils

        if not hasattr(transformers_utils, "FLAX_WEIGHTS_NAME"):
            transformers_utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
    except Exception:
        pass

    train_file = train_dir / "train_cogvideox_inpainting_i2v_video.py"
    if not train_file.exists():
        raise FileNotFoundError(f"VideoPainter train file missing: {train_file}")

    spec = importlib.util.spec_from_file_location("vp_train_cogvideox", train_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import VideoPainter train file: {train_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def list_frame_files(frame_dir: str) -> List[Path]:
    path = Path(frame_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Frame directory missing: {frame_dir}")
    files = sorted(
        p for p in path.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    )
    if not files:
        raise FileNotFoundError(f"No image frames found in {frame_dir}")
    return files


def normalize_frame_count(files: List[Path], max_frames: int) -> List[Path]:
    if len(files) > max_frames:
        files = files[:max_frames]
    # CogVideoX video length convention follows upstream VideoPainter:
    # after trimming, F should be 4k + 1. For D3's 16 frames this becomes 13.
    remainder = (3 + (len(files) % 4)) % 4
    if remainder:
        files = files[:-remainder]
    if not files:
        raise ValueError("No frames remain after CogVideoX 4k+1 normalization")
    return files


def load_rgb_frames(frame_dir: str, height: int, width: int, max_frames: int) -> torch.Tensor:
    files = normalize_frame_count(list_frame_files(frame_dir), max_frames)
    frames = []
    for p in files:
        img = Image.open(p).convert("RGB").resize((width, height), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        frames.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(frames, dim=0).contiguous()


def load_mask_frames(mask_dir: str, height: int, width: int, max_frames: int) -> torch.Tensor:
    files = normalize_frame_count(list_frame_files(mask_dir), max_frames)
    frames = []
    for p in files:
        img = Image.open(p).convert("L").resize((width, height), Image.NEAREST)
        # Current D3 manifest convention: 255 = inpaint / hole, 0 = keep.
        arr = (np.asarray(img, dtype=np.float32) > 127.0).astype(np.float32)
        frames.append(torch.from_numpy(arr)[None, ...])
    return torch.stack(frames, dim=0).contiguous()


class VideoPainterPairDataset(Dataset):
    """Reads D3 frame-directory winner / loser / mask pairs."""

    def __init__(self, manifest: str, height: int, width: int, num_frames: int, limit: int = 0):
        self.manifest = manifest
        self.height = height
        self.width = width
        self.num_frames = num_frames
        rows = []
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("status", "OK") != "OK":
                    continue
                rows.append(obj)
                if limit and len(rows) >= limit:
                    break
        if not rows:
            raise ValueError(f"No usable rows in manifest: {manifest}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        winner = load_rgb_frames(row["win_video_path"], self.height, self.width, self.num_frames)
        loser_path = row.get("final_loser_video_path") or row.get("comp_loser_video_path") or row["raw_loser_video_path"]
        loser = load_rgb_frames(loser_path, self.height, self.width, self.num_frames)
        mask = load_mask_frames(row["mask_path"], self.height, self.width, self.num_frames)

        f = min(winner.shape[0], loser.shape[0], mask.shape[0])
        winner = winner[:f]
        loser = loser[:f]
        mask = mask[:f]
        conditioning = winner * (1.0 - mask)
        if f > 0:
            # Match VideoPainter's first_frame_gt conditioning behavior.
            conditioning[0] = winner[0]
            mask[0] = torch.zeros_like(mask[0])

        return {
            "winner": winner,
            "loser": loser,
            "conditioning": conditioning,
            "mask": mask,
            "prompt": row.get("prompt") or "",
            "sample_id": row.get("sample_id") or f"row_{idx}",
        }


def collate_pairs(examples: List[Dict[str, object]]) -> VideoPainterBatch:
    min_f = min(x["winner"].shape[0] for x in examples)  # type: ignore[index]
    winner = torch.stack([x["winner"][:min_f] for x in examples])  # type: ignore[index]
    loser = torch.stack([x["loser"][:min_f] for x in examples])  # type: ignore[index]
    conditioning = torch.stack([x["conditioning"][:min_f] for x in examples])  # type: ignore[index]
    mask = torch.stack([x["mask"][:min_f] for x in examples])  # type: ignore[index]
    prompts = [str(x["prompt"]) for x in examples]
    sample_ids = [str(x["sample_id"]) for x in examples]
    return VideoPainterBatch(winner, loser, conditioning, mask, prompts, sample_ids)


def get_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return torch.float32


def resolve_branch_dir(path: str) -> Tuple[str, Optional[str]]:
    p = Path(path)
    if (p / "branch").is_dir():
        return str(p), "branch"
    return str(p), None


def load_branch(vp, transformer, checkpoint: str, args: argparse.Namespace, trainable: bool):
    root, subfolder = resolve_branch_dir(checkpoint)
    if Path(root).exists():
        if subfolder is None:
            branch = vp.CogvideoXBranchModel.from_pretrained(root)
        else:
            branch = vp.CogvideoXBranchModel.from_pretrained(root, subfolder=subfolder)
    else:
        raise FileNotFoundError(f"Branch checkpoint missing: {checkpoint}")
    branch.requires_grad_(trainable)
    branch.train(trainable)
    return branch


def make_region_weight_map(
    mask: torch.Tensor,
    mask_weight: float,
    boundary_weight: float,
    outside_weight: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # mask: [B, F, 1, H, W], 1 = hole.
    b, f, c, h, w = mask.shape
    flat = mask.reshape(b * f, c, h, w)
    dilated = F.max_pool2d(flat, kernel_size=3, stride=1, padding=1)
    boundary_outer = (dilated - flat).clamp(0, 1).reshape(b, f, c, h, w)
    outside = ((1.0 - mask) * (1.0 - boundary_outer)).clamp(0, 1)
    weight_map = mask_weight * mask + boundary_weight * boundary_outer + outside_weight * outside
    stats = {
        "mask_area_ratio": mask.mean(),
        "boundary_area_ratio": boundary_outer.mean(),
        "outside_area_ratio": outside.mean(),
        "region_weight_sum": weight_map.sum(dim=(1, 2, 3, 4)).mean(),
    }
    return weight_map, stats


def region_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    scheduler_weights: torch.Tensor,
    region_weight_map: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    while scheduler_weights.ndim < pred.ndim:
        scheduler_weights = scheduler_weights.unsqueeze(-1)
    per = scheduler_weights * (pred - target).pow(2)
    region = region_weight_map.to(dtype=per.dtype)
    numerator = (per * region).flatten(1).sum(dim=1)
    denominator = region.expand_as(per).flatten(1).sum(dim=1).clamp_min(eps)
    return numerator / denominator


class VideoPainterDPOTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.vp = setup_videopainter_imports(args.videopainter_root)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_dtype = get_dtype(args.mixed_precision)
        if self.device.type != "cuda":
            self.weight_dtype = torch.float32

        if args.pretrained_model_name_or_path is None:
            args.pretrained_model_name_or_path = str(Path(args.videopainter_root) / "ckpt" / "CogVideoX-5b-I2V")

        self._load_models()

    def _load_models(self) -> None:
        vp = self.vp
        args = self.args

        self.tokenizer = vp.AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=None
        )
        self.text_encoder = vp.T5EncoderModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=None
        )
        load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
        self.transformer = vp.CogVideoXTransformer3DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=load_dtype,
            revision=None,
            variant=None,
        )
        self.vae = vp.AutoencoderKLCogVideoX.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=None, variant=None
        )
        self.scheduler = vp.CogVideoXDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )

        self.policy_branch = load_branch(self.vp, self.transformer, args.policy_checkpoint, args, trainable=True)
        self.reference_branch = load_branch(self.vp, self.transformer, args.reference_checkpoint, args, trainable=False)

        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.reference_branch.requires_grad_(False)
        self.reference_branch.eval()

        if args.enable_slicing:
            self.vae.enable_slicing()
        if args.enable_tiling:
            self.vae.enable_tiling()
        if args.gradient_checkpointing and hasattr(self.policy_branch, "enable_gradient_checkpointing"):
            self.policy_branch.enable_gradient_checkpointing()

        for model in (self.text_encoder, self.transformer, self.vae, self.policy_branch, self.reference_branch):
            model.to(self.device, dtype=self.weight_dtype)

        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device, dtype=torch.float32)

    def encode_latents(self, video: torch.Tensor) -> torch.Tensor:
        video = video.to(self.device, dtype=self.weight_dtype)
        latents = self.vae.encode(video.permute(0, 2, 1, 3, 4)).latent_dist.sample()
        latents = latents.permute(0, 2, 1, 3, 4) * self.vae.config.scaling_factor
        return latents.to(memory_format=torch.contiguous_format, dtype=self.weight_dtype)

    def prepare_conditioning(self, conditioning: torch.Tensor, mask: torch.Tensor, target_frames: int) -> Tuple[torch.Tensor, torch.Tensor]:
        args = self.args
        vae_scale_factor_spatial = 2 ** (len(self.vae.config.block_out_channels) - 1)
        vae_scale_factor_temporal = self.vae.config.temporal_compression_ratio

        conditioning_latents = self.encode_latents(conditioning)
        masks = mask.to(self.device, dtype=self.weight_dtype).permute(0, 2, 1, 3, 4)
        masks = F.interpolate(
            masks,
            size=(
                (masks.shape[-3] - 1) // vae_scale_factor_temporal + 1,
                args.height // vae_scale_factor_spatial,
                args.width // vae_scale_factor_spatial,
            ),
            mode="nearest",
        ).permute(0, 2, 1, 3, 4).to(dtype=self.weight_dtype)
        conditioning_latents = torch.cat([conditioning_latents, masks], dim=-3)
        return conditioning_latents, masks

    def prepare_image_latents(self, conditioning: torch.Tensor, latent_frames: int) -> torch.Tensor:
        images = conditioning[:, :1].permute(0, 2, 1, 3, 4).to(self.device, dtype=self.weight_dtype)
        sigma = torch.normal(mean=-3.0, std=0.5, size=(images.size(0),), device=self.device, dtype=self.weight_dtype)
        sigma = torch.exp(sigma)
        noisy_images = images + torch.randn_like(images) * sigma[:, None, None, None, None]
        image_latents = self.vae.encode(noisy_images).latent_dist.sample()
        image_latents = image_latents.permute(0, 2, 1, 3, 4) * self.vae.config.scaling_factor
        padding_shape = (image_latents.shape[0], latent_frames - 1, *image_latents.shape[2:])
        if padding_shape[1] > 0:
            image_latents = torch.cat([image_latents, image_latents.new_zeros(padding_shape)], dim=1)
        return image_latents.to(memory_format=torch.contiguous_format, dtype=self.weight_dtype)

    def prompt_embeds(self, prompts: List[str]) -> torch.Tensor:
        return self.vp.compute_prompt_embeddings(
            self.tokenizer,
            self.text_encoder,
            prompts,
            self.args.prompt_max_sequence_length,
            self.device,
            self.weight_dtype,
            requires_grad=False,
            token_ids=None,
        )

    def branch_forward(
        self,
        branch,
        model_input: torch.Tensor,
        conditioning_latents: torch.Tensor,
        image_latents: torch.Tensor,
        masks: torch.Tensor,
        prompt_embeds: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        args = self.args
        noisy_video_latents = self.scheduler.add_noise(model_input, noise, timesteps)
        noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=2)
        model_config = self.transformer.config
        image_rotary_emb = (
            self.vp.prepare_rotary_positional_embeddings(
                height=args.height,
                width=args.width,
                num_frames=model_input.shape[1],
                vae_scale_factor_spatial=2 ** (len(self.vae.config.block_out_channels) - 1),
                patch_size=model_config.patch_size,
                attention_head_dim=model_config.attention_head_dim,
                device=self.device,
            )
            if model_config.use_rotary_positional_embeddings
            else None
        )
        branch_block_samples = branch(
            hidden_states=noisy_video_latents,
            encoder_hidden_states=prompt_embeds,
            branch_cond=conditioning_latents,
            timestep=timesteps,
            image_rotary_emb=image_rotary_emb,
            mask_add=args.mask_add,
            wo_text=args.wo_text,
            return_dict=False,
        )[0]
        branch_block_samples = [x.to(dtype=self.weight_dtype) for x in branch_block_samples]
        model_output = self.transformer(
            hidden_states=noisy_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
            branch_block_samples=branch_block_samples,
            branch_block_masks=masks if args.mask_add else None,
            add_first=args.add_first,
        )[0]
        return self.scheduler.get_velocity(model_output, noisy_video_latents, timesteps)

    def compute_losses(self, batch: VideoPainterBatch) -> Tuple[torch.Tensor, Dict[str, float]]:
        args = self.args
        winner = batch.winner.to(self.device)
        loser = batch.loser.to(self.device)
        conditioning = batch.conditioning.to(self.device)
        mask = batch.mask.to(self.device)

        with torch.no_grad():
            prompt_embeds = self.prompt_embeds(batch.prompts)
            conditioning_latents, latent_mask = self.prepare_conditioning(conditioning, mask, winner.shape[1])
            winner_latents = self.encode_latents(winner)
            loser_latents = self.encode_latents(loser)
            image_latents = self.prepare_image_latents(conditioning, winner_latents.shape[1])
            noise = torch.randn_like(winner_latents).to(dtype=self.weight_dtype)
            timesteps = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (winner_latents.shape[0],),
                device=self.device,
            ).long()
            scheduler_weights = 1 / (1 - self.alphas_cumprod[timesteps])
            region_weight_map, region_stats = make_region_weight_map(
                latent_mask.float(),
                args.mask_weight,
                args.boundary_weight,
                args.outside_weight,
            )

        policy_w = self.branch_forward(
            self.policy_branch, winner_latents, conditioning_latents, image_latents, latent_mask,
            prompt_embeds, noise, timesteps
        )
        policy_l = self.branch_forward(
            self.policy_branch, loser_latents, conditioning_latents, image_latents, latent_mask,
            prompt_embeds, noise, timesteps
        )
        with torch.no_grad():
            ref_w = self.branch_forward(
                self.reference_branch, winner_latents, conditioning_latents, image_latents, latent_mask,
                prompt_embeds, noise, timesteps
            )
            ref_l = self.branch_forward(
                self.reference_branch, loser_latents, conditioning_latents, image_latents, latent_mask,
                prompt_embeds, noise, timesteps
            )

        m_w = region_weighted_mse(policy_w, winner_latents, scheduler_weights, region_weight_map, args.gap_eps)
        m_l = region_weighted_mse(policy_l, loser_latents, scheduler_weights, region_weight_map, args.gap_eps)
        m_w_ref = region_weighted_mse(ref_w, winner_latents, scheduler_weights, region_weight_map, args.gap_eps)
        m_l_ref = region_weighted_mse(ref_l, loser_latents, scheduler_weights, region_weight_map, args.gap_eps)

        raw_win_gap = m_w - m_w_ref
        raw_lose_gap = m_l - m_l_ref
        norm_win_gap = torch.log((m_w + args.gap_eps) / (m_w_ref + args.gap_eps))
        norm_lose_gap = torch.log((m_l + args.gap_eps) / (m_l_ref + args.gap_eps))
        norm_lose_gap_clipped = torch.clamp(norm_lose_gap, max=args.lose_gap_clip_tau)

        inside = -0.5 * args.beta_dpo * (norm_win_gap - args.lose_gap_weight * norm_lose_gap_clipped)
        dpo_loss = -F.logsigmoid(inside).mean()
        winner_abs_reg = m_w.mean()
        winner_gap_reg = F.relu(norm_win_gap - args.winner_gap_reg_margin).mean()
        loss = (
            dpo_loss
            + args.winner_abs_reg_weight * winner_abs_reg
            + args.winner_gap_reg_weight * winner_gap_reg
        )

        implicit_acc = (inside > 0).float().mean()
        loser_dominant_ratio = (m_l > m_w).float().mean()
        diag = {
            "loss": float(loss.detach().float().cpu()),
            "dpo_loss": float(dpo_loss.detach().float().cpu()),
            "implicit_acc": float(implicit_acc.detach().float().cpu()),
            "m_w": float(m_w.mean().detach().float().cpu()),
            "m_l": float(m_l.mean().detach().float().cpu()),
            "m_w_ref": float(m_w_ref.mean().detach().float().cpu()),
            "m_l_ref": float(m_l_ref.mean().detach().float().cpu()),
            "raw_win_gap": float(raw_win_gap.mean().detach().float().cpu()),
            "raw_lose_gap": float(raw_lose_gap.mean().detach().float().cpu()),
            "norm_win_gap": float(norm_win_gap.mean().detach().float().cpu()),
            "norm_lose_gap": float(norm_lose_gap.mean().detach().float().cpu()),
            "norm_lose_gap_clipped": float(norm_lose_gap_clipped.mean().detach().float().cpu()),
            "winner_abs_reg": float(winner_abs_reg.detach().float().cpu()),
            "winner_gap_reg": float(winner_gap_reg.detach().float().cpu()),
            "mse_w_over_ref_mse_w": float((m_w / (m_w_ref + args.gap_eps)).mean().detach().float().cpu()),
            "mse_l_over_ref_mse_l": float((m_l / (m_l_ref + args.gap_eps)).mean().detach().float().cpu()),
            "loser_dominant_ratio": float(loser_dominant_ratio.detach().float().cpu()),
            "mask_area_ratio": float(region_stats["mask_area_ratio"].detach().float().cpu()),
            "boundary_area_ratio": float(region_stats["boundary_area_ratio"].detach().float().cpu()),
            "outside_area_ratio": float(region_stats["outside_area_ratio"].detach().float().cpu()),
            "region_weight_sum": float(region_stats["region_weight_sum"].detach().float().cpu()),
        }
        return loss, diag

    def save_branch_checkpoint(self, step: int, optimizer: torch.optim.Optimizer) -> None:
        output_dir = Path(self.args.output_dir)
        ckpt = output_dir / f"checkpoint-{step}"
        ckpt.mkdir(parents=True, exist_ok=True)
        self.policy_branch.save_pretrained(ckpt / "branch", safe_serialization=True, max_shard_size="5GB")
        torch.save({"step": step, "optimizer": optimizer.state_dict()}, ckpt / "trainer_state.pt")

        checkpoints = sorted(
            [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
            key=lambda p: int(p.name.split("-")[-1]),
        )
        excess = len(checkpoints) - self.args.checkpoints_total_limit
        for p in checkpoints[: max(0, excess)]:
            shutil.rmtree(p)

    def save_last_weights(self, step: int) -> None:
        last = Path(self.args.output_dir) / "last_weights"
        if last.exists():
            shutil.rmtree(last)
        last.mkdir(parents=True, exist_ok=True)
        self.policy_branch.save_pretrained(last / "branch", safe_serialization=True, max_shard_size="5GB")
        with open(last / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump({"step": step, "adapter_type": "direct_diff_dpo"}, f, indent=2)


def ensure_diag_header(path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        with p.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=DIAG_COLUMNS).writeheader()


def append_diag(path: str, step: int, diag: Dict[str, float], grad_norm: Optional[float], lr: float, args: argparse.Namespace) -> None:
    row = {k: "" for k in DIAG_COLUMNS}
    row.update({k: diag.get(k, "") for k in DIAG_COLUMNS})
    row.update(
        {
            "step": step,
            "grad_norm": "" if grad_norm is None else grad_norm,
            "lr": lr,
            "boundary_mode": args.boundary_mode,
            "mask_weight": args.mask_weight,
            "boundary_weight": args.boundary_weight,
            "outside_weight": args.outside_weight,
        }
    )
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=DIAG_COLUMNS).writerow(row)


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        total += float(p.grad.detach().data.norm(2).cpu()) ** 2
    return math.sqrt(total)


def run(args: argparse.Namespace) -> None:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    dataset = VideoPainterPairDataset(
        args.pair_manifest,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        limit=args.limit_train_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=not args.preflight_only,
        num_workers=args.num_workers,
        collate_fn=collate_pairs,
    )
    trainer = VideoPainterDPOTrainer(args)
    optimizer = torch.optim.AdamW(trainer.policy_branch.parameters(), lr=args.learning_rate)

    ensure_diag_header(args.dpo_diag_csv)

    iterator = iter(loader) if args.preflight_only else itertools.cycle(loader)
    progress = tqdm(range(1, (1 if args.preflight_only else args.max_train_steps) + 1), desc="Exp14 VideoPainter DPO")
    for step in progress:
        batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        loss, diag = trainer.compute_losses(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite DPO loss at step {step}: {loss}")
        loss.backward()
        gnorm = grad_norm(trainer.policy_branch.parameters())
        torch.nn.utils.clip_grad_norm_(trainer.policy_branch.parameters(), args.max_grad_norm)
        if not args.preflight_only:
            optimizer.step()
        lr = optimizer.param_groups[0]["lr"]

        if args.preflight_only or step % args.dpo_diag_log_every == 0 or step == 1:
            append_diag(args.dpo_diag_csv, step, diag, gnorm, lr, args)
        progress.set_postfix(loss=diag["loss"], dpo=diag["dpo_loss"], acc=diag["implicit_acc"])

        if args.preflight_only:
            report = {
                "status": "passed",
                "step": step,
                "diag": diag,
                "grad_norm": gnorm,
                "reference_has_grad": any(p.grad is not None for p in trainer.reference_branch.parameters()),
            }
            if report["reference_has_grad"]:
                raise RuntimeError("Reference branch received gradients")
            with open(Path(args.output_dir) / "preflight_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            return

        if step % args.checkpointing_steps == 0:
            trainer.save_branch_checkpoint(step, optimizer)

    trainer.save_last_weights(args.max_train_steps)


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
