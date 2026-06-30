from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file


@dataclass
class VoidPaths:
    repo: Path
    base_model: Path
    void_weights: Path
    transformer_path: Path


def add_void_repo(repo: Path) -> None:
    repo_s = str(repo)
    if repo_s not in sys.path:
        sys.path.insert(0, repo_s)


def read_prompt(prompt_json: str | Path) -> str:
    try:
        data = json.loads(Path(prompt_json).read_text())
        return data.get("bg") or data.get("prompt") or data.get("text") or "Remove the target object while preserving the background."
    except Exception:
        return "Remove the target object while preserving the background."


def _read_video_rgb(path: str | Path, frames: int, size: Tuple[int, int], nearest: bool = False) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    got = []
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    width, height = size
    while len(got) < frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=interp)
        if nearest:
            got.append(frame)
        else:
            got.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not got:
        raise RuntimeError(f"no frames decoded: {path}")
    while len(got) < frames:
        got.append(got[-1].copy())
    return np.stack(got[:frames], axis=0)


def video_tensor(path: str | Path, frames: int, size: Tuple[int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    arr = _read_video_rgb(path, frames, size, nearest=False).astype(np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).contiguous()
    ten = ten * 2.0 - 1.0
    return ten.to(device=device, dtype=dtype)


def quadmask_training_tensor(path: str | Path, frames: int, size: Tuple[int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    arr = _read_video_rgb(path, frames, size, nearest=True)
    gray = arr[..., 0].astype(np.float32)
    inv = (255.0 - gray) / 255.0
    mask = torch.from_numpy(inv).unsqueeze(1).unsqueeze(0).contiguous().to(device=device, dtype=dtype)
    mask = torch.where(mask <= 31 / 255.0, torch.zeros_like(mask), mask)
    mask = torch.where((mask > 31 / 255.0) & (mask <= 95 / 255.0), torch.full_like(mask, 63 / 255.0), mask)
    mask = torch.where((mask > 95 / 255.0) & (mask <= 191 / 255.0), torch.full_like(mask, 127 / 255.0), mask)
    mask = torch.where(mask > 191 / 255.0, torch.ones_like(mask), mask)
    return mask


def load_micro_row(manifest: str | Path, index: int = 0) -> Dict[str, Any]:
    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"empty manifest: {manifest}")
    return rows[index % len(rows)]


def load_components(paths: VoidPaths, device: torch.device, dtype: torch.dtype, load_transformer: bool = True) -> Dict[str, Any]:
    add_void_repo(paths.repo)
    from diffusers import DDPMScheduler
    from transformers import T5EncoderModel, T5Tokenizer
    from videox_fun.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel

    scheduler = DDPMScheduler.from_pretrained(str(paths.base_model), subfolder="scheduler")
    tokenizer = T5Tokenizer.from_pretrained(str(paths.base_model), subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(str(paths.base_model), subfolder="text_encoder", torch_dtype=dtype).to(device)
    text_encoder.eval().requires_grad_(False)
    vae = AutoencoderKLCogVideoX.from_pretrained(str(paths.base_model), subfolder="vae").to(device=device, dtype=dtype)
    vae.eval().requires_grad_(False)
    transformer = None
    if load_transformer:
        transformer = CogVideoXTransformer3DModel.from_pretrained(str(paths.base_model), subfolder="transformer", use_vae_mask=True).to(device=device, dtype=dtype)
        state = load_file(str(paths.transformer_path), device="cpu")
        missing, unexpected = transformer.load_state_dict(state, strict=False)
        transformer.eval()
    else:
        missing, unexpected = [], []
    return {
        "scheduler": scheduler,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "transformer": transformer,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def encode_vae(vae: Any, pixel_values: torch.Tensor, mini_batch: int = 1) -> torch.Tensor:
    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
    outs = []
    for i in range(0, pixel_values.shape[0], mini_batch):
        enc = vae.encode(pixel_values[i : i + mini_batch])[0]
        outs.append(enc.sample())
    return torch.cat(outs, dim=0)


def make_target_pack(
    row: Dict[str, Any],
    components: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    frames: int = 17,
    size: Tuple[int, int] = (672, 384),
    seed: int = 1234,
    timestep: int = 500,
    target_key: str = "winner_path",
) -> Dict[str, Any]:
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]
    vae = components["vae"]
    scheduler = components["scheduler"]
    prompt = read_prompt(row.get("prompt_json", ""))
    target_path = row[target_key] if target_key in row else row.get("rgb_removed_path") or row.get("winner_path")
    condition = video_tensor(row["condition_path"], frames, size, device, dtype)
    target_video = video_tensor(target_path, frames, size, device, dtype)
    dataset_mask = quadmask_training_tensor(row["quadmask_0_path"], frames, size, device, dtype)
    train_loop_mask = 1.0 - dataset_mask
    with torch.no_grad():
        latents = encode_vae(vae, target_video) * vae.config.scaling_factor
        mask_for_vae = train_loop_mask.repeat(1, 1, 3, 1, 1).to(dtype=target_video.dtype, device=device)
        mask_latents = encode_vae(vae, mask_for_vae)
        condition_latents = encode_vae(vae, condition)
        inpaint_latents = torch.cat([mask_latents, condition_latents], dim=1) * vae.config.scaling_factor
        inpaint_latents = rearrange(inpaint_latents, "b c f h w -> b f c h w")
        latents = rearrange(latents, "b c f h w -> b f c h w")
        prompt_ids = tokenizer([prompt], max_length=226, padding="max_length", add_special_tokens=True, truncation=True, return_tensors="pt")
        prompt_embeds = text_encoder(prompt_ids.input_ids.to(device), return_dict=False)[0]
    gen = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(latents.size(), device=device, generator=gen, dtype=dtype)
    timesteps = torch.tensor([timestep], device=device, dtype=torch.long)
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        target = noise
    elif prediction_type == "v_prediction":
        target = scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"unknown prediction_type {prediction_type}")
    return {
        "sample_id": row.get("sample_id"),
        "prompt": prompt,
        "latents": latents,
        "noisy_latents": noisy_latents,
        "target": target,
        "inpaint_latents": inpaint_latents,
        "prompt_embeds": prompt_embeds,
        "timesteps": timesteps,
        "noise": noise,
        "prediction_type": prediction_type,
        "condition_shape": tuple(condition.shape),
        "target_video_shape": tuple(target_video.shape),
        "mask_shape": tuple(train_loop_mask.shape),
        "latent_shape": tuple(latents.shape),
        "inpaint_shape": tuple(inpaint_latents.shape),
        "target_path": str(target_path),
    }


def rotary_embeddings(transformer: Any, vae: Any, height: int, width: int, num_frames: int, device: torch.device):
    add_void_repo(Path("/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/VOID/Netflix_void-model"))
    from videox_fun.pipeline.pipeline_cogvideox_fun_inpaint import get_3d_rotary_pos_embed, get_resize_crop_region_for_grid
    cfg = transformer.config
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
    p = cfg.patch_size
    p_t = cfg.patch_size_t
    grid_height = height // (vae_scale_factor_spatial * p)
    grid_width = width // (vae_scale_factor_spatial * p)
    base_size_height = cfg.sample_height // p
    base_size_width = cfg.sample_width // p
    if p_t is None:
        crops = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=cfg.attention_head_dim,
            crops_coords=crops,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
            use_real=True,
        )
    else:
        base_num_frames = (num_frames + p_t - 1) // p_t
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=cfg.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(base_size_height, base_size_width),
        )
    return freqs_cos.to(device=device), freqs_sin.to(device=device)


def sft_forward_loss(components: Dict[str, Any], pack: Dict[str, Any], height: int = 384, width: int = 672) -> Dict[str, Any]:
    transformer = components["transformer"]
    vae = components["vae"]
    if transformer is None:
        raise RuntimeError("transformer not loaded")
    image_rotary_emb = rotary_embeddings(transformer, vae, height, width, pack["latents"].shape[1], pack["latents"].device)
    with torch.no_grad():
        pred = transformer(
            hidden_states=pack["noisy_latents"],
            encoder_hidden_states=pack["prompt_embeds"].to(pack["latents"].device),
            timestep=pack["timesteps"],
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
            inpaint_latents=pack["inpaint_latents"],
        )[0]
        loss = F.mse_loss(pred.float(), pack["target"].float(), reduction="mean")
    return {
        "loss": float(loss.detach().cpu()),
        "noise_pred_shape": tuple(pred.shape),
        "target_shape": tuple(pack["target"].shape),
        "finite": bool(torch.isfinite(loss).item()),
    }


def load_transformer_clone(paths: VoidPaths, device: torch.device, dtype: torch.dtype, trainable_filter: Optional[str] = None, gradient_checkpointing: bool = False):
    add_void_repo(paths.repo)
    from videox_fun.models import CogVideoXTransformer3DModel
    model = CogVideoXTransformer3DModel.from_pretrained(str(paths.base_model), subfolder="transformer", use_vae_mask=True).to(device=device, dtype=dtype)
    state = load_file(str(paths.transformer_path), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if gradient_checkpointing:
        try:
            if hasattr(model, "enable_gradient_checkpointing"):
                model.enable_gradient_checkpointing()
            elif hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
        except TypeError:
            for module in model.modules():
                if hasattr(module, "gradient_checkpointing"):
                    module.gradient_checkpointing = True
    if trainable_filter is None:
        model.requires_grad_(False)
    else:
        model.requires_grad_(False)
        for name, param in model.named_parameters():
            if trainable_filter in name:
                param.requires_grad_(True)
    return model, list(missing), list(unexpected)


def trainable_parameter_summary(model: Any) -> Dict[str, Any]:
    total = 0
    trainable = 0
    trainable_names = []
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
            if len(trainable_names) < 20:
                trainable_names.append(name)
    return {"total_parameters": total, "trainable_parameters": trainable, "trainable_names_head": trainable_names}


def forward_noise_pred(model: Any, components: Dict[str, Any], pack: Dict[str, Any], height: int = 384, width: int = 672) -> torch.Tensor:
    image_rotary_emb = rotary_embeddings(model, components["vae"], height, width, pack["latents"].shape[1], pack["latents"].device)
    return model(
        hidden_states=pack["noisy_latents"],
        encoder_hidden_states=pack["prompt_embeds"].to(pack["latents"].device),
        timestep=pack["timesteps"],
        image_rotary_emb=image_rotary_emb,
        return_dict=False,
        inpaint_latents=pack["inpaint_latents"],
    )[0]


def latent_region_weights(row: Dict[str, Any], frames: int, size: Tuple[int, int], latent_shape: Iterable[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    arr = _read_video_rgb(row["quadmask_0_path"], frames, size, nearest=True)[..., 0]
    q = torch.from_numpy(arr.astype(np.float32)).to(device=device)
    obj = (q <= 31).float()
    affected = ((q > 31) & (q <= 191)).float()
    outside = (q > 191).float()
    boundary_frames = []
    kernel = np.ones((9, 9), np.uint8)
    for fr in obj.detach().cpu().numpy().astype(np.uint8):
        dil = cv2.dilate(fr, kernel, iterations=1)
        ero = cv2.erode(fr, kernel, iterations=1)
        boundary_frames.append(np.clip(dil - ero, 0, 1))
    boundary = torch.from_numpy(np.stack(boundary_frames).astype(np.float32)).to(device=device)
    weight = obj * 1.0 + affected * 0.75 + boundary * 0.75 + outside * 0.05
    weight = weight.clamp_min(0.05).unsqueeze(0).unsqueeze(0).to(dtype=dtype)
    _, latent_f, _, latent_h, latent_w = tuple(latent_shape)
    weight = F.interpolate(weight, size=(latent_f, latent_h, latent_w), mode="trilinear", align_corners=False)
    weight = rearrange(weight, "b c f h w -> b f c h w")
    return weight


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return ((pred.float() - target.float()) ** 2 * weight.float()).sum() / (weight.float().sum() * pred.shape[2]).clamp_min(1.0)
