#!/usr/bin/env python3
"""DAVIS flow-cache and context-window helpers for Exp19 inference."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffueraser.pipeline_diffueraser import get_frames_context_swap  # noqa: E402

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from export_propainter_completed_flow import compute_completed_flow  # noqa: E402
from flow_confidence import flow_to_color, save_flow_npy  # noqa: E402


@dataclass
class DavisFlowCacheResult:
    video: str
    status: str
    sample_root: Path
    forward_flow_path: Path
    backward_flow_path: Path
    confidence_path: Path
    fb_error_path: Path
    flow_conf_mean: float
    valid_flow_ratio: float
    mean_flow_magnitude: float
    forward_backward_error: float


def context_sequence(video_length: int, num_frames: int, num_inference_steps: int) -> list[list[int]]:
    overlap = int(num_frames) // 4
    context_list, context_list_swap = get_frames_context_swap(
        total_frames=int(video_length),
        overlap=overlap,
        num_frames_per_clip=int(num_frames),
    )
    sequence: list[list[int]] = []
    for step in range(int(num_inference_steps)):
        chosen = context_list_swap if step % 2 == 1 else context_list
        sequence.extend([list(map(int, context)) for context in chosen])
    return sequence


def masks_to_hole_tensor(masks: Sequence[np.ndarray], size: tuple[int, int]) -> torch.Tensor:
    width, height = size
    out = []
    for mask in masks:
        arr = np.asarray(mask)
        if arr.shape[:2] != (height, width):
            arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_NEAREST)
        out.append(torch.from_numpy((arr > 0).astype(np.float32))[None])
    return torch.stack(out, dim=0)


def outer_boundary(hole_mask: torch.Tensor, pixels: int = 1) -> torch.Tensor:
    kernel = pixels * 2 + 1
    dil = F.max_pool2d(hole_mask, kernel_size=kernel, stride=1, padding=pixels)
    return (dil - hole_mask).clamp(0.0, 1.0)


def flow_condition_from_cache(
    forward_flow_path: str | Path,
    backward_flow_path: str | Path,
    confidence_path: str | Path,
    masks: Sequence[np.ndarray],
    size: tuple[int, int],
    video_length: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    width, height = size
    fwd = torch.from_numpy(np.load(forward_flow_path).astype(np.float32))
    bwd = torch.from_numpy(np.load(backward_flow_path).astype(np.float32))
    conf = torch.from_numpy(np.load(confidence_path).astype(np.float32))
    if conf.ndim == 3:
        conf = conf[:, None]
    if fwd.shape[0] < video_length - 1 or bwd.shape[0] < video_length - 1:
        raise ValueError(f"flow cache too short: fwd={tuple(fwd.shape)} bwd={tuple(bwd.shape)} video_length={video_length}")
    fwd = fwd[: video_length - 1]
    bwd = bwd[: video_length - 1]
    conf = conf[: video_length - 1]
    h0, w0 = fwd.shape[-2:]
    if (h0, w0) != (height, width):
        scale_x = float(width) / max(float(w0), 1.0)
        scale_y = float(height) / max(float(h0), 1.0)
        fwd = F.interpolate(fwd, size=(height, width), mode="bilinear", align_corners=False)
        bwd = F.interpolate(bwd, size=(height, width), mode="bilinear", align_corners=False)
        fwd[:, 0] *= scale_x
        bwd[:, 0] *= scale_x
        fwd[:, 1] *= scale_y
        bwd[:, 1] *= scale_y
        conf = F.interpolate(conf, size=(height, width), mode="bilinear", align_corners=False)
    zero_flow = torch.zeros(1, 2, height, width, dtype=fwd.dtype)
    zero_conf = torch.zeros(1, 1, height, width, dtype=conf.dtype)
    fwd_frame = torch.cat([fwd, zero_flow], dim=0)
    bwd_frame = torch.cat([zero_flow, bwd], dim=0)
    conf_frame = torch.cat([conf, zero_conf], dim=0).clamp(0.0, 1.0)
    hole = masks_to_hole_tensor(masks[:video_length], (width, height)).float()
    boundary = outer_boundary(hole)
    f_norm = fwd_frame.clone()
    b_norm = bwd_frame.clone()
    f_norm[:, 0] /= max(float(width), 1.0)
    b_norm[:, 0] /= max(float(width), 1.0)
    f_norm[:, 1] /= max(float(height), 1.0)
    b_norm[:, 1] /= max(float(height), 1.0)
    cond = torch.cat([f_norm, b_norm, conf_frame, hole, boundary], dim=1).unsqueeze(0)
    gate = conf_frame * torch.clamp(hole + 0.75 * boundary, 0.0, 1.0)
    stats = {
        "flow_conf_mean": float(conf_frame.mean()),
        "valid_flow_ratio": float((conf_frame > 0.05).float().mean()),
        "mean_flow_magnitude": float(torch.sqrt((fwd_frame.pow(2).sum(dim=1) + bwd_frame.pow(2).sum(dim=1)) * 0.5).mean()),
        "gate_mean": float(gate.mean()),
        "gate_p10": float(torch.quantile(gate.flatten(), 0.10)),
        "gate_p50": float(torch.quantile(gate.flatten(), 0.50)),
        "gate_p90": float(torch.quantile(gate.flatten(), 0.90)),
        "nonzero_gate_ratio": float((gate > 1e-6).float().mean()),
    }
    return cond, stats


def _to_pil_frames(frames: Sequence[np.ndarray], size: tuple[int, int]) -> list[Image.Image]:
    width, height = size
    out = []
    for frame in frames:
        arr = np.asarray(frame, dtype=np.uint8)
        if arr.shape[:2] != (height, width):
            arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)
        out.append(Image.fromarray(arr, mode="RGB"))
    return out


def _to_pil_masks(masks: Sequence[np.ndarray], size: tuple[int, int]) -> list[Image.Image]:
    width, height = size
    out = []
    for mask in masks:
        arr = np.asarray(mask)
        if arr.shape[:2] != (height, width):
            arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_NEAREST)
        out.append(Image.fromarray(((arr > 0).astype(np.uint8) * 255), mode="L"))
    return out


def ensure_davis_flow_cache(
    video_name: str,
    frames: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    cache_root: str | Path,
    propainter_flow_model,
    size: tuple[int, int],
    video_length: int,
    raft_iter: int = 20,
    fp16: bool = True,
    save_visuals: bool = True,
) -> DavisFlowCacheResult:
    cache_root = Path(cache_root)
    sample_root = cache_root / "samples" / video_name
    fwd_path = sample_root / "completed_forward_flow.npy"
    bwd_path = sample_root / "completed_backward_flow.npy"
    conf_path = sample_root / "flow_confidence.npy"
    fb_path = sample_root / "forward_backward_error.npy"
    meta_path = sample_root / "metadata.json"
    done = sample_root / ".done"
    if done.exists() and fwd_path.exists() and bwd_path.exists() and conf_path.exists() and fb_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return DavisFlowCacheResult(
            video=video_name,
            status="reused",
            sample_root=sample_root,
            forward_flow_path=fwd_path,
            backward_flow_path=bwd_path,
            confidence_path=conf_path,
            fb_error_path=fb_path,
            flow_conf_mean=float(meta.get("flow_conf_mean", np.nan)),
            valid_flow_ratio=float(meta.get("valid_flow_ratio", np.nan)),
            mean_flow_magnitude=float(meta.get("mean_flow_magnitude", np.nan)),
            forward_backward_error=float(meta.get("forward_backward_error", np.nan)),
        )

    sample_root.mkdir(parents=True, exist_ok=True)
    frame_pil = _to_pil_frames(frames[:video_length], size)
    mask_pil = _to_pil_masks(masks[:video_length], size)
    fwd, bwd, conf, fb_error = compute_completed_flow(propainter_flow_model, frame_pil, mask_pil, raft_iter, fp16)
    save_flow_npy(fwd_path, fwd.numpy())
    save_flow_npy(bwd_path, bwd.numpy())
    save_flow_npy(conf_path, conf.numpy())
    save_flow_npy(fb_path, fb_error.numpy())
    conf_np = conf.numpy()
    fmag = np.sqrt((fwd.numpy() ** 2).sum(axis=1))
    result = DavisFlowCacheResult(
        video=video_name,
        status="ok",
        sample_root=sample_root,
        forward_flow_path=fwd_path,
        backward_flow_path=bwd_path,
        confidence_path=conf_path,
        fb_error_path=fb_path,
        flow_conf_mean=float(conf_np.mean()),
        valid_flow_ratio=float((conf_np > 0.05).mean()),
        mean_flow_magnitude=float(fmag.mean()),
        forward_backward_error=float(fb_error.numpy().mean()),
    )
    meta_path.write_text(json.dumps(result.__dict__ | {"sample_root": str(sample_root), "forward_flow_path": str(fwd_path), "backward_flow_path": str(bwd_path), "confidence_path": str(conf_path), "fb_error_path": str(fb_path)}, indent=2), encoding="utf-8")
    done.write_text("ok\n", encoding="utf-8")
    if save_visuals and len(frame_pil) > 1:
        vis_root = sample_root / "visuals"
        vis_root.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(frame_pil[0]), mode="RGB").save(vis_root / "frame0_input.png")
        flow_rgb = flow_to_color(fwd[0].permute(1, 2, 0).numpy())
        Image.fromarray(flow_rgb.astype(np.uint8), mode="RGB").save(vis_root / "flow_forward_0.png")
        conf_rgb = np.repeat((conf[0, 0].numpy().clip(0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)
        Image.fromarray(conf_rgb, mode="RGB").save(vis_root / "confidence_0.png")
    return result

