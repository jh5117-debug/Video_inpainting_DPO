#!/usr/bin/env python3
"""Flow confidence helpers for Exp19.

The functions here are deliberately independent of DiffuEraser training code so
the flow cache can be audited before any DPO run starts.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def flow_warp_tensor(x: torch.Tensor, flow: torch.Tensor, mode: str = "bilinear") -> tuple[torch.Tensor, torch.Tensor]:
    """Warp ``x`` by pixel-displacement ``flow``.

    Args:
        x: Tensor ``[B,C,H,W]``.
        flow: Tensor ``[B,2,H,W]`` with ``dx, dy`` in pixel units.

    Returns:
        warped tensor and a float valid mask ``[B,1,H,W]``.
    """
    if x.ndim != 4 or flow.ndim != 4:
        raise ValueError(f"Expected x/flow rank 4, got {x.shape=} {flow.shape=}")
    b, _, h, w = x.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=x.device, dtype=x.dtype),
        torch.arange(w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    grid_x = xx[None] + flow[:, 0]
    grid_y = yy[None] + flow[:, 1]
    valid = (grid_x >= 0) & (grid_x <= w - 1) & (grid_y >= 0) & (grid_y <= h - 1)
    grid = torch.stack(
        [
            2.0 * grid_x / max(w - 1, 1) - 1.0,
            2.0 * grid_y / max(h - 1, 1) - 1.0,
        ],
        dim=-1,
    )
    warped = F.grid_sample(x, grid, mode=mode, padding_mode="zeros", align_corners=True)
    return warped, valid[:, None].to(dtype=x.dtype)


def forward_backward_confidence(
    forward: torch.Tensor,
    backward: torch.Tensor,
    tau_flow: float = 1.0,
    source_valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute forward-backward consistency confidence.

    ``forward`` and ``backward`` are ``[B,T-1,2,H,W]`` tensors. ``backward`` is
    the reverse adjacent flow returned by ProPainter/RAFT, aligned so
    ``backward[:, t]`` corresponds to ``t+1 -> t``.
    """
    if forward.shape != backward.shape:
        raise ValueError(f"Flow shape mismatch: {forward.shape} vs {backward.shape}")
    b, tm1, _, h, w = forward.shape
    f = forward.reshape(b * tm1, 2, h, w)
    br = backward.reshape(b * tm1, 2, h, w)
    warped_b, valid = flow_warp_tensor(br, f, mode="bilinear")
    fb_err = (f + warped_b).abs().sum(dim=1, keepdim=True)
    conf = torch.exp(-fb_err / float(tau_flow)).clamp(0.0, 1.0) * valid
    if source_valid is not None:
        sv = source_valid.reshape(b * tm1, 1, h, w).to(conf.dtype)
        warped_sv, _ = flow_warp_tensor(sv, f, mode="nearest")
        conf = conf * (warped_sv > 0.5).to(conf.dtype)
    return {
        "confidence": conf.reshape(b, tm1, 1, h, w),
        "fb_error": fb_err.reshape(b, tm1, 1, h, w),
        "valid": valid.reshape(b, tm1, 1, h, w),
    }


def pad_pairwise_to_frames(pairwise: torch.Tensor, last_value: float = 0.0) -> torch.Tensor:
    """Pad a ``[B,T-1,C,H,W]`` adjacent tensor to ``[B,T,C,H,W]``."""
    if pairwise.ndim != 5:
        raise ValueError(f"Expected [B,T-1,C,H,W], got {pairwise.shape}")
    b, _, c, h, w = pairwise.shape
    tail = torch.full((b, 1, c, h, w), float(last_value), dtype=pairwise.dtype, device=pairwise.device)
    return torch.cat([pairwise, tail], dim=1)


def resize_flow_pair(flow: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize flow and scale displacement units to the new spatial size."""
    if flow.ndim != 4:
        raise ValueError(f"Expected [N,2,H,W], got {flow.shape}")
    _, _, h, w = flow.shape
    new_h, new_w = size
    out = F.interpolate(flow, size=size, mode="bilinear", align_corners=False)
    out[:, 0] *= float(new_w) / max(float(w), 1.0)
    out[:, 1] *= float(new_h) / max(float(h), 1.0)
    return out


def flow_to_color(flow: np.ndarray) -> np.ndarray:
    """Small HSV flow visualizer returning RGB uint8."""
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"Expected HWC flow with 2 channels, got {flow.shape}")
    mag, ang = cv2.cartToPolar(flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32))
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = np.uint8(np.mod(ang * 180 / math.pi / 2, 180))
    hsv[..., 1] = 255
    hsv[..., 2] = np.uint8(np.clip(mag / (np.percentile(mag, 95) + 1e-6), 0, 1) * 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def save_flow_npy(path: str | Path, flow: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, flow.astype(np.float32))


def load_flow_npy(path: str | Path) -> torch.Tensor:
    arr = np.load(Path(path)).astype(np.float32)
    return torch.from_numpy(arr)
