#!/usr/bin/env python3
"""Zero-initialized multi-scale flow residual adapters for Exp19."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ResidualShape:
    channels: int
    height: int
    width: int


def tensor_shape_to_residual_shape(tensor: torch.Tensor) -> ResidualShape:
    if tensor.ndim != 4:
        raise ValueError(f"Expected [N,C,H,W], got {tensor.shape}")
    return ResidualShape(int(tensor.shape[1]), int(tensor.shape[2]), int(tensor.shape[3]))


class ZeroConvProjector(nn.Module):
    """1x1 zero-conv projection.

    There is intentionally no zero-initialized alpha multiplier. With a zero
    conv alone, the initial residual is exactly zero while the first backward
    pass still produces non-zero gradients for the projection weights.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MultiScaleFlowResidualBuilder(nn.Module):
    """Build down/mid residual tensors matching UNetMotionModel shapes.

    The builder is shape-driven: a preflight pass records the real residual
    shapes of the frozen Exp11 Stage2 UNet, then this module creates a zero-conv
    projector for each shape. Initial residuals are exactly zero because both
    the 1x1 conv and alpha are zero-initialized.
    """

    def __init__(self, in_channels: int = 7):
        super().__init__()
        self.in_channels = int(in_channels)
        self.down_shapes: list[ResidualShape] = []
        self.mid_shape: ResidualShape | None = None
        self.down = nn.ModuleList()
        self.mid: ZeroConvProjector | None = None

    @property
    def is_built(self) -> bool:
        return bool(self.down_shapes) and self.mid_shape is not None and len(self.down) == len(self.down_shapes)

    def build(self, down_shapes: list[ResidualShape], mid_shape: ResidualShape) -> None:
        self.down_shapes = list(down_shapes)
        self.mid_shape = mid_shape
        self.down = nn.ModuleList([ZeroConvProjector(self.in_channels, s.channels) for s in self.down_shapes])
        self.mid = ZeroConvProjector(self.in_channels, mid_shape.channels)

    def _project(self, flow_cond: torch.Tensor, shape: ResidualShape, projector: ZeroConvProjector) -> torch.Tensor:
        b, t, c, _, _ = flow_cond.shape
        x = flow_cond.reshape(b * t, c, flow_cond.shape[-2], flow_cond.shape[-1])
        x = F.interpolate(x, size=(shape.height, shape.width), mode="bilinear", align_corners=False)
        return projector(x)

    def forward(self, flow_cond: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        if not self.is_built or self.mid is None or self.mid_shape is None:
            raise RuntimeError("MultiScaleFlowResidualBuilder must be built from shape preflight before forward")
        if flow_cond.ndim != 5:
            raise ValueError(f"Expected flow_cond [B,T,C,H,W], got {flow_cond.shape}")
        down = tuple(self._project(flow_cond, s, p) for s, p in zip(self.down_shapes, self.down))
        mid = self._project(flow_cond, self.mid_shape, self.mid)
        return down, mid

    def alpha_values(self) -> dict[str, float]:
        return {}


def record_unet_residual_shapes(unet, *args, **kwargs) -> tuple[list[ResidualShape], ResidualShape]:
    """Run a no-grad UNet forward and record residual shapes via hooks."""
    down_shapes: list[ResidualShape] = []
    mid_shape: ResidualShape | None = None

    def conv_hook(_module, _inputs, output):
        down_shapes.append(tensor_shape_to_residual_shape(output))

    def down_hook(_module, _inputs, output):
        if isinstance(output, tuple) and len(output) == 2:
            _sample, res_samples = output
            for tensor in res_samples:
                down_shapes.append(tensor_shape_to_residual_shape(tensor))

    def mid_hook(_module, _inputs, output):
        nonlocal mid_shape
        mid_shape = tensor_shape_to_residual_shape(output)

    handles = [unet.conv_in.register_forward_hook(conv_hook)]
    handles += [block.register_forward_hook(down_hook) for block in unet.down_blocks]
    if unet.mid_block is None:
        raise RuntimeError("UNetMotionModel has no mid_block; Exp19 adapter injection is not supported")
    handles.append(unet.mid_block.register_forward_hook(mid_hook))
    try:
        with torch.no_grad():
            unet(*args, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
    if mid_shape is None:
        raise RuntimeError("Failed to record UNet mid-block shape")
    return down_shapes, mid_shape
