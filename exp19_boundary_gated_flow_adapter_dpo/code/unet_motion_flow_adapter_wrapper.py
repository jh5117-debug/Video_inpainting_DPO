#!/usr/bin/env python3
"""Hook-based Stage2 flow-adapter wrapper for Exp19.

This wrapper never uses ControlNet/T2I-Adapter ``additional_residuals``. It
registers forward hooks on selected temporal/motion modules and adds a
zero-initialized, boundary-gated flow residual directly to the hidden-state
tensor returned by those modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


DEFAULT_TARGET_MODULES = (
    "mid_block.motion_modules.0",
    "up_blocks.0.motion_modules.0",
    "up_blocks.1.motion_modules.0",
)


@dataclass
class HookShape:
    name: str
    channels: int
    height: int
    width: int
    output_type: str


class FlowResidualProjector(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # ZeroConv starts from an exact no-op. Alpha is initialized to one, not
        # zero, so the first backward pass can update the projection weights.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.alpha = nn.Parameter(torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.proj(x)


class UNetMotionFlowAdapterWrapper(nn.Module):
    """Wrap a frozen UNetMotionModel with hook-based flow residual adapters."""

    def __init__(
        self,
        unet: nn.Module,
        target_module_names: tuple[str, ...] = DEFAULT_TARGET_MODULES,
        flow_channels: int = 7,
        gate_mode: str = "boundary",
    ):
        super().__init__()
        self.unet = unet
        self.target_module_names = tuple(target_module_names)
        self.flow_channels = int(flow_channels)
        self.gate_mode = gate_mode
        self.projectors = nn.ModuleDict()
        self.hook_shapes: list[HookShape] = []
        self._handles = []
        self._flow_context: torch.Tensor | None = None
        self._num_frames: int | None = None
        self._adapter_enabled = False
        self._auto_full_flow_context: torch.Tensor | None = None
        self._auto_context_sequence: list[list[int]] = []
        self._auto_context_pos = 0
        self._auto_adapter_enabled = False
        self.last_residual_norms: dict[str, float] = {}
        self.last_gate_stats: dict[str, float] = {}
        self._install_hooks()

    @property
    def config(self):
        return self.unet.config

    @property
    def dtype(self):
        return self.unet.dtype

    @property
    def device(self):
        return self.unet.device

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _install_hooks(self) -> None:
        modules = dict(self.unet.named_modules())
        missing = [name for name in self.target_module_names if name not in modules]
        if missing:
            raise ValueError(f"Exp19 target motion modules not found: {missing}")
        for name in self.target_module_names:
            self._handles.append(modules[name].register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            tensor, rebuild = self._extract_tensor(output)
            if tensor is None:
                return output
            shape = HookShape(name=name, channels=int(tensor.shape[1]), height=int(tensor.shape[2]), width=int(tensor.shape[3]), output_type=type(output).__name__)
            key = name.replace(".", "__")
            if key not in self.projectors:
                self.projectors[key] = FlowResidualProjector(self.flow_channels, shape.channels).to(
                    device=tensor.device, dtype=tensor.dtype
                )
                self.hook_shapes.append(shape)
            if not self._adapter_enabled or self._flow_context is None:
                return output
            residual, gate = self._build_residual(name, tensor)
            self.last_residual_norms[name] = float(residual.detach().float().norm().cpu())
            self._update_gate_stats(gate)
            return rebuild(tensor + residual)

        return hook

    @staticmethod
    def _extract_tensor(output: Any):
        if torch.is_tensor(output):
            return output, lambda tensor: tensor
        if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
            return output[0], lambda tensor: (tensor, *output[1:])
        if hasattr(output, "sample") and torch.is_tensor(output.sample):
            def rebuild(tensor):
                try:
                    return output.__class__(sample=tensor)
                except Exception:
                    output.sample = tensor
                    return output
            return output.sample, rebuild
        return None, lambda tensor: output

    def _build_residual(self, name: str, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._flow_context is None or self._num_frames is None:
            raise RuntimeError("Exp19 flow context is not set")
        bt, _c, h, w = hidden.shape
        t = int(self._num_frames)
        if bt % t != 0:
            raise RuntimeError(f"Cannot recover batch size from hidden shape {hidden.shape} and num_frames={t}")
        b = bt // t
        flow = self._flow_context
        if flow.shape[0] != b or flow.shape[1] != t:
            raise RuntimeError(f"Flow context shape {flow.shape} does not match hidden B={b}, T={t}")
        flat = flow.reshape(b * t, flow.shape[2], flow.shape[3], flow.shape[4]).to(device=hidden.device, dtype=hidden.dtype)
        flat = F.interpolate(flat, size=(h, w), mode="bilinear", align_corners=False)
        gate = self._gate_from_flat_flow(flat)
        proj = self.projectors[name.replace(".", "__")]
        residual = proj(flat) * gate
        return residual, gate

    def _gate_from_flat_flow(self, flat: torch.Tensor) -> torch.Tensor:
        conf = flat[:, 4:5].clamp(0.0, 1.0)
        if self.gate_mode == "global":
            return conf
        hole = flat[:, 5:6].clamp(0.0, 1.0)
        boundary = flat[:, 6:7].clamp(0.0, 1.0)
        return (conf * torch.clamp(hole + 0.75 * boundary, 0.0, 1.0)).clamp(0.0, 1.0)

    def _update_gate_stats(self, gate: torch.Tensor) -> None:
        flat = gate.detach().float().flatten()
        if flat.numel() == 0:
            return
        self.last_gate_stats = {
            "gate_mean": float(flat.mean().cpu()),
            "gate_p10": float(torch.quantile(flat, 0.10).cpu()),
            "gate_p50": float(torch.quantile(flat, 0.50).cpu()),
            "gate_p90": float(torch.quantile(flat, 0.90).cpu()),
            "nonzero_gate_ratio": float((flat > 1e-6).float().mean().cpu()),
        }

    def set_flow_context(self, flow_condition: torch.Tensor | None, num_frames: int, enabled: bool) -> None:
        self._flow_context = flow_condition
        self._num_frames = int(num_frames)
        self._adapter_enabled = bool(enabled)
        self.last_residual_norms = {}
        self.last_gate_stats = {}

    def clear_flow_context(self) -> None:
        self._flow_context = None
        self._num_frames = None
        self._adapter_enabled = False

    def adapter_parameters(self):
        return self.projectors.parameters()

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return self.projectors.state_dict()

    def load_adapter_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.materialize_projectors_from_state_dict(state_dict)
        self.projectors.load_state_dict(state_dict)

    def materialize_projectors_from_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Create lazy projectors before strict checkpoint loading.

        Training discovers motion-module channel counts during the first
        forward hook. Inference needs to load weights before the first forward,
        so we recover the projector output channels from each ``proj.weight``.
        """
        for key, value in state_dict.items():
            if not key.endswith(".proj.weight"):
                continue
            name = key[: -len(".proj.weight")]
            if name in self.projectors:
                continue
            if value.ndim != 4:
                raise ValueError(f"Unexpected projector weight shape for {key}: {tuple(value.shape)}")
            out_channels = int(value.shape[0])
            in_channels = int(value.shape[1])
            if in_channels != self.flow_channels:
                raise ValueError(
                    f"Checkpoint flow channels for {key} = {in_channels}, expected {self.flow_channels}"
                )
            self.projectors[name] = FlowResidualProjector(self.flow_channels, out_channels)

    def alpha_values(self) -> dict[str, float]:
        values = {}
        for name, module in self.projectors.items():
            if hasattr(module, "alpha"):
                values[name] = float(module.alpha.detach().float().cpu())
        return values

    def set_auto_flow_context(
        self,
        full_flow_condition: torch.Tensor,
        context_sequence: list[list[int]],
        enabled: bool = True,
    ) -> None:
        if full_flow_condition.ndim != 5:
            raise ValueError(f"full_flow_condition must be [B,T,C,H,W], got {tuple(full_flow_condition.shape)}")
        self._auto_full_flow_context = full_flow_condition
        self._auto_context_sequence = [list(map(int, context)) for context in context_sequence]
        self._auto_context_pos = 0
        self._auto_adapter_enabled = bool(enabled)

    def clear_auto_flow_context(self) -> None:
        self._auto_full_flow_context = None
        self._auto_context_sequence = []
        self._auto_context_pos = 0
        self._auto_adapter_enabled = False

    def auto_context_consumed(self) -> tuple[int, int]:
        return self._auto_context_pos, len(self._auto_context_sequence)

    def _pop_auto_flow_condition(self, num_frames: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self._auto_full_flow_context is None:
            return None
        if self._auto_context_pos >= len(self._auto_context_sequence):
            raise RuntimeError(
                "Exp19 flow context queue exhausted: "
                f"pos={self._auto_context_pos}, total={len(self._auto_context_sequence)}"
            )
        context = self._auto_context_sequence[self._auto_context_pos]
        self._auto_context_pos += 1
        if len(context) != int(num_frames):
            raise RuntimeError(f"Context length {len(context)} does not match num_frames={num_frames}: {context}")
        full = self._auto_full_flow_context
        if max(context) >= full.shape[1] or min(context) < 0:
            raise RuntimeError(f"Context {context} out of range for flow length {full.shape[1]}")
        return full[:, context].to(device=device, dtype=dtype)

    def forward(self, *args, flow_condition: torch.Tensor | None = None, adapter_enabled: bool = True, num_frames: int = 24, **kwargs):
        if flow_condition is None and self._auto_full_flow_context is not None:
            sample = args[0] if args and torch.is_tensor(args[0]) else None
            if sample is None:
                raise RuntimeError("Cannot infer device/dtype for automatic Exp19 flow context")
            flow_condition = self._pop_auto_flow_condition(num_frames, sample.device, sample.dtype)
            adapter_enabled = self._auto_adapter_enabled
        self.set_flow_context(flow_condition, num_frames=num_frames, enabled=adapter_enabled)
        try:
            return self.unet(*args, num_frames=num_frames, **kwargs)
        finally:
            self.clear_flow_context()
