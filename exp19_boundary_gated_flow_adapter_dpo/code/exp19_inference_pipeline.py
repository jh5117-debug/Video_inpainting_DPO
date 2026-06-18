#!/usr/bin/env python3
"""Exp19-aware DiffuEraser inference wrapper.

This module wraps the existing DiffuEraser runtime without modifying shared
pipeline code. The Stage2 UNet is replaced by the isolated Exp19 hook wrapper,
which consumes a precomputed flow context queue during denoising.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.run_BR import DiffuEraser  # noqa: E402

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from unet_motion_flow_adapter_wrapper import DEFAULT_TARGET_MODULES, UNetMotionFlowAdapterWrapper  # noqa: E402


def tensor_checksum(tensor: torch.Tensor) -> str:
    arr = tensor.detach().float().cpu().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()[:16]


def load_flow_adapter_checkpoint(
    wrapper: UNetMotionFlowAdapterWrapper,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing Exp19 flow adapter checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    state = ckpt.get("adapter_state_dict", ckpt)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"invalid flow adapter checkpoint: {checkpoint_path}")
    wrapper.load_adapter_state_dict(state)
    expected_modules = tuple(ckpt.get("target_module_names", DEFAULT_TARGET_MODULES))
    if tuple(wrapper.target_module_names) != expected_modules:
        raise ValueError(
            "target module mismatch: "
            f"wrapper={wrapper.target_module_names}, checkpoint={expected_modules}"
        )
    total_params = 0
    total_norm_sq = 0.0
    key_norms: dict[str, float] = {}
    key_checksums: dict[str, str] = {}
    nonzero_tensors = 0
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        total_params += int(value.numel())
        norm = float(value.detach().float().norm().cpu())
        key_norms[key] = norm
        key_checksums[key] = tensor_checksum(value)
        total_norm_sq += norm * norm
        if bool((value.detach().float().abs() > 0).any().item()):
            nonzero_tensors += 1
    return {
        "checkpoint_path": str(checkpoint_path),
        "target_module_names": list(expected_modules),
        "state_dict_keys": sorted(state.keys()),
        "num_tensors": len(state),
        "num_nonzero_tensors": nonzero_tensors,
        "parameter_count": total_params,
        "parameter_l2_norm": total_norm_sq ** 0.5,
        "key_norms": key_norms,
        "key_checksums": key_checksums,
        "hook_shapes": ckpt.get("hook_shapes", []),
    }


class Exp19FlowAdapterDiffuEraser:
    """DiffuEraser runtime with strict Exp19 flow-adapter injection."""

    def __init__(
        self,
        device: str,
        base_model_path: str,
        vae_path: str,
        diffueraser_path: str,
        adapter_checkpoint: str,
        pcm_weights_path: str,
        use_pcm: bool = False,
        num_inference_steps_override: int = 6,
        target_module_names: tuple[str, ...] = DEFAULT_TARGET_MODULES,
    ) -> None:
        self.device = device
        self.runtime = DiffuEraser(
            device,
            base_model_path,
            vae_path,
            diffueraser_path,
            pcm_weights_path=pcm_weights_path,
            use_pcm=use_pcm,
            num_inference_steps_override=num_inference_steps_override,
        )
        self.wrapper = UNetMotionFlowAdapterWrapper(
            self.runtime.pipeline.unet,
            target_module_names=target_module_names,
            gate_mode="boundary",
        )
        self.checkpoint_audit = load_flow_adapter_checkpoint(self.wrapper, adapter_checkpoint)
        self.wrapper.to(device=device, dtype=self.runtime.pipeline.unet.dtype)
        self.wrapper.eval()
        self.runtime.pipeline.unet = self.wrapper
        self.runtime.unet_main = self.wrapper

    def write_checkpoint_audit(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Exp19 Inference Checkpoint Loading Audit",
            "",
            f"- checkpoint: `{self.checkpoint_audit['checkpoint_path']}`",
            f"- target_modules: `{self.checkpoint_audit['target_module_names']}`",
            f"- tensors: `{self.checkpoint_audit['num_tensors']}`",
            f"- nonzero_tensors: `{self.checkpoint_audit['num_nonzero_tensors']}`",
            f"- parameter_count: `{self.checkpoint_audit['parameter_count']}`",
            f"- parameter_l2_norm: `{self.checkpoint_audit['parameter_l2_norm']}`",
            "",
            "## Keys",
            "",
        ]
        for key in self.checkpoint_audit["state_dict_keys"]:
            lines.append(
                f"- `{key}` norm={self.checkpoint_audit['key_norms'].get(key)} "
                f"sha256={self.checkpoint_audit['key_checksums'].get(key)}"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        (path.with_suffix(".json")).write_text(json.dumps(self.checkpoint_audit, indent=2), encoding="utf-8")

    def forward(
        self,
        *,
        validation_image: str,
        validation_mask: str,
        priori_frames,
        output_path: str,
        flow_condition: torch.Tensor,
        context_sequence: list[list[int]],
        video_length: int,
        nframes: int = 22,
        seed: int | None = 1234,
        adapter_enabled: bool = True,
        **kwargs,
    ):
        self.wrapper.set_auto_flow_context(
            flow_condition.to(device=self.device, dtype=self.runtime.pipeline.unet.dtype),
            context_sequence,
            enabled=adapter_enabled,
        )
        try:
            frames = self.runtime.forward(
                validation_image=validation_image,
                validation_mask=validation_mask,
                priori="__unused__",
                output_path=output_path,
                video_length=video_length,
                nframes=nframes,
                seed=seed,
                blended=False,
                priori_frames=priori_frames,
                return_frames=True,
                apply_composite=True,
                **kwargs,
            )
            consumed, total = self.wrapper.auto_context_consumed()
            if consumed != total:
                raise RuntimeError(f"Exp19 flow context queue consumed {consumed}/{total}")
            return frames
        finally:
            self.wrapper.clear_auto_flow_context()

