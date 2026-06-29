"""Precision and runtime policy for Exp43 MiniMax BF16-safe runs."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PrecisionPolicy:
    """Resolved dtype/backend contract recorded for each Exp43 run."""

    name: str = "bf16_safe"
    transformer_dtype: str = "bf16"
    vae_dtype: str = "fp32"
    loss_dtype: str = "fp32"
    reduction_dtype: str = "fp32"
    grad_norm_dtype: str = "fp32"
    use_grad_scaler: bool = False
    disable_flash_sdp: bool = True
    disable_mem_efficient_sdp: bool = True
    enable_math_sdp: bool = True
    disable_xformers: bool = True
    timestep_min: float = 0.05
    timestep_max: float = 0.95
    silent_fallback_allowed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def dtype_from_name(name: str) -> torch.dtype:
    if name in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    if name in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def clamp_timestep(value: float, policy: PrecisionPolicy) -> float:
    return max(policy.timestep_min, min(policy.timestep_max, float(value)))


def apply_runtime_policy(policy: PrecisionPolicy | None = None) -> PrecisionPolicy:
    resolved = policy or PrecisionPolicy()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if resolved.disable_xformers:
        os.environ.setdefault("XFORMERS_DISABLED", "1")
        os.environ.setdefault("DISABLE_XFORMERS", "1")
        os.environ.setdefault("FLASH_ATTENTION_FORCE_DISABLE", "1")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if resolved.disable_flash_sdp and hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if resolved.disable_mem_efficient_sdp and hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if resolved.enable_math_sdp and hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    return resolved


def classify_runtime_failure(text: str) -> str:
    lowered = text.lower()
    if "sigfpe" in lowered or "floating point exception" in lowered:
        return "UNKNOWN_SIGFPE"
    if "flash" in lowered or "flashattention" in lowered or "flash-attn" in lowered:
        return "FLASH_ATTN"
    if "xformers" in lowered:
        return "XFORMERS"
    if "vae" in lowered or "autoencoder" in lowered:
        return "VAE"
    if "nan" in lowered or "inf" in lowered or "nonfinite" in lowered:
        return "LOSS_REDUCTION"
    if "checkpoint" in lowered and ("replay" in lowered or "reload" in lowered):
        return "CHECKPOINT_REPLAY"
    if "timestep" in lowered:
        return "TIMESTEP_EDGE"
    if "out of memory" in lowered or "cuda oom" in lowered:
        return "OOM"
    if "cuda" in lowered:
        return "CUDA"
    return "UNKNOWN"
