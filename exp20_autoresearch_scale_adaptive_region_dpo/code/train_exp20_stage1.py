#!/usr/bin/env python
# coding=utf-8
"""
DPO Stage 1 Training — DiffuEraser DPO Finetune

训练目标: UNet2D + BrushNet (可训练)
冻结: VAE, text_encoder, ref_model (BrushNet_ref + UNet2D_ref)
损失函数: Diffusion-DPO loss

基于 train_DiffuEraser_stage1.py 改造，核心变化:
1. 使用 DPODataset 替代 FinetuneDataset
2. 加载冻结的 ref_model (BrushNet_ref + UNet2D_ref)
3. pos/neg 共享 noise + timestep，concat 后一次前向
4. DPO loss 替代 MSE loss
5. 最多保存 2 个权重 (best + last)
"""

import argparse
import atexit
import csv
import contextlib
import gc
import gzip
import warnings
import logging
import math
import json
import os
import sys
import shutil
import glob
import subprocess
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image
import imageio
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange, repeat
from packaging import version
from tqdm.auto import tqdm
import ast

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import transformers
from transformers import AutoTokenizer, PretrainedConfig
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

# 需要从项目根目录 import. Copied experiment entrypoints live under
# exp*/code, so two dirname levels would point at the experiment folder rather
# than the repository root.
PROJECT_ROOT = os.environ.get("PROJECT_ROOT") or str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffueraser.pipeline_diffueraser_stage1 import StableDiffusionDiffuEraserPipelineStageOne
from libs.brushnet_CA import BrushNetModel
from libs.unet_2d_condition import UNet2DConditionModel
from libs.unet_motion_model import UNetMotionModel
from training.common.dataset_imports import import_dataset_file_helpers
from training.dpo.dataset.factory import build_dpo_dataset
from exp20_autoresearch_scale_adaptive_region_dpo.code.boundary_maps import (
    DistanceCache,
    build_region_maps as build_exp20_region_maps,
    summarize_maps as summarize_exp20_region_maps,
)

FileClient, imfrombytes = import_dataset_file_helpers(PROJECT_ROOT)


if is_wandb_available():
    import wandb

check_min_version("0.27.0.dev0")

logger = get_logger(__name__)


def parse_bool_arg(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false value, got {value!r}")


def set_process_title_from_env():
    name = os.environ.get("LINGBOT_PROCESS_NAME") or os.environ.get("PROCESS_TITLE")
    if not name:
        return
    try:
        import setproctitle

        setproctitle.setproctitle(name)
        return
    except Exception:
        pass
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        libc.prctl(15, name[:15].encode("utf-8"), 0, 0, 0)
    except Exception:
        pass


def forward_stage1_pair_member(
    brushnet,
    unet,
    noisy_latents,
    timesteps,
    encoder_hidden_states,
    brushnet_cond,
    weight_dtype,
):
    down_samples, mid_sample, up_samples = brushnet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        brushnet_cond=brushnet_cond,
        return_dict=False,
    )

    model_pred = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        down_block_add_samples=[s.to(dtype=weight_dtype) for s in down_samples],
        mid_block_add_sample=mid_sample.to(dtype=weight_dtype),
        up_block_add_samples=[s.to(dtype=weight_dtype) for s in up_samples],
        return_dict=True,
    ).sample

    del down_samples, mid_sample, up_samples
    return model_pred


def resolve_torch_dtype(dtype_name, default_dtype):
    if dtype_name == "auto":
        return default_dtype
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


class TeeStream:
    """Mirror a text stream to both the original stream and a log file."""

    def __init__(self, original_stream, log_fp, lock):
        self.original_stream = original_stream
        self.log_fp = log_fp
        self.lock = lock

    def write(self, data):
        if not data:
            return 0
        with self.lock:
            written = self.original_stream.write(data)
            try:
                self.log_fp.write(data)
                self.log_fp.flush()
            except Exception:
                pass
        return written

    def flush(self):
        with self.lock:
            self.original_stream.flush()
            try:
                self.log_fp.flush()
            except Exception:
                pass

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def isatty(self):
        return self.original_stream.isatty()

    def fileno(self):
        return self.original_stream.fileno()

    @property
    def encoding(self):
        return getattr(self.original_stream, "encoding", None)

    @property
    def errors(self):
        return getattr(self.original_stream, "errors", None)

    def __getattr__(self, name):
        return getattr(self.original_stream, name)


def get_console_log_dir(output_dir):
    external_log_dir = os.environ.get("DPO_EXTERNAL_LOG_DIR")
    if external_log_dir:
        return os.path.join(external_log_dir, "console_logs")
    return os.path.join(output_dir, "console_logs")


def setup_process_console_capture(output_dir):
    """
    将当前进程的 stdout/stderr 显式 tee 到 rank 专属文本文件。
    这样即使 W&B Logs 页面不完整，run Files 里也能拿到完整文本日志。
    """
    rank = os.environ.get("RANK", "0")
    log_dir = get_console_log_dir(output_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"rank{rank}.log")

    # 避免重复包裹 stdout/stderr
    if getattr(sys.stdout, "_wandb_console_log_path", None) == log_path and \
       getattr(sys.stderr, "_wandb_console_log_path", None) == log_path:
        return log_path

    log_fp = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")
    lock = threading.Lock()

    stdout_tee = TeeStream(sys.stdout, log_fp, lock)
    stderr_tee = TeeStream(sys.stderr, log_fp, lock)
    stdout_tee._wandb_console_log_path = log_path
    stderr_tee._wandb_console_log_path = log_path

    sys.stdout = stdout_tee
    sys.stderr = stderr_tee

    def _cleanup():
        # 先恢复原始流，再关文件，防止解释器退出时 flush 已关闭的 log_fp
        try:
            sys.stdout = stdout_tee.original_stream
        except Exception:
            pass
        try:
            sys.stderr = stderr_tee.original_stream
        except Exception:
            pass
        try:
            log_fp.flush()
        finally:
            try:
                log_fp.close()
            except Exception:
                pass

    atexit.register(_cleanup)
    return log_path


def sync_console_logs_to_wandb(output_dir, policy="live"):
    """将 console log 文件同步到 W&B。内部完全容错，不抛异常。"""
    if not is_wandb_available() or wandb.run is None:
        return

    try:
        log_dir = get_console_log_dir(output_dir)
        log_paths = sorted(glob.glob(os.path.join(log_dir, "*.log")))
        for log_path in log_paths:
            wandb.save(log_path, policy=policy)

        if log_paths:
            wandb.run.summary["console_log_dir"] = log_dir
            wandb.run.summary["console_log_count"] = len(log_paths)
    except Exception as e:
        # 不让 W&B 同步失败掩盖原始训练错误，或把成功训练变成失败退出
        try:
            logger.warning(f"sync_console_logs_to_wandb failed (non-fatal): {e}")
        except Exception:
            pass


def save_wandb_run_info(output_dir, args):
    """
    将当前 W&B run 的关键信息落盘，供 launcher 进程在 crash 后恢复同一个 run，
    并补传完整的 Slurm stdout/stderr 日志。
    """
    if not is_wandb_available() or wandb.run is None:
        return

    run_info = {
        "id": wandb.run.id,
        "project": wandb.run.project,
        "entity": getattr(wandb.run, "entity", None) or args.wandb_entity,
        "name": wandb.run.name,
        "url": wandb.run.url,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }

    save_dirs = [output_dir]
    external_log_dir = os.environ.get("DPO_EXTERNAL_LOG_DIR")
    if external_log_dir:
        save_dirs.append(external_log_dir)

    for save_dir in save_dirs:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "wandb_run_info.json")
        tmp_path = save_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(run_info, f, indent=2)
        os.replace(tmp_path, save_path)


# ============================================================
# DPO Loss + Reg-DPO Diagnostics
# ============================================================
def compute_dpo_loss(
    model_pred,
    ref_pred,
    noise,
    loss_weight_map=None,
    loss_region_maps=None,
    loss_region_mode="full",
    region_stats=None,
    aggregation="legacy_global_weighted_mean",
    gap_normalization="raw",
    gap_eps=1e-6,
    lose_gap_clip_tau=None,
    beta_dpo=500.0,
    sft_reg_weight=0.0,
    lose_gap_weight=1.0,
    winner_abs_reg_weight=0.0,
    winner_gap_reg_weight=0.0,
    winner_gap_reg_margin=0.0,
    winner_gap_reg_mode="relu",
    nframes=None,
):
    """
    Diffusion-DPO loss + Reg-DPO 论文要求的诊断指标。

    model_pred: [2*B*F, 4, H, W] — policy 的 ε prediction (pos+neg concat)
    ref_pred:   [2*B*F, 4, H, W] — ref 的 ε prediction
    noise:      [B*F, 4, H, W]   — 共享 noise (target ε)
    nframes:    每个 video pair 的帧数；诊断指标按 video-pair 聚合

    Returns:
        loss:         总 loss = DPO + winner-side SFT regularization
        diagnostics:  dict 包含所有诊断指标 + 跨卡聚合所需的原始 tensor
    """
    target = noise.repeat(2, 1, 1, 1)  # repeat 匹配 pos+neg

    def per_sample_mse(pred, tgt, weight_map=None):
        err = (pred.float() - tgt.float()).pow(2)
        if weight_map is None:
            return err.mean(dim=[1, 2, 3])
        weight = weight_map.to(device=err.device, dtype=err.dtype)
        if weight.shape[0] * 2 == err.shape[0]:
            weight = weight.repeat(2, 1, 1, 1)
        if weight.shape[0] != err.shape[0]:
            raise ValueError(
                f"loss_weight_map batch={weight.shape[0]} does not match prediction batch={err.shape[0]}"
            )
        if weight.shape[-2:] != err.shape[-2:]:
            weight = F.interpolate(weight, size=err.shape[-2:], mode="nearest")
        if weight.shape[1] == 1 and err.shape[1] != 1:
            weight = weight.expand(-1, err.shape[1], -1, -1)
        numer = (err * weight).sum(dim=[1, 2, 3])
        denom = weight.sum(dim=[1, 2, 3]).clamp(min=1e-8)
        return numer / denom

    # ‖ε - ε_θ‖² per-sample. In region mode this is region-weighted MSE.
    model_losses = per_sample_mse(model_pred, target, loss_weight_map)
    model_losses_w, model_losses_l = model_losses.chunk(2)

    # ‖ε - ε_ref‖² (常量)
    with torch.no_grad():
        ref_losses = per_sample_mse(ref_pred, target, loss_weight_map)
        ref_losses_w, ref_losses_l = ref_losses.chunk(2)

    if loss_weight_map is not None and region_stats is not None:
        with torch.no_grad():
            err_w_map = (model_pred[: noise.shape[0]].float() - noise.float()).pow(2).mean(dim=1, keepdim=True)
            weight = loss_weight_map.to(device=err_w_map.device, dtype=err_w_map.dtype)
            if weight.shape[-2:] != err_w_map.shape[-2:]:
                weight = F.interpolate(weight, size=err_w_map.shape[-2:], mode="nearest")

            def region_mean_mse(region_mask):
                denom = region_mask.sum().clamp(min=1.0)
                return ((err_w_map * region_mask).sum() / denom).detach().item()

            mask_w = float(region_stats.get("region_mask_weight", 1.0))
            boundary_w = float(region_stats.get("region_boundary_weight", 0.5))
            outside_w = float(region_stats.get("region_outside_weight", 0.05))
            def region_selector(value):
                value_tensor = torch.tensor(value, device=weight.device, dtype=weight.dtype)
                return torch.isclose(weight, value_tensor, rtol=0.0, atol=1e-4).float()

            region_stats["mask_region_mse"] = region_mean_mse(region_selector(mask_w))
            region_stats["boundary_region_mse"] = region_mean_mse(region_selector(boundary_w))
            region_stats["outside_region_mse"] = region_mean_mse(region_selector(outside_w))

    aggregation = str(aggregation or "legacy_global_weighted_mean")
    gap_normalization = str(gap_normalization or "raw").lower()
    if gap_normalization in {"none", "identity"}:
        gap_normalization = "raw"
    if gap_normalization not in {"raw", "log_ratio"}:
        raise ValueError(f"Unsupported gap_normalization={gap_normalization!r}")

    scale_term = -0.5 * beta_dpo
    raw_win_gap_vec = model_losses_w - ref_losses_w
    raw_lose_gap_vec = model_losses_l - ref_losses_l
    eps = float(gap_eps)
    norm_win_gap_vec = torch.log((model_losses_w.clamp(min=0) + eps) / (ref_losses_w.clamp(min=0) + eps))
    norm_lose_gap_vec = torch.log((model_losses_l.clamp(min=0) + eps) / (ref_losses_l.clamp(min=0) + eps))

    region_balanced_diag = {}
    if aggregation == "region_balanced":
        if loss_region_maps is None:
            raise ValueError("aggregation=region_balanced requires loss_region_maps")
        err_policy = (model_pred.float() - target.float()).pow(2)
        with torch.no_grad():
            err_ref = (ref_pred.float() - target.float()).pow(2)
        err_w, err_l = err_policy.chunk(2)
        err_w_ref, err_l_ref = err_ref.chunk(2)

        def region_mean(error, region_weight):
            weight = region_weight.to(device=error.device, dtype=error.dtype)
            if weight.shape[-2:] != error.shape[-2:]:
                weight = F.interpolate(weight, size=error.shape[-2:], mode="area")
            if weight.ndim != 4:
                raise ValueError(f"region map must be [N,1,H,W], got {tuple(weight.shape)}")
            if weight.shape[0] != error.shape[0]:
                raise ValueError(f"region map batch={weight.shape[0]} does not match error batch={error.shape[0]}")
            if weight.shape[1] == 1 and error.shape[1] != 1:
                weight = weight.expand(-1, error.shape[1], -1, -1)
            numer = (error * weight).sum(dim=[1, 2, 3])
            denom = weight.sum(dim=[1, 2, 3]).clamp(min=1e-8)
            return numer / denom

        alpha_values = {
            "mask": float(region_stats.get("region_mask_weight", 1.0) if region_stats else 1.0),
            "boundary": float(region_stats.get("region_boundary_weight", 0.75) if region_stats else 0.75),
            "outside": float(region_stats.get("region_outside_weight", 0.05) if region_stats else 0.05),
        }
        valid_alpha = {}
        for name in ("mask", "boundary", "outside"):
            region = loss_region_maps[name].to(device=err_w.device, dtype=err_w.dtype)
            valid = (region.sum(dim=[1, 2, 3]) > 1e-8).to(dtype=err_w.dtype)
            valid_alpha[name] = valid * alpha_values[name]
        alpha_total = (valid_alpha["mask"] + valid_alpha["boundary"] + valid_alpha["outside"]).clamp(min=1e-8)
        for name in valid_alpha:
            valid_alpha[name] = valid_alpha[name] / alpha_total

        dpo_win_gap_vec = torch.zeros_like(model_losses_w)
        dpo_lose_gap_vec = torch.zeros_like(model_losses_l)
        for name in ("mask", "boundary", "outside"):
            region = loss_region_maps[name]
            w_policy = region_mean(err_w, region)
            w_ref = region_mean(err_w_ref, region)
            l_policy = region_mean(err_l, region)
            l_ref = region_mean(err_l_ref, region)
            win_gap_region = torch.log((w_policy.clamp(min=0) + eps) / (w_ref.clamp(min=0) + eps))
            lose_gap_region = torch.log((l_policy.clamp(min=0) + eps) / (l_ref.clamp(min=0) + eps))
            dpo_win_gap_vec = dpo_win_gap_vec + valid_alpha[name] * win_gap_region
            dpo_lose_gap_vec = dpo_lose_gap_vec + valid_alpha[name] * lose_gap_region
            region_balanced_diag[f"{name}_policy_mse_w"] = w_policy.detach().mean().item()
            region_balanced_diag[f"{name}_ref_mse_w"] = w_ref.detach().mean().item()
            region_balanced_diag[f"{name}_policy_mse_l"] = l_policy.detach().mean().item()
            region_balanced_diag[f"{name}_ref_mse_l"] = l_ref.detach().mean().item()
            region_balanced_diag[f"{name}_win_gap"] = win_gap_region.detach().mean().item()
            region_balanced_diag[f"{name}_lose_gap"] = lose_gap_region.detach().mean().item()
            region_balanced_diag[f"{name}_alpha_effective"] = valid_alpha[name].detach().mean().item()
        norm_win_gap_vec = dpo_win_gap_vec
        norm_lose_gap_vec = dpo_lose_gap_vec
    elif aggregation == "legacy_global_weighted_mean" and gap_normalization == "log_ratio":
        dpo_win_gap_vec = norm_win_gap_vec
        dpo_lose_gap_vec = norm_lose_gap_vec
    elif aggregation == "legacy_global_weighted_mean":
        dpo_win_gap_vec = raw_win_gap_vec
        dpo_lose_gap_vec = raw_lose_gap_vec
    else:
        raise ValueError(f"Unsupported aggregation={aggregation!r}")

    clip_tau = None
    if lose_gap_clip_tau is not None and str(lose_gap_clip_tau).strip().lower() not in {"", "none", "inf", "infinity"}:
        clip_tau = float(lose_gap_clip_tau)
    dpo_lose_gap_clipped_vec = torch.clamp(dpo_lose_gap_vec, max=clip_tau) if clip_tau is not None else dpo_lose_gap_vec
    norm_lose_gap_clipped_vec = torch.clamp(norm_lose_gap_vec, max=clip_tau) if clip_tau is not None else norm_lose_gap_vec
    frame_inside_term = scale_term * (dpo_win_gap_vec - float(lose_gap_weight) * dpo_lose_gap_clipped_vec)

    # 训练 loss 保持原有 frame-level 目标；诊断统一按 video-pair 统计。
    dpo_loss = (-1.0 * F.logsigmoid(frame_inside_term)).mean()
    sft_loss = model_losses_w.mean()
    winner_abs_reg = model_losses_w.mean()

    winner_gap_reg_mode = str(winner_gap_reg_mode or "relu").lower()
    if winner_gap_reg_mode != "relu":
        raise ValueError(f"Unsupported winner_gap_reg_mode={winner_gap_reg_mode!r}; only 'relu' is supported.")
    win_gap_vec = dpo_win_gap_vec
    relu_win_gap_vec = F.relu(win_gap_vec - float(winner_gap_reg_margin))
    winner_gap_reg = relu_win_gap_vec.mean()

    loss = (
        dpo_loss
        + float(sft_reg_weight) * sft_loss
        + float(winner_abs_reg_weight) * winner_abs_reg
        + float(winner_gap_reg_weight) * winner_gap_reg
    )

    if nframes is None:
        nframes = raw_win_gap_vec.numel()
    nframes = int(nframes)
    if nframes <= 0 or raw_win_gap_vec.numel() % nframes != 0:
        raise ValueError(
            f"nframes={nframes} must divide the number of frame losses "
            f"({raw_win_gap_vec.numel()}) for pair-level DPO diagnostics."
        )

    pair_model_losses_w = model_losses_w.view(-1, nframes).mean(dim=1)
    pair_ref_losses_w = ref_losses_w.view(-1, nframes).mean(dim=1)
    pair_model_losses_l = model_losses_l.view(-1, nframes).mean(dim=1)
    pair_ref_losses_l = ref_losses_l.view(-1, nframes).mean(dim=1)
    pair_raw_win_gap = raw_win_gap_vec.view(-1, nframes).mean(dim=1)
    pair_raw_lose_gap = raw_lose_gap_vec.view(-1, nframes).mean(dim=1)
    pair_norm_win_gap = norm_win_gap_vec.view(-1, nframes).mean(dim=1)
    pair_norm_lose_gap = norm_lose_gap_vec.view(-1, nframes).mean(dim=1)
    pair_norm_lose_gap_clipped = norm_lose_gap_clipped_vec.view(-1, nframes).mean(dim=1)
    pair_dpo_win_gap = dpo_win_gap_vec.view(-1, nframes).mean(dim=1)
    pair_dpo_lose_gap_clipped = dpo_lose_gap_clipped_vec.view(-1, nframes).mean(dim=1)
    inside_term = scale_term * (pair_dpo_win_gap - float(lose_gap_weight) * pair_dpo_lose_gap_clipped)

    # 本卡的 pair-level implicit_acc (后续会跨卡 gather 得到全局值)
    implicit_acc_local = (inside_term > 0).sum().float() / inside_term.size(0)

    # === Reg-DPO 诊断指标 ===
    win_gap = (model_losses_w - ref_losses_w).mean()
    lose_gap = (model_losses_l - ref_losses_l).mean()
    # Match official VideoDPO dpo_diag: policy winner MSE minus policy loser MSE.
    reward_margin = (model_losses_w - model_losses_l).mean()
    sigma_term = torch.sigmoid(inside_term).mean()
    all_model_losses = model_losses.mean()
    all_ref_losses = ref_losses.mean()
    kl_div = 0.5 * (all_model_losses - all_ref_losses)

    # === inside_term 统计 + loser-degradation 分析 ===
    # “靠 loser 变差赢的” = DPO 判断正确的样本中, loser 退化的贡献 > winner 提升的贡献
    #   winner_improvement = max(0, -(model_loss_w - ref_loss_w))  # 正值 = policy 在 winner 上变好
    #   loser_degradation  = max(0,   model_loss_l - ref_loss_l)   # 正值 = policy 在 loser  上变差
    #   loser_dominant iff loser_degradation > winner_improvement
    with torch.no_grad():
        correct_mask = inside_term > 0
        winner_improvement = (-pair_dpo_win_gap).clamp(min=0)  # 取正部: policy 在 winner 上改善的量
        loser_degradation = pair_dpo_lose_gap_clipped.clamp(min=0)     # 取正部: policy 在 loser 上退化的量
        loser_dominant_mask = loser_degradation > winner_improvement
        loser_dominant_wins = (correct_mask & loser_dominant_mask).sum().float()
        n_correct = correct_mask.sum().float()
        loser_degrade_ratio = loser_dominant_wins / n_correct.clamp(min=1)

    diagnostics = {
        "loss_region_mode": str(loss_region_mode),
        "aggregation": aggregation,
        "gap_normalization": gap_normalization,
        "gap_eps": eps,
        "lose_gap_clip_tau": clip_tau if clip_tau is not None else "",
        "dpo_loss": dpo_loss.detach().item(),
        "total_loss": loss.detach().item(),
        "anchored_total_loss": loss.detach().item(),
        "sft_loss": sft_loss.detach().item(),
        "sft_reg_weight": float(sft_reg_weight),
        "lose_gap_weight": float(lose_gap_weight),
        "winner_abs_reg": winner_abs_reg.detach().item(),
        "winner_abs_reg_weight": float(winner_abs_reg_weight),
        "winner_gap_reg": winner_gap_reg.detach().item(),
        "winner_gap_reg_weight": float(winner_gap_reg_weight),
        "winner_gap_reg_margin": float(winner_gap_reg_margin),
        "winner_gap_reg_mode": winner_gap_reg_mode,
        "relu_win_gap_mean": relu_win_gap_vec.detach().mean().item(),
        "relu_win_gap_max": relu_win_gap_vec.detach().max().item(),
        "win_gap_positive_ratio": (win_gap_vec > float(winner_gap_reg_margin)).float().mean().detach().item(),
        "mse_w_over_ref_mse_w": (model_losses_w.mean() / ref_losses_w.mean().clamp(min=1e-8)).detach().item(),
        "mse_l_over_ref_mse_l": (model_losses_l.mean() / ref_losses_l.mean().clamp(min=1e-8)).detach().item(),
        "implicit_acc": implicit_acc_local.detach().item(),
        "mse_w": model_losses_w.mean().detach().item(),
        "mse_l": model_losses_l.mean().detach().item(),
        "ref_mse_w": ref_losses_w.mean().detach().item(),
        "ref_mse_l": ref_losses_l.mean().detach().item(),
        "win_gap": dpo_win_gap_vec.mean().detach().item(),
        "lose_gap": dpo_lose_gap_clipped_vec.mean().detach().item(),
        "raw_win_gap": raw_win_gap_vec.mean().detach().item(),
        "raw_lose_gap": raw_lose_gap_vec.mean().detach().item(),
        "norm_win_gap": norm_win_gap_vec.mean().detach().item(),
        "norm_lose_gap": norm_lose_gap_vec.mean().detach().item(),
        "norm_lose_gap_clipped": norm_lose_gap_clipped_vec.mean().detach().item(),
        "adaptive_norm_mode": "null",
        "adaptive_norm_win_gap": "null",
        "adaptive_norm_lose_gap": "null",
        "adaptive_norm_lose_gap_clipped": "null",
        "reward_margin": reward_margin.detach().item(),
        "sigma_term": sigma_term.detach().item(),
        "kl_divergence": kl_div.detach().item(),
        # inside_term 统计
        "inside_term_mean": inside_term.detach().mean().item(),
        "inside_term_min": inside_term.detach().min().item(),
        "inside_term_max": inside_term.detach().max().item(),
        # loser-dominant: 主要靠 loser 退化获胜的比例
        "loser_degrade_ratio": loser_degrade_ratio.detach().item(),
        "loser_degrade_count": int(loser_dominant_wins.detach().item()),
        "n_correct_local": int(n_correct.detach().item()),
        "n_total_local": inside_term.size(0),
    }
    if region_stats:
        region_diag = {}
        for k, v in region_stats.items():
            try:
                region_diag[k] = float(v)
            except (TypeError, ValueError):
                region_diag[k] = v
        diagnostics.update(region_diag)
        diagnostics.update(
            {
                "region_weighted_mse_w": model_losses_w.mean().detach().item(),
                "region_weighted_mse_l": model_losses_l.mean().detach().item(),
                "region_weighted_ref_mse_w": ref_losses_w.mean().detach().item(),
                "region_weighted_ref_mse_l": ref_losses_l.mean().detach().item(),
            }
        )
    diagnostics.update(region_balanced_diag)
    # 保留原始 tensor 供跨卡 gather
    diagnostics["_inside_term"] = inside_term.detach()
    diagnostics["_winner_improvement"] = winner_improvement.detach()
    diagnostics["_loser_degradation"] = loser_degradation.detach()
    diagnostics["_pair_raw_win_gap"] = pair_raw_win_gap.detach()
    diagnostics["_pair_raw_lose_gap"] = pair_raw_lose_gap.detach()
    diagnostics["_pair_norm_win_gap"] = pair_norm_win_gap.detach()
    diagnostics["_pair_norm_lose_gap"] = pair_norm_lose_gap.detach()
    diagnostics["_pair_norm_lose_gap_clipped"] = pair_norm_lose_gap_clipped.detach()
    diagnostics["_pair_m_w"] = pair_model_losses_w.detach()
    diagnostics["_pair_m_w_ref"] = pair_ref_losses_w.detach()
    diagnostics["_pair_m_l"] = pair_model_losses_l.detach()
    diagnostics["_pair_m_l_ref"] = pair_ref_losses_l.detach()

    return loss, diagnostics


def build_region_loss_weight_map(
    brushnet_masks,
    mask_region_weight=1.0,
    boundary_region_weight=0.5,
    outside_region_weight=0.05,
):
    """Build latent-resolution region weights from DiffuEraser/BrushNet masks.

    `brushnet_masks` follows the repository convention used by the current
    manifest dataset: 0 means unknown/hole, 1 means known outside context.
    Therefore the DPO mask region is `1 - brushnet_masks`.
    """
    if brushnet_masks.ndim != 5:
        raise ValueError(f"expected masks as [B,F,1,H,W], got shape={tuple(brushnet_masks.shape)}")
    if brushnet_masks.shape[2] != 1:
        raise ValueError(f"expected one mask channel, got shape={tuple(brushnet_masks.shape)}")
    boundary_mode = os.environ.get("BOUNDARY_MODE", "both").strip().lower()
    if boundary_mode not in {"inner", "outer", "both", "exp10_default"}:
        raise ValueError(
            "BOUNDARY_MODE must be one of inner, outer, both, exp10_default; "
            f"got {boundary_mode!r}"
        )

    hole = 1.0 - rearrange(brushnet_masks.float(), "b f c h w -> (b f) c h w")
    mask_region = (hole > 0.5).float()
    dilated = F.max_pool2d(mask_region, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool2d(1.0 - mask_region, kernel_size=3, stride=1, padding=1)
    boundary_inner = torch.clamp(mask_region - eroded, 0.0, 1.0)
    boundary_outer = torch.clamp(dilated - mask_region, 0.0, 1.0)
    if boundary_mode == "inner":
        boundary_region = boundary_inner
        mask_core_region = torch.clamp(mask_region - boundary_inner, 0.0, 1.0)
    elif boundary_mode == "outer":
        boundary_region = boundary_outer
        mask_core_region = mask_region
    elif boundary_mode == "both":
        boundary_region = torch.clamp(boundary_inner + boundary_outer, 0.0, 1.0)
        mask_core_region = torch.clamp(mask_region - boundary_inner, 0.0, 1.0)
    else:
        boundary_region = ((dilated - eroded) > 0.0).float()
        mask_core_region = mask_region

    outside_region = (1.0 - torch.clamp(mask_core_region + boundary_region, 0.0, 1.0)).float()
    weight = torch.full_like(mask_region, float(outside_region_weight))
    weight = torch.where(mask_core_region > 0.5, torch.full_like(weight, float(mask_region_weight)), weight)
    weight = torch.where(boundary_region > 0.5, torch.full_like(weight, float(boundary_region_weight)), weight)
    stats = {
        "boundary_mode": boundary_mode,
        "mask_weight": float(mask_region_weight),
        "boundary_weight": float(boundary_region_weight),
        "outside_weight": float(outside_region_weight),
        "region_mask_weight": float(mask_region_weight),
        "region_boundary_weight": float(boundary_region_weight),
        "region_outside_weight": float(outside_region_weight),
        "mask_area_ratio": mask_core_region.mean().detach().item(),
        "boundary_area_ratio": boundary_region.mean().detach().item(),
        "outside_area_ratio": outside_region.mean().detach().item(),
        "region_weight_sum": weight.sum(dim=[1, 2, 3]).mean().detach().item(),
    }
    return weight, stats


def build_exp20_loss_weight_map(
    brushnet_masks_image,
    latent_hw,
    args,
    distance_cache=None,
    cache_identities=None,
):
    """Build Exp20 region weights without touching the legacy Exp11 path.

    `brushnet_masks_image` follows DiffuEraser convention: 0 = hole, 1 = known.
    Exp20 boundary utilities use 1 = hole, so convert once at the boundary.
    """
    if bool(getattr(args, "legacy_exact", False)) or str(args.radius_mode) == "legacy_latent_exact":
        latent_masks = torch.nn.functional.interpolate(
            brushnet_masks_image.to(dtype=torch.float32),
            size=(1, latent_hw[0], latent_hw[1]),
        )
        return build_region_loss_weight_map(
            latent_masks,
            mask_region_weight=args.mask_region_weight,
            boundary_region_weight=args.boundary_region_weight,
            outside_region_weight=args.outside_region_weight,
        )

    hole_image = 1.0 - brushnet_masks_image.float()
    maps = build_exp20_region_maps(
        hole_image,
        latent_hw,
        radius_mode=args.radius_mode,
        radius_value=args.radius_value,
        adaptive_k=args.adaptive_k,
        boundary_mode="outer",
        distance_cache=distance_cache,
        cache_identities=cache_identities,
    )
    mask_map = maps["mask"].to(device=brushnet_masks_image.device, dtype=brushnet_masks_image.dtype)
    boundary_map = maps["boundary"].to(device=brushnet_masks_image.device, dtype=brushnet_masks_image.dtype)
    outside_map = maps["outside"].to(device=brushnet_masks_image.device, dtype=brushnet_masks_image.dtype)
    weight = (
        float(args.outside_region_weight) * outside_map
        + float(args.mask_region_weight) * mask_map
        + float(args.boundary_region_weight) * boundary_map
    )
    weight = rearrange(weight, "b f h w -> (b f) 1 h w")
    stats = summarize_exp20_region_maps(maps)
    stats.update(
        {
            "boundary_mode": "outer_image_space",
            "mask_weight": float(args.mask_region_weight),
            "boundary_weight": float(args.boundary_region_weight),
            "outside_weight": float(args.outside_region_weight),
            "region_mask_weight": float(args.mask_region_weight),
            "region_boundary_weight": float(args.boundary_region_weight),
            "region_outside_weight": float(args.outside_region_weight),
            "radius_mode": str(args.radius_mode),
            "adaptive_k": float(args.adaptive_k),
            "region_weight_sum": weight.sum(dim=[1, 2, 3]).mean().detach().item(),
        }
    )
    if distance_cache is not None:
        stats.update(distance_cache.stats())
    return weight, stats


def build_exp20_loss_weight_map_and_regions(
    brushnet_masks_image,
    latent_hw,
    args,
    distance_cache=None,
    cache_identities=None,
):
    """Return legacy-compatible weight map plus explicit region maps.

    The legacy path remains byte-for-byte controlled by
    ``build_region_loss_weight_map``. The explicit region maps are only consumed
    when ``--aggregation region_balanced`` is selected.
    """
    weight, stats = build_exp20_loss_weight_map(
        brushnet_masks_image,
        latent_hw,
        args,
        distance_cache=distance_cache,
        cache_identities=cache_identities,
    )
    hole_image = 1.0 - brushnet_masks_image.float()
    maps = build_exp20_region_maps(
        hole_image,
        latent_hw,
        radius_mode=args.radius_mode,
        radius_value=args.radius_value,
        adaptive_k=args.adaptive_k,
        boundary_mode="outer",
        distance_cache=distance_cache,
        cache_identities=cache_identities,
    )
    region_maps = {
        name: rearrange(maps[name].to(device=brushnet_masks_image.device, dtype=brushnet_masks_image.dtype), "b f h w -> (b f) 1 h w")
        for name in ("mask", "boundary", "outside")
    }
    return weight, stats, region_maps


def compute_dpo_grad_norm(loss, params_to_optimize):
    """
    计算 DPO loss 对可训练参数的梯度 L2 范数 (DGR 诊断)。
    需要在 backward 之后、optimizer.step 之前调用。
    """
    total_norm = 0.0
    for p in params_to_optimize:
        if p.grad is not None:
            total_norm += p.grad.data.float().norm(2).item() ** 2
    return total_norm ** 0.5


def gather_dpo_diagnostics(diagnostics, accelerator):
    """
    跨卡 gather inside_term，计算全局 implicit_acc / inside_term 统计 / loser_degrade。
    失败时在主进程打 warning，并标记 scope=rank0。
    """
    inside_term_local = diagnostics.get("_inside_term")
    winner_imp_local = diagnostics.get("_winner_improvement")
    loser_deg_local = diagnostics.get("_loser_degradation")
    pair_gap_keys = {
        "_pair_raw_win_gap": "raw_win_gap",
        "_pair_raw_lose_gap": "raw_lose_gap",
        "_pair_norm_win_gap": "norm_win_gap",
        "_pair_norm_lose_gap": "norm_lose_gap",
        "_pair_norm_lose_gap_clipped": "norm_lose_gap_clipped",
    }

    def add_gap_stats(prefix, tensor):
        tensor = tensor.detach().float()
        diagnostics[f"{prefix}_mean"] = tensor.mean().item()
        diagnostics[f"{prefix}_p50"] = tensor.quantile(0.5).item()
        diagnostics[f"{prefix}_p90"] = tensor.quantile(0.9).item()
        diagnostics[f"{prefix}_max"] = tensor.max().item()

    if inside_term_local is None:
        diagnostics["_scope"] = "rank0"
        return diagnostics

    try:
        all_inside = accelerator.gather(inside_term_local)
        all_winner_imp = accelerator.gather(winner_imp_local)
        all_loser_deg = accelerator.gather(loser_deg_local)

        n_total_global = all_inside.size(0)
        n_correct_global = (all_inside > 0).sum().float()
        global_acc = n_correct_global / n_total_global

        global_inside_mean = all_inside.mean().item()
        global_inside_min = all_inside.min().item()
        global_inside_max = all_inside.max().item()

        # 全局 loser-dominant: loser 退化贡献 > winner 提升贡献
        correct_mask_global = all_inside > 0
        loser_dominant_global = (correct_mask_global & (all_loser_deg > all_winner_imp)).sum().float()
        global_loser_ratio = loser_dominant_global / n_correct_global.clamp(min=1)

        diagnostics["implicit_acc"] = global_acc.item()
        diagnostics["inside_term_mean"] = global_inside_mean
        diagnostics["inside_term_min"] = global_inside_min
        diagnostics["inside_term_max"] = global_inside_max
        diagnostics["loser_degrade_ratio"] = global_loser_ratio.item()
        diagnostics["loser_degrade_count"] = int(loser_dominant_global.item())
        diagnostics["n_correct_global"] = int(n_correct_global.item())
        diagnostics["n_total_global"] = n_total_global
        diagnostics["_scope"] = "global"
        for tensor_key, prefix in pair_gap_keys.items():
            tensor = diagnostics.get(tensor_key)
            if tensor is not None:
                add_gap_stats(prefix, accelerator.gather(tensor))
        diagnostics["mask_area_ratio_mean"] = diagnostics.get("mask_area_ratio")
        diagnostics["boundary_area_ratio_mean"] = diagnostics.get("boundary_area_ratio")
    except Exception as e:
        # gather 失败: 打 warning + 标记 scope 为 rank0，不静默吞掉
        if accelerator.is_main_process:
            logger.warning(f"gather_dpo_diagnostics failed, falling back to rank0 values: {e}")
        diagnostics["_scope"] = "rank0"

    # 清理不可序列化的 tensor
    diagnostics.pop("_inside_term", None)
    diagnostics.pop("_winner_improvement", None)
    diagnostics.pop("_loser_degradation", None)
    for key in [
        "_pair_raw_win_gap",
        "_pair_raw_lose_gap",
        "_pair_norm_win_gap",
        "_pair_norm_lose_gap",
        "_pair_norm_lose_gap_clipped",
        "_pair_m_w",
        "_pair_m_w_ref",
        "_pair_m_l",
        "_pair_m_l_ref",
    ]:
        diagnostics.pop(key, None)
    return diagnostics


def format_dpo_diagnostics(step, diag, grad_norm=None, extra=None):
    """
    格式化 DPO 诊断指标为美观的 ASCII 表格，每 300 步输出。
    指标名称前缀: [R0] = rank0 本卡, [G] = 全局 gather。
    """
    scope = diag.get("_scope", "rank0")
    scope_tag_local = "[R0]" if scope != "global" else "[R0]"
    scope_tag_global = "[G]" if scope == "global" else "[R0]"

    sep = "═" * 76
    thin = "─" * 76
    lines = [
        "",
        f"  {sep}",
        f"  ║  DPO Diagnostics @ Step {step:>6d}  (scope: {scope})".ljust(77) + "║",
        f"  {sep}",
        f"  ║  {'Metric':<24s}  {'Value':>12s}  {'Ideal':>12s}  {'Status':>10s}  ║",
        f"  ║  {thin[:72]}  ║",
    ]

    def status_icon(val, ideal_dir, threshold=0):
        if ideal_dir == "neg":
            return "✅" if val < threshold else "⚠️"
        elif ideal_dir == "pos":
            return "✅" if val > threshold else "⚠️"
        elif ideal_dir == "mid":
            return "✅" if 0.5 < val < 0.95 else "⚠️"
        return "—"

    rows = [
        (f"{scope_tag_local} L_total",       diag["total_loss"],   "↓ decreasing", "—"),
        (f"{scope_tag_local} L_dpo",         diag["dpo_loss"],     "↓ decreasing", status_icon(diag["dpo_loss"], "neg", 0.693)),
        (f"{scope_tag_local} L_sft",         diag["sft_loss"],     "↓ stable",     "—"),
        (f"{scope_tag_local} Winner abs",    diag.get("winner_abs_reg", 0.0), "↓ bounded", "—"),
        (f"{scope_tag_local} Winner gap reg",diag.get("winner_gap_reg", 0.0), "↓ near 0",  "—"),
        (f"{scope_tag_global} Implicit Acc",  diag["implicit_acc"], "0.5~0.9",      status_icon(diag["implicit_acc"], "mid")),
        (f"{scope_tag_local} Win Gap",       diag["win_gap"],      "< 0 (neg)",    status_icon(diag["win_gap"], "neg")),
        (f"{scope_tag_local} Lose Gap",      diag["lose_gap"],     "> 0 (pos)",    status_icon(diag["lose_gap"], "pos")),
        (f"{scope_tag_local} Reward Margin", diag["reward_margin"],"< 0 (neg)",    status_icon(diag["reward_margin"], "neg")),
        (f"{scope_tag_local} Sigma Term",    diag["sigma_term"],   "0.5~0.9",      status_icon(diag["sigma_term"], "mid")),
        (f"{scope_tag_local} KL Divergence", diag["kl_divergence"], "small",        "—"),
        (f"{scope_tag_local} MSE (win)",     diag["mse_w"],        "↓",            "—"),
        (f"{scope_tag_local} MSE (lose)",    diag["mse_l"],        "↑ or stable",  "—"),
        (f"{scope_tag_local} Ref MSE (win)", diag["ref_mse_w"],    "baseline",     "—"),
        (f"{scope_tag_local} Ref MSE (lose)",diag["ref_mse_l"],    "baseline",     "—"),
    ]

    if grad_norm is not None:
        rows.append((f"{scope_tag_local} DGR (grad norm)", grad_norm, "> 0 (alive)", status_icon(grad_norm, "pos", 1e-6)))
        if diag.get("grad_norm_ratio") is not None:
            rows.append((f"{scope_tag_local} Grad Norm Ratio", diag["grad_norm_ratio"], "> 0.01", status_icon(diag["grad_norm_ratio"], "pos", 0.01)))

    # inside_term 统计 (gather 后为全局)
    if "inside_term_mean" in diag:
        rows.append((f"{scope_tag_global} InsideTerm mean", diag["inside_term_mean"], "moderate", "—"))
        rows.append((f"{scope_tag_global} InsideTerm min",  diag["inside_term_min"],  "—",       "—"))
        rows.append((f"{scope_tag_global} InsideTerm max",  diag["inside_term_max"],  "—",       "—"))

    # loser-dominant 分析 (gather 后为全局)
    if "loser_degrade_ratio" in diag:
        ldr = diag["loser_degrade_ratio"]
        rows.append((f"{scope_tag_global} Loser Dominant %", ldr, "< 0.5", status_icon(ldr, "neg", 0.5)))

    for name, val, ideal, st in rows:
        lines.append(f"  ║  {name:<24s}  {val:>12.6f}  {ideal:>12s}  {st:>10s}  ║")

    if extra:
        lines.append(f"  ║  {thin[:72]}  ║")
        for k, v in extra.items():
            lines.append(f"  ║  {k:<24s}  {v:>12.4f}  {'':>12s}  {'':>10s}  ║")

    # 样本计数摘要行
    n_correct = diag.get("n_correct_global", diag.get("n_correct_local", "?"))
    n_total = diag.get("n_total_global", diag.get("n_total_local", "?"))
    ldc = diag.get("loser_degrade_count", "?")
    lines.append(f"  ║  {scope_tag_global} Samples: {n_correct}/{n_total} correct, {ldc} loser-dominant".ljust(77) + "║")

    lines.append(f"  {sep}")
    lines.append("")
    return "\n".join(lines)


def format_dpo_diagnostics_line(step, diag):
    """Official VideoDPO-compatible one-line diagnostics for grep/CSV parsing."""
    n_correct = diag.get("n_correct_global", diag.get("n_correct_local", 0))
    n_total = diag.get("n_total_global", diag.get("n_total_local", 0))
    return (
        "[dpo_diag] "
        f"global_step={step} "
        f"implicit_acc={diag['implicit_acc']:.6f} "
        f"implicit_acc_count={int(n_correct)}/{int(n_total)} "
        f"inside_term_mean={diag.get('inside_term_mean', 0.0):.6f} "
        f"inside_term_min={diag.get('inside_term_min', 0.0):.6f} "
        f"inside_term_max={diag.get('inside_term_max', 0.0):.6f} "
        f"loser_dominant_ratio={diag.get('loser_degrade_ratio', 0.0):.6f} "
        f"gap_normalization={diag.get('gap_normalization', 'raw')} "
        f"lose_gap_clip_tau={diag.get('lose_gap_clip_tau', '')} "
        f"dpo_loss={diag['dpo_loss']:.6f} "
        f"anchored_total_loss={diag.get('anchored_total_loss', diag.get('total_loss', 0.0)):.6f} "
        f"winner_abs_reg={diag.get('winner_abs_reg', 0.0):.6f} "
        f"winner_gap_reg={diag.get('winner_gap_reg', 0.0):.6f} "
        f"winner_gap_reg_weight={diag.get('winner_gap_reg_weight', 0.0):.6f} "
        f"winner_gap_reg_margin={diag.get('winner_gap_reg_margin', 0.0):.6f} "
        f"relu_win_gap_mean={diag.get('relu_win_gap_mean', 0.0):.6f} "
        f"relu_win_gap_max={diag.get('relu_win_gap_max', 0.0):.6f} "
        f"win_gap_positive_ratio={diag.get('win_gap_positive_ratio', 0.0):.6f} "
        f"mse_w_over_ref_mse_w={diag.get('mse_w_over_ref_mse_w', 0.0):.6f} "
        f"mse_l_over_ref_mse_l={diag.get('mse_l_over_ref_mse_l', 0.0):.6f} "
        f"lose_gap_weight={diag.get('lose_gap_weight', 1.0):.6f} "
        f"mse_w={diag['mse_w']:.6f} "
        f"ref_mse_w={diag['ref_mse_w']:.6f} "
        f"mse_l={diag['mse_l']:.6f} "
        f"ref_mse_l={diag['ref_mse_l']:.6f} "
        f"win_gap={diag['win_gap']:.6f} "
        f"lose_gap={diag['lose_gap']:.6f} "
        f"raw_win_gap={diag.get('raw_win_gap', diag['win_gap']):.6f} "
        f"raw_lose_gap={diag.get('raw_lose_gap', diag['lose_gap']):.6f} "
        f"norm_win_gap={diag.get('norm_win_gap', 0.0):.6f} "
        f"norm_lose_gap={diag.get('norm_lose_gap', 0.0):.6f} "
        f"norm_lose_gap_clipped={diag.get('norm_lose_gap_clipped', 0.0):.6f} "
        f"reward_margin={diag['reward_margin']:.6f} "
        f"sigma_term={diag['sigma_term']:.6f} "
        f"kl_divergence={diag['kl_divergence']:.6f}"
    )


DPO_DIAG_CSV_FIELDS = [
    "global_step",
    "dpo_loss",
    "implicit_acc",
    "win_gap",
    "lose_gap",
    "raw_win_gap",
    "raw_lose_gap",
    "norm_win_gap",
    "norm_lose_gap",
    "norm_lose_gap_clipped",
    "adaptive_norm_win_gap",
    "adaptive_norm_lose_gap",
    "adaptive_norm_lose_gap_clipped",
    "adaptive_norm_mode",
    "gap_normalization",
    "gap_eps",
    "lose_gap_clip_tau",
    "mse_w",
    "ref_mse_w",
    "mse_l",
    "ref_mse_l",
    "loser_dominant_ratio",
    "sigma_term",
    "grad_norm",
    "total_loss",
    "anchored_total_loss",
    "sft_loss",
    "sft_reg_weight",
    "lose_gap_weight",
    "winner_abs_reg",
    "winner_abs_reg_weight",
    "winner_gap_reg",
    "winner_gap_reg_weight",
    "winner_gap_reg_margin",
    "relu_win_gap_mean",
    "relu_win_gap_max",
    "win_gap_positive_ratio",
    "mse_w_over_ref_mse_w",
    "mse_l_over_ref_mse_l",
    "reward_margin",
    "kl_divergence",
    "inside_term_mean",
    "inside_term_min",
    "inside_term_max",
    "loss_region_mode",
    "region_mask_weight",
    "region_boundary_weight",
    "region_outside_weight",
    "mask_area_ratio",
    "boundary_area_ratio",
    "outside_area_ratio",
    "region_weight_sum",
    "boundary_mode",
    "mask_weight",
    "boundary_weight",
    "outside_weight",
    "aggregation",
    "mask_policy_mse_w",
    "mask_ref_mse_w",
    "mask_policy_mse_l",
    "mask_ref_mse_l",
    "mask_win_gap",
    "mask_lose_gap",
    "mask_alpha_effective",
    "boundary_policy_mse_w",
    "boundary_ref_mse_w",
    "boundary_policy_mse_l",
    "boundary_ref_mse_l",
    "boundary_win_gap",
    "boundary_lose_gap",
    "boundary_alpha_effective",
    "outside_policy_mse_w",
    "outside_ref_mse_w",
    "outside_policy_mse_l",
    "outside_ref_mse_l",
    "outside_win_gap",
    "outside_lose_gap",
    "outside_alpha_effective",
    "mask_region_mse",
    "boundary_region_mse",
    "outside_region_mse",
    "region_weighted_mse_w",
    "region_weighted_mse_l",
    "region_weighted_ref_mse_w",
    "region_weighted_ref_mse_l",
]


def dpo_diagnostics_enabled(args):
    return bool(getattr(args, "enable_dpo_diag", True)) and not bool(getattr(args, "disable_dpo_diagnostics", False))


def dpo_diagnostics_interval(args):
    interval = int(getattr(args, "dpo_diag_log_every", 0) or 0)
    if interval > 0:
        return interval
    return int(getattr(args, "logging_steps", 300) or 300)


def append_dpo_diagnostics_csv(output_dir, step, diag, grad_norm=None):
    path = os.path.join(output_dir, "dpo_diagnostics.csv")
    exists = os.path.exists(path)
    row = {
        "global_step": step,
        "dpo_loss": diag.get("dpo_loss"),
        "implicit_acc": diag.get("implicit_acc"),
        "win_gap": diag.get("win_gap"),
        "lose_gap": diag.get("lose_gap"),
        "raw_win_gap": diag.get("raw_win_gap"),
        "raw_lose_gap": diag.get("raw_lose_gap"),
        "norm_win_gap": diag.get("norm_win_gap"),
        "norm_lose_gap": diag.get("norm_lose_gap"),
        "norm_lose_gap_clipped": diag.get("norm_lose_gap_clipped"),
        "adaptive_norm_win_gap": diag.get("adaptive_norm_win_gap"),
        "adaptive_norm_lose_gap": diag.get("adaptive_norm_lose_gap"),
        "adaptive_norm_lose_gap_clipped": diag.get("adaptive_norm_lose_gap_clipped"),
        "adaptive_norm_mode": diag.get("adaptive_norm_mode"),
        "gap_normalization": diag.get("gap_normalization"),
        "gap_eps": diag.get("gap_eps"),
        "lose_gap_clip_tau": diag.get("lose_gap_clip_tau"),
        "mse_w": diag.get("mse_w"),
        "ref_mse_w": diag.get("ref_mse_w"),
        "mse_l": diag.get("mse_l"),
        "ref_mse_l": diag.get("ref_mse_l"),
        "loser_dominant_ratio": diag.get("loser_degrade_ratio"),
        "sigma_term": diag.get("sigma_term"),
        "grad_norm": grad_norm,
        "total_loss": diag.get("total_loss"),
        "anchored_total_loss": diag.get("anchored_total_loss"),
        "sft_loss": diag.get("sft_loss"),
        "sft_reg_weight": diag.get("sft_reg_weight"),
        "lose_gap_weight": diag.get("lose_gap_weight"),
        "winner_abs_reg": diag.get("winner_abs_reg"),
        "winner_abs_reg_weight": diag.get("winner_abs_reg_weight"),
        "winner_gap_reg": diag.get("winner_gap_reg"),
        "winner_gap_reg_weight": diag.get("winner_gap_reg_weight"),
        "winner_gap_reg_margin": diag.get("winner_gap_reg_margin"),
        "relu_win_gap_mean": diag.get("relu_win_gap_mean"),
        "relu_win_gap_max": diag.get("relu_win_gap_max"),
        "win_gap_positive_ratio": diag.get("win_gap_positive_ratio"),
        "mse_w_over_ref_mse_w": diag.get("mse_w_over_ref_mse_w"),
        "mse_l_over_ref_mse_l": diag.get("mse_l_over_ref_mse_l"),
        "reward_margin": diag.get("reward_margin"),
        "kl_divergence": diag.get("kl_divergence"),
        "inside_term_mean": diag.get("inside_term_mean"),
        "inside_term_min": diag.get("inside_term_min"),
        "inside_term_max": diag.get("inside_term_max"),
        "loss_region_mode": diag.get("loss_region_mode"),
        "region_mask_weight": diag.get("region_mask_weight"),
        "region_boundary_weight": diag.get("region_boundary_weight"),
        "region_outside_weight": diag.get("region_outside_weight"),
        "mask_area_ratio": diag.get("mask_area_ratio"),
        "boundary_area_ratio": diag.get("boundary_area_ratio"),
        "outside_area_ratio": diag.get("outside_area_ratio"),
        "region_weight_sum": diag.get("region_weight_sum"),
        "boundary_mode": diag.get("boundary_mode"),
        "mask_weight": diag.get("mask_weight"),
        "boundary_weight": diag.get("boundary_weight"),
        "outside_weight": diag.get("outside_weight"),
        "aggregation": diag.get("aggregation"),
        "mask_policy_mse_w": diag.get("mask_policy_mse_w"),
        "mask_ref_mse_w": diag.get("mask_ref_mse_w"),
        "mask_policy_mse_l": diag.get("mask_policy_mse_l"),
        "mask_ref_mse_l": diag.get("mask_ref_mse_l"),
        "mask_win_gap": diag.get("mask_win_gap"),
        "mask_lose_gap": diag.get("mask_lose_gap"),
        "mask_alpha_effective": diag.get("mask_alpha_effective"),
        "boundary_policy_mse_w": diag.get("boundary_policy_mse_w"),
        "boundary_ref_mse_w": diag.get("boundary_ref_mse_w"),
        "boundary_policy_mse_l": diag.get("boundary_policy_mse_l"),
        "boundary_ref_mse_l": diag.get("boundary_ref_mse_l"),
        "boundary_win_gap": diag.get("boundary_win_gap"),
        "boundary_lose_gap": diag.get("boundary_lose_gap"),
        "boundary_alpha_effective": diag.get("boundary_alpha_effective"),
        "outside_policy_mse_w": diag.get("outside_policy_mse_w"),
        "outside_ref_mse_w": diag.get("outside_ref_mse_w"),
        "outside_policy_mse_l": diag.get("outside_policy_mse_l"),
        "outside_ref_mse_l": diag.get("outside_ref_mse_l"),
        "outside_win_gap": diag.get("outside_win_gap"),
        "outside_lose_gap": diag.get("outside_lose_gap"),
        "outside_alpha_effective": diag.get("outside_alpha_effective"),
        "mask_region_mse": diag.get("mask_region_mse"),
        "boundary_region_mse": diag.get("boundary_region_mse"),
        "outside_region_mse": diag.get("outside_region_mse"),
        "region_weighted_mse_w": diag.get("region_weighted_mse_w"),
        "region_weighted_mse_l": diag.get("region_weighted_mse_l"),
        "region_weighted_ref_mse_w": diag.get("region_weighted_ref_mse_w"),
        "region_weighted_ref_mse_l": diag.get("region_weighted_ref_mse_l"),
    }
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DPO_DIAG_CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


DPO_GAP_TRACE_FIELDS = [
    "step",
    "raw_win_gap_mean",
    "raw_win_gap_p50",
    "raw_win_gap_p90",
    "raw_win_gap_max",
    "raw_lose_gap_mean",
    "raw_lose_gap_p50",
    "raw_lose_gap_p90",
    "raw_lose_gap_max",
    "norm_win_gap_mean",
    "norm_win_gap_p50",
    "norm_win_gap_p90",
    "norm_win_gap_max",
    "norm_lose_gap_mean",
    "norm_lose_gap_p50",
    "norm_lose_gap_p90",
    "norm_lose_gap_max",
    "norm_lose_gap_clipped_mean",
    "norm_lose_gap_clipped_p90",
    "mask_area_ratio_mean",
    "boundary_area_ratio_mean",
]


def append_dpo_gap_trace_csv(output_dir, step, diag, explicit_path=""):
    path = explicit_path or os.path.join(output_dir, "dpo_gap_trace.csv")
    exists = os.path.exists(path)
    row = {"step": step}
    for key in DPO_GAP_TRACE_FIELDS:
        if key == "step":
            continue
        row[key] = diag.get(key)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DPO_GAP_TRACE_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def append_dpo_gap_samples_jsonl_gz(output_dir, step, diag, batch, nframes, explicit_path=""):
    path = explicit_path or os.path.join(output_dir, "dpo_gap_samples.jsonl.gz")
    sample_ids = batch.get("sample_id") or []
    pair_indices = batch.get("pair_index") or []
    mask_area_ratios = batch.get("mask_area_ratio")
    tensors = {
        "raw_win_gap": diag.get("_pair_raw_win_gap"),
        "raw_lose_gap": diag.get("_pair_raw_lose_gap"),
        "norm_win_gap": diag.get("_pair_norm_win_gap"),
        "norm_lose_gap": diag.get("_pair_norm_lose_gap"),
        "norm_lose_gap_clipped": diag.get("_pair_norm_lose_gap_clipped"),
        "m_w": diag.get("_pair_m_w"),
        "m_w_ref": diag.get("_pair_m_w_ref"),
        "m_l": diag.get("_pair_m_l"),
        "m_l_ref": diag.get("_pair_m_l_ref"),
    }
    if any(value is None for value in tensors.values()):
        return
    n = int(tensors["raw_win_gap"].numel())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as f:
        for i in range(n):
            row = {
                "step": int(step),
                "sample_id": str(sample_ids[i]) if i < len(sample_ids) and sample_ids[i] is not None else "",
                "pair_index": pair_indices[i] if i < len(pair_indices) else "",
                "mask_area_ratio": float(mask_area_ratios[i]) if mask_area_ratios is not None and i < len(mask_area_ratios) else None,
            }
            for key, tensor in tensors.items():
                row[key] = float(tensor.detach().float().cpu()[i].item())
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


# ============================================================
# Utility Functions
# ============================================================
def save_videos_grid(video, path: str, fps=8):
    outputs = [np.array(img) for img in video]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


def print_model_info(models_dict, logger):
    logger.info("====== Model Parameters ======")
    for name, model in models_dict.items():
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        status = "🔥 trainable" if trainable > 0 else "❄️ frozen"
        logger.info(f"  {name:15s}: total={total/1e6:>7.1f}M  trainable={trainable/1e6:>7.1f}M  frozen={frozen/1e6:>7.1f}M  [{status}]")
    logger.info("==============================")


def format_metrics_table(step, metrics_dict, n_videos):
    headers = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    col_w = max(10, max(len(h) for h in headers) + 2)
    n_cols = len(headers)
    table_w = (col_w + 1) * n_cols + 1

    lines = []
    title = f" Validation @ Step {step} ({n_videos} videos) "
    lines.append(f"\n{'=' * table_w}")
    lines.append(title.center(table_w))
    lines.append('=' * table_w)
    lines.append('|'.join(h.center(col_w) for h in [''] + headers))
    lines.append('-' * table_w)
    lines.append('|'.join(v.center(col_w) for v in ['Avg'] + [f'{v:.4f}' for v in values]))
    lines.append('=' * table_w)
    return '\n'.join(lines)


# ============================================================
# Validation (Stage 1: PSNR + SSIM only)
# ============================================================
def log_validation(
    vae, text_encoder, tokenizer, unet, brushnet, args, accelerator, weight_dtype, step, is_final_validation=False
):
    logger.info("Running validation...")

    from inference.metrics import compute_psnr, compute_ssim

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*scale.*deprecated.*")
        warnings.filterwarnings("ignore", message=".*was not found in config.*")
        pipeline = StableDiffusionDiffuEraserPipelineStageOne.from_pretrained(
            args.base_model_name_or_path,
            vae=accelerator.unwrap_model(vae),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            unet=accelerator.unwrap_model(unet),
            brushnet=accelerator.unwrap_model(brushnet),
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    if is_final_validation or args.mixed_precision == "no":
        inference_ctx = contextlib.nullcontext()
    elif args.mixed_precision == "bf16":
        inference_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        inference_ctx = torch.autocast("cuda", dtype=torch.float16)

    val_data_dir = args.val_data_dir
    images_root = os.path.join(val_data_dir, "JPEGImages_432_240")
    masks_root = os.path.join(val_data_dir, "test_masks")

    if not os.path.isdir(images_root):
        logger.warning(f"Validation image dir not found: {images_root}, skipping.")
        del pipeline; gc.collect(); torch.cuda.empty_cache()
        return None, None

    video_dirs = sorted([d for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, d))])
    logger.info(f"Found {len(video_dirs)} validation videos in {images_root}")

    all_psnr, all_ssim = [], []
    file_client = FileClient('disk')

    for video_name in video_dirs:
        video_image_dir = os.path.join(images_root, video_name)
        video_mask_dir = os.path.join(masks_root, video_name)

        if not os.path.isdir(video_mask_dir):
            continue

        frame_list = sorted(os.listdir(video_image_dir))
        selected_index = list(range(len(frame_list)))[:args.nframes]

        frames, masks, masked_images = [], [], []
        for idx in selected_index:
            frame_path = os.path.join(video_image_dir, frame_list[idx])
            img_bytes = file_client.get(frame_path, 'input')
            img = imfrombytes(img_bytes, float32=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            frames.append(img)

            mask_path = os.path.join(video_mask_dir, str(idx).zfill(5) + '.png')
            if not os.path.exists(mask_path):
                break
            mask = Image.open(mask_path).convert('L')
            mask = np.asarray(mask)
            m = np.array(mask > 0).astype(np.uint8)
            if args.val_mask_dilation_iter > 0:
                m = cv2.dilate(
                    m,
                    cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                    iterations=args.val_mask_dilation_iter,
                )
            mask = Image.fromarray(m * 255)
            masks.append(mask)

            masked_image = np.array(img) * (1 - (np.array(mask)[:, :, np.newaxis].astype(np.float32) / 255))
            masked_images.append(Image.fromarray(masked_image.astype(np.uint8)))

        if len(frames) != len(selected_index) or len(masks) != len(selected_index):
            continue

        try:
            with inference_ctx:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*scale.*deprecated.*")
                    images = pipeline(
                        num_frames=args.nframes, prompt="clean background",
                        images=masked_images, masks=masks,
                        num_inference_steps=args.val_num_inference_steps, generator=generator,
                        guidance_scale=0.0,
                    ).frames
        except Exception as e:
            logger.warning(f"Inference failed for {video_name}: {e}")
            continue

        pred_np = [np.array(img, dtype=np.uint8) for img in images]
        gt_np = [
            np.array(img.resize((pred_np[0].shape[1], pred_np[0].shape[0]), Image.BILINEAR), dtype=np.uint8)
            for img in frames[:len(pred_np)]
        ]

        v_psnr = [compute_psnr(gt, pred) for gt, pred in zip(gt_np, pred_np)]
        v_ssim = [compute_ssim(gt, pred) for gt, pred in zip(gt_np, pred_np)]
        all_psnr.append(float(np.mean(v_psnr)))
        all_ssim.append(float(np.mean(v_ssim)))

    avg_psnr, avg_ssim = None, None
    if all_psnr and all_ssim:
        avg_psnr = float(np.mean(all_psnr))
        avg_ssim = float(np.mean(all_ssim))
        table = format_metrics_table(step, {'PSNR': avg_psnr, 'SSIM': avg_ssim}, len(all_psnr))
        logger.info(table)

        if is_wandb_available() and wandb.run is not None and accelerator.is_main_process:
            wandb.log({"val/psnr_mean": avg_psnr, "val/ssim_mean": avg_ssim}, step=step)
    else:
        logger.warning("No valid validation results collected.")

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return avg_psnr, avg_ssim


# ============================================================
# Helper
# ============================================================
def import_model_class_from_model_name_or_path(base_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        base_model_name_or_path, subfolder="text_encoder", revision=revision,
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def ensure_list(input_variable):
    if isinstance(input_variable, list):
        return input_variable
    elif isinstance(input_variable, str):
        try:
            parsed_list = ast.literal_eval(input_variable)
            if isinstance(parsed_list, list):
                return parsed_list
            else:
                raise ValueError(f"The input string didn't evaluate to a list: {input_variable}")
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid input string: {input_variable}") from e
    else:
        raise TypeError(f"Input must be a list or string representing a list, got {type(input_variable)}")


# ============================================================
# Parse Args
# ============================================================
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="DiffuEraser DPO Stage 1 Training")
    parser.add_argument("--base_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="experiments/dpo/stage1/manual")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_height", type=int, default=None,
                        help="Optional training frame height. Used by videodpo_fullmask to match VideoDPO [320,512].")
    parser.add_argument("--train_width", type=int, default=None,
                        help="Optional training frame width. Used by videodpo_fullmask to match VideoDPO [320,512].")
    parser.add_argument("--nframes", type=int, default=16)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=2000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument(
        "--policy_init_path",
        type=str,
        default=None,
        help=(
            "Optional DiffuEraser weights directory used to initialize only the trainable "
            "policy UNet/BrushNet. The frozen reference model still comes from ref_model_path."
        ),
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--logging_dir", type=str, default="logs-dpo-stage1")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--vae_dtype", type=str, default="auto", choices=["auto", "fp32"],
                        help="VAE encode dtype. Use fp32 on H20 if half-precision VAE hits SIGFPE.")
    parser.add_argument("--policy_dtype", type=str, default="auto", choices=["auto", "fp32"],
                        help="Policy forward dtype. Use fp32 with mixed_precision=no if bf16 backward SIGFPEs on H20.")
    parser.add_argument("--ref_dtype", type=str, default="auto", choices=["auto", "fp32", "bf16", "fp16"],
                        help="Frozen ref forward dtype.")
    parser.add_argument("--text_dtype", type=str, default="auto", choices=["auto", "fp32", "bf16", "fp16"],
                        help="Frozen text encoder dtype.")
    parser.add_argument("--debug_first_batch_stages", action="store_true",
                        help="Print first-batch stage markers to locate hard crashes such as SIGFPE.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")
    parser.add_argument("--proportion_empty_prompts", type=float, default=0)

    # Validation
    parser.add_argument("--validation_prompt", type=str, default=["clean background", "clean background"])
    parser.add_argument("--validation_image", type=str, default=["data/external/davis_432_240/JPEGImages_432_240/bear", "data/external/davis_432_240/JPEGImages_432_240/boat"])
    parser.add_argument("--validation_mask", type=str, default=["data/external/davis_432_240/test_masks/bear", "data/external/davis_432_240/test_masks/boat"])
    parser.add_argument("--val_data_dir", type=str, default="data/external/davis_432_240")
    parser.add_argument("--validation_steps", type=int, default=300)
    parser.add_argument("--val_num_inference_steps", type=int, default=6)
    parser.add_argument("--val_mask_dilation_iter", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=300,
                        help="每隔多少步输出详细 DPO 诊断日志")
    parser.add_argument("--disable_dpo_diagnostics", action="store_true",
                        help="Disable detailed DPO diagnostic tables and keep tqdm progress only.")

    # W&B
    parser.add_argument("--tracker_project_name", type=str, default="DPO_Diffueraser")
    parser.add_argument("--wandb_entity", type=str, default=None)

    # ===== DPO 特有参数 =====
    parser.add_argument("--dpo_data_root", type=str, default="data/external/DPO_Finetune_data",
                        help="DPO 偏好对数据集根目录")
    parser.add_argument("--dpo_dataset_type", type=str, default="diffueraser_inpainting",
                        choices=["diffueraser_inpainting", "videodpo_fullmask", "generated_loser_manifest"],
                        help="DPO dataset adapter. videodpo_fullmask keeps VideoDPO data/task and feeds DiffuEraser with a full-hole mask.")
    parser.add_argument("--preference_manifest", type=str, default="",
                        help="Repaired generated-loser JSONL manifest for --dpo_dataset_type generated_loser_manifest.")
    parser.add_argument("--train_mask_mode", type=str, default="full", choices=["full", "partial"],
                        help="Training mask policy for generated-loser manifests.")
    parser.add_argument("--mask_from_manifest", type=parse_bool_arg, default=False,
                        help="Use mask_path from the manifest as M_train.")
    parser.add_argument("--loss_region_mode", type=str, default="full", choices=["full", "region"],
                        help="DPO loss region mode. region uses sum(weight*mse)/sum(weight).")
    parser.add_argument("--gap_normalization", type=str, default="raw", choices=["raw", "log_ratio"],
                        help="Gap transform used inside DPO. log_ratio uses log((m+eps)/(m_ref+eps)).")
    parser.add_argument("--gap_eps", type=float, default=1e-6)
    parser.add_argument("--lose_gap_clip_tau", type=str, default="",
                        help="Optional max clamp for the DPO loser gap after normalization.")
    parser.add_argument("--mask_region_weight", type=float, default=1.0)
    parser.add_argument("--boundary_region_weight", type=float, default=0.5)
    parser.add_argument("--outside_region_weight", type=float, default=0.05)
    parser.add_argument("--radius_mode", type=str, default="legacy_latent_exact",
                        choices=["legacy_latent_exact", "fixed_image_px", "adaptive_area_perimeter", "adaptive_sqrt_area"],
                        help="Exp20 boundary radius mode. legacy_latent_exact preserves Exp11 maps exactly.")
    parser.add_argument("--radius_value", type=float, default=0.0,
                        help="Fixed image-space radius in pixels for --radius_mode fixed_image_px.")
    parser.add_argument("--adaptive_k", type=float, default=1.0,
                        help="Adaptive radius multiplier for image-space adaptive modes.")
    parser.add_argument("--aggregation", type=str, default="legacy_global_weighted_mean",
                        choices=["legacy_global_weighted_mean", "region_balanced"],
                        help="Exp20 loss aggregation: legacy global weighted mean or region-balanced region log-ratio.")
    parser.add_argument("--boundary_contribution", type=float, default=None,
                        help="Alias for --boundary_region_weight used by immutable trial configs.")
    parser.add_argument("--mask_contribution", type=float, default=None,
                        help="Alias for --mask_region_weight used by immutable trial configs.")
    parser.add_argument("--outside_contribution", type=float, default=None,
                        help="Alias for --outside_region_weight used by immutable trial configs.")
    parser.add_argument("--distance_cache_dir", type=str, default="",
                        help="Exp20-only cache dir for image-space distance transforms.")
    parser.add_argument("--max_train_seconds", type=float, default=0.0,
                        help="Gracefully stop after this many training seconds after warmup. 0 disables.")
    parser.add_argument("--legacy_exact", type=parse_bool_arg, default=True,
                        help="When true, force Exp11 legacy latent map semantics.")
    parser.add_argument("--trial_config", type=str, default="",
                        help="Immutable trial config path copied into the output metadata.")
    parser.add_argument("--trial_id", type=str, default="",
                        help="Immutable trial id copied into the output metadata.")
    parser.add_argument("--dpo_gap_trace_csv", type=str, default="",
                        help="Optional path for aggregate raw/norm gap trace CSV.")
    parser.add_argument("--dpo_gap_samples_jsonl_gz", type=str, default="",
                        help="Optional path for per-diagnostic-batch raw/norm gap samples.")
    parser.add_argument("--enable_dpo_diag", type=parse_bool_arg, default=True,
                        help="Enable DPO diagnostics. Prefer this over --disable_dpo_diagnostics for new runs.")
    parser.add_argument("--dpo_diag_log_every", type=int, default=10,
                        help="Print/write detailed DPO diagnostics every N optimizer steps.")
    parser.add_argument("--dpo_diag_save_csv", type=parse_bool_arg, default=True)
    parser.add_argument("--dpo_diag_save_wandb", type=parse_bool_arg, default=True)
    parser.add_argument("--videodpo_frame_stride", type=int, default=1,
                        help="Frame stride for videodpo_fullmask dataset.")
    parser.add_argument("--videodpo_clip_length", type=float, default=1.0,
                        help="Minimum clip duration for videodpo_fullmask metadata compatibility.")
    parser.add_argument("--videodpo_full_mask_value", type=float, default=0.0,
                        help="Mask tensor value for full-hole DiffuEraser generation mode; 0 follows the existing inverted-mask convention.")
    parser.add_argument("--ref_model_path", type=str, default=None,
                        help="Ref model 权重路径 (SFT 后的 DiffuEraser 权重，含 unet_main/ 和 brushnet/)")
    parser.add_argument("--beta_dpo", type=float, default=500.0,
                        help="DPO 温度系数 beta (推荐 500~1000，过大导致 sigmoid 饱和)")
    parser.add_argument("--sft_reg_weight", type=float, default=0.0,
                        help="Reg-DPO 风格的 winner-side SFT regularization 权重 rho")
    parser.add_argument("--lose_gap_weight", type=float, default=1.0,
                        help="DPO loss 中 loser/negative gap 的权重；1.0 为原始 DPO，0.0 为 winner-only ablation")
    parser.add_argument("--winner_abs_reg_weight", type=float, default=0.0,
                        help="Winner-anchor absolute policy winner MSE regularization weight.")
    parser.add_argument("--winner_gap_reg_weight", type=float, default=0.0,
                        help="Winner-anchor ReLU(policy winner MSE - ref winner MSE - margin) regularization weight.")
    parser.add_argument("--winner_gap_reg_margin", type=float, default=0.0,
                        help="Margin for winner gap regularization.")
    parser.add_argument("--winner_gap_reg_mode", type=str, default="relu", choices=["relu"],
                        help="Winner gap regularization mode.")
    parser.add_argument("--davis_oversample", type=int, default=10,
                        help="DAVIS 视频过采样倍数")
    parser.add_argument("--chunk_aligned", action="store_true",
                        help="是否根据 meta.json chunk 边界对齐采样")
    parser.add_argument("--split_pos_neg_forward", action="store_true",
                        help="分别 forward positive/negative，降低显存峰值，DPO loss 保持不变")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    args.validation_image = ensure_list(args.validation_image)
    args.validation_mask = ensure_list(args.validation_mask)
    args.validation_prompt = ensure_list(args.validation_prompt)

    if args.resolution % 8 != 0:
        raise ValueError("`--resolution` must be divisible by 8.")
    if args.boundary_contribution is not None:
        args.boundary_region_weight = float(args.boundary_contribution)
    if args.mask_contribution is not None:
        args.mask_region_weight = float(args.mask_contribution)
    if args.outside_contribution is not None:
        args.outside_region_weight = float(args.outside_contribution)
    if args.radius_mode != "legacy_latent_exact":
        args.legacy_exact = False

    return args


# ============================================================
# Collate
# ============================================================
def collate_fn(examples):
    pixel_values_pos = torch.stack([e["pixel_values_pos"] for e in examples]).float()
    pixel_values_neg = torch.stack([e["pixel_values_neg"] for e in examples]).float()
    conditioning_pixel_values = torch.stack([e["conditioning_pixel_values"] for e in examples]).float()
    masks = torch.stack([e["masks"] for e in examples]).float()
    input_ids = torch.stack([e["input_ids"] for e in examples])

    return {
        "pixel_values_pos": pixel_values_pos,
        "pixel_values_neg": pixel_values_neg,
        "conditioning_pixel_values": conditioning_pixel_values,
        "masks": masks,
        "input_ids": input_ids,
        "sample_id": [e.get("sample_id") for e in examples],
        "pair_index": [e.get("pair_index") for e in examples],
        "mask_area_ratio": [
            float((1.0 - e["masks"].float()).mean().item()) if "masks" in e else None
            for e in examples
        ],
    }


def _run_git_command(args_list):
    try:
        return subprocess.check_output(args_list, cwd=PROJECT_ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def _safe_sha256_file(path):
    if not path:
        return ""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    import hashlib

    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_exp20_run_metadata(args, accelerator):
    if not accelerator.is_main_process:
        return
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    resolved_config = {k: str(v) if isinstance(v, Path) else v for k, v in sorted(vars(args).items())}
    (out / "resolved_config.yaml").write_text(json.dumps(resolved_config, indent=2, sort_keys=True) + "\n")
    git_info = {
        "branch": _run_git_command(["git", "branch", "--show-current"]),
        "head": _run_git_command(["git", "rev-parse", "HEAD"]),
        "status_short": _run_git_command(["git", "status", "--short"]),
        "diff_stat": _run_git_command(["git", "diff", "--stat"]),
    }
    (out / "git_info.txt").write_text(json.dumps(git_info, indent=2, sort_keys=True) + "\n")
    env = {
        "hostname": os.uname().nodename,
        "cwd": os.getcwd(),
        "python": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "mixed_precision": args.mixed_precision,
    }
    (out / "environment.txt").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n")
    dataset_identity = {
        "preference_manifest": args.preference_manifest,
        "preference_manifest_sha256": _safe_sha256_file(args.preference_manifest),
        "dpo_dataset_type": args.dpo_dataset_type,
        "train_mask_mode": args.train_mask_mode,
        "mask_from_manifest": bool(args.mask_from_manifest),
    }
    (out / "dataset_manifest_identity.json").write_text(json.dumps(dataset_identity, indent=2, sort_keys=True) + "\n")
    evaluator_identity = {
        "eval_protocol": "raw6_hard_comp_no_pcm_no_dilation_no_gaussian_framewise",
        "metrics_source": "inference/metrics.py",
        "trainer": "exp20_train_exp20_stage1.py",
    }
    (out / "evaluator_identity.json").write_text(json.dumps(evaluator_identity, indent=2, sort_keys=True) + "\n")


# ============================================================
# Main
# ============================================================
def main(args):
    set_process_title_from_env()
    if not args.enable_dpo_diag:
        args.disable_dpo_diagnostics = True

    report_to = args.report_to
    if report_to is not None and str(report_to).strip().lower() in {"none", "off", "no", "false", "disabled"}:
        report_to = None
        args.report_to = "none"

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=report_to,
        project_config=accelerator_project_config,
    )
    setup_process_console_capture(args.output_dir)
    accelerator.wait_for_everyone()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    write_exp20_run_metadata(args, accelerator)
    accelerator.wait_for_everyone()

    distance_cache = None
    if args.distance_cache_dir and not bool(args.legacy_exact):
        distance_cache = DistanceCache(args.distance_cache_dir)

    # ===== Tracker 初始化提前: 确保后续任何报错都能在启用的 tracker 中可见 =====
    if accelerator.is_main_process and report_to is not None:
        tracker_config = dict(vars(args))
        for key in ["validation_prompt", "validation_image", "validation_mask"]:
            tracker_config.pop(key, None)

        init_kwargs = {}
        if report_to == "wandb":
            init_kwargs["wandb"] = {"name": f"dpo-stage1-{args.max_train_steps or 'auto'}steps"}
            if args.wandb_entity:
                init_kwargs["wandb"]["entity"] = args.wandb_entity

        try:
            accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs=init_kwargs)
            logger.info("Tracker initialized successfully (early init).")
            save_wandb_run_info(args.output_dir, args)
            sync_console_logs_to_wandb(args.output_dir, policy="live")
        except Exception as e:
            logger.error(f"Failed to init tracker: {e}")
            raise RuntimeError(
                "Tracker initialization failed before training started. "
                "Aborting to avoid running without the requested tracker."
            ) from e

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.base_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False,
        )

    text_encoder_cls = import_model_class_from_model_name_or_path(args.base_model_name_or_path, args.revision)
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.base_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.vae_path, subfolder="vae", revision=args.revision, variant=args.variant
    )

    # ===== 辅助函数: 从 UNetMotionModel 提取 2D 权重到 UNet2DConditionModel =====
    def _extract_2d_from_motion(motion_unet, base_model_path, revision=None, variant=None):
        """
        SFT 权重的 config.json 声明为 UNetMotionModel，不能直接用 UNet2DConditionModel 加载。
        必须先创建空的 UNet2D (从 SD base model 获取正确 config)，然后逐模块拷贝 2D 权重。
        """
        unet_2d = UNet2DConditionModel.from_pretrained(
            base_model_path, subfolder="unet", revision=revision, variant=variant
        )
        # 从 UNetMotionModel 拷贝 2D 权重
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
        if hasattr(motion_unet, 'conv_act') and motion_unet.conv_act is not None:
            unet_2d.conv_act.load_state_dict(motion_unet.conv_act.state_dict())
        unet_2d.conv_out.load_state_dict(motion_unet.conv_out.state_dict())
        return unet_2d

    # ===== 检测 ref_model_path 的 unet config 类型 =====
    ref_unet_config_path = os.path.join(args.ref_model_path, "unet_main", "config.json")
    is_motion_model = False
    if os.path.exists(ref_unet_config_path):
        import json
        with open(ref_unet_config_path) as f:
            unet_cfg = json.load(f)
        is_motion_model = (unet_cfg.get("_class_name", "") == "UNetMotionModel")
        logger.info(f"Ref unet config _class_name: {unet_cfg.get('_class_name')}, is_motion_model={is_motion_model}")

    # ===== Policy model: 从 ref_model_path 初始化 (可训练) =====
    if is_motion_model:
        logger.info(f"Loading policy UNet via UNetMotionModel from {args.ref_model_path} (extracting 2D weights)")
        _motion_unet = UNetMotionModel.from_pretrained(args.ref_model_path, subfolder="unet_main")
        unet_main = _extract_2d_from_motion(_motion_unet, args.base_model_name_or_path, args.revision, args.variant)
        del _motion_unet
        logger.info("Successfully extracted policy 2D UNet weights from UNetMotionModel")
    else:
        logger.info(f"Loading policy UNet2D directly from {args.ref_model_path}")
        unet_main = UNet2DConditionModel.from_pretrained(args.ref_model_path, subfolder="unet_main")

    logger.info(f"Loading policy BrushNet from {args.ref_model_path}")
    brushnet = BrushNetModel.from_pretrained(args.ref_model_path, subfolder="brushnet")

    if args.policy_init_path:
        policy_init_path = str(args.policy_init_path)
        policy_unet_config_path = os.path.join(policy_init_path, "unet_main", "config.json")
        policy_is_motion_model = False
        if os.path.exists(policy_unet_config_path):
            import json

            with open(policy_unet_config_path) as f:
                policy_unet_cfg = json.load(f)
            policy_is_motion_model = policy_unet_cfg.get("_class_name", "") == "UNetMotionModel"
        logger.info(f"Overriding trainable policy weights from policy_init_path={policy_init_path}")
        del unet_main
        del brushnet
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if policy_is_motion_model:
            logger.info("Loading policy init UNet via UNetMotionModel (extracting 2D weights)")
            _policy_motion_unet = UNetMotionModel.from_pretrained(policy_init_path, subfolder="unet_main")
            unet_main = _extract_2d_from_motion(
                _policy_motion_unet,
                args.base_model_name_or_path,
                args.revision,
                args.variant,
            )
            del _policy_motion_unet
        else:
            logger.info("Loading policy init UNet2D directly")
            unet_main = UNet2DConditionModel.from_pretrained(policy_init_path, subfolder="unet_main")
        logger.info("Loading policy init BrushNet")
        brushnet = BrushNetModel.from_pretrained(policy_init_path, subfolder="brushnet")

    # ===== Ref model: 从同一路径加载，冻结 =====
    if is_motion_model:
        logger.info(f"Loading ref UNet via UNetMotionModel from {args.ref_model_path} (extracting 2D weights)")
        _motion_unet_ref = UNetMotionModel.from_pretrained(args.ref_model_path, subfolder="unet_main")
        unet_ref = _extract_2d_from_motion(_motion_unet_ref, args.base_model_name_or_path, args.revision, args.variant)
        del _motion_unet_ref
        logger.info("Successfully extracted ref 2D UNet weights from UNetMotionModel")
    else:
        logger.info(f"Loading ref UNet2D directly from {args.ref_model_path}")
        unet_ref = UNet2DConditionModel.from_pretrained(args.ref_model_path, subfolder="unet_main")

    unet_ref.requires_grad_(False)
    unet_ref.eval()

    logger.info(f"Loading ref BrushNet from {args.ref_model_path}")
    brushnet_ref = BrushNetModel.from_pretrained(args.ref_model_path, subfolder="brushnet")
    brushnet_ref.requires_grad_(False)
    brushnet_ref.eval()

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet_main.train()
    brushnet.train()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            unet_main.enable_xformers_memory_efficient_attention()
            brushnet.enable_xformers_memory_efficient_attention()
            unet_ref.enable_xformers_memory_efficient_attention()
            brushnet_ref.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available.")

    if args.gradient_checkpointing:
        unet_main.enable_gradient_checkpointing()
        brushnet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = list(filter(lambda p: p.requires_grad, unet_main.parameters())) + \
                         list(filter(lambda p: p.requires_grad, brushnet.parameters()))

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = build_dpo_dataset(args, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn,
        batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers,
    )
    train_dataset_len = len(train_dataset)
    train_dataloader_len = train_dataset_len // args.train_batch_size

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(train_dataloader_len / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,
    )

    unet_main, brushnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet_main, brushnet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae_dtype = resolve_torch_dtype(args.vae_dtype, weight_dtype)
    policy_dtype = resolve_torch_dtype(args.policy_dtype, weight_dtype)
    ref_dtype = resolve_torch_dtype(args.ref_dtype, weight_dtype)
    text_dtype = resolve_torch_dtype(args.text_dtype, weight_dtype)

    vae.to(accelerator.device, dtype=vae_dtype)
    text_encoder.to(accelerator.device, dtype=text_dtype)
    unet_ref.to(accelerator.device, dtype=ref_dtype)
    brushnet_ref.to(accelerator.device, dtype=ref_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # WandB init 已在前面提前完成

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running DPO Stage 1 Training *****")
    logger.info(f"  Num examples = {train_dataset_len}")
    logger.info(f"  Num batches each epoch (per GPU) = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  VideoDPO-compatible epoch target = {args.num_train_epochs}")
    logger.info(f"  Beta DPO = {args.beta_dpo}")
    logger.info(f"  SFT Reg Weight = {args.sft_reg_weight}")
    logger.info(f"  Lose Gap Weight = {args.lose_gap_weight}")
    logger.info(f"  Winner Abs Reg Weight = {args.winner_abs_reg_weight}")
    logger.info(f"  Winner Gap Reg Weight = {args.winner_gap_reg_weight}")
    logger.info(f"  Winner Gap Reg Margin = {args.winner_gap_reg_margin}")
    logger.info(f"  Winner Gap Reg Mode = {args.winner_gap_reg_mode}")
    logger.info(f"  VAE dtype = {vae_dtype}")
    logger.info(f"  Policy forward dtype = {policy_dtype}")
    logger.info(f"  Ref dtype = {ref_dtype}")
    logger.info(f"  Text dtype = {text_dtype}")
    logger.info(f"  Policy negative grad enabled = {float(args.lose_gap_weight) != 0.0}")
    print_model_info({
        'unet_main (policy)': unet_main, 'brushnet (policy)': brushnet,
        'unet_ref (frozen)': unet_ref, 'brushnet_ref (frozen)': brushnet_ref,
        'vae': vae, 'text_encoder': text_encoder,
    }, logger)

    warnings.filterwarnings("ignore", message=".*scale.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*was not found in config.*")
    global_step = 0
    first_epoch = 0

    # 最佳权重追踪
    best_composite_score = -float('inf')
    initial_grad_norm = None  # DGR 归一化基准
    train_timer_start = None
    train_stop_reason = "max_steps"
    should_stop_after_step = False

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting fresh.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path), map_location="cpu")
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps), initial=initial_global_step,
        desc=f"Epoch {first_epoch + 1}/{args.num_train_epochs}",
        disable=not accelerator.is_local_main_process,
    )

    def debug_stage(message):
        if args.debug_first_batch_stages and global_step == initial_global_step and accelerator.is_local_main_process:
            print(f"[debug-stage] step={global_step} {message}", flush=True)

    for epoch in range(first_epoch, args.num_train_epochs):
        if accelerator.is_local_main_process:
            progress_bar.set_description(f"Epoch {epoch + 1}/{args.num_train_epochs}")
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet_main, brushnet):
                torch.cuda.empty_cache()
                gc.collect()

                # === VAE Encode ===
                debug_stage("before vae pos encode")
                pos_latents = vae.encode(
                    rearrange(batch["pixel_values_pos"], "b f c h w -> (b f) c h w").to(dtype=vae_dtype)
                ).latent_dist.sample() * vae.config.scaling_factor
                pos_latents = pos_latents.to(dtype=policy_dtype)
                debug_stage("after vae pos encode")

                debug_stage("before vae neg encode")
                neg_latents = vae.encode(
                    rearrange(batch["pixel_values_neg"], "b f c h w -> (b f) c h w").to(dtype=vae_dtype)
                ).latent_dist.sample() * vae.config.scaling_factor
                neg_latents = neg_latents.to(dtype=policy_dtype)
                debug_stage("after vae neg encode")

                # BrushNet conditioning: GT masked image + mask
                n_batch = batch["conditioning_pixel_values"].shape[0]
                debug_stage("before vae conditioning encode")
                cond_latents = vae.encode(
                    rearrange(batch["conditioning_pixel_values"], "b f c h w -> (b f) c h w").to(dtype=vae_dtype)
                ).latent_dist.sample() * vae.config.scaling_factor
                cond_latents = cond_latents.to(dtype=policy_dtype)
                cond_latents = rearrange(cond_latents, "(b f) c h w -> b f c h w", b=n_batch)
                debug_stage("after vae conditioning encode")

                masks_image_for_region = batch["masks"].detach()
                masks = torch.nn.functional.interpolate(
                    batch["masks"].to(dtype=policy_dtype),
                    size=(1, pos_latents.shape[-2], pos_latents.shape[-1])
                )
                loss_weight_map = None
                region_stats = None
                loss_region_maps = None
                if args.loss_region_mode == "region":
                    cache_identities = [
                        f"{sid}:{idx}" for sid, idx in zip(batch.get("sample_id", []), batch.get("pair_index", []))
                    ]
                    loss_weight_map, region_stats, loss_region_maps = build_exp20_loss_weight_map_and_regions(
                        masks_image_for_region,
                        (pos_latents.shape[-2], pos_latents.shape[-1]),
                        args,
                        distance_cache=distance_cache,
                        cache_identities=cache_identities if cache_identities else None,
                    )

                # VAE encode 完毕，释放原始像素 tensor 节省显存
                del batch["pixel_values_pos"], batch["pixel_values_neg"], batch["conditioning_pixel_values"]

                brushnet_cond = rearrange(
                    torch.concat([cond_latents, masks], 2),
                    "b f c h w -> (b f) c h w"
                )  # [(b f), 5, h, w]
                debug_stage("after brushnet conditioning build")

                # === Shared noise + timestep ===
                noise = torch.randn_like(pos_latents)
                bsz = pos_latents.shape[0] // args.nframes
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=pos_latents.device
                ).long()
                # 展开 timesteps 到 per-frame 维度，匹配 latent shape (bsz*nframes, ...)
                timesteps_expanded = timesteps.repeat_interleave(args.nframes, dim=0)

                # 加噪: pos 和 neg 使用相同的 noise 但不同的 latent
                noisy_pos = noise_scheduler.add_noise(pos_latents, noise, timesteps_expanded)
                noisy_neg = noise_scheduler.add_noise(neg_latents, noise, timesteps_expanded)
                debug_stage("after add noise")

                debug_stage("before text encoder")
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]
                encoder_hidden_states_expanded = rearrange(
                    repeat(encoder_hidden_states, "b c d -> b t c d", t=args.nframes),
                    'b t c d -> (b t) c d'
                )
                debug_stage("after text encoder")
                encoder_hidden_states_policy = encoder_hidden_states_expanded.to(dtype=policy_dtype)
                encoder_hidden_states_ref = encoder_hidden_states_expanded.to(dtype=ref_dtype)
                brushnet_cond_ref = brushnet_cond.to(dtype=ref_dtype)

                if args.split_pos_neg_forward:
                    # 顺序跑 win/lose，保持 DPO 数学等价，同时避免 pos+neg concat 的激活峰值。
                    with torch.no_grad():
                        debug_stage("before ref pos forward")
                        ref_pred_pos = forward_stage1_pair_member(
                            brushnet_ref, unet_ref, noisy_pos.to(dtype=ref_dtype), timesteps_expanded,
                            encoder_hidden_states_ref, brushnet_cond_ref, ref_dtype,
                        )
                        debug_stage("after ref pos forward")
                        torch.cuda.empty_cache()
                        gc.collect()
                        debug_stage("before ref neg forward")
                        ref_pred_neg = forward_stage1_pair_member(
                            brushnet_ref, unet_ref, noisy_neg.to(dtype=ref_dtype), timesteps_expanded,
                            encoder_hidden_states_ref, brushnet_cond_ref, ref_dtype,
                        )
                        debug_stage("after ref neg forward")
                        ref_pred = torch.cat([ref_pred_pos, ref_pred_neg], dim=0)
                    del ref_pred_pos, ref_pred_neg
                    torch.cuda.empty_cache()
                    gc.collect()

                    debug_stage("before policy pos forward")
                    model_pred_pos = forward_stage1_pair_member(
                        brushnet, unet_main, noisy_pos, timesteps_expanded,
                        encoder_hidden_states_policy, brushnet_cond, policy_dtype,
                    )
                    debug_stage("after policy pos forward")
                    torch.cuda.empty_cache()
                    gc.collect()
                    if float(args.lose_gap_weight) == 0.0:
                        debug_stage("before policy neg forward no_grad")
                        with torch.no_grad():
                            model_pred_neg = forward_stage1_pair_member(
                                brushnet, unet_main, noisy_neg, timesteps_expanded,
                                encoder_hidden_states_policy, brushnet_cond, policy_dtype,
                            )
                        debug_stage("after policy neg forward no_grad")
                    else:
                        debug_stage("before policy neg forward")
                        model_pred_neg = forward_stage1_pair_member(
                            brushnet, unet_main, noisy_neg, timesteps_expanded,
                            encoder_hidden_states_policy, brushnet_cond, policy_dtype,
                        )
                        debug_stage("after policy neg forward")
                    model_pred = torch.cat([model_pred_pos, model_pred_neg], dim=0)
                    del model_pred_pos, model_pred_neg
                else:
                    # Concat pos + neg: batch dim 翻倍
                    noisy_all = torch.cat([noisy_pos, noisy_neg], dim=0)  # [2*b*f, 4, h, w]
                    brushnet_cond_all = torch.cat([brushnet_cond, brushnet_cond], dim=0)
                    timesteps_all = timesteps_expanded.repeat(2)  # [2*b*f]
                    encoder_hidden_states_all = torch.cat([encoder_hidden_states_policy, encoder_hidden_states_policy], dim=0)
                    encoder_hidden_states_ref_all = torch.cat([encoder_hidden_states_ref, encoder_hidden_states_ref], dim=0)

                    # === Ref forward (no_grad) ===
                    # 先算 ref，避免 policy 反向图驻留时再叠加 frozen ref 的 forward 峰值显存。
                    with torch.no_grad():
                        ref_pred = forward_stage1_pair_member(
                            brushnet_ref, unet_ref, noisy_all.to(dtype=ref_dtype), timesteps_all,
                            encoder_hidden_states_ref_all, brushnet_cond_all.to(dtype=ref_dtype), ref_dtype,
                        )

                    torch.cuda.empty_cache()
                    gc.collect()

                    # === Policy forward ===
                    model_pred = forward_stage1_pair_member(
                        brushnet, unet_main, noisy_all, timesteps_all,
                        encoder_hidden_states_all, brushnet_cond_all, policy_dtype,
                    )
                    del noisy_all, brushnet_cond_all, timesteps_all, encoder_hidden_states_all, encoder_hidden_states_ref_all

                # === DPO Loss ===
                debug_stage("before dpo loss")
                loss, diagnostics = compute_dpo_loss(
                    model_pred, ref_pred, noise,
                    loss_weight_map=loss_weight_map,
                    loss_region_maps=loss_region_maps,
                    loss_region_mode=args.loss_region_mode,
                    region_stats=region_stats,
                    aggregation=args.aggregation,
                    gap_normalization=args.gap_normalization,
                    gap_eps=args.gap_eps,
                    lose_gap_clip_tau=args.lose_gap_clip_tau,
                    beta_dpo=args.beta_dpo,
                    sft_reg_weight=args.sft_reg_weight,
                    lose_gap_weight=args.lose_gap_weight,
                    winner_abs_reg_weight=args.winner_abs_reg_weight,
                    winner_gap_reg_weight=args.winner_gap_reg_weight,
                    winner_gap_reg_margin=args.winner_gap_reg_margin,
                    winner_gap_reg_mode=args.winner_gap_reg_mode,
                    nframes=args.nframes,
                )
                debug_stage("after dpo loss")
                local_gap_diagnostics = dict(diagnostics)
                next_global_step = global_step + 1
                if (
                    accelerator.is_main_process
                    and dpo_diagnostics_enabled(args)
                    and (next_global_step % dpo_diagnostics_interval(args) == 0 or next_global_step == 1)
                    and args.dpo_gap_samples_jsonl_gz
                ):
                    append_dpo_gap_samples_jsonl_gz(
                        args.output_dir,
                        next_global_step,
                        local_gap_diagnostics,
                        batch,
                        args.nframes,
                        explicit_path=args.dpo_gap_samples_jsonl_gz,
                    )
                # 跨卡 gather: 全局 implicit_acc / inside_term 统计
                diagnostics = gather_dpo_diagnostics(diagnostics, accelerator)
                debug_stage("after diagnostics gather")

                torch.cuda.empty_cache()
                gc.collect()

                debug_stage("before backward")
                accelerator.backward(loss)
                debug_stage("after backward")

                # DGR: 计算梯度范数（检测梯度消失）
                grad_norm = None
                if accelerator.sync_gradients:
                    debug_stage("before grad norm and clip")
                    grad_norm = compute_dpo_grad_norm(loss, params_to_optimize)
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                    debug_stage("after grad norm and clip")

                debug_stage("before optimizer step")
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                debug_stage("after optimizer step")

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if train_timer_start is None:
                    train_timer_start = time.monotonic()
                elif args.max_train_seconds and args.max_train_seconds > 0:
                    elapsed_train_seconds = time.monotonic() - train_timer_start
                    if elapsed_train_seconds >= float(args.max_train_seconds):
                        should_stop_after_step = True
                        train_stop_reason = "max_train_seconds"

                if accelerator.is_main_process:
                    # === 每 300 步: 详细诊断日志 ===
                    if (
                        dpo_diagnostics_enabled(args)
                        and (global_step % dpo_diagnostics_interval(args) == 0 or global_step == 1)
                    ):
                        logger.info(format_dpo_diagnostics_line(global_step, diagnostics))
                        diag_table = format_dpo_diagnostics(
                            global_step, diagnostics, grad_norm=grad_norm
                        )
                        logger.info(diag_table)
                        if args.dpo_diag_save_csv:
                            append_dpo_diagnostics_csv(args.output_dir, global_step, diagnostics, grad_norm=grad_norm)
                        if args.dpo_gap_trace_csv:
                            append_dpo_gap_trace_csv(
                                args.output_dir,
                                global_step,
                                diagnostics,
                                explicit_path=args.dpo_gap_trace_csv,
                            )

                    # Checkpoint 保存
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                for removing in checkpoints[:num_to_remove]:
                                    shutil.rmtree(os.path.join(args.output_dir, removing))

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    # Validation + 权重保存 (最多 best + last)
                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        avg_psnr, avg_ssim = log_validation(
                            vae, text_encoder, tokenizer, unet_main, brushnet,
                            args, accelerator, weight_dtype, global_step,
                        )

                        if avg_psnr is not None and avg_ssim is not None:
                            composite = 0.5 * (avg_psnr / 50.0) + 0.5 * avg_ssim

                            if composite > best_composite_score:
                                best_composite_score = composite
                                try:
                                    best_dir = os.path.join(args.output_dir, "best_weights")
                                    os.makedirs(os.path.join(best_dir, "unet_main"), exist_ok=True)
                                    os.makedirs(os.path.join(best_dir, "brushnet"), exist_ok=True)
                                    unwrap_model(unet_main).save_pretrained(os.path.join(best_dir, "unet_main"))
                                    unwrap_model(brushnet).save_pretrained(os.path.join(best_dir, "brushnet"))
                                    logger.info(f"New best weights saved (composite={composite:.4f})")

                                    if is_wandb_available() and wandb.run is not None:
                                        artifact = wandb.Artifact(
                                            "dpo-stage1-best", type="model",
                                            metadata={"step": global_step, "psnr": avg_psnr, "ssim": avg_ssim, "composite": composite}
                                        )
                                        artifact.add_dir(best_dir)
                                        wandb.log_artifact(artifact)
                                except Exception as e:
                                    logger.warning(f"Failed to save best weights: {e}")

                # === WandB + progress bar logging (optimizer step only) ===
                # With gradient accumulation, code outside sync_gradients runs once per
                # micro-batch. Keep tqdm/log refresh tied to optimizer steps so one
                # visible progress update corresponds to one real training step.
                scope_prefix = "global/" if diagnostics.get("_scope") == "global" else "rank0/"
                logs = {
                    "rank0/dpo_loss": diagnostics["dpo_loss"],
                    "rank0/total_loss": diagnostics["total_loss"],
                    "rank0/sft_loss": diagnostics["sft_loss"],
                    "rank0/sft_reg_weight": diagnostics["sft_reg_weight"],
                    "rank0/lose_gap_weight": diagnostics["lose_gap_weight"],
                    "rank0/winner_abs_reg": diagnostics["winner_abs_reg"],
                    "rank0/winner_abs_reg_weight": diagnostics["winner_abs_reg_weight"],
                    "rank0/winner_gap_reg": diagnostics["winner_gap_reg"],
                    "rank0/winner_gap_reg_weight": diagnostics["winner_gap_reg_weight"],
                    "rank0/winner_gap_reg_margin": diagnostics["winner_gap_reg_margin"],
                    "rank0/relu_win_gap_mean": diagnostics["relu_win_gap_mean"],
                    "rank0/relu_win_gap_max": diagnostics["relu_win_gap_max"],
                    "rank0/win_gap_positive_ratio": diagnostics["win_gap_positive_ratio"],
                    "rank0/mse_w_over_ref_mse_w": diagnostics["mse_w_over_ref_mse_w"],
                    "rank0/mse_l_over_ref_mse_l": diagnostics["mse_l_over_ref_mse_l"],
                    f"{scope_prefix}implicit_acc": diagnostics["implicit_acc"],
                    "rank0/mse_w": diagnostics["mse_w"],
                    "rank0/mse_l": diagnostics["mse_l"],
                    "rank0/win_gap": diagnostics["win_gap"],
                    "rank0/lose_gap": diagnostics["lose_gap"],
                    "rank0/raw_win_gap": diagnostics.get("raw_win_gap", diagnostics["win_gap"]),
                    "rank0/raw_lose_gap": diagnostics.get("raw_lose_gap", diagnostics["lose_gap"]),
                    "rank0/norm_win_gap": diagnostics.get("norm_win_gap", 0.0),
                    "rank0/norm_lose_gap": diagnostics.get("norm_lose_gap", 0.0),
                    "rank0/norm_lose_gap_clipped": diagnostics.get("norm_lose_gap_clipped", 0.0),
                    "rank0/reward_margin": diagnostics["reward_margin"],
                    "rank0/sigma_term": diagnostics["sigma_term"],
                    "rank0/kl_divergence": diagnostics["kl_divergence"],
                    f"{scope_prefix}inside_term_mean": diagnostics.get("inside_term_mean", 0),
                    f"{scope_prefix}inside_term_min": diagnostics.get("inside_term_min", 0),
                    f"{scope_prefix}inside_term_max": diagnostics.get("inside_term_max", 0),
                    f"{scope_prefix}loser_dominant_ratio": diagnostics.get("loser_degrade_ratio", 0),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                if grad_norm is not None:
                    logs["rank0/dgr_grad_norm"] = grad_norm
                    if initial_grad_norm is None:
                        initial_grad_norm = grad_norm
                    if initial_grad_norm > 0:
                        ratio = grad_norm / initial_grad_norm
                        logs["rank0/grad_norm_ratio"] = ratio
                        diagnostics["grad_norm_ratio"] = ratio
                if accelerator.is_local_main_process:
                    progress_bar.set_postfix(
                        loss=f"{diagnostics['total_loss']:.4f}",
                        dpo=f"{diagnostics['dpo_loss']:.4f}",
                        acc=f"{diagnostics['implicit_acc']:.4f}",
                        lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    )
                if args.dpo_diag_save_wandb:
                    accelerator.log(logs, step=global_step)

            if should_stop_after_step:
                logger.info(
                    f"Graceful Exp20 stop requested after optimizer step {global_step}: "
                    f"max_train_seconds={args.max_train_seconds}"
                )
                break

            if global_step >= args.max_train_steps:
                break
        if should_stop_after_step:
            break

    # 保存 last 权重
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        last_dir = os.path.join(args.output_dir, "last_weights")
        os.makedirs(os.path.join(last_dir, "unet_main"), exist_ok=True)
        os.makedirs(os.path.join(last_dir, "brushnet"), exist_ok=True)
        accelerator.unwrap_model(unet_main).save_pretrained(os.path.join(last_dir, "unet_main"))
        accelerator.unwrap_model(brushnet).save_pretrained(os.path.join(last_dir, "brushnet"))
        logger.info(f"Last weights saved to {last_dir}")
        train_timing = {
            "global_step": int(global_step),
            "stop_reason": train_stop_reason,
            "max_train_seconds": float(args.max_train_seconds or 0.0),
            "train_elapsed_seconds": float(time.monotonic() - train_timer_start) if train_timer_start is not None else 0.0,
            "timer_started_after_first_optimizer_step": train_timer_start is not None,
        }
        Path(args.output_dir, "train_timing.json").write_text(json.dumps(train_timing, indent=2, sort_keys=True) + "\n")

        if is_wandb_available() and wandb.run is not None:
            try:
                artifact = wandb.Artifact(
                    "dpo-stage1-last", type="model",
                    metadata={"step": global_step}
                )
                artifact.add_dir(last_dir)
                wandb.log_artifact(artifact)
            except Exception as e:
                logger.warning(f"Failed to upload last weights: {e}")
        sync_console_logs_to_wandb(args.output_dir, policy="now")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            logger.error(f"Training crashed!\n{tb}")
        except Exception:
            print(f"Training crashed!\n{tb}", file=sys.stderr)
        if is_wandb_available() and wandb.run is not None:
            try:
                sync_console_logs_to_wandb(args.output_dir, policy="now")
                crash_file = os.path.join(wandb.run.dir, "train_process_traceback.txt")
                with open(crash_file, "w") as f:
                    f.write(tb)
                wandb.save(crash_file, policy="now")
                wandb.alert(
                    title="DPO Stage 1 Crashed",
                    text=f"```\n{tb}\n```",
                    level=wandb.AlertLevel.ERROR,
                )
                wandb.finish(exit_code=1)
            except Exception:
                pass
        raise
