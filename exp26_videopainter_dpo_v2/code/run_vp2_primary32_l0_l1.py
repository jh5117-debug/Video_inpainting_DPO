#!/usr/bin/env python3
"""Run Exp26 VideoPainter primary-32 L0/L1 gates.

This runner intentionally stays tiny: one real primary-32 batch, one optimizer
step, strict branch checkpoint reload, and fixed-noise diagnostics. It does not
start 10-step or longer training.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from train_videopainter_dpo_adapter import (  # noqa: E402
    VideoPainterDPOTrainer,
    VideoPainterPairDataset,
    apply_official_config_to_args,
    collate_pairs,
    make_vp2_optimizer,
)
from vp2_official_config import write_locked_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Exp26 primary32 L0/L1 gate")
    p.add_argument("--videopainter_root", required=True)
    p.add_argument("--pretrained_model_name_or_path", required=True)
    p.add_argument("--policy_checkpoint", required=True)
    p.add_argument("--reference_checkpoint", required=True)
    p.add_argument("--official_train_file", required=True)
    p.add_argument("--pair_manifest", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--seed", type=int, default=20260625)
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    p.add_argument("--limit_train_samples", type=int, default=1)
    p.add_argument("--first_frame_gt", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def state_digest(state: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        t = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def max_state_diff(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    if set(a) != set(b):
        missing = sorted(set(a) - set(b))
        unexpected = sorted(set(b) - set(a))
        raise RuntimeError(f"state key mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    out = 0.0
    for key in a:
        av = a[key].detach().float().cpu()
        bv = b[key].detach().float().cpu()
        if av.shape != bv.shape:
            raise RuntimeError(f"state shape mismatch {key}: {tuple(av.shape)} vs {tuple(bv.shape)}")
        if av.numel():
            out = max(out, float((av - bv).abs().max()))
    return out


def state_delta_norm(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key in a:
        av = a[key].detach().float().cpu()
        bv = b[key].detach().float().cpu()
        total += float((av - bv).pow(2).sum())
    return math.sqrt(total)


def grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.detach().float().norm().cpu()) ** 2
    return math.sqrt(total)


def make_trainer_args(base: argparse.Namespace, output_dir: Path, diag_csv: Path, manifest: Path) -> argparse.Namespace:
    ns = argparse.Namespace(
        videopainter_root=base.videopainter_root,
        pretrained_model_name_or_path=base.pretrained_model_name_or_path,
        policy_checkpoint=base.policy_checkpoint,
        reference_checkpoint=base.reference_checkpoint,
        pair_manifest=str(manifest),
        youtubevos_root="",
        davis_root="",
        output_dir=str(output_dir),
        dpo_diag_csv=str(diag_csv),
        official_train_file=base.official_train_file,
        locked_official_config_json=str(output_dir / "locked_official_optimizer_scheduler.json"),
        max_train_steps=1,
        checkpointing_steps=1,
        checkpoints_total_limit=2,
        train_batch_size=1,
        num_workers=0,
        learning_rate=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        weight_decay=1e-4,
        lr_scheduler="constant",
        lr_warmup_steps=500,
        max_grad_norm=1.0,
        seed=base.seed,
        mixed_precision=base.mixed_precision,
        report_to="none",
        dpo_diag_log_every=1,
        preflight_only=False,
        limit_train_samples=base.limit_train_samples,
        height=base.height,
        width=base.width,
        num_frames=base.num_frames,
        plumbing_only_13f=False,
        first_frame_gt=base.first_frame_gt,
        prompt_max_sequence_length=226,
        branch_layer_num=2,
        enable_slicing=True,
        enable_tiling=True,
        gradient_checkpointing=True,
        mask_add=True,
        wo_text=False,
        add_first=False,
        noised_image_dropout=0.0,
        beta_dpo=10.0,
        lose_gap_weight=0.25,
        lose_gap_clip_tau=1.0,
        winner_abs_reg_weight=0.05,
        winner_gap_reg_weight=1.0,
        winner_gap_reg_margin=0.0,
        gap_eps=1e-6,
        boundary_mode="outer",
        mask_weight=1.0,
        boundary_weight=0.75,
        outside_weight=0.05,
    )
    return apply_official_config_to_args(ns)


def fixed_noise_timestep(trainer: VideoPainterDPOTrainer, batch, seed: int):
    torch.manual_seed(seed)
    with torch.no_grad():
        latents = trainer.encode_latents(batch.winner.to(trainer.device))
    noise = torch.randn_like(latents).to(dtype=trainer.weight_dtype)
    timestep = torch.tensor([min(500, trainer.scheduler.config.num_train_timesteps - 1)], device=trainer.device).long()
    return noise, timestep


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.pair_manifest)
    diag_csv = out_dir / "vp_l0_l1_diagnostics.csv"
    locked_cfg = write_locked_config(Path(args.official_train_file), out_dir / "locked_official_optimizer_scheduler.json")

    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)
    targs = make_trainer_args(args, out_dir / "one_step_run", diag_csv, manifest)
    write_json(out_dir / "resolved_config.json", vars(targs))

    dataset = VideoPainterPairDataset(
        str(manifest),
        height=targs.height,
        width=targs.width,
        num_frames=targs.num_frames,
        limit=targs.limit_train_samples,
        plumbing_only_13f=False,
        first_frame_gt=targs.first_frame_gt,
    )
    batch = collate_pairs([dataset[0]])
    trainer = VideoPainterDPOTrainer(targs)
    optimizer = make_vp2_optimizer(trainer.policy_branch.parameters(), targs)

    noise, timesteps = fixed_noise_timestep(trainer, batch, args.seed + 17)
    torch.manual_seed(args.seed + 23)
    loss0, diag0 = trainer.compute_losses(batch, fixed_noise=noise, fixed_timesteps=timesteps)
    if not torch.isfinite(loss0):
        raise FloatingPointError(f"L0 non-finite loss: {loss0}")
    optimizer.zero_grad(set_to_none=True)
    loss0.backward()
    policy_grad_norm = grad_norm(trainer.policy_branch.parameters())
    ref_has_grad = any(p.grad is not None for p in trainer.reference_branch.parameters())
    if ref_has_grad:
        raise RuntimeError("Reference branch received gradients in L0")
    if policy_grad_norm <= 0 or not math.isfinite(policy_grad_norm):
        raise RuntimeError(f"Policy grad norm invalid: {policy_grad_norm}")

    before_state = {k: v.detach().cpu().clone() for k, v in trainer.policy_branch.state_dict().items()}
    before_ref_state = {k: v.detach().cpu().clone() for k, v in trainer.reference_branch.state_dict().items()}
    torch.nn.utils.clip_grad_norm_(trainer.policy_branch.parameters(), targs.max_grad_norm)
    optimizer.step()
    after_state = {k: v.detach().cpu().clone() for k, v in trainer.policy_branch.state_dict().items()}
    after_ref_state = {k: v.detach().cpu().clone() for k, v in trainer.reference_branch.state_dict().items()}
    policy_delta_norm = state_delta_norm(before_state, after_state)
    reference_delta_norm = state_delta_norm(before_ref_state, after_ref_state)
    if policy_delta_norm <= 0:
        raise RuntimeError("L1 optimizer step did not change policy")
    if reference_delta_norm != 0.0:
        raise RuntimeError(f"Reference changed during L1: {reference_delta_norm}")

    trainer.save_branch_checkpoint(1, optimizer)
    trainer.save_last_weights(1)
    ckpt = out_dir / "one_step_run" / "checkpoint-1"
    saved_state = {k: v.detach().cpu().clone() for k, v in trainer.policy_branch.state_dict().items()}
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reload_args = make_trainer_args(args, out_dir / "reload_run", out_dir / "reload_diag.csv", manifest)
    reload_args.policy_checkpoint = str(ckpt)
    reload = VideoPainterDPOTrainer(reload_args)
    reloaded_state = {k: v.detach().cpu().clone() for k, v in reload.policy_branch.state_dict().items()}
    reload_max_abs_diff = max_state_diff(saved_state, reloaded_state)
    if reload_max_abs_diff != 0.0:
        raise RuntimeError(f"Strict reload state diff is not zero: {reload_max_abs_diff}")
    torch.manual_seed(args.seed + 23)
    loss_reload, diag_reload = reload.compute_losses(batch, fixed_noise=noise.to(reload.device), fixed_timesteps=timesteps.to(reload.device))
    if not torch.isfinite(loss_reload):
        raise FloatingPointError(f"Reloaded branch non-finite loss: {loss_reload}")

    report = {
        "status": "passed",
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "manifest": str(manifest),
        "sample_id": batch.sample_ids[0],
        "frames": int(batch.winner.shape[1]),
        "official_optimizer_scheduler": locked_cfg.__dict__,
        "L0": {
            "status": "passed",
            "loss": float(loss0.detach().float().cpu()),
            "diag": diag0,
            "policy_grad_norm": policy_grad_norm,
            "reference_has_grad": ref_has_grad,
        },
        "L1": {
            "status": "passed",
            "checkpoint": str(ckpt),
            "last_weights": str(out_dir / "one_step_run" / "last_weights"),
            "policy_delta_norm": policy_delta_norm,
            "reference_delta_norm": reference_delta_norm,
            "saved_state_digest": state_digest(saved_state),
            "reloaded_state_digest": state_digest(reloaded_state),
            "reload_max_abs_diff": reload_max_abs_diff,
            "reload_loss": float(loss_reload.detach().float().cpu()),
            "reload_diag": diag_reload,
        },
    }
    write_json(out_dir / "vp_l0_l1_report.json", report)
    with (out_dir / "vp_l0_l1_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["gate", "loss", "dpo_loss", "implicit_acc", "policy_grad_norm", "policy_delta_norm", "reference_delta_norm", "reload_max_abs_diff"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({
            "gate": "L0",
            "loss": report["L0"]["loss"],
            "dpo_loss": diag0["dpo_loss"],
            "implicit_acc": diag0["implicit_acc"],
            "policy_grad_norm": policy_grad_norm,
            "policy_delta_norm": "",
            "reference_delta_norm": "",
            "reload_max_abs_diff": "",
        })
        w.writerow({
            "gate": "L1",
            "loss": report["L1"]["reload_loss"],
            "dpo_loss": diag_reload["dpo_loss"],
            "implicit_acc": diag_reload["implicit_acc"],
            "policy_grad_norm": "",
            "policy_delta_norm": policy_delta_norm,
            "reference_delta_norm": reference_delta_norm,
            "reload_max_abs_diff": reload_max_abs_diff,
        })
    (out_dir / "vp_l0_l1_report.md").write_text(
        "\n".join([
            "# Exp26 VideoPainter Primary-32 L0/L1",
            "",
            "status: passed",
            f"sample_id: `{batch.sample_ids[0]}`",
            f"frames: `{int(batch.winner.shape[1])}`",
            f"policy_grad_norm: `{policy_grad_norm}`",
            f"policy_delta_norm: `{policy_delta_norm}`",
            f"reference_delta_norm: `{reference_delta_norm}`",
            f"reload_max_abs_diff: `{reload_max_abs_diff}`",
            "",
        ]),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
