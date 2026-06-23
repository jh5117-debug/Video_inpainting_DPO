#!/usr/bin/env python3
"""Run Exp26 VideoPainter v2 L0-L4 gates.

The gates intentionally stay short. They validate semantics, checkpoint
identity, and optimizer/update plumbing without starting any long training.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_videopainter_dpo_adapter import (  # noqa: E402
    VideoPainterBatch,
    VideoPainterDPOTrainer,
    VideoPainterPairDataset,
    apply_official_config_to_args,
    collate_pairs,
    make_vp2_optimizer,
    repeating_epoch_iterator,
)
from vp2_official_config import parse_official_optimizer_scheduler_config, write_locked_config  # noqa: E402


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Exp26 VideoPainter v2 L0-L4 gates")
    parser.add_argument("--videopainter_root", required=True)
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--policy_checkpoint", required=True)
    parser.add_argument("--reference_checkpoint", required=True)
    parser.add_argument("--official_train_file", required=True)
    parser.add_argument("--plumbing_pair_manifest", required=True)
    parser.add_argument("--davis_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--formal_video", default="bear")
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=432)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--l4_steps", type=int, default=10)
    return parser.parse_args()


def image_files(path: Path) -> List[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_formal_davis_pair(args: argparse.Namespace, out_dir: Path) -> Path:
    """Create one real 49-frame pair manifest from DAVIS frames.

    Winner is GT. Loser is black-filled hole pixels from the corresponding
    DAVIS mask. This is only a gate sample, not a preference dataset.
    """

    image_dir = Path(args.davis_root) / "JPEGImages_432_240" / args.formal_video
    mask_dir = Path(args.davis_root) / "test_masks" / args.formal_video
    images = image_files(image_dir)
    masks = image_files(mask_dir)
    if len(images) < 49 or len(masks) < 49:
        raise RuntimeError(f"{args.formal_video}: formal gate needs >=49 frames, got image={len(images)} mask={len(masks)}")

    sample_root = out_dir / "formal_49f" / args.formal_video
    loser_dir = sample_root / "loser_masked"
    loser_dir.mkdir(parents=True, exist_ok=True)
    for idx, (img_path, mask_path) in enumerate(zip(images[:49], masks[:49])):
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L").resize(img.size, Image.NEAREST)
        arr = np.asarray(img, dtype=np.uint8).copy()
        marr = np.asarray(mask, dtype=np.uint8) > 127
        arr[marr] = 0
        Image.fromarray(arr).save(loser_dir / f"{idx:05d}.png")

    row = {
        "sample_id": f"davis49_{args.formal_video}",
        "status": "OK",
        "prompt": "",
        "win_video_path": str(image_dir),
        "raw_loser_video_path": str(loser_dir),
        "comp_loser_video_path": str(loser_dir),
        "final_loser_video_path": str(loser_dir),
        "mask_path": str(mask_dir),
        "num_frames": 49,
        "gate_use": "FORMAL_49F_ONLY",
    }
    manifest = sample_root / "formal_49f_manifest.jsonl"
    manifest.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def make_args(base: argparse.Namespace, output_dir: Path, diag_csv: Path, manifest: Path, *, frames: int, plumbing_only: bool) -> argparse.Namespace:
    ns = argparse.Namespace(
        videopainter_root=base.videopainter_root,
        pretrained_model_name_or_path=base.pretrained_model_name_or_path,
        policy_checkpoint=base.policy_checkpoint,
        reference_checkpoint=base.reference_checkpoint,
        pair_manifest=str(manifest),
        youtubevos_root="",
        davis_root=base.davis_root,
        output_dir=str(output_dir),
        dpo_diag_csv=str(diag_csv),
        official_train_file=base.official_train_file,
        locked_official_config_json=str(Path(base.output_dir) / "locked_official_optimizer_scheduler.json"),
        max_train_steps=1,
        checkpointing_steps=1,
        checkpoints_total_limit=3,
        train_batch_size=1,
        num_workers=0,
        learning_rate=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        lr_scheduler="constant",
        lr_warmup_steps=0,
        max_grad_norm=1.0,
        seed=base.seed,
        mixed_precision=base.mixed_precision,
        report_to="none",
        dpo_diag_log_every=1,
        preflight_only=False,
        limit_train_samples=1,
        height=base.height,
        width=base.width,
        num_frames=frames,
        plumbing_only_13f=plumbing_only,
        first_frame_gt=True,
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


def one_batch(manifest: Path, args: argparse.Namespace) -> VideoPainterBatch:
    dataset = VideoPainterPairDataset(
        str(manifest),
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        limit=1,
        plumbing_only_13f=args.plumbing_only_13f,
        first_frame_gt=args.first_frame_gt,
    )
    return collate_pairs([dataset[0]])


def make_fixed_noise_timestep(trainer: VideoPainterDPOTrainer, batch: VideoPainterBatch, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    with torch.no_grad():
        latents = trainer.encode_latents(batch.winner.to(trainer.device))
    noise = torch.randn_like(latents).to(dtype=trainer.weight_dtype)
    timestep = torch.tensor([min(500, trainer.scheduler.config.num_train_timesteps - 1)], device=trainer.device).long()
    return noise, timestep


def branch_digest(trainer: VideoPainterDPOTrainer, batch: VideoPainterBatch, noise: torch.Tensor, timesteps: torch.Tensor) -> Dict[str, object]:
    trainer.policy_branch.eval()
    with torch.no_grad():
        winner = batch.winner.to(trainer.device)
        conditioning = batch.conditioning.to(trainer.device)
        mask = batch.mask.to(trainer.device)
        prompt_embeds = trainer.prompt_embeds(batch.prompts)
        conditioning_latents, latent_mask = trainer.prepare_conditioning(conditioning, mask, winner.shape[1])
        winner_latents = trainer.encode_latents(winner)
        image_latents = trainer.prepare_image_latents(conditioning, winner_latents.shape[1])
        out = trainer.branch_forward(
            trainer.policy_branch,
            winner_latents,
            conditioning_latents,
            image_latents,
            latent_mask,
            prompt_embeds,
            noise,
            timesteps,
        )
    arr = out.detach().float().cpu()
    return {
        "shape": list(arr.shape),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "l2": float(torch.linalg.vector_norm(arr)),
        "finite": bool(torch.isfinite(arr).all()),
        "checksum16": hashlib.sha256(arr.numpy().tobytes()).hexdigest()[:16],
    }


def native_policy_loss(
    trainer: VideoPainterDPOTrainer,
    batch: VideoPainterBatch,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Single-policy native denoising loss for L3 optimizer/update parity."""

    winner = batch.winner.to(trainer.device)
    conditioning = batch.conditioning.to(trainer.device)
    mask = batch.mask.to(trainer.device)
    with torch.no_grad():
        prompt_embeds = trainer.prompt_embeds(batch.prompts)
        conditioning_latents, latent_mask = trainer.prepare_conditioning(conditioning, mask, winner.shape[1])
        winner_latents = trainer.encode_latents(winner)
        image_latents = trainer.prepare_image_latents(conditioning, winner_latents.shape[1])
    pred = trainer.branch_forward(
        trainer.policy_branch,
        winner_latents,
        conditioning_latents,
        image_latents,
        latent_mask,
        prompt_embeds,
        noise,
        timesteps,
    )
    return (pred.float() - winner_latents.float()).pow(2).mean()


def state_max_abs_diff(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    if set(a) != set(b):
        missing = sorted(set(a) - set(b))
        unexpected = sorted(set(b) - set(a))
        raise RuntimeError(f"state_dict key mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    max_diff = 0.0
    for key in a:
        av = a[key].detach().float().cpu()
        bv = b[key].detach().float().cpu()
        if av.shape != bv.shape:
            raise RuntimeError(f"state_dict shape mismatch for {key}: {tuple(av.shape)} vs {tuple(bv.shape)}")
        if av.numel():
            max_diff = max(max_diff, float((av - bv).abs().max()))
    return max_diff


def digest_delta(a: Dict[str, object], b: Dict[str, object]) -> float:
    return abs(float(a["mean"]) - float(b["mean"])) + abs(float(a["l2"]) - float(b["l2"]))


def release_trainer(trainer: VideoPainterDPOTrainer | None) -> None:
    if trainer is not None:
        del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_l0_l3(args: argparse.Namespace, formal_manifest: Path, out_dir: Path) -> Dict[str, object]:
    report: Dict[str, object] = {}
    diag = out_dir / "formal_dpo_diag.csv"
    targs = make_args(args, out_dir / "formal_run", diag, formal_manifest, frames=49, plumbing_only=False)

    trainer = VideoPainterDPOTrainer(targs)
    batch = one_batch(formal_manifest, targs)
    noise, timesteps = make_fixed_noise_timestep(trainer, batch, args.seed)

    l0_digest = branch_digest(trainer, batch, noise, timesteps)
    if not l0_digest["finite"]:
        raise RuntimeError("L0 native forward produced non-finite output")
    report["L0"] = {
        "status": "passed",
        "strict_load": "from_pretrained branch checkpoint loaded without fallback",
        "native_inference_digest": l0_digest,
        "manifest": str(formal_manifest),
        "frames": 49,
    }

    with torch.no_grad():
        loss1, diag1 = trainer.compute_losses(batch, fixed_noise=noise, fixed_timesteps=timesteps)
        loss2, diag2 = trainer.compute_losses(batch, fixed_noise=noise, fixed_timesteps=timesteps)
    l1_diff = max(abs(float(loss1.detach().cpu()) - float(loss2.detach().cpu())), abs(diag1["dpo_loss"] - diag2["dpo_loss"]))
    # BF16 attention kernels can differ at the few-e-5 scalar level even with
    # identical batch/noise/timestep; larger drift would indicate protocol
    # mismatch rather than numerical noise.
    if l1_diff > 5e-5:
        raise RuntimeError(f"L1 same-batch/noise/timestep parity failed: diff={l1_diff}")
    report["L1"] = {
        "status": "passed",
        "total_loss_1": float(loss1.detach().float().cpu()),
        "total_loss_2": float(loss2.detach().float().cpu()),
        "dpo_loss_1": diag1["dpo_loss"],
        "dpo_loss_2": diag2["dpo_loss"],
        "max_abs_diff": l1_diff,
    }

    l2_gap = abs(diag1["dpo_loss"] - math.log(2.0))
    if l2_gap > 0.02:
        raise RuntimeError(f"L2 zero-gap failed: dpo_loss={diag1['dpo_loss']} log2_gap={l2_gap}")
    report["L2"] = {
        "status": "passed",
        "dpo_loss": diag1["dpo_loss"],
        "target_log2": math.log(2.0),
        "abs_gap": l2_gap,
    }

    trainer.policy_branch.train()
    optimizer = make_vp2_optimizer(trainer.policy_branch.parameters(), targs)
    before = branch_digest(trainer, batch, noise, timesteps)
    optimizer.zero_grad(set_to_none=True)
    loss = native_policy_loss(trainer, batch, noise, timesteps)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainer.policy_branch.parameters(), targs.max_grad_norm)
    optimizer.step()
    after = branch_digest(trainer, batch, noise, timesteps)
    delta = digest_delta(before, after)
    if delta <= 0.0:
        raise RuntimeError("L3 one-step update did not change native output digest")
    trainer.save_branch_checkpoint(1, optimizer)
    ckpt = Path(targs.output_dir) / "checkpoint-1"
    saved_state = {k: v.detach().cpu() for k, v in trainer.policy_branch.state_dict().items()}
    post_digest = after
    release_trainer(trainer)
    trainer = None

    reload_args = make_args(args, out_dir / "reload_run", out_dir / "reload_dpo_diag.csv", formal_manifest, frames=49, plumbing_only=False)
    reload_args.policy_checkpoint = str(ckpt)
    reload_args.reference_checkpoint = args.reference_checkpoint
    reloaded = VideoPainterDPOTrainer(reload_args)
    loaded_state = {k: v.detach().cpu() for k, v in reloaded.policy_branch.state_dict().items()}
    max_state_diff = state_max_abs_diff(saved_state, loaded_state)
    if max_state_diff != 0.0:
        raise RuntimeError(f"L3 strict reload parameter diff not zero: {max_state_diff}")
    reload_batch = one_batch(formal_manifest, reload_args)
    reload_noise = noise.to(device=reloaded.device, dtype=reloaded.weight_dtype)
    reload_timesteps = timesteps.to(device=reloaded.device)
    reload_digest = branch_digest(reloaded, reload_batch, reload_noise, reload_timesteps)
    reload_output_delta = digest_delta(post_digest, reload_digest)
    if reload_output_delta > 1e-3:
        raise RuntimeError(f"L3 reload output drift too large: {reload_output_delta}")
    report["L3"] = {
        "status": "passed",
        "one_step_loss": float(loss.detach().float().cpu()),
        "loss_type": "formal_49f_native_policy_loss",
        "output_delta_after_update": delta,
        "checkpoint": str(ckpt),
        "state_max_abs_diff_after_reload": max_state_diff,
        "reload_output_delta": reload_output_delta,
        "reload_native_inference_digest": reload_digest,
    }
    release_trainer(reloaded)
    return report


def run_l4(args: argparse.Namespace, out_dir: Path) -> Dict[str, object]:
    manifest = Path(args.plumbing_pair_manifest)
    diag = out_dir / "plumbing_13f_dpo_diag.csv"
    targs = make_args(args, out_dir / "plumbing_13f_run", diag, manifest, frames=13, plumbing_only=True)
    targs.max_train_steps = args.l4_steps
    targs.checkpointing_steps = max(1, args.l4_steps)
    targs.checkpoints_total_limit = 2
    trainer = VideoPainterDPOTrainer(targs)
    dataset = VideoPainterPairDataset(
        str(manifest),
        height=targs.height,
        width=targs.width,
        num_frames=targs.num_frames,
        limit=1,
        plumbing_only_13f=True,
        first_frame_gt=targs.first_frame_gt,
    )
    loader = [collate_pairs([dataset[0]])]
    iterator = repeating_epoch_iterator(loader)
    optimizer = make_vp2_optimizer(trainer.policy_branch.parameters(), targs)
    diag.parent.mkdir(parents=True, exist_ok=True)
    with diag.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "dpo_loss", "grad_norm", "finite", "plumbing_only"])
        writer.writeheader()
        rows = []
        for step in range(1, args.l4_steps + 1):
            batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            loss, row = trainer.compute_losses(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"L4 non-finite loss at step {step}: {loss}")
            loss.backward()
            grad_norm = 0.0
            for p in trainer.policy_branch.parameters():
                if p.grad is not None:
                    grad_norm += float(p.grad.detach().float().norm().cpu()) ** 2
            grad_norm = math.sqrt(grad_norm)
            torch.nn.utils.clip_grad_norm_(trainer.policy_branch.parameters(), targs.max_grad_norm)
            optimizer.step()
            out_row = {
                "step": step,
                "loss": row["loss"],
                "dpo_loss": row["dpo_loss"],
                "grad_norm": grad_norm,
                "finite": True,
                "plumbing_only": "PLUMBING_ONLY_13F",
            }
            writer.writerow(out_row)
            rows.append(out_row)
    trainer.save_branch_checkpoint(args.l4_steps, optimizer)
    trainer.save_last_weights(args.l4_steps)
    ckpt = Path(targs.output_dir) / f"checkpoint-{args.l4_steps}"
    if not ckpt.exists():
        raise RuntimeError(f"L4 checkpoint missing: {ckpt}")
    report = {
        "status": "passed",
        "label": "PLUMBING_ONLY_13F",
        "steps": args.l4_steps,
        "diag_csv": str(diag),
        "last_loss": rows[-1]["loss"],
        "last_dpo_loss": rows[-1]["dpo_loss"],
        "checkpoint": str(ckpt),
        "last_weights": str(Path(targs.output_dir) / "last_weights"),
    }
    release_trainer(trainer)
    return report


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))

    locked = out_dir / "locked_official_optimizer_scheduler.json"
    cfg = write_locked_config(Path(args.official_train_file), locked)
    formal_manifest = make_formal_davis_pair(args, out_dir / "gate_data")

    report: Dict[str, object] = {
        "status": "running",
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "official_optimizer_scheduler": asdict(cfg),
        "locked_official_config_json": str(locked),
        "formal_manifest": str(formal_manifest),
        "plumbing_manifest": args.plumbing_pair_manifest,
    }
    write_json(out_dir / "l0_l4_report.json", report)

    try:
        l0_l3 = run_l0_l3(args, formal_manifest, out_dir)
        report.update(l0_l3)
        write_json(out_dir / "l0_l4_report.json", report)

        report["L4"] = run_l4(args, out_dir)
        report["status"] = "passed"
        write_json(out_dir / "l0_l4_report.json", report)
    except Exception as exc:  # noqa: BLE001 - gate runner must preserve blocker details.
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        write_json(out_dir / "l0_l4_report.json", report)
        raise

    lines = [
        "# Exp26 VideoPainter v2 L0-L4 Gate Report",
        "",
        f"status: {report['status']}",
        f"pid: {report['pid']}",
        f"cuda_visible_devices: {report['cuda_visible_devices']}",
        f"formal_manifest: {formal_manifest}",
        f"plumbing_manifest: {args.plumbing_pair_manifest}",
        "",
        "## Gates",
    ]
    for gate in ("L0", "L1", "L2", "L3", "L4"):
        gate_obj = report.get(gate, {})
        lines.append(f"- {gate}: {gate_obj.get('status', 'missing') if isinstance(gate_obj, dict) else gate_obj}")
    lines.extend([
        "",
        "L4 is explicitly marked `PLUMBING_ONLY_13F`; it is not a formal 49-frame result.",
        "",
    ])
    write_text(out_dir / "l0_l4_report.md", "\n".join(lines))


if __name__ == "__main__":
    main()
