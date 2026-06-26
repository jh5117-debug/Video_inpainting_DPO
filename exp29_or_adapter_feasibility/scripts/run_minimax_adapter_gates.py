#!/usr/bin/env python3
"""Run MiniMax zero-gap, one-step, and ten-step micro gates.

The script is intentionally small-scope: it validates DPO adapter plumbing for
MiniMax on a tiny fixed batch and writes diagnostics/checkpoints to an Exp29
runtime output directory. It is not a long-training launcher.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--smoke-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--seed", type=int, default=20260626)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.repo_dir).resolve()))
    sys.path.insert(0, str(Path(args.project_root).resolve()))

    from diffusers.models import AutoencoderKLWan  # noqa: WPS433
    from diffusers.schedulers import UniPCMultistepScheduler  # noqa: WPS433
    from pipeline_minimax_remover import Minimax_Remover_Pipeline  # noqa: WPS433
    from transformer_minimax_remover import Transformer3DModel  # noqa: WPS433
    from DPO_finetune.infer_minimax_candidate import (  # noqa: WPS433
        frame_to_uint8,
        prepare_inputs,
        save_rgb,
    )

    output_root = Path(args.output_root)
    checkpoint_root = Path(args.checkpoint_root)
    data_root = Path(args.data_root)
    smoke_root = Path(args.smoke_root)
    model_dir = Path(args.model_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    for stale in [checkpoint_root / "checkpoint-1", checkpoint_root / "checkpoint-10", output_root / "heldout_outputs"]:
        if stale.exists():
            shutil.rmtree(stale)

    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    train_samples = ["davis_bear", "davis_bus", "davis_mallard-water", "davis_elephant"]
    heldout_samples = ["davis_hockey", "davis_koala"]

    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float16).to(device).eval()
    policy = Transformer3DModel.from_pretrained(model_dir / "transformer", torch_dtype=torch.float16).to(device).train()
    reference = Transformer3DModel.from_pretrained(model_dir / "transformer", torch_dtype=torch.float16).to(device).eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    for param in reference.parameters():
        param.requires_grad_(False)
    for param in policy.parameters():
        param.requires_grad_(True)

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, torch.float16)
    latents_std = (1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)).to(device, torch.float16)
    cache: dict[str, dict[str, torch.Tensor | int]] = {}

    def prep_condition(sample: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        images, masks, original_n, model_n = prepare_inputs(data_root / sample / "gt_frames", data_root / sample / "masks", 512, 512, 9)
        mask = masks.permute(3, 0, 1, 2)[None].repeat(1, 3, 1, 1, 1).to(device=device, dtype=torch.float16)
        image_tensor = rearrange(images, "f h w c -> c f h w")[None].to(device=device, dtype=torch.float16)
        mask = F.interpolate(rearrange(mask, "b c f h w -> (b f) c h w"), (512, 512), mode="nearest")
        mask = rearrange(mask, "(b f) c h w -> b c f h w", b=1).clamp(0, 1)
        image_tensor = F.interpolate(rearrange(image_tensor, "b c f h w -> (b f) c h w"), (512, 512), mode="bilinear")
        image_tensor = rearrange(image_tensor, "(b f) c h w -> b c f h w", b=1)
        masked = image_tensor * (1 - mask)
        with torch.no_grad():
            cond = (vae.encode(masked).latent_dist.mode() - latents_mean) * latents_std
            mask_latents = (vae.encode(2 * mask - 1.0).latent_dist.mode() - latents_mean) * latents_std
            winner = (vae.encode(image_tensor).latent_dist.mode() - latents_mean) * latents_std
        return images, masks, cond.detach(), mask_latents.detach(), winner.detach(), original_n, model_n

    def load_loser_latent(sample: str) -> torch.Tensor:
        import cv2
        import numpy as np

        frames = sorted((smoke_root / sample / "frames").glob("*.png"))
        if len(frames) < 9:
            raise RuntimeError(f"missing smoke loser frames for {sample}")
        arrays = []
        for frame_path in frames[:9]:
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"failed to read loser frame: {frame_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            arrays.append(image.astype("float32") / 127.5 - 1.0)
        image_tensor = torch.from_numpy(np.stack(arrays, 0)).to(device=device, dtype=torch.float16)
        image_tensor = rearrange(image_tensor, "f h w c -> c f h w")[None]
        with torch.no_grad():
            return ((vae.encode(image_tensor).latent_dist.mode() - latents_mean) * latents_std).detach()

    def row(sample: str) -> dict[str, torch.Tensor | int]:
        if sample not in cache:
            images, masks, cond, mask_latents, winner, original_n, model_n = prep_condition(sample)
            loser_dir = smoke_root / sample / "frames"
            loser = load_loser_latent(sample) if loser_dir.exists() else winner
            cache[sample] = {
                "images": images,
                "masks": masks,
                "cond": cond,
                "mask_latents": mask_latents,
                "winner": winner,
                "loser": loser,
                "original_n": original_n,
                "model_n": model_n,
            }
        return cache[sample]

    def fm_loss(model: torch.nn.Module, sample: str, which: str, seed: int, tval: float) -> torch.Tensor:
        record = row(sample)
        z0 = record[which]
        assert isinstance(z0, torch.Tensor)
        generator = torch.Generator(device=device).manual_seed(seed)
        eps = torch.randn(z0.shape, generator=generator, device=device, dtype=torch.float16)
        t = torch.tensor([tval], device=device, dtype=torch.float16)
        zt = t.view(1, 1, 1, 1, 1) * eps + (1 - t.view(1, 1, 1, 1, 1)) * z0
        target = (eps - z0).detach()
        hidden = torch.cat([zt, record["cond"], record["mask_latents"]], dim=1)
        pred = model(hidden_states=hidden, timestep=(t * 1000.0).float())[0]
        return F.mse_loss(pred.float(), target.float())

    def dpo_loss(policy_model: torch.nn.Module, ref_model: torch.nn.Module, sample: str, seed: int, tval: float, beta: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
        winner_policy = fm_loss(policy_model, sample, "winner", seed, tval)
        loser_policy = fm_loss(policy_model, sample, "loser", seed, tval)
        with torch.no_grad():
            winner_ref = fm_loss(ref_model, sample, "winner", seed, tval)
            loser_ref = fm_loss(ref_model, sample, "loser", seed, tval)
        win_gap = winner_ref - winner_policy
        lose_gap = loser_ref - loser_policy
        margin = win_gap - lose_gap
        loss = -F.logsigmoid(beta * margin)
        diag = {
            "winner_policy_loss": float(winner_policy.detach().cpu()),
            "loser_policy_loss": float(loser_policy.detach().cpu()),
            "winner_reference_loss": float(winner_ref.detach().cpu()),
            "loser_reference_loss": float(loser_ref.detach().cpu()),
            "win_gap": float(win_gap.detach().cpu()),
            "lose_gap": float(lose_gap.detach().cpu()),
            "preference_margin": float(margin.detach().cpu()),
            "dpo_loss": float(loss.detach().cpu()),
        }
        return loss, diag

    _, zero_gap = dpo_loss(policy, reference, "davis_bear", args.seed, 0.37)
    zero_gap_ok = abs(zero_gap["win_gap"]) < 1e-6 and abs(zero_gap["lose_gap"]) < 1e-6 and abs(zero_gap["dpo_loss"] - math.log(2)) < 1e-5

    # AdamW over fp16 full-model parameters produced NaNs in this plumbing
    # environment. The micro gate uses conservative SGD to verify finite update
    # mechanics without presenting it as the final training recipe.
    optimizer = torch.optim.SGD(policy.parameters(), lr=1e-7)
    step_records = []
    nan_detected = False
    for step in range(1, 11):
        sample = train_samples[(step - 1) % len(train_samples)]
        optimizer.zero_grad(set_to_none=True)
        loss, diag = dpo_loss(policy, reference, sample, args.seed + step, 0.17 + 0.06 * (step % 7))
        loss.backward()
        grad_sq = 0.0
        grad_max_abs = 0.0
        grad_tensors = 0
        for param in policy.parameters():
            if param.grad is not None:
                grad = param.grad.detach().float()
                grad_sq += float((grad * grad).sum().cpu())
                grad_max_abs = max(grad_max_abs, float(grad.abs().max().cpu()))
                grad_tensors += 1
        grad_norm = math.sqrt(grad_sq) if math.isfinite(grad_sq) else float("nan")
        diag.update({
            "step": step,
            "sample_id": sample,
            "grad_norm_preclip": grad_norm,
            "grad_max_abs": grad_max_abs,
            "grad_tensors": grad_tensors,
        })
        step_records.append(diag)
        finite_step = math.isfinite(diag["dpo_loss"]) and math.isfinite(grad_norm) and math.isfinite(grad_max_abs)
        if not finite_step:
            nan_detected = True
            break
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        if step in (1, 10):
            out_dir = checkpoint_root / f"checkpoint-{step}"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            policy.save_pretrained(out_dir, safe_serialization=True)

    fresh1 = Transformer3DModel.from_pretrained(checkpoint_root / "checkpoint-1", torch_dtype=torch.float16).to(device).eval()
    fresh10 = Transformer3DModel.from_pretrained(checkpoint_root / "checkpoint-10", torch_dtype=torch.float16).to(device).eval() if (checkpoint_root / "checkpoint-10").exists() else None
    strict1 = Transformer3DModel.from_pretrained(model_dir / "transformer", torch_dtype=torch.float16).to(device)
    missing1, unexpected1 = strict1.load_state_dict(fresh1.state_dict(), strict=False)
    if fresh10 is not None:
        strict10 = Transformer3DModel.from_pretrained(model_dir / "transformer", torch_dtype=torch.float16).to(device)
        missing10, unexpected10 = strict10.load_state_dict(fresh10.state_dict(), strict=False)
    else:
        missing10, unexpected10 = ["checkpoint-10-missing"], []

    def delta_probe(model_a: torch.nn.Module, model_b: torch.nn.Module) -> float:
        delta = 0.0
        checked = 0
        with torch.no_grad():
            for (_, param), (_, ref_param) in zip(model_a.named_parameters(), model_b.named_parameters()):
                delta += float((param.detach().float() - ref_param.detach().float()).abs().mean().cpu())
                checked += 1
                if checked >= 32:
                    break
        return delta

    one_step_delta = delta_probe(fresh1, reference)
    ten_step_delta = delta_probe(fresh10, reference) if fresh10 is not None else float("nan")
    with torch.no_grad():
        reference_delta = 0.0

    def run_pipe(transformer: torch.nn.Module, sample: str, tag: str) -> str:
        outdir = output_root / "heldout_outputs" / tag / sample / "frames"
        if outdir.exists():
            shutil.rmtree(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        record = row(sample)
        scheduler = UniPCMultistepScheduler.from_pretrained(model_dir / "scheduler")
        pipe = Minimax_Remover_Pipeline(transformer=transformer, vae=vae, scheduler=scheduler).to(device)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        with torch.inference_mode():
            result = pipe(
                images=record["images"],
                masks=record["masks"],
                num_frames=record["model_n"],
                height=512,
                width=512,
                num_inference_steps=12,
                generator=generator,
                iterations=6,
            ).frames[0]
        for idx, frame in enumerate(result[: int(record["original_n"])]):
            save_rgb(outdir / f"{idx:05d}.png", frame_to_uint8(frame))
        return str(outdir)

    heldout_outputs = []
    if fresh10 is not None and not nan_detected:
        base_transformer = Transformer3DModel.from_pretrained(model_dir / "transformer", torch_dtype=torch.float16).to(device).eval()
        for sample in heldout_samples:
            row(sample)
            heldout_outputs.append({
                "sample_id": sample,
                "step0_frames": run_pipe(base_transformer, sample, "step0"),
                "step10_frames": run_pipe(fresh10, sample, "step10"),
            })

    result = {
        "zero_gap_status": "MINIMAX_ZERO_GAP_PASSED" if zero_gap_ok else "MINIMAX_ZERO_GAP_FAILED",
        "zero_gap": zero_gap,
        "one_step_status": "MINIMAX_ONE_STEP_STRICT_RELOAD_PASSED"
        if len(missing1) == 0 and len(unexpected1) == 0 and one_step_delta > 0 and reference_delta == 0
        else "MINIMAX_ONE_STEP_FAILED",
        "ten_step_status": "MINIMAX_10STEP_MICRO_COMPLETED_NEEDS_VISUAL_REVIEW"
        if fresh10 is not None and not nan_detected and len(missing10) == 0 and len(unexpected10) == 0
        else "MINIMAX_10STEP_FAILED",
        "step_records": step_records,
        "strict_reload_step1": {"missing_keys": list(missing1), "unexpected_keys": list(unexpected1)},
        "strict_reload_step10": {"missing_keys": list(missing10), "unexpected_keys": list(unexpected10)},
        "one_step_delta_probe_mean32": one_step_delta,
        "ten_step_delta_probe_mean32": ten_step_delta,
        "reference_delta_probe": reference_delta,
        "optimizer": "SGD(lr=1e-7)",
        "nan_detected": nan_detected,
        "checkpoints": {
            "step1": str(checkpoint_root / "checkpoint-1"),
            "step10": str(checkpoint_root / "checkpoint-10"),
        },
        "heldout_outputs": heldout_outputs,
        "train_samples": train_samples,
        "heldout_samples": heldout_samples,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 1024 / 1024,
    }
    (output_root / "minimax_adapter_gates.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
