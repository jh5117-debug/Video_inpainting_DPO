#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from exp50_pai_void_adapter_feasibility.void_preference_wrapper.void_sft_wrapper import (
    VoidPaths,
    load_components,
    load_micro_row,
    make_target_pack,
    sft_forward_loss,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--void-weights", required=True)
    ap.add_argument("--manifest", default="manifests/exp50_void_adapter_train4.jsonl")
    ap.add_argument("--row-index", type=int, default=0)
    ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--width", type=int, default=672)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--timestep", type=int, default=500)
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    paths = VoidPaths(
        repo=Path(args.repo),
        base_model=Path(args.base_model),
        void_weights=Path(args.void_weights),
        transformer_path=Path(args.void_weights) / "void_pass1.safetensors",
    )
    row = load_micro_row(args.manifest, args.row_index)
    components = load_components(paths, device=device, dtype=dtype, load_transformer=True)
    frames = args.frames
    temporal_compression = int(getattr(components["vae"].config, "temporal_compression_ratio", 4))
    patch_size_t = getattr(components["transformer"].config, "patch_size_t", None)
    if patch_size_t is not None:
        latent_len = (frames - 1) // temporal_compression + 1
        remainder = latent_len % int(patch_size_t)
        if remainder:
            frames = max(1, frames - remainder * temporal_compression)
    pack = make_target_pack(row, components, device, dtype, frames=frames, size=(args.width, args.height), seed=args.seed, timestep=args.timestep)
    result = sft_forward_loss(components, pack, height=args.height, width=args.width)
    summary = {
        "status": "VOID_SFT_FORWARD_PARITY_EXPLAINED" if result["finite"] else "VOID_SFT_FORWARD_PARITY_BLOCKED",
        "sample_id": row.get("sample_id"),
        "device": str(device),
        "dtype": str(dtype),
        "requested_frames": args.frames,
        "frames": frames,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "timestep": args.timestep,
        "prediction_type": pack["prediction_type"],
        "sft_loss": result["loss"],
        "loss_finite": result["finite"],
        "missing_keys_count": len(components["missing_keys"]),
        "unexpected_keys_count": len(components["unexpected_keys"]),
        "latent_shape": list(pack["latent_shape"]),
        "inpaint_shape": list(pack["inpaint_shape"]),
        "target_shape": list(result["target_shape"]),
        "noise_pred_shape": list(result["noise_pred_shape"]),
        "official_equivalence": "Code mirrors train.py VAE encode, inpaint_latents, scheduler add_noise, epsilon/v_prediction target, transformer forward, and MSE loss. Direct train.py helper is not callable without entering its training loop/Accelerator side effects.",
        "training_run": False,
        "optimizer_step": False,
        "vor_eval_used": False,
        "hard_comp_used": False,
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with Path(args.output_csv).open("w", newline="") as f:
        fields = ["sample_id", "status", "prediction_type", "sft_loss", "loss_finite", "latent_shape", "inpaint_shape", "noise_pred_shape", "missing_keys_count", "unexpected_keys_count"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({k: summary[k] for k in fields})
    md = f"""# Exp50 VOID SFT Forward Parity

Status: `{summary['status']}`

## Run

- Sample: `{summary['sample_id']}`
- Device: `{summary['device']}`
- dtype: `{summary['dtype']}`
- Requested frames: {args.frames}
- Effective frames after official patch-size truncation: {frames}
- Resolution: {args.width}x{args.height}
- Timestep: {args.timestep}
- Seed: {args.seed}
- Prediction type: `{summary['prediction_type']}`
- SFT loss: {summary['sft_loss']}
- Loss finite: {summary['loss_finite']}

## Equivalence

The wrapper mirrors the official `scripts/cogvideox_fun/train.py` path for the core forward: VAE encode target video, construct VAE-mask inpaint latents from condition and quadmask, sample scheduler noise/timestep, build the scheduler target (`epsilon` or `v_prediction`), run `CogVideoXTransformer3DModel`, and compute mean MSE. The official helper is not directly callable without entering the full Accelerator training loop, so this is marked `EXPLAINED` rather than strict byte-for-byte parity.

## Safety

No training, optimizer step, DPO, VOR-Eval, hard comp, official source modification, or deepspeed install was performed.
"""
    Path(args.output_md).write_text(md)


if __name__ == "__main__":
    main()
