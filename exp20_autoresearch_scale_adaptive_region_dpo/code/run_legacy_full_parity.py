#!/usr/bin/env python
"""Exp20 legacy_exact parity harness against Exp11 copied trainer code.

This script intentionally compares the old Exp11 implementation and the
isolated Exp20 copy on the same real manifest mask and identical prediction
tensors. The heavier real-model smoke is handled by the subsequent smoke gate.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_first_row_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    raise RuntimeError(f"empty manifest: {path}")


def load_brushnet_masks(row: dict, nframes: int, latent_hw: tuple[int, int]) -> torch.Tensor:
    mask_dir = Path(row["mask_path"]) / str(row.get("mask_id", "mask_000")) / "mask"
    paths = sorted(mask_dir.glob("*.png"))[:nframes]
    if len(paths) < nframes:
        raise RuntimeError(f"expected {nframes} mask frames under {mask_dir}, found {len(paths)}")
    masks = []
    for p in paths:
        arr = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        hole = (arr > 0.5).astype(np.float32)
        brushnet_mask = 1.0 - hole
        masks.append(torch.from_numpy(brushnet_mask))
    image = torch.stack(masks, dim=0)[None, :, None]
    latent = F.interpolate(image, size=(1, latent_hw[0], latent_hw[1]), mode="nearest")
    return latent


def tensor_max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def scalar_abs(a, b) -> float:
    return abs(float(a) - float(b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-md", default="reports/exp20_legacy_full_parity.md")
    parser.add_argument("--output-csv", default="reports/exp20_legacy_full_parity.csv")
    parser.add_argument("--registry-json", default="experiment_registry/exp20_autoresearch_scale_adaptive_region_dpo/parity.json")
    parser.add_argument("--nframes", type=int, default=16)
    parser.add_argument("--latent-height", type=int, default=40)
    parser.add_argument("--latent-width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260619)
    args = parser.parse_args()

    os.environ["BOUNDARY_MODE"] = "outer"
    exp11 = import_module_from_path("exp11_stage1_parity", PROJECT_ROOT / "exp11_region_boundary_ablation/code/train_stage1.py")
    exp20 = import_module_from_path(
        "exp20_stage1_parity",
        PROJECT_ROOT / "exp20_autoresearch_scale_adaptive_region_dpo/code/train_exp20_stage1.py",
    )

    row = load_first_row_manifest(Path(args.manifest))
    masks = load_brushnet_masks(row, args.nframes, (args.latent_height, args.latent_width))

    old_weight, old_stats = exp11.build_region_loss_weight_map(
        masks,
        mask_region_weight=1.0,
        boundary_region_weight=0.75,
        outside_region_weight=0.05,
    )

    exp20_args = argparse.Namespace(
        legacy_exact=True,
        radius_mode="legacy_latent_exact",
        radius_value=0.0,
        adaptive_k=1.0,
        mask_region_weight=1.0,
        boundary_region_weight=0.75,
        outside_region_weight=0.05,
    )
    new_weight, new_stats = exp20.build_exp20_loss_weight_map(
        masks,
        (args.latent_height, args.latent_width),
        exp20_args,
    )

    torch.manual_seed(args.seed)
    shape = (2 * args.nframes, 4, args.latent_height, args.latent_width)
    model_pred_old = torch.randn(shape, dtype=torch.float32, requires_grad=True)
    ref_pred = torch.randn(shape, dtype=torch.float32)
    noise = torch.randn((args.nframes, 4, args.latent_height, args.latent_width), dtype=torch.float32)

    model_pred_new = model_pred_old.detach().clone().requires_grad_(True)
    old_loss, old_diag = exp11.compute_dpo_loss(
        model_pred_old,
        ref_pred,
        noise,
        loss_weight_map=old_weight,
        loss_region_mode="region",
        region_stats=dict(old_stats),
        gap_normalization="log_ratio",
        gap_eps=1e-6,
        lose_gap_clip_tau="1.0",
        beta_dpo=10,
        sft_reg_weight=0.0,
        lose_gap_weight=0.25,
        winner_abs_reg_weight=0.05,
        winner_gap_reg_weight=1.0,
        winner_gap_reg_margin=0.0,
        winner_gap_reg_mode="relu",
        nframes=args.nframes,
    )
    new_loss, new_diag = exp20.compute_dpo_loss(
        model_pred_new,
        ref_pred,
        noise,
        loss_weight_map=new_weight,
        loss_region_mode="region",
        region_stats=dict(new_stats),
        gap_normalization="log_ratio",
        gap_eps=1e-6,
        lose_gap_clip_tau="1.0",
        beta_dpo=10,
        sft_reg_weight=0.0,
        lose_gap_weight=0.25,
        winner_abs_reg_weight=0.05,
        winner_gap_reg_weight=1.0,
        winner_gap_reg_margin=0.0,
        winner_gap_reg_mode="relu",
        nframes=args.nframes,
    )
    old_loss.backward()
    new_loss.backward()
    old_grad = model_pred_old.grad.detach().float()
    new_grad = model_pred_new.grad.detach().float()
    grad_cos = torch.nn.functional.cosine_similarity(old_grad.flatten(), new_grad.flatten(), dim=0).item()
    grad_rel_l2 = (old_grad - new_grad).norm().item() / max(old_grad.norm().item(), 1e-12)

    scalar_keys = [
        ("model_losses_w", "mse_w"),
        ("model_losses_l", "mse_l"),
        ("ref_losses_w", "ref_mse_w"),
        ("ref_losses_l", "ref_mse_l"),
        ("raw_win_gap", "raw_win_gap"),
        ("raw_lose_gap", "raw_lose_gap"),
        ("norm_win_gap", "norm_win_gap"),
        ("norm_lose_gap", "norm_lose_gap"),
        ("norm_lose_gap_clipped", "norm_lose_gap_clipped"),
        ("inside_term", "inside_term_mean"),
        ("dpo_loss", "dpo_loss"),
        ("winner_abs_reg", "winner_abs_reg"),
        ("winner_gap_reg", "winner_gap_reg"),
        ("total_loss", "total_loss"),
        ("implicit_acc", "implicit_acc"),
        ("loser_degrade_ratio", "loser_degrade_ratio"),
    ]
    rows = []
    for label, key in scalar_keys:
        rows.append(
            {
                "metric": label,
                "old": old_diag.get(key, old_loss.item() if key == "total_loss" else ""),
                "new": new_diag.get(key, new_loss.item() if key == "total_loss" else ""),
                "abs_diff": scalar_abs(old_diag.get(key, old_loss.item()), new_diag.get(key, new_loss.item())),
            }
        )
    rows.extend(
        [
            {"metric": "weight_map_max_abs_diff", "old": "", "new": "", "abs_diff": tensor_max_abs(old_weight, new_weight)},
            {"metric": "prediction_grad_cosine", "old": "", "new": "", "abs_diff": 1.0 - grad_cos},
            {"metric": "prediction_grad_relative_l2", "old": "", "new": "", "abs_diff": grad_rel_l2},
        ]
    )
    passed = (
        tensor_max_abs(old_weight, new_weight) == 0.0
        and max(float(r["abs_diff"]) for r in rows if r["metric"] not in {"prediction_grad_cosine", "prediction_grad_relative_l2"}) <= 1e-7
        and grad_cos >= 0.999999
        and grad_rel_l2 <= 1e-5
    )

    csv_path = PROJECT_ROOT / args.output_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "old", "new", "abs_diff"])
        writer.writeheader()
        writer.writerows(rows)

    md = [
        "# Exp20 Legacy Full Parity",
        "",
        f"- manifest: `{args.manifest}`",
        f"- sample_id: `{row.get('sample_id')}`",
        f"- mask_id: `{row.get('mask_id')}`",
        f"- status: `{'LEGACY_FULL_PARITY_PASSED' if passed else 'LEGACY_FULL_PARITY_FAILED'}`",
        "",
        "## Summary",
        "",
        f"- legacy map max_abs_diff: `{tensor_max_abs(old_weight, new_weight):.12g}`",
        f"- prediction-grad cosine: `{grad_cos:.12g}`",
        f"- prediction-grad relative L2: `{grad_rel_l2:.12g}`",
        "",
        "Model-parameter backward and one-step optimizer delta parity are deferred to the real 10-step smoke because this lightweight harness does not instantiate two full training models simultaneously.",
        "",
        "## Scalar Parity",
        "",
        "| metric | old | new | abs_diff |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        md.append(f"| {r['metric']} | {r['old']} | {r['new']} | {r['abs_diff']} |")
    (PROJECT_ROOT / args.output_md).write_text("\n".join(md) + "\n", encoding="utf-8")

    reg_path = PROJECT_ROOT / args.registry_json
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(
        json.dumps(
            {
                "status": "LEGACY_FULL_PARITY_PASSED" if passed else "LEGACY_FULL_PARITY_FAILED",
                "manifest": args.manifest,
                "sample_id": row.get("sample_id"),
                "mask_id": row.get("mask_id"),
                "map_max_abs_diff": tensor_max_abs(old_weight, new_weight),
                "prediction_grad_cosine": grad_cos,
                "prediction_grad_relative_l2": grad_rel_l2,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"status={'LEGACY_FULL_PARITY_PASSED' if passed else 'LEGACY_FULL_PARITY_FAILED'}")
    print(f"report={PROJECT_ROOT / args.output_md}")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
