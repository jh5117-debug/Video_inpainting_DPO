#!/usr/bin/env python3
"""Mine bounded bad-noise / hard-timestep states for Exp35 MiniMax rescue."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", required=True)
    p.add_argument("--project-root", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--heldout-manifest", required=True)
    p.add_argument("--manifests-root", required=True)
    p.add_argument("--reports-root", required=True)
    p.add_argument("--seed", type=int, default=20260627)
    p.add_argument("--noise-count", type=int, default=4)
    p.add_argument("--timesteps", default="0.15,0.35,0.55,0.75")
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--heartbeat", default="")
    return p.parse_args()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def heartbeat(path: Path | None, text: str) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{time.time():.0f}\t{text}\n", encoding="utf-8")


def torch_dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "float16" else torch.bfloat16


def latent_region_masks(record: dict[str, object], z0: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    mask_frames = np.stack(record["mask_frames_uint8"], axis=0)
    mask = torch.from_numpy((mask_frames > 20).astype(np.float32))[None, None].to(device=device)
    mask = F.interpolate(mask, size=tuple(z0.shape[2:]), mode="nearest").clamp(0, 1)
    dil = F.max_pool3d(mask, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
    ero = 1.0 - F.max_pool3d(1.0 - mask, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
    boundary = (dil - ero).clamp(0, 1)
    outside = (1.0 - dil).clamp(0, 1)
    return {
        "mask": mask.bool().expand_as(z0),
        "boundary": boundary.bool().expand_as(z0),
        "outside": outside.bool().expand_as(z0),
    }


def mean_region(sq: torch.Tensor, region: torch.Tensor) -> float:
    denom = int(region.sum().detach().cpu())
    if denom <= 0:
        return float(sq.mean().detach().cpu())
    return float(sq[region].mean().detach().cpu())


@torch.no_grad()
def target_residual(
    model: torch.nn.Module,
    cache,
    row: dict[str, object],
    which: str,
    seed: int,
    tval: float,
) -> dict[str, float]:
    record = cache.row(row)
    z0 = record[which]
    assert isinstance(z0, torch.Tensor)
    gen = torch.Generator(device=cache.device).manual_seed(seed)
    eps = torch.randn(z0.shape, generator=gen, device=cache.device, dtype=cache.dtype)
    t = torch.tensor([tval], device=cache.device, dtype=cache.dtype)
    t_view = t.view(1, 1, 1, 1, 1)
    zt = t_view * eps + (1 - t_view) * z0
    target = (eps - z0).detach()
    hidden = torch.cat([zt, record["cond"], record["mask_latents"]], dim=1)
    pred = model(hidden_states=hidden, timestep=(t * 1000.0).float())[0]
    sq = (pred.float() - target.float()).pow(2)
    regions = latent_region_masks(record, z0, cache.device)
    return {
        "full_residual": float(sq.mean().detach().cpu()),
        "mask_residual": mean_region(sq, regions["mask"]),
        "boundary_residual": mean_region(sq, regions["boundary"]),
        "outside_residual": mean_region(sq, regions["outside"]),
        "pred_norm": float(pred.float().norm().detach().cpu()),
        "target_norm": float(target.float().norm().detach().cpu()),
        "noise_norm": float(eps.float().norm().detach().cpu()),
        "z0_norm": float(z0.float().norm().detach().cpu()),
    }


def choose_state(states: list[dict[str, object]], score_key: str, outside_key: str) -> dict[str, object]:
    outside_values = np.array([float(s[outside_key]) for s in states], dtype=np.float64)
    outside_values = np.nan_to_num(outside_values, nan=float(np.nanmax(outside_values[np.isfinite(outside_values)])) if np.isfinite(outside_values).any() else 0.0)
    sane_cutoff = float(np.quantile(outside_values, 0.75))
    sane = [s for s in states if math.isfinite(float(s[outside_key])) and float(s[outside_key]) <= sane_cutoff]
    pool = sane or states
    return max(pool, key=lambda s: float(s[score_key]))


def finite_mean(values: list[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def finite_min(values: list[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.min(arr))


def finite_max(values: list[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.max(arr))


def summarize_row(row: dict[str, object], states: list[dict[str, object]]) -> dict[str, object]:
    for s in states:
        s["score_A_loser_local"] = float(s["loser_mask_residual"]) + 0.5 * float(s["loser_boundary_residual"])
        s["score_B_preference_violation"] = float(s["winner_mask_residual"]) - float(s["loser_mask_residual"])
        s["score_C_winner_risk"] = float(s["winner_mask_residual"]) + 0.5 * float(s["winner_boundary_residual"])
        s["outside_sanity_residual"] = max(float(s["winner_outside_residual"]), float(s["loser_outside_residual"]))
        s["winner_advantage_mask"] = float(s["loser_mask_residual"]) - float(s["winner_mask_residual"])
    state_a = choose_state(states, "score_A_loser_local", "outside_sanity_residual")
    state_b = choose_state(states, "score_B_preference_violation", "outside_sanity_residual")
    state_c = choose_state(states, "score_C_winner_risk", "outside_sanity_residual")
    return {
        "sample_id": row.get("sample_id"),
        "source_group": row.get("source_group", row.get("scene_group", "")),
        "condition_path": row.get("condition_path"),
        "winner_path": row.get("winner_path"),
        "loser_path": row.get("loser_path"),
        "mask_path": row.get("mask_path"),
        "num_candidate_states": len(states),
        "selection_policy": {
            "H0": "fixed hard_state_A per row",
            "H1": "online K=4 bad-noise sampler per batch, choose worst valid state",
            "H2": "mixed 50% random state + 50% hard_state_A",
        },
        "hard_state_A": state_a,
        "hard_state_B": state_b,
        "hard_state_C": state_c,
        "state_summary": {
            "winner_mask_residual_mean": finite_mean([float(s["winner_mask_residual"]) for s in states]),
            "loser_mask_residual_mean": finite_mean([float(s["loser_mask_residual"]) for s in states]),
            "outside_sanity_residual_mean": finite_mean([float(s["outside_sanity_residual"]) for s in states]),
            "winner_advantage_mask_mean": finite_mean([float(s["winner_advantage_mask"]) for s in states]),
            "winner_advantage_mask_min": finite_min([float(s["winner_advantage_mask"]) for s in states]),
            "winner_advantage_mask_max": finite_max([float(s["winner_advantage_mask"]) for s in states]),
        },
    }


def mine_split(
    model: torch.nn.Module,
    cache,
    rows: list[dict[str, object]],
    split: str,
    noise_count: int,
    timesteps: list[float],
    base_seed: int,
    hb: Path | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    manifest_rows: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []
    for row_idx, row in enumerate(rows):
        sample_id = str(row.get("sample_id", f"{split}_{row_idx:04d}"))
        heartbeat(hb, f"{split}:{row_idx + 1}/{len(rows)}:{sample_id}")
        states: list[dict[str, object]] = []
        for noise_idx in range(noise_count):
            noise_seed = base_seed + row_idx * 1009 + noise_idx * 9173
            for timestep_idx, tval in enumerate(timesteps):
                winner = target_residual(model, cache, row, "winner", noise_seed, tval)
                loser = target_residual(model, cache, row, "loser", noise_seed, tval)
                state = {
                    "split": split,
                    "sample_id": sample_id,
                    "source_group": row.get("source_group", row.get("scene_group", "")),
                    "noise_index": noise_idx,
                    "timestep_index": timestep_idx,
                    "noise_seed": noise_seed,
                    "t": tval,
                }
                for key, value in winner.items():
                    state[f"winner_{key}"] = value
                for key, value in loser.items():
                    state[f"loser_{key}"] = value
                states.append(state)
                csv_rows.append(dict(state))
        manifest_rows.append({"split": split, **summarize_row(row, states)})
    return manifest_rows, csv_rows


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.repo_dir).resolve()))
    sys.path.insert(0, str(Path(args.project_root).resolve()))
    from diffusers.models import AutoencoderKLWan  # noqa: WPS433
    from transformer_minimax_remover import Transformer3DModel  # noqa: WPS433
    from exp30_vor_or_multimodel_minimax.scripts.run_minimax_gate64_adapter_gate_v3 import (  # noqa: WPS433
        BatchCache,
        read_jsonl,
    )

    timesteps = [float(x) for x in args.timesteps.split(",") if x.strip()]
    if len(timesteps) != 4 or args.noise_count != 4:
        raise ValueError("Exp35 bad-noise miner is preregistered for K_noise=4 and K_timestep=4")
    manifests_root = Path(args.manifests_root)
    reports_root = Path(args.reports_root)
    manifests_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    hb = Path(args.heartbeat) if args.heartbeat else reports_root / "exp35_bad_noise_miner.heartbeat"
    dtype = torch_dtype(args.dtype)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir)

    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=dtype).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    model = Transformer3DModel.from_pretrained(model_dir / "transformer", torch_dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    latents_std = (1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)).to(device, dtype)
    cache = BatchCache(vae, latents_mean, latents_std, device, dtype)

    train_rows = read_jsonl(Path(args.train_manifest))
    heldout_rows = read_jsonl(Path(args.heldout_manifest))
    train_manifest, train_csv = mine_split(
        model, cache, train_rows, "train32", args.noise_count, timesteps, args.seed, hb
    )
    heldout_manifest, heldout_csv = mine_split(
        model, cache, heldout_rows, "heldout16", args.noise_count, timesteps, args.seed + 500_000, hb
    )

    train_path = manifests_root / "train32_bad_noise_states.jsonl"
    heldout_path = manifests_root / "heldout16_eval_states.jsonl"
    write_jsonl(train_path, train_manifest)
    write_jsonl(heldout_path, heldout_manifest)
    all_csv = train_csv + heldout_csv
    csv_path = reports_root / "exp35_minimax_bad_noise_miner.csv"
    write_csv(csv_path, all_csv)

    summary = {
        "status": "MINIMAX_BAD_NOISE_STATES_READY",
        "training_launched": False,
        "model_update": False,
        "device": str(device),
        "dtype": args.dtype,
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
        "noise_count": args.noise_count,
        "timesteps": timesteps,
        "candidate_states_per_row": args.noise_count * len(timesteps),
        "train_manifest": str(train_path),
        "train_manifest_sha256": sha256_file(train_path),
        "heldout_manifest": str(heldout_path),
        "heldout_manifest_sha256": sha256_file(heldout_path),
        "csv": str(csv_path),
        "csv_rows": len(all_csv),
        "train_winner_advantage_mask_mean": finite_mean([float(r["state_summary"]["winner_advantage_mask_mean"]) for r in train_manifest]),
        "heldout_winner_advantage_mask_mean": finite_mean([float(r["state_summary"]["winner_advantage_mask_mean"]) for r in heldout_manifest]),
    }
    write_json(reports_root / "exp35_minimax_bad_noise_summary.json", summary)
    md = [
        "# Exp35 MiniMax Bad-Noise / Hard-Timestep Miner",
        "",
        "Status: `MINIMAX_BAD_NOISE_STATES_READY`",
        "",
        "- Model update: false.",
        "- Candidate states per row: `16` (`K_noise=4`, `K_timestep=4`).",
        f"- Train rows: `{len(train_rows)}`.",
        f"- Heldout rows: `{len(heldout_rows)}`.",
        f"- Timesteps: `{', '.join(str(t) for t in timesteps)}`.",
        f"- Train manifest SHA256: `{summary['train_manifest_sha256']}`.",
        f"- Heldout manifest SHA256: `{summary['heldout_manifest_sha256']}`.",
        "",
        "Selection policy:",
        "",
        "- `hard_state_A`: max loser local residual with outside sanity filter.",
        "- `hard_state_B`: max preference violation / weakest winner advantage.",
        "- `hard_state_C`: max winner-risk with outside sanity filter.",
        "- Training-time preregistration options: H0 fixed A, H1 online K=4 worst valid, H2 50% random + 50% fixed A.",
        "",
        "This milestone mines states only. It does not train, does not evaluate a recipe, and does not unlock 30-step.",
    ]
    (reports_root / "exp35_minimax_bad_noise_miner.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    heartbeat(hb, "completed")


if __name__ == "__main__":
    main()
