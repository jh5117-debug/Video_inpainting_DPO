#!/usr/bin/env python3
"""Mine Exp37 MiniMax hard-noise states without training."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--manifests-root", required=True)
    parser.add_argument("--reports-root", required=True)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--noise-count", type=int, default=8)
    parser.add_argument("--timesteps", default="0.05,0.15,0.25,0.35,0.5,0.65,0.8,0.95")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--heartbeat", default="")
    return parser.parse_args()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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
    cond_frames = np.stack(record["condition_frames_uint8"], axis=0)
    win_frames = np.stack(record["winner_frames_uint8"], axis=0)
    mask = torch.from_numpy((mask_frames > 20).astype(np.float32))[None, None].to(device=device)
    affected_np = (np.mean(np.abs(cond_frames.astype(np.float32) - win_frames.astype(np.float32)), axis=3) > 10.0).astype(np.float32)
    affected = torch.from_numpy(affected_np)[None, None].to(device=device)
    target_size = tuple(z0.shape[2:])
    mask = F.interpolate(mask, size=target_size, mode="nearest").clamp(0, 1)
    affected = F.interpolate(affected, size=target_size, mode="nearest").clamp(0, 1)
    dil = F.max_pool3d(mask, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
    ero = 1.0 - F.max_pool3d(1.0 - mask, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
    boundary = (dil - ero).clamp(0, 1)
    local = torch.maximum(torch.maximum(mask, boundary), affected)
    outside = (1.0 - F.max_pool3d(local, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))).clamp(0, 1)
    return {
        "mask": mask.bool().expand_as(z0),
        "boundary": boundary.bool().expand_as(z0),
        "affected": affected.bool().expand_as(z0),
        "local": local.bool().expand_as(z0),
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
        "affected_residual": mean_region(sq, regions["affected"]),
        "local_residual": mean_region(sq, regions["local"]),
        "outside_residual": mean_region(sq, regions["outside"]),
        "pred_norm": float(pred.float().norm().detach().cpu()),
        "target_norm": float(target.float().norm().detach().cpu()),
        "noise_norm": float(eps.float().norm().detach().cpu()),
        "z0_norm": float(z0.float().norm().detach().cpu()),
    }


def finite_mean(values: list[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else 0.0


def finite_max(values: list[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.max()) if arr.size else 0.0


def choose_state(states: list[dict[str, object]], score_key: str, outside_key: str) -> dict[str, object]:
    outside_values = np.array([float(s[outside_key]) for s in states], dtype=np.float64)
    finite = outside_values[np.isfinite(outside_values)]
    sane_cutoff = float(np.quantile(finite, 0.75)) if finite.size else float("inf")
    sane = [s for s in states if math.isfinite(float(s[outside_key])) and float(s[outside_key]) <= sane_cutoff]
    return max(sane or states, key=lambda s: float(s[score_key]))


def score_states(states: list[dict[str, object]]) -> None:
    for state in states:
        state["winner_local_score"] = (
            float(state["winner_mask_residual"])
            + float(state["winner_affected_residual"])
            + 0.5 * float(state["winner_boundary_residual"])
        )
        state["loser_local_score"] = (
            float(state["loser_mask_residual"])
            + float(state["loser_affected_residual"])
            + 0.5 * float(state["loser_boundary_residual"])
        )
        state["preference_margin_proxy"] = float(state["loser_local_score"]) - float(state["winner_local_score"])
        state["preference_violation_proxy"] = -float(state["preference_margin_proxy"])
        state["outside_sanity_residual"] = max(
            float(state["winner_outside_residual"]),
            float(state["loser_outside_residual"]),
        )
        state["gradient_proxy_norm"] = math.sqrt(
            max(0.0, float(state["winner_local_score"]))
            + max(0.0, float(state["loser_local_score"]))
            + abs(float(state["preference_margin_proxy"]))
        )


def summarize_row(row: dict[str, object], states: list[dict[str, object]]) -> dict[str, object]:
    score_states(states)
    random_state = states[0]
    state_a = choose_state(states, "loser_local_score", "outside_sanity_residual")
    state_b = choose_state(states, "preference_violation_proxy", "outside_sanity_residual")
    state_c = choose_state(states, "winner_local_score", "outside_sanity_residual")
    return {
        "sample_id": row.get("sample_id"),
        "source_group": row.get("source_group", row.get("scene_group", "")),
        "source_type": row.get("source_type", ""),
        "condition_path": row.get("condition_path"),
        "winner_path": row.get("winner_path"),
        "loser_path": row.get("loser_path"),
        "mask_path": row.get("mask_path"),
        "localdpo_profile": row.get("profile", ""),
        "num_candidate_states": len(states),
        "hard_state_A": state_a,
        "hard_state_B": state_b,
        "hard_state_C": state_c,
        "random_state_baseline": random_state,
        "state_summary": {
            "winner_local_score_mean": finite_mean([float(s["winner_local_score"]) for s in states]),
            "loser_local_score_mean": finite_mean([float(s["loser_local_score"]) for s in states]),
            "preference_margin_proxy_mean": finite_mean([float(s["preference_margin_proxy"]) for s in states]),
            "gradient_proxy_norm_mean": finite_mean([float(s["gradient_proxy_norm"]) for s in states]),
            "gradient_proxy_norm_max": finite_max([float(s["gradient_proxy_norm"]) for s in states]),
            "hard_A_vs_random_gradient_proxy_ratio": float(state_a["gradient_proxy_norm"]) / max(1e-12, float(random_state["gradient_proxy_norm"])),
            "hard_A_vs_random_loser_local_ratio": float(state_a["loser_local_score"]) / max(1e-12, float(random_state["loser_local_score"])),
        },
    }


def mine_rows(
    model: torch.nn.Module,
    cache,
    rows: list[dict[str, object]],
    noise_count: int,
    timesteps: list[float],
    base_seed: int,
    hb: Path | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    manifest_rows: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []
    for row_idx, row in enumerate(rows):
        sample_id = str(row.get("sample_id", f"train32_{row_idx:04d}"))
        heartbeat(hb, f"train32:{row_idx + 1}/{len(rows)}:{sample_id}")
        states: list[dict[str, object]] = []
        for noise_idx in range(noise_count):
            noise_seed = base_seed + row_idx * 1009 + noise_idx * 9173
            for timestep_idx, tval in enumerate(timesteps):
                winner = target_residual(model, cache, row, "winner", noise_seed, tval)
                loser = target_residual(model, cache, row, "loser", noise_seed, tval)
                state: dict[str, object] = {
                    "split": "train32",
                    "sample_id": sample_id,
                    "source_group": row.get("source_group", row.get("scene_group", "")),
                    "profile": row.get("profile", ""),
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
        score_states(states)
        csv_rows.extend(dict(state) for state in states)
        manifest_rows.append({"split": "train32", **summarize_row(row, states)})
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
    if args.noise_count != 8 or len(timesteps) != 8:
        raise ValueError("Exp37 bad-noise scan is preregistered for K_noise=8 and K_timestep=8")
    manifests_root = Path(args.manifests_root)
    reports_root = Path(args.reports_root)
    manifests_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    hb = Path(args.heartbeat) if args.heartbeat else reports_root / "exp37_badnoise_scan.heartbeat"
    dtype = torch_dtype(args.dtype)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir)

    heartbeat(hb, "load_model")
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=dtype).to(device).eval()
    for param in vae.parameters():
        param.requires_grad_(False)
    model = Transformer3DModel.from_pretrained(model_dir / "transformer", torch_dtype=dtype).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    latents_std = (1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)).to(device, dtype)
    cache = BatchCache(vae, latents_mean, latents_std, device, dtype)

    train_rows = read_jsonl(Path(args.train_manifest))
    manifest_rows, csv_rows = mine_rows(model, cache, train_rows, args.noise_count, timesteps, args.seed, hb)

    manifest_path = manifests_root / "exp37_badnoise_states.jsonl"
    csv_path = reports_root / "exp37_minimax_badnoise_diagnostic_scan.csv"
    write_jsonl(manifest_path, manifest_rows)
    write_csv(csv_path, csv_rows)
    ratios = [float(row["state_summary"]["hard_A_vs_random_gradient_proxy_ratio"]) for row in manifest_rows]
    loser_ratios = [float(row["state_summary"]["hard_A_vs_random_loser_local_ratio"]) for row in manifest_rows]
    summary = {
        "status": "MINIMAX_BAD_NOISE_STATES_READY",
        "training_launched": False,
        "model_update": False,
        "device": str(device),
        "dtype": args.dtype,
        "train_rows": len(train_rows),
        "noise_count": args.noise_count,
        "timesteps": timesteps,
        "candidate_states_per_row": args.noise_count * len(timesteps),
        "total_candidate_states": len(csv_rows),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "csv": str(csv_path),
        "hard_A_vs_random_gradient_proxy_ratio_mean": finite_mean(ratios),
        "hard_A_vs_random_gradient_proxy_ratio_max": finite_max(ratios),
        "hard_A_vs_random_loser_local_ratio_mean": finite_mean(loser_ratios),
        "hard_A_vs_random_loser_local_ratio_max": finite_max(loser_ratios),
    }
    write_json(reports_root / "exp37_minimax_badnoise_summary.json", summary)
    md = [
        "# Exp37 MiniMax Bad-Noise Diagnostic Scan",
        "",
        "Status: `MINIMAX_BAD_NOISE_STATES_READY`",
        "",
        "- Training launched: false.",
        "- Model update: false.",
        "- Rows: `32` train rows from the LocalDPO-style pool.",
        "- Candidate states per row: `64` (`K_noise=8`, `K_timestep=8`).",
        f"- Timesteps: `{', '.join(str(t) for t in timesteps)}`.",
        f"- Total states: `{len(csv_rows)}`.",
        f"- Manifest SHA256: `{summary['manifest_sha256']}`.",
        f"- Mean hard-A/random gradient proxy ratio: `{summary['hard_A_vs_random_gradient_proxy_ratio_mean']:.6f}`.",
        f"- Max hard-A/random gradient proxy ratio: `{summary['hard_A_vs_random_gradient_proxy_ratio_max']:.6f}`.",
        "",
        "Selection policy:",
        "",
        "- `hard_state_A`: maximum local loser residual with outside sanity filter.",
        "- `hard_state_B`: weakest winner advantage / largest preference violation proxy.",
        "- `hard_state_C`: highest winner-loss risk with outside sanity filter.",
        "",
        "This milestone mines states only. It does not train, evaluate a recipe, or unlock 30-step.",
    ]
    (reports_root / "exp37_minimax_badnoise_diagnostic_scan.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    heartbeat(hb, "done")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
