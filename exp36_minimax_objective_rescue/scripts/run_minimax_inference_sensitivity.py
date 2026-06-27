#!/usr/bin/env python3
"""Exp36 MiniMax inference-sensitivity positive control.

This diagnostic performs no training.  It runs a fixed Step0 checkpoint twice
with the same seed, then runs a temporary perturbed checkpoint saved under the
Exp36 output root.  The goal is to prove whether MiniMax inference is sensitive
to the trainable transformer weights before changing recipes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--step0-checkpoint", required=True)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--heldout-manifest", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--reports-root", required=True)
    p.add_argument("--seed", type=int, default=20260627)
    p.add_argument("--num-inference-steps", type=int, default=12)
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--train-rows", type=int, default=2)
    p.add_argument("--heldout-rows", type=int, default=2)
    p.add_argument("--perturb-scale", type=float, default=1.01)
    p.add_argument("--max-perturb-tensors", type=int, default=16)
    p.add_argument("--heartbeat", default="")
    return p.parse_args()


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def read_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError(f"failed to read RGB frame: {path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def read_gray(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"failed to read mask frame: {path}")
    return arr


def save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


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
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree_prefix(path: Path) -> str:
    h = hashlib.sha256()
    for file_path in sorted(path.glob("*.png")):
        h.update(file_path.name.encode("utf-8"))
        h.update(file_path.read_bytes())
    return h.hexdigest()[:16]


def heartbeat(path: Path | None, text: str) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{time.time():.0f}\t{text}\n", encoding="utf-8")


def compatible_temporal_length(n: int) -> int:
    if n <= 1:
        return n
    remainder = (n - 1) % 4
    return n if remainder == 0 else n + (4 - remainder)


def frame_to_uint8(frame: object) -> np.ndarray:
    arr = np.array(frame)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.5:
            arr *= 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"unexpected output frame shape: {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    return arr


def prepare_inputs(video_dir: Path, mask_dir: Path, width: int, height: int, num_frames: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    frame_files = image_files(video_dir)
    mask_files = image_files(mask_dir)
    n = min(len(frame_files), len(mask_files), num_frames)
    if n <= 0:
        raise RuntimeError(f"no frames found in {video_dir} / {mask_dir}")
    frames: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for frame_path, mask_path in zip(frame_files[:n], mask_files[:n]):
        frame = read_rgb(frame_path)
        mask = read_gray(mask_path)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        frames.append(frame.astype(np.float32) / 127.5 - 1.0)
        masks.append((mask > 20).astype(np.float32)[:, :, None])
    model_n = compatible_temporal_length(n)
    while len(frames) < model_n:
        frames.append(frames[-1].copy())
        masks.append(masks[-1].copy())
    return torch.from_numpy(np.stack(frames, axis=0)), torch.from_numpy(np.stack(masks, axis=0)), n, model_n


def write_mp4(path: Path, frames: list[np.ndarray], fps: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open mp4 writer: {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()


def mask_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = frame.copy()
    mask_bool = mask > 20
    out[mask_bool] = (0.55 * out[mask_bool] + 0.45 * np.array([255, 40, 40])).astype(np.uint8)
    return out


def sample_indices(n: int, count: int = 16) -> list[int]:
    if n <= count:
        return list(range(n))
    return sorted({int(round(i * (n - 1) / (count - 1))) for i in range(count)})


def labeled_tile(frame: np.ndarray, label: str, tile_w: int = 192) -> np.ndarray:
    h, w = frame.shape[:2]
    tile_h = max(1, int(round(h * tile_w / w)))
    tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    cv2.putText(tile, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
    return tile


def contact_sheet(frames: Iterable[np.ndarray], labels: Iterable[str], cols: int = 1, tile_w: int = 768) -> np.ndarray:
    tiles = [labeled_tile(frame, label, tile_w=tile_w) for frame, label in zip(frames, labels)]
    rows = []
    for start in range(0, len(tiles), cols):
        row_tiles = tiles[start : start + cols]
        if len(row_tiles) < cols:
            row_tiles += [np.zeros_like(row_tiles[0]) for _ in range(cols - len(row_tiles))]
        rows.append(np.concatenate(row_tiles, axis=1))
    return np.concatenate(rows, axis=0)


def psnr_from_mse(mse: float) -> float:
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10((255.0 * 255.0) / mse)


def region_diff(a: list[np.ndarray], b: list[np.ndarray], masks: list[np.ndarray]) -> dict[str, float]:
    rows = []
    for fa, fb, mask in zip(a, b, masks):
        diff = fa.astype(np.float32) - fb.astype(np.float32)
        mask_bool = mask > 20
        kernel = np.ones((9, 9), np.uint8)
        dil = cv2.dilate(mask_bool.astype(np.uint8), kernel, iterations=1).astype(bool)
        ero = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=1).astype(bool)
        boundary = np.logical_xor(dil, ero)
        outside = ~dil
        frame_row: dict[str, float] = {}
        for name, region in (("full", np.ones(mask_bool.shape, dtype=bool)), ("mask", mask_bool), ("boundary", boundary), ("outside", outside)):
            if not np.any(region):
                frame_row[f"{name}_mae"] = float("nan")
                frame_row[f"{name}_max_abs"] = float("nan")
                frame_row[f"{name}_psnr"] = float("nan")
                continue
            vals = diff[region]
            frame_row[f"{name}_mae"] = float(np.mean(np.abs(vals)))
            frame_row[f"{name}_max_abs"] = float(np.max(np.abs(vals)))
            frame_row[f"{name}_psnr"] = psnr_from_mse(float(np.mean(vals * vals)))
        rows.append(frame_row)
    out: dict[str, float] = {}
    for key in rows[0]:
        vals = [r[key] for r in rows if math.isfinite(r[key])]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def load_uint8_frames(path: Path, n: int, width: int, height: int, gray: bool = False) -> list[np.ndarray]:
    files = image_files(path)
    out = []
    for file_path in files[:n]:
        frame = read_gray(file_path) if gray else read_rgb(file_path)
        if frame.shape[:2] != (height, width):
            interp = cv2.INTER_NEAREST if gray else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (width, height), interpolation=interp)
        out.append(frame)
    return out


def perturb_transformer(model: torch.nn.Module, scale: float, max_tensors: int) -> list[dict[str, object]]:
    candidates = []
    for name, param in model.named_parameters():
        if not param.requires_grad or not param.is_floating_point() or param.ndim < 2:
            continue
        # Prefer the high-throughput transformer blocks, but keep the code
        # robust to upstream naming differences.
        score = 1 if ("blocks" in name or "transformer_blocks" in name) else 0
        candidates.append((score, name, param))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    touched: list[dict[str, object]] = []
    with torch.no_grad():
        for _, name, param in candidates[:max_tensors]:
            before = float(param.detach().float().abs().mean().cpu())
            param.mul_(scale)
            after = float(param.detach().float().abs().mean().cpu())
            touched.append({
                "tensor": name,
                "shape": "x".join(str(x) for x in param.shape),
                "mean_abs_before": before,
                "mean_abs_after": after,
                "scale": scale,
            })
    return touched


def run_pipeline(pipe, row: dict[str, object], outdir: Path, seed: int, steps: int, iterations: int) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    frames_dir = outdir / "frames"
    evidence_dir = outdir / "evidence"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    if evidence_dir.exists():
        shutil.rmtree(evidence_dir)
    frames_dir.mkdir(parents=True)
    evidence_dir.mkdir(parents=True)

    width = int(row.get("width", 512))
    height = int(row.get("height", 512))
    num_frames = int(row.get("num_frames", 17))
    condition_dir = Path(str(row["condition_path"]))
    mask_dir = Path(str(row["mask_path"]))
    winner_dir = Path(str(row["winner_path"]))
    images, masks, original_n, model_n = prepare_inputs(condition_dir, mask_dir, width, height, num_frames)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    with torch.inference_mode():
        result = pipe(
            images=images,
            masks=masks,
            num_frames=model_n,
            height=height,
            width=width,
            num_inference_steps=steps,
            generator=generator,
            iterations=iterations,
        ).frames[0]
    output_frames = [frame_to_uint8(frame) for frame in result[:original_n]]
    for idx, frame in enumerate(output_frames):
        save_rgb(frames_dir / f"{idx:05d}.png", frame)
    write_mp4(evidence_dir / "raw_output.mp4", output_frames)

    cond = load_uint8_frames(condition_dir, original_n, width, height)
    winner = load_uint8_frames(winner_dir, original_n, width, height)
    mask_frames = load_uint8_frames(mask_dir, original_n, width, height, gray=True)
    side = [np.concatenate([cond[i], mask_overlay(cond[i], mask_frames[i]), winner[i], output_frames[i]], axis=1) for i in range(original_n)]
    write_mp4(evidence_dir / "side_by_side.mp4", side)
    idxs = sample_indices(original_n, 16)
    strip = contact_sheet([side[i] for i in idxs], [f"f{i:02d}" for i in idxs], cols=1, tile_w=768)
    save_rgb(evidence_dir / "temporal_strip_16.jpg", strip)
    mid = original_n // 2
    review = np.concatenate([cond[mid], mask_overlay(cond[mid], mask_frames[mid]), winner[mid], output_frames[mid]], axis=1)
    save_rgb(evidence_dir / "midframe_review_sheet.jpg", review)
    return {
        "frames": output_frames,
        "mask_frames": mask_frames,
        "frames_dir": str(frames_dir),
        "raw_output_mp4": str(evidence_dir / "raw_output.mp4"),
        "side_by_side_mp4": str(evidence_dir / "side_by_side.mp4"),
        "temporal_strip_16": str(evidence_dir / "temporal_strip_16.jpg"),
        "review_sheet": str(evidence_dir / "midframe_review_sheet.jpg"),
        "frames_sha256": sha256_tree_prefix(frames_dir),
    }


def main() -> None:
    args = parse_args()
    repo_dir = Path(args.repo_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    step0_checkpoint = Path(args.step0_checkpoint).resolve()
    output_root = Path(args.output_root).resolve()
    reports_root = Path(args.reports_root).resolve()
    heartbeat_path = Path(args.heartbeat) if args.heartbeat else output_root / "inference_sensitivity.heartbeat"
    output_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo_dir))

    from diffusers.models import AutoencoderKLWan  # noqa: WPS433
    from diffusers.schedulers import UniPCMultistepScheduler  # noqa: WPS433
    from pipeline_minimax_remover import Minimax_Remover_Pipeline  # noqa: WPS433
    from transformer_minimax_remover import Transformer3DModel  # noqa: WPS433

    rows = []
    for role, manifest_path, limit in (
        ("heldout", Path(args.heldout_manifest), args.heldout_rows),
        ("train", Path(args.train_manifest), args.train_rows),
    ):
        for row in read_jsonl(manifest_path)[:limit]:
            item = dict(row)
            item["exp36_sensitivity_role"] = role
            rows.append(item)
    if not rows:
        raise RuntimeError("no rows selected for sensitivity audit")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    heartbeat(heartbeat_path, "loading_step0")
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float16).to(device).eval()
    step0_a = Transformer3DModel.from_pretrained(step0_checkpoint, torch_dtype=torch.float16).to(device).eval()
    step0_b = Transformer3DModel.from_pretrained(step0_checkpoint, torch_dtype=torch.float16).to(device).eval()
    perturbed = Transformer3DModel.from_pretrained(step0_checkpoint, torch_dtype=torch.float16).to(device).eval()
    touched = perturb_transformer(perturbed, args.perturb_scale, args.max_perturb_tensors)
    temp_ckpt = output_root / "temporary_perturbed_checkpoint"
    if temp_ckpt.exists():
        shutil.rmtree(temp_ckpt)
    perturbed.save_pretrained(temp_ckpt, safe_serialization=True)
    write_json(reports_root / "exp36_minimax_inference_sensitivity_perturb_tensors.json", touched)

    scheduler_a = UniPCMultistepScheduler.from_pretrained(model_dir / "scheduler")
    scheduler_b = UniPCMultistepScheduler.from_pretrained(model_dir / "scheduler")
    scheduler_p = UniPCMultistepScheduler.from_pretrained(model_dir / "scheduler")
    pipe_a = Minimax_Remover_Pipeline(transformer=step0_a, vae=vae, scheduler=scheduler_a).to(device)
    pipe_b = Minimax_Remover_Pipeline(transformer=step0_b, vae=vae, scheduler=scheduler_b).to(device)
    pipe_p = Minimax_Remover_Pipeline(transformer=perturbed, vae=vae, scheduler=scheduler_p).to(device)

    metric_rows: list[dict[str, object]] = []
    visual_rows: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        sample_id = str(row["sample_id"])
        role = str(row["exp36_sensitivity_role"])
        sample_root = output_root / "rows" / f"{role}_{sample_id}"
        heartbeat(heartbeat_path, f"sample={idx + 1}/{len(rows)} {role}:{sample_id}:step0_a")
        out_a = run_pipeline(pipe_a, row, sample_root / "step0_identity_a", args.seed, args.num_inference_steps, args.iterations)
        heartbeat(heartbeat_path, f"sample={idx + 1}/{len(rows)} {role}:{sample_id}:step0_b")
        out_b = run_pipeline(pipe_b, row, sample_root / "step0_identity_b", args.seed, args.num_inference_steps, args.iterations)
        heartbeat(heartbeat_path, f"sample={idx + 1}/{len(rows)} {role}:{sample_id}:perturbed")
        out_p = run_pipeline(pipe_p, row, sample_root / "perturbed", args.seed, args.num_inference_steps, args.iterations)

        identity = region_diff(out_a["frames"], out_b["frames"], out_a["mask_frames"])
        pert = region_diff(out_a["frames"], out_p["frames"], out_a["mask_frames"])
        metric_row: dict[str, object] = {
            "sample_id": sample_id,
            "source_group": row.get("source_group", ""),
            "role": role,
            "classification_final": row.get("classification_final", ""),
            "condition_path": row.get("condition_path", ""),
            "winner_path": row.get("winner_path", ""),
            "loser_path": row.get("loser_path", ""),
            "mask_path": row.get("mask_path", ""),
            "step0_identity_a_frames": out_a["frames_dir"],
            "step0_identity_b_frames": out_b["frames_dir"],
            "perturbed_frames": out_p["frames_dir"],
            "step0_identity_a_mp4": out_a["raw_output_mp4"],
            "step0_identity_b_mp4": out_b["raw_output_mp4"],
            "perturbed_mp4": out_p["raw_output_mp4"],
            "step0_identity_a_strip": out_a["temporal_strip_16"],
            "step0_identity_b_strip": out_b["temporal_strip_16"],
            "perturbed_strip": out_p["temporal_strip_16"],
            "step0_identity_a_sha": out_a["frames_sha256"],
            "step0_identity_b_sha": out_b["frames_sha256"],
            "perturbed_sha": out_p["frames_sha256"],
        }
        for key, value in identity.items():
            metric_row[f"identity_{key}"] = value
        for key, value in pert.items():
            metric_row[f"perturb_{key}"] = value
        metric_rows.append(metric_row)

        # Combined temporal evidence: condition/mask/winner/base/perturbed/diff.
        a_frames = out_a["frames"]
        p_frames = out_p["frames"]
        mask_frames = out_a["mask_frames"]
        width = int(row.get("width", 512))
        height = int(row.get("height", 512))
        n = len(a_frames)
        cond = load_uint8_frames(Path(str(row["condition_path"])), n, width, height)
        winner = load_uint8_frames(Path(str(row["winner_path"])), n, width, height)
        idxs = sample_indices(n, 16)
        combined_frames = []
        for frame_idx in idxs:
            diff = np.abs(a_frames[frame_idx].astype(np.int16) - p_frames[frame_idx].astype(np.int16)).clip(0, 255).astype(np.uint8)
            combined_frames.append(np.concatenate([
                cond[frame_idx],
                mask_overlay(cond[frame_idx], mask_frames[frame_idx]),
                winner[frame_idx],
                a_frames[frame_idx],
                p_frames[frame_idx],
                diff,
            ], axis=1))
        combined_strip = sample_root / "evidence" / "sensitivity_comparison_strip_16.jpg"
        save_rgb(combined_strip, contact_sheet(combined_frames, [f"f{i:02d}" for i in idxs], cols=1, tile_w=1152))
        visual_rows.append({
            "sample_id": sample_id,
            "source_group": row.get("source_group", ""),
            "role": role,
            "condition_path": row.get("condition_path", ""),
            "winner_path": row.get("winner_path", ""),
            "loser_path": row.get("loser_path", ""),
            "mask_path": row.get("mask_path", ""),
            "step0_identity_a_output": out_a["raw_output_mp4"],
            "step0_identity_b_output": out_b["raw_output_mp4"],
            "perturbed_output": out_p["raw_output_mp4"],
            "frames_reviewed": "0,mid,last,16-strip",
            "comparison_strip": str(combined_strip),
            "identity_control_visual": "PENDING_CODEX_REVIEW",
            "perturbation_visual": "PENDING_CODEX_REVIEW",
            "collapse": "PENDING_CODEX_REVIEW",
            "classification": "PENDING_CODEX_VISUAL_REVIEW",
            "reason": "",
        })

    write_csv(reports_root / "exp36_minimax_inference_sensitivity.csv", metric_rows)
    write_csv(reports_root / "exp36_minimax_inference_sensitivity_visual_review.csv", visual_rows)
    identity_full = [float(r.get("identity_full_mae", float("nan"))) for r in metric_rows]
    perturb_mask = [float(r.get("perturb_mask_mae", float("nan"))) for r in metric_rows]
    perturb_full = [float(r.get("perturb_full_mae", float("nan"))) for r in metric_rows]
    identity_max = max([v for v in identity_full if math.isfinite(v)] or [float("nan")])
    perturb_mask_mean = float(np.mean([v for v in perturb_mask if math.isfinite(v)])) if perturb_mask else float("nan")
    perturb_full_mean = float(np.mean([v for v in perturb_full if math.isfinite(v)])) if perturb_full else float("nan")
    identity_ok = math.isfinite(identity_max) and identity_max <= 0.01
    perturb_nonzero = math.isfinite(perturb_mask_mean) and perturb_mask_mean > 0.01
    status = "MINIMAX_INFERENCE_SENSITIVITY_PASS" if identity_ok and perturb_nonzero else (
        "MINIMAX_INFERENCE_IGNORES_ADAPTER" if identity_ok and not perturb_nonzero else "MINIMAX_INFERENCE_SENSITIVITY_BLOCKED"
    )
    summary = {
        "status": status,
        "rows": len(metric_rows),
        "seed": args.seed,
        "step0_checkpoint": str(step0_checkpoint),
        "temporary_perturbed_checkpoint": str(temp_ckpt),
        "perturb_scale": args.perturb_scale,
        "max_perturb_tensors": args.max_perturb_tensors,
        "perturbed_tensors": len(touched),
        "identity_full_mae_max": identity_max,
        "perturb_full_mae_mean": perturb_full_mean,
        "perturb_mask_mae_mean": perturb_mask_mean,
        "adapter_scale_sweep_status": "NOT_APPLICABLE_NO_LORA_SCOPE_IN_EXP30",
        "train_manifest_sha256": sha256_file(Path(args.train_manifest)),
        "heldout_manifest_sha256": sha256_file(Path(args.heldout_manifest)),
    }
    write_json(reports_root / "exp36_minimax_inference_sensitivity_summary.json", summary)
    md = [
        "# Exp36 MiniMax Inference Sensitivity",
        "",
        f"Status: `{status}`",
        "",
        "This diagnostic performed no training and did not modify Exp30 outputs.",
        "",
        f"- Rows: `{len(metric_rows)}` (heldout first, then train).",
        f"- Step0 checkpoint: `{step0_checkpoint}`.",
        f"- Temporary perturbed checkpoint: `{temp_ckpt}`.",
        f"- Perturb scale: `{args.perturb_scale}` over `{len(touched)}` transformer tensors.",
        f"- Identity control max full MAE: `{identity_max}`.",
        f"- Perturbed mean full MAE: `{perturb_full_mean}`.",
        f"- Perturbed mean mask MAE: `{perturb_mask_mean}`.",
        "- Adapter scale sweep: `NOT_APPLICABLE_NO_LORA_SCOPE_IN_EXP30`.",
        "",
        "Codex visual review of the generated strips is required before treating this as final.",
    ]
    (reports_root / "exp36_minimax_inference_sensitivity.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
