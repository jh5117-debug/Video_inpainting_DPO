#!/usr/bin/env python3
"""Generate MiniMax OR candidates for the Exp29 micro data-quality gate.

The script loads MiniMax once per worker, runs a deterministic source/seed
shard, and writes raw output frames plus temporal evidence. It performs no
training and does not select winners.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", default="20260626,20260627,20260628")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--num-inference-steps", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--heartbeat", default="")
    parser.add_argument("--status-jsonl", default="")
    return parser.parse_args()


def image_files(path: Path) -> list[Path]:
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def read_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError(f"failed to read image: {path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def read_gray(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"failed to read mask: {path}")
    return arr


def save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR))


def compatible_temporal_length(n: int) -> int:
    if n <= 1:
        return n
    remainder = (n - 1) % 4
    return n if remainder == 0 else n + (4 - remainder)


def frame_to_uint8(frame: object) -> np.ndarray:
    if not hasattr(frame, "__array__"):
        raise TypeError(f"unsupported frame type: {type(frame).__name__}")
    arr = np.array(frame)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.5:
            arr = arr * 255.0
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
    mask_bool = mask > 20
    overlay = frame.copy()
    overlay[mask_bool] = (0.55 * overlay[mask_bool] + 0.45 * np.array([255, 40, 40])).astype(np.uint8)
    return overlay


def side_by_side_frames(condition_dir: Path, winner_dir: Path, mask_dir: Path, output_frames: list[np.ndarray], n: int) -> list[np.ndarray]:
    cond_files = image_files(condition_dir)
    win_files = image_files(winner_dir)
    mask_files = image_files(mask_dir)
    frames = []
    for idx in range(n):
        cond = read_rgb(cond_files[idx])
        win = read_rgb(win_files[idx])
        mask = read_gray(mask_files[idx])
        overlay = mask_overlay(cond, mask)
        frames.append(np.concatenate([cond, overlay, win, output_frames[idx]], axis=1))
    return frames


def sample_indices(n: int, count: int = 16) -> list[int]:
    if n <= count:
        return list(range(n))
    return sorted({int(round(i * (n - 1) / (count - 1))) for i in range(count)})


def contact_sheet(frames: Iterable[np.ndarray], labels: Iterable[str], tile_w: int = 192) -> np.ndarray:
    tiles = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for frame, label in zip(frames, labels):
        h, w = frame.shape[:2]
        tile_h = int(round(h * tile_w / w))
        tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, label, (6, 20), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    rows = []
    for start in range(0, len(tiles), 4):
        row_tiles = tiles[start : start + 4]
        if len(row_tiles) < 4:
            blank = np.zeros_like(row_tiles[0])
            row_tiles.extend([blank] * (4 - len(row_tiles)))
        rows.append(np.concatenate(row_tiles, axis=1))
    return np.concatenate(rows, axis=0)


def sha256_tree_prefix(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(path.glob("*.png")):
        digest.update(file_path.name.encode("utf-8"))
        digest.update(file_path.read_bytes())
    return digest.hexdigest()[:16]


def load_manifest(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def heartbeat(path: Path | None, text: str) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{time.time():.0f}\t{text}\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_dir = Path(args.repo_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    heartbeat_path = Path(args.heartbeat) if args.heartbeat else output_root / f"worker{args.shard_index}.heartbeat"
    status_jsonl = Path(args.status_jsonl) if args.status_jsonl else output_root / f"worker{args.shard_index}_status.jsonl"
    status_csv = output_root / f"worker{args.shard_index}_status.csv"

    if args.num_shards < 1 or args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("invalid shard configuration")
    for child in ("vae", "transformer", "scheduler"):
        if not (model_dir / child).exists():
            raise FileNotFoundError(f"missing MiniMax weight folder: {model_dir / child}")

    sys.path.insert(0, str(repo_dir))
    from diffusers.models import AutoencoderKLWan  # noqa: WPS433
    from diffusers.schedulers import UniPCMultistepScheduler  # noqa: WPS433
    from pipeline_minimax_remover import Minimax_Remover_Pipeline  # noqa: WPS433
    from transformer_minimax_remover import Transformer3DModel  # noqa: WPS433

    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    manifest = load_manifest(Path(args.manifest))
    assigned = [row for idx, row in enumerate(manifest) if idx % args.num_shards == args.shard_index]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float16)
    transformer = Transformer3DModel.from_pretrained(model_dir / "transformer", torch_dtype=torch.float16)
    scheduler = UniPCMultistepScheduler.from_pretrained(model_dir / "scheduler")
    pipe = Minimax_Remover_Pipeline(transformer=transformer, vae=vae, scheduler=scheduler).to(device)

    start_time = time.time()
    for row_index, row in enumerate(assigned):
        sample_id = str(row["sample_id"])
        condition_dir = Path(str(row["condition_frame_dir"]))
        winner_dir = Path(str(row["winner_frame_dir"]))
        mask_dir = Path(str(row["mask_frame_dir"]))
        num_frames = int(row["num_frames"])
        width = int(row["width"])
        height = int(row["height"])
        heartbeat(heartbeat_path, f"sample={sample_id} {row_index + 1}/{len(assigned)}")
        images, masks, original_n, model_n = prepare_inputs(condition_dir, mask_dir, width, height, num_frames)

        for seed in seeds:
            candidate_root = output_root / "candidates" / sample_id / f"seed_{seed}"
            frames_dir = candidate_root / "frames"
            done = candidate_root / "DONE.json"
            if done.exists() and frames_dir.exists() and len(image_files(frames_dir)) == original_n:
                status = json.loads(done.read_text(encoding="utf-8"))
                append_jsonl(status_jsonl, status)
                append_csv(status_csv, status)
                continue

            heartbeat(heartbeat_path, f"sample={sample_id} seed={seed} running")
            generator = torch.Generator(device=device).manual_seed(seed) if device.type == "cuda" else None
            with torch.inference_mode():
                result = pipe(
                    images=images,
                    masks=masks,
                    num_frames=model_n,
                    height=height,
                    width=width,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    iterations=args.iterations,
                ).frames[0]
            output_frames = [frame_to_uint8(frame) for frame in result[:original_n]]
            frames_dir.mkdir(parents=True, exist_ok=True)
            for idx, frame in enumerate(output_frames):
                save_rgb(frames_dir / f"{idx:05d}.png", frame)

            evidence_root = candidate_root / "evidence"
            raw_mp4 = evidence_root / "raw_output.mp4"
            side_mp4 = evidence_root / "side_by_side.mp4"
            write_mp4(raw_mp4, output_frames)
            side_frames = side_by_side_frames(condition_dir, winner_dir, mask_dir, output_frames, original_n)
            write_mp4(side_mp4, side_frames)
            strip_indices = sample_indices(original_n, 16)
            strip_frames = [side_frames[idx] for idx in strip_indices]
            strip = contact_sheet(strip_frames, [f"f{idx:03d}" for idx in strip_indices], tile_w=384)
            strip_path = evidence_root / "temporal_strip_16.jpg"
            strip_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(strip_path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))

            status = {
                "sample_id": sample_id,
                "seed": seed,
                "status": "OK",
                "source_type": row.get("source_type"),
                "scene_group": row.get("scene_group"),
                "mask_bucket": row.get("mask_bucket"),
                "mask_area_mean": row.get("mask_area_mean"),
                "condition_frame_dir": str(condition_dir),
                "winner_frame_dir": str(winner_dir),
                "mask_frame_dir": str(mask_dir),
                "output_frame_dir": str(frames_dir),
                "raw_mp4": str(raw_mp4),
                "side_by_side_mp4": str(side_mp4),
                "temporal_strip_16": str(strip_path),
                "num_frames": original_n,
                "model_frames": model_n,
                "width": width,
                "height": height,
                "num_inference_steps": args.num_inference_steps,
                "iterations": args.iterations,
                "output_sha256_prefix": sha256_tree_prefix(frames_dir),
                "worker_shard": args.shard_index,
                "num_shards": args.num_shards,
                "elapsed_sec": round(time.time() - start_time, 3),
            }
            done.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            append_jsonl(status_jsonl, status)
            append_csv(status_csv, status)
            heartbeat(heartbeat_path, f"sample={sample_id} seed={seed} done")

    heartbeat(heartbeat_path, "complete")
    print(json.dumps({"status": "MINIMAX_MICRO_CANDIDATES_COMPLETE", "assigned_sources": len(assigned), "seeds": seeds, "output_root": str(output_root)}, indent=2))


if __name__ == "__main__":
    main()
