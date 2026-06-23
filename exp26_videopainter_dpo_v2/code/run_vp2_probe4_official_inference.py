#!/usr/bin/env python3
"""Run official VideoPainter branch inference on locked Exp26 Probe4 49F rows.

This is an inference-only gate. It loads the official VideoPainter branch
checkpoint and CogVideoX base model, consumes the already materialized 49-frame
Probe4 source/mask manifest, and writes raw outputs plus contact sheets. It
does not train and it does not use the old Exp14 adapter output as a proxy.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from eval_videopainter_adapter_davis import (  # noqa: E402
    add_label,
    load_pipeline,
    make_contact_sheet,
    np_frames_to_uint8,
    save_frame_sequence,
    write_mp4,
)


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Exp26 Probe4 official VideoPainter inference")
    p.add_argument("--videopainter-root", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--branch-checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--num-frames", type=int, default=49)
    p.add_argument("--num-inference-steps", type=int, default=20)
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=20260623)
    p.add_argument("--prompt", default="")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def load_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def load_probe_sample(row: dict, width: int, height: int, num_frames: int) -> tuple[list[Image.Image], list[Image.Image], list[Image.Image], list[np.ndarray]]:
    frame_dir = Path(row["frame_dir"])
    mask_dir = Path(row["mask_dir"])
    frame_files = list_images(frame_dir)
    mask_files = list_images(mask_dir)
    if len(frame_files) != num_frames:
        raise ValueError(f"{row.get('sample_id')}: expected exactly {num_frames} frames, got {len(frame_files)}")
    if len(mask_files) != num_frames:
        raise ValueError(f"{row.get('sample_id')}: expected exactly {num_frames} masks, got {len(mask_files)}")
    frames: list[Image.Image] = []
    masks_rgb: list[Image.Image] = []
    masked: list[Image.Image] = []
    mask_arrays: list[np.ndarray] = []
    for idx, (fp, mp) in enumerate(zip(frame_files, mask_files)):
        img = Image.open(fp).convert("RGB").resize((width, height), Image.BICUBIC)
        mask_l = Image.open(mp).convert("L").resize((width, height), Image.NEAREST)
        mask = (np.asarray(mask_l, dtype=np.uint8) > 127).astype(np.uint8)
        if idx == 0:
            mask = np.zeros_like(mask)
        arr = np.asarray(img, dtype=np.uint8)
        cond = arr.copy()
        cond[mask > 0] = 0
        frames.append(img)
        masks_rgb.append(Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB"))
        masked.append(Image.fromarray(cond).convert("RGB"))
        mask_arrays.append(mask)
    return frames, masks_rgb, masked, mask_arrays


def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = frame.astype(np.float32).copy()
    red = np.zeros_like(out)
    red[..., 0] = 255
    alpha = (mask > 0).astype(np.float32)[..., None] * 0.45
    return np.clip(out * (1 - alpha) + red * alpha, 0, 255).astype(np.uint8)


def make_probe_visuals(
    sample_id: str,
    winner: Sequence[Image.Image],
    masked: Sequence[Image.Image],
    masks: Sequence[np.ndarray],
    pred: Sequence[np.ndarray],
    out_dir: Path,
    fps: int,
) -> None:
    frames = []
    n = min(len(winner), len(masked), len(masks), len(pred))
    for i in range(n):
        gt = np.asarray(winner[i].convert("RGB"), dtype=np.uint8)
        cond = np.asarray(masked[i].convert("RGB"), dtype=np.uint8)
        row = np.concatenate(
            [
                add_label(gt, "winner/source"),
                add_label(overlay_mask(gt, masks[i]), "mask overlay"),
                add_label(cond, "condition"),
                add_label(pred[i], "VideoPainter official"),
            ],
            axis=1,
        )
        frames.append(row)
    frame_dir = out_dir / "frame_by_frame" / sample_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(frame_dir / f"{i:05d}.jpg", quality=92)
    write_mp4(frames, out_dir / "side_by_side" / f"{sample_id}.mp4", fps=fps)
    make_contact_sheet(frames, out_dir / "contact_sheets" / f"{sample_id}.jpg")


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.manifest), args.limit)
    if not rows:
        raise ValueError("empty Probe4 manifest")
    pipe_args = argparse.Namespace(
        videopainter_root=args.videopainter_root,
        base_model=args.base_model,
        dtype=args.dtype,
        device=args.device,
    )
    pipe = load_pipeline(pipe_args, Path(args.branch_checkpoint))
    status_rows = []
    try:
        for row in rows:
            sid = row["sample_id"]
            raw_dir = out / "raw_frames" / sid
            if args.skip_existing and raw_dir.is_dir() and len(list_images(raw_dir)) == args.num_frames:
                status_rows.append({"sample_id": sid, "status": "SKIP_EXISTING", "frames": args.num_frames})
                continue
            winner, mask_frames, masked, masks = load_probe_sample(row, args.width, args.height, args.num_frames)
            generator = torch.Generator(device=args.device if args.device.startswith("cuda") else "cpu").manual_seed(args.seed)
            result = pipe(
                prompt=args.prompt,
                image=masked[0],
                num_videos_per_prompt=1,
                num_inference_steps=args.num_inference_steps,
                num_frames=args.num_frames,
                use_dynamic_cfg=True,
                guidance_scale=args.guidance_scale,
                generator=generator,
                video=masked,
                masks=mask_frames,
                strength=1.0,
                replace_gt=False,
                mask_add=True,
                stride=args.num_frames,
                prev_clip_weight=0.0,
                output_type="np",
            ).frames[0]
            pred = np_frames_to_uint8(result)[: args.num_frames]
            save_frame_sequence(pred, raw_dir)
            write_mp4(pred, out / "videos" / f"{sid}.mp4", fps=args.fps)
            make_probe_visuals(sid, winner, masked, masks, pred, out, args.fps)
            status_rows.append({"sample_id": sid, "status": "OK", "frames": len(pred)})
    finally:
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    with (out / "probe4_official_inference_status.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "status", "frames"])
        writer.writeheader()
        writer.writerows(status_rows)
    (out / "probe4_official_inference_summary.json").write_text(
        json.dumps(
            {
                "status": "passed" if all(r["status"] in {"OK", "SKIP_EXISTING"} and int(r["frames"]) == args.num_frames for r in status_rows) else "failed",
                "num_rows": len(status_rows),
                "num_frames": args.num_frames,
                "num_inference_steps": args.num_inference_steps,
                "branch_checkpoint": args.branch_checkpoint,
                "base_model": args.base_model,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out), "rows": status_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
