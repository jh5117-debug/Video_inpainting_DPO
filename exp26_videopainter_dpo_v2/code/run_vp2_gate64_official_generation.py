#!/usr/bin/env python3
"""Run official VideoPainter 49F self-loser generation for Exp26 Gate64."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

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
from run_vp2_probe4_official_inference import list_images, overlay_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Exp26 Gate64 official VideoPainter generation")
    p.add_argument("--videopainter-root", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--branch-checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--num-frames", type=int, default=49)
    p.add_argument("--num-inference-steps", type=int, default=20)
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=20260624)
    p.add_argument("--prompt", default="")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def read_jsonl(path: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_gate_sample(row: dict, width: int, height: int, num_frames: int) -> tuple[list[Image.Image], list[Image.Image], list[Image.Image], list[np.ndarray]]:
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
        if idx == 0 and row.get("first_frame_gt", True):
            mask = np.zeros_like(mask)
        arr = np.asarray(img, dtype=np.uint8)
        cond = arr.copy()
        cond[mask > 0] = 0
        frames.append(img)
        masks_rgb.append(Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB"))
        masked.append(Image.fromarray(cond).convert("RGB"))
        mask_arrays.append(mask)
    return frames, masks_rgb, masked, mask_arrays


def composite(pred: Sequence[np.ndarray], winner: Sequence[Image.Image], masks: Sequence[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for pr, gt_img, mask in zip(pred, winner, masks):
        gt = np.asarray(gt_img.convert("RGB").resize((pr.shape[1], pr.shape[0]), Image.BICUBIC), dtype=np.uint8)
        if mask.shape != pr.shape[:2]:
            mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize((pr.shape[1], pr.shape[0]), Image.NEAREST)
            mask = (np.asarray(mask_img, dtype=np.uint8) > 127).astype(np.uint8)
        w = mask[..., None].astype(bool)
        out.append(np.where(w, pr, gt).astype(np.uint8))
    return out


def save_visuals(
    sample_id: str,
    winner: Sequence[Image.Image],
    masked: Sequence[Image.Image],
    masks: Sequence[np.ndarray],
    raw: Sequence[np.ndarray],
    comp: Sequence[np.ndarray],
    out_dir: Path,
    fps: int,
) -> None:
    frames = []
    n = min(len(winner), len(masked), len(masks), len(raw), len(comp))
    for i in range(n):
        gt = np.asarray(winner[i].convert("RGB"), dtype=np.uint8)
        cond = np.asarray(masked[i].convert("RGB"), dtype=np.uint8)
        row = np.concatenate(
            [
                add_label(gt, "winner/BG"),
                add_label(overlay_mask(gt, masks[i]), "mask overlay"),
                add_label(cond, "condition"),
                add_label(raw[i], "VideoPainter raw"),
                add_label(comp[i], "diagnostic comp"),
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
    if args.num_frames != 49:
        raise ValueError("Gate64 formal generation requires exactly 49 frames")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(Path(args.manifest), args.limit)
    if not rows:
        raise ValueError("empty Gate64 manifest")
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
            comp_dir = out / "comp_frames" / sid
            if (
                args.skip_existing
                and raw_dir.is_dir()
                and comp_dir.is_dir()
                and len(list_images(raw_dir)) == args.num_frames
                and len(list_images(comp_dir)) == args.num_frames
            ):
                status_rows.append({"sample_id": sid, "status": "SKIP_EXISTING", "frames": args.num_frames, "seconds": 0.0, "raw_hash": "", "comp_hash": "", "error": ""})
                continue
            start = time.time()
            winner, mask_frames, masked, masks = load_gate_sample(row, args.width, args.height, args.num_frames)
            generator = torch.Generator(device=args.device if args.device.startswith("cuda") else "cpu").manual_seed(args.seed)
            result = pipe(
                prompt=row.get("prompt", args.prompt),
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
            raw = np_frames_to_uint8(result)[: args.num_frames]
            comp = composite(raw, winner, masks)
            save_frame_sequence(raw, raw_dir)
            save_frame_sequence(comp, comp_dir)
            write_mp4(raw, out / "videos_raw" / f"{sid}.mp4", fps=args.fps)
            write_mp4(comp, out / "videos_comp" / f"{sid}.mp4", fps=args.fps)
            save_visuals(sid, winner, masked, masks, raw, comp, out, args.fps)
            raw_hash = sha256_file(sorted(raw_dir.iterdir())[0]) if raw_dir.exists() else ""
            comp_hash = sha256_file(sorted(comp_dir.iterdir())[0]) if comp_dir.exists() else ""
            status_rows.append({"sample_id": sid, "status": "OK", "frames": len(raw), "seconds": round(time.time() - start, 3), "raw_hash": raw_hash, "comp_hash": comp_hash, "error": ""})
    finally:
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    csv_path = out / "gate64_generation_status.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["sample_id", "status", "frames", "seconds", "raw_hash", "comp_hash", "error"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in fields} for r in status_rows])
    ok = sum(1 for r in status_rows if r["status"] in {"OK", "SKIP_EXISTING"} and int(r["frames"]) == args.num_frames)
    summary = {
        "status": "passed" if ok == len(status_rows) and ok == len(rows) else "failed",
        "ok": ok,
        "num_rows": len(rows),
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "branch_checkpoint": args.branch_checkpoint,
        "base_model": args.base_model,
        "manifest": args.manifest,
        "status_csv": str(csv_path),
    }
    (out / "gate64_generation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
