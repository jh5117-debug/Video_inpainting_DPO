#!/usr/bin/env python3
"""Exp14-only VideoPainter adapter DAVIS evaluator.

This is a thin adapter around VideoPainter inference and the existing project
metric backend. It does not modify upstream VideoPainter, shared DPO training,
or inference/metrics.py.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from inference import metrics as metric_backend  # noqa: E402


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class DavisSample:
    name: str
    gt_frames: List[Image.Image]
    mask_frames: List[Image.Image]
    masked_frames: List[Image.Image]
    mask_arrays: List[np.ndarray]
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("VideoPainter baseline vs DPO adapter DAVIS eval")
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--videopainter_root", default=None)
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--baseline_branch", default=None)
    parser.add_argument("--adapter_checkpoint", default=None)
    parser.add_argument("--davis_root", default="/mnt/workspace/hj/nas_hj/data/external/davis_432_240")
    parser.add_argument("--output_dir", default="logs/target_eval/exp14_videopainter_adapter_gate2000_davis")
    parser.add_argument("--video_names", default="", help="Comma-separated DAVIS video names. Empty means all.")
    parser.add_argument("--limit_videos", type=int, default=0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip_generation_if_exists", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compute_lpips", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def list_image_files(path: Path) -> List[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def trim_to_4k_plus_1(files: Sequence[Path], max_frames: int) -> List[Path]:
    selected = list(files[:max_frames])
    while len(selected) > 1 and (len(selected) - 1) % 4 != 0:
        selected = selected[:-1]
    return selected


def available_videos(davis_root: Path) -> List[str]:
    image_root = davis_root / "JPEGImages_432_240"
    mask_root = davis_root / "test_masks"
    names = []
    if not image_root.is_dir() or not mask_root.is_dir():
        return names
    for item in sorted(image_root.iterdir()):
        if item.is_dir() and (mask_root / item.name).is_dir():
            names.append(item.name)
    return names


def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = frame.copy().astype(np.float32)
    red = np.zeros_like(out)
    red[..., 0] = 255
    alpha = (mask > 0).astype(np.float32)[..., None] * 0.45
    out = out * (1.0 - alpha) + red * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def load_davis_sample(davis_root: Path, name: str, width: int, height: int, max_frames: int) -> DavisSample:
    image_dir = davis_root / "JPEGImages_432_240" / name
    mask_dir = davis_root / "test_masks" / name
    image_files = trim_to_4k_plus_1(list_image_files(image_dir), max_frames)
    mask_files = trim_to_4k_plus_1(list_image_files(mask_dir), max_frames)
    n = min(len(image_files), len(mask_files))
    if n < 5:
        raise ValueError(f"{name}: not enough frames after 4k+1 trim (n={n})")
    image_files = image_files[:n]
    mask_files = mask_files[:n]

    gt_frames: List[Image.Image] = []
    mask_frames: List[Image.Image] = []
    masked_frames: List[Image.Image] = []
    mask_arrays: List[np.ndarray] = []
    for idx, (image_path, mask_path) in enumerate(zip(image_files, mask_files)):
        gt = Image.open(image_path).convert("RGB").resize((width, height), Image.BICUBIC)
        mask_l = Image.open(mask_path).convert("L").resize((width, height), Image.NEAREST)
        mask = (np.asarray(mask_l, dtype=np.uint8) > 127).astype(np.uint8)
        if idx == 0:
            mask = np.zeros_like(mask)
        gt_np = np.asarray(gt, dtype=np.uint8)
        masked_np = gt_np.copy()
        masked_np[mask > 0] = 0
        gt_frames.append(gt)
        mask_frames.append(Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB"))
        masked_frames.append(Image.fromarray(masked_np).convert("RGB"))
        mask_arrays.append(mask)
    return DavisSample(
        name=name,
        gt_frames=gt_frames,
        mask_frames=mask_frames,
        masked_frames=masked_frames,
        mask_arrays=mask_arrays,
        width=width,
        height=height,
    )


def np_frames_to_uint8(frames: Sequence[object]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            arr = np.asarray(frame.convert("RGB"), dtype=np.uint8)
        else:
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0.0, 1.0) * 255.0 if arr.max() <= 1.5 else np.clip(arr, 0.0, 255.0)
                arr = arr.astype(np.uint8)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=-1)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
        out.append(arr)
    return out


def save_frame_sequence(frames: Sequence[np.ndarray], out_dir: Path) -> None:
    ensure_dir(out_dir)
    for idx, frame in enumerate(frames):
        Image.fromarray(frame.astype(np.uint8)).save(out_dir / f"{idx:05d}.png")


def read_frame_sequence(path: Path) -> List[np.ndarray]:
    frames = []
    for item in list_image_files(path):
        frames.append(np.asarray(Image.open(item).convert("RGB"), dtype=np.uint8))
    return frames


def hard_comp(pred_frames: Sequence[np.ndarray], gt_frames: Sequence[Image.Image], masks: Sequence[np.ndarray]) -> List[np.ndarray]:
    comps = []
    for pred, gt_pil, mask in zip(pred_frames, gt_frames, masks):
        gt = np.asarray(gt_pil.convert("RGB"), dtype=np.uint8)
        if pred.shape[:2] != gt.shape[:2]:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        mask3 = (mask > 0)[..., None]
        comp = np.where(mask3, pred, gt)
        comps.append(comp.astype(np.uint8))
    return comps


def write_mp4(frames: Sequence[np.ndarray], path: Path, fps: int) -> None:
    ensure_dir(path.parent)
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()


def add_label(frame: np.ndarray, label: str) -> np.ndarray:
    img = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, img.width, 34], fill=(0, 0, 0))
    draw.text((8, 5), label, fill=(255, 255, 255), font=font)
    return np.asarray(img, dtype=np.uint8)


def make_side_by_side(
    sample: DavisSample,
    baseline_frames: Sequence[np.ndarray],
    adapter_frames: Sequence[np.ndarray],
    out_video: Path,
    frame_dir: Path,
    contact_sheet: Path,
    fps: int,
) -> None:
    ensure_dir(frame_dir)
    rows = []
    n = min(len(sample.gt_frames), len(baseline_frames), len(adapter_frames), len(sample.mask_arrays))
    for idx in range(n):
        gt = np.asarray(sample.gt_frames[idx].convert("RGB"), dtype=np.uint8)
        mask_overlay = overlay_mask(gt, sample.mask_arrays[idx])
        row = np.concatenate(
            [
                add_label(gt, "GT / winner"),
                add_label(mask_overlay, "mask overlay"),
                add_label(baseline_frames[idx], "VideoPainter baseline"),
                add_label(adapter_frames[idx], "VideoPainter + DPO adapter"),
            ],
            axis=1,
        )
        rows.append(row)
        Image.fromarray(row).save(frame_dir / f"{idx:05d}.jpg", quality=92)
    write_mp4(rows, out_video, fps=fps)
    make_contact_sheet(rows, contact_sheet)


def make_contact_sheet(frames: Sequence[np.ndarray], path: Path, max_items: int = 8) -> None:
    ensure_dir(path.parent)
    if not frames:
        return
    if len(frames) <= max_items:
        idxs = list(range(len(frames)))
    else:
        idxs = sorted(set(np.linspace(0, len(frames) - 1, max_items).round().astype(int).tolist()))
    tiles = [Image.fromarray(frames[i]).resize((max(1, frames[i].shape[1] // 2), max(1, frames[i].shape[0] // 2))) for i in idxs]
    w = max(t.width for t in tiles)
    h = sum(t.height for t in tiles)
    sheet = Image.new("RGB", (w, h), (20, 20, 20))
    y = 0
    for tile in tiles:
        sheet.paste(tile, (0, y))
        y += tile.height
    sheet.save(path, quality=92)


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def resolve_branch_dir(path: str) -> Tuple[str, Optional[str]]:
    p = Path(path)
    if (p / "branch").is_dir():
        return str(p), "branch"
    return str(p), None


def setup_videopainter_eval_imports(videopainter_root: str):
    root = Path(videopainter_root).resolve()
    diffusers_src = root / "diffusers" / "src"
    for p in (str(diffusers_src), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # The vendored Diffusers snapshot imports this constant from Transformers.
    # Newer Transformers builds no longer re-export it.
    try:
        import transformers.utils as transformers_utils

        if not hasattr(transformers_utils, "FLAX_WEIGHTS_NAME"):
            transformers_utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
    except Exception:
        pass

    from diffusers import CogVideoXDPMScheduler, CogVideoXI2VDualInpaintAnyLPipeline, CogvideoXBranchModel

    class VideoPainterImports:
        pass

    vp = VideoPainterImports()
    vp.CogVideoXDPMScheduler = CogVideoXDPMScheduler
    vp.CogVideoXI2VDualInpaintAnyLPipeline = CogVideoXI2VDualInpaintAnyLPipeline
    vp.CogvideoXBranchModel = CogvideoXBranchModel
    return vp


def load_pipeline(args: argparse.Namespace, branch_checkpoint: Path):
    vp = setup_videopainter_eval_imports(args.videopainter_root)
    dtype = dtype_from_name(args.dtype)
    branch_root, subfolder = resolve_branch_dir(str(branch_checkpoint))
    if subfolder:
        branch = vp.CogvideoXBranchModel.from_pretrained(branch_root, subfolder=subfolder, torch_dtype=dtype)
    else:
        branch = vp.CogvideoXBranchModel.from_pretrained(branch_root, torch_dtype=dtype)
    branch = branch.to(dtype=dtype)
    pipe = vp.CogVideoXI2VDualInpaintAnyLPipeline.from_pretrained(
        args.base_model,
        branch=branch,
        torch_dtype=dtype,
    )
    pipe.scheduler = vp.CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.text_encoder.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.branch.requires_grad_(False)
    pipe.to(args.device)
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    return pipe


def generate_with_pipe(pipe, args: argparse.Namespace, sample: DavisSample) -> List[np.ndarray]:
    generator = torch.Generator().manual_seed(args.seed)
    result = pipe(
        prompt=args.prompt,
        image=sample.masked_frames[0],
        num_videos_per_prompt=1,
        num_inference_steps=args.num_inference_steps,
        num_frames=len(sample.gt_frames),
        use_dynamic_cfg=True,
        guidance_scale=args.guidance_scale,
        generator=generator,
        video=sample.masked_frames,
        masks=sample.mask_frames,
        strength=1.0,
        replace_gt=False,
        mask_add=True,
        stride=len(sample.gt_frames),
        prev_clip_weight=0.0,
        output_type="np",
    ).frames[0]
    return np_frames_to_uint8(result)


def maybe_generate_model(args: argparse.Namespace, label: str, branch: Path, samples: Sequence[DavisSample]) -> None:
    raw_root = Path(args.output_dir) / label / "raw_frames"
    comp_root = Path(args.output_dir) / label / "comp_frames"
    video_root = Path(args.output_dir) / label / "videos"
    missing = [s for s in samples if not (comp_root / s.name).is_dir()]
    if args.skip_generation_if_exists and not missing:
        print(f"[eval] {label}: comp frames already exist, skip generation")
        return
    pipe = load_pipeline(args, branch)
    try:
        for sample in samples:
            if args.skip_generation_if_exists and (comp_root / sample.name).is_dir():
                continue
            print(f"[eval] {label}: generating {sample.name} frames={len(sample.gt_frames)} steps={args.num_inference_steps}")
            raw = generate_with_pipe(pipe, args, sample)
            raw = raw[: len(sample.gt_frames)]
            comp = hard_comp(raw, sample.gt_frames, sample.mask_arrays)
            save_frame_sequence(raw, raw_root / sample.name)
            save_frame_sequence(comp, comp_root / sample.name)
            write_mp4(comp, video_root / f"{sample.name}.mp4", fps=args.fps)
    finally:
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def compute_metrics(args: argparse.Namespace, samples: Sequence[DavisSample]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    per_frame_rows: List[Dict[str, object]] = []
    for sample in samples:
        gt = [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in sample.gt_frames]
        baseline = read_frame_sequence(Path(args.output_dir) / "baseline" / "comp_frames" / sample.name)
        adapter = read_frame_sequence(Path(args.output_dir) / "adapter" / "comp_frames" / sample.name)
        n = min(len(gt), len(baseline), len(adapter), len(sample.mask_arrays))
        for model_label, pred_frames in (("baseline", baseline), ("adapter", adapter)):
            psnrs, ssims, mask_psnrs = [], [], []
            lpips_vals: List[float] = []
            for idx in range(n):
                pred = pred_frames[idx]
                if pred.shape != gt[idx].shape:
                    pred = cv2.resize(pred, (gt[idx].shape[1], gt[idx].shape[0]), interpolation=cv2.INTER_CUBIC)
                psnr = metric_backend.compute_psnr(gt[idx], pred)
                ssim = metric_backend.compute_ssim(gt[idx], pred)
                mask_psnr = strict_mask_psnr(gt[idx], pred, sample.mask_arrays[idx])
                psnrs.append(psnr)
                ssims.append(ssim)
                mask_psnrs.append(mask_psnr)
                if args.compute_lpips:
                    try:
                        lpips_vals.append(float(metric_backend.LPIPSMetric.compute(gt[idx], pred, device=args.device)))
                    except Exception:
                        lpips_vals.append(float("nan"))
                per_frame_rows.append(
                    {
                        "video": sample.name,
                        "model": model_label,
                        "frame": idx,
                        "psnr": psnr,
                        "ssim": ssim,
                        "strict_mask_pixel_psnr": mask_psnr,
                    }
                )
            rows.append(
                {
                    "video": sample.name,
                    "model": model_label,
                    "frames": n,
                    "PSNR": finite_mean(psnrs),
                    "SSIM": finite_mean(ssims),
                    "strict_mask_pixel_psnr": finite_mean(mask_psnrs),
                    "LPIPS": finite_mean(lpips_vals) if args.compute_lpips else "",
                }
            )
    return rows, per_frame_rows


def strict_mask_psnr(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return float("nan")
    diff = gt.astype(np.float64) - pred.astype(np.float64)
    vals = diff[mask_bool]
    if vals.size == 0:
        return float("nan")
    mse = float(np.mean(vals ** 2))
    return float("inf") if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def finite_mean(values: Iterable[object]) -> float:
    vals = []
    for v in values:
        try:
            f = float(v)
        except Exception:
            continue
        if math.isfinite(f):
            vals.append(f)
    return float(np.mean(vals)) if vals else float("nan")


def summarize_metrics(per_video_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_model: Dict[str, List[Dict[str, object]]] = {}
    for row in per_video_rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    summary = []
    for model, rows in sorted(by_model.items()):
        summary.append(
            {
                "method": f"VideoPainter {model}",
                "model": model,
                "PSNR": finite_mean(row["PSNR"] for row in rows),
                "SSIM": finite_mean(row["SSIM"] for row in rows),
                "strict_mask_pixel_psnr": finite_mean(row["strict_mask_pixel_psnr"] for row in rows),
                "LPIPS": finite_mean(row["LPIPS"] for row in rows),
                "number_of_videos": len(rows),
                "number_of_frames": int(sum(int(row["frames"]) for row in rows)),
                "eval_protocol": "DAVIS frame-wise hard-comp, no mask dilation, no Gaussian blur, no VBench",
            }
        )
    return summary


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_pair_manifest(args: argparse.Namespace, samples: Sequence[DavisSample]) -> None:
    rows = []
    for sample in samples:
        gt_dir = ensure_dir(Path(args.output_dir) / "gt_frames" / sample.name)
        mask_dir = ensure_dir(Path(args.output_dir) / "mask_frames" / sample.name)
        for idx, frame in enumerate(sample.gt_frames):
            frame.save(gt_dir / f"{idx:05d}.png")
            Image.fromarray((sample.mask_arrays[idx] * 255).astype(np.uint8)).save(mask_dir / f"{idx:05d}.png")
        for model in ("baseline", "adapter"):
            rows.append(
                {
                    "sample_id": sample.name,
                    "model_label": f"VideoPainter_{model}",
                    "gt_video_path": str(gt_dir),
                    "prediction_video_path": str(Path(args.output_dir) / model / "comp_frames" / sample.name),
                    "mask_path": str(mask_dir),
                }
            )
    write_csv(Path(args.output_dir) / "pair_manifest.csv", rows)


def make_visuals(args: argparse.Namespace, samples: Sequence[DavisSample]) -> None:
    for sample in samples:
        baseline = read_frame_sequence(Path(args.output_dir) / "baseline" / "comp_frames" / sample.name)
        adapter = read_frame_sequence(Path(args.output_dir) / "adapter" / "comp_frames" / sample.name)
        make_side_by_side(
            sample,
            baseline,
            adapter,
            Path(args.output_dir) / "side_by_side" / f"{sample.name}.mp4",
            Path(args.output_dir) / "frame_by_frame" / sample.name,
            Path(args.output_dir) / "contact_sheets" / f"{sample.name}.jpg",
            fps=args.fps,
        )


def sha256_file(path: Path, max_bytes: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = max_bytes
        while remaining > 0:
            chunk = f.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def checkpoint_audit(args: argparse.Namespace) -> Dict[str, object]:
    baseline_weight = Path(args.baseline_branch) / "diffusion_pytorch_model.safetensors"
    adapter_branch = Path(args.adapter_checkpoint) / "branch"
    adapter_weight = adapter_branch / "diffusion_pytorch_model.safetensors"
    if not baseline_weight.exists():
        # The baseline branch may be passed one level above branch.
        baseline_weight = Path(args.baseline_branch) / "branch" / "diffusion_pytorch_model.safetensors"
    if not adapter_weight.exists():
        adapter_weight = Path(args.adapter_checkpoint) / "diffusion_pytorch_model.safetensors"
    audit = {
        "baseline_checkpoint": args.baseline_branch,
        "adapter_checkpoint": args.adapter_checkpoint,
        "baseline_weight": str(baseline_weight),
        "adapter_weight": str(adapter_weight),
        "baseline_weight_exists": baseline_weight.exists(),
        "adapter_weight_exists": adapter_weight.exists(),
        "adapter_load_path": str(args.adapter_checkpoint),
        "fallback_used": False,
    }
    if baseline_weight.exists() and adapter_weight.exists():
        audit["baseline_sha256_head"] = sha256_file(baseline_weight)
        audit["adapter_sha256_head"] = sha256_file(adapter_weight)
        audit["weights_different"] = audit["baseline_sha256_head"] != audit["adapter_sha256_head"]
    else:
        audit["weights_different"] = False
    return audit


def write_reports(args: argparse.Namespace, videos: Sequence[str], checkpoint_info: Dict[str, object]) -> None:
    davis_root = Path(args.davis_root)
    all_videos = available_videos(davis_root)
    precheck = f"""# VideoPainter Adapter Eval Precheck

- project_root: `{args.project_root}`
- videopainter_root: `{args.videopainter_root}`
- base_model: `{args.base_model}`
- baseline_branch: `{args.baseline_branch}`
- adapter_checkpoint: `{args.adapter_checkpoint}`
- output_dir: `{args.output_dir}`
- DAVIS root: `{args.davis_root}`
- DAVIS available videos: {len(all_videos)}
- selected videos: {len(videos)}
- debug: {args.debug}
- num_frames: {args.num_frames}
- num_inference_steps: {args.num_inference_steps}
- hard comp: yes
- mask dilation: no
- Gaussian blur: no
- VBench: no
- metric backend: `inference/metrics.py`
"""
    write_text(PROJECT_ROOT / "reports" / "videopainter_adapter_eval_precheck.md", precheck)

    ckpt_report = "# VideoPainter Adapter Checkpoint Loading Audit\n\n"
    ckpt_report += "\n".join(f"- {k}: `{v}`" for k, v in checkpoint_info.items())
    ckpt_report += "\n\nConclusion: adapter checkpoint is considered safe for eval only if `weights_different=True` and both weight files exist.\n"
    write_text(PROJECT_ROOT / "reports" / "videopainter_adapter_checkpoint_loading_audit.md", ckpt_report)

    davis_report = f"""# VideoPainter DAVIS Eval Set Audit

- davis_root: `{args.davis_root}`
- image_dir: `{davis_root / 'JPEGImages_432_240'}`
- mask_dir: `{davis_root / 'test_masks'}`
- available paired videos: {len(all_videos)}
- selected videos: {len(videos)}
- selected names: {', '.join(videos)}
- eval is full DAVIS50: {len(videos) == 50 and not args.debug}
- frame convention: each clip trimmed to 4k+1 frames, capped at {args.num_frames}
- resize for VideoPainter inference: {args.width}x{args.height}
"""
    write_text(PROJECT_ROOT / "reports" / "videopainter_davis_eval_set_audit.md", davis_report)


def classify_cases(per_video_rows: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    paired: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in per_video_rows:
        paired.setdefault(str(row["video"]), {})[str(row["model"])] = row
    deltas = []
    for video, pair in paired.items():
        if "baseline" not in pair or "adapter" not in pair:
            continue
        base = pair["baseline"]
        adapt = pair["adapter"]
        delta_psnr = float(adapt["PSNR"]) - float(base["PSNR"])
        delta_ssim = float(adapt["SSIM"]) - float(base["SSIM"])
        deltas.append(
            {
                "video": video,
                "delta_psnr": delta_psnr,
                "delta_ssim": delta_ssim,
                "baseline_psnr": base["PSNR"],
                "adapter_psnr": adapt["PSNR"],
                "baseline_ssim": base["SSIM"],
                "adapter_ssim": adapt["SSIM"],
            }
        )
    successes = sorted([x for x in deltas if x["delta_psnr"] > 0 and x["delta_ssim"] >= -1e-4], key=lambda x: x["delta_psnr"], reverse=True)
    failures = sorted([x for x in deltas if x["delta_psnr"] < 0 or x["delta_ssim"] < -1e-4], key=lambda x: x["delta_psnr"])
    ties = sorted([x for x in deltas if abs(x["delta_psnr"]) <= 0.05 and abs(x["delta_ssim"]) <= 0.001], key=lambda x: abs(x["delta_psnr"]))
    return successes[:8], ties[:8], failures[:8]


def write_final_reports(args: argparse.Namespace, summary_rows: Sequence[Dict[str, object]], per_video_rows: Sequence[Dict[str, object]]) -> None:
    successes, ties, failures = classify_cases(per_video_rows)
    summary_json = json.dumps(list(summary_rows), indent=2, ensure_ascii=False)
    success_text = "\n".join(f"- {x['video']}: PSNR {x['baseline_psnr']:.4f} -> {x['adapter_psnr']:.4f}, SSIM {x['baseline_ssim']:.4f} -> {x['adapter_ssim']:.4f}" for x in successes) or "- none"
    tie_text = "\n".join(f"- {x['video']}: dPSNR={x['delta_psnr']:.4f}, dSSIM={x['delta_ssim']:.5f}" for x in ties) or "- none"
    failure_text = "\n".join(f"- {x['video']}: PSNR {x['baseline_psnr']:.4f} -> {x['adapter_psnr']:.4f}, SSIM {x['baseline_ssim']:.4f} -> {x['adapter_ssim']:.4f}" for x in failures) or "- none"
    report = f"""# VideoPainter Adapter Gate2000 DAVIS Eval

## Protocol

- Baseline: VideoPainter official branch checkpoint.
- Adapter: Exp14 gate2000 `last_weights`.
- Dataset: DAVIS under `{args.davis_root}`.
- Output: `{args.output_dir}`.
- VideoPainter inference steps: {args.num_inference_steps}.
- VideoPainter frames per clip cap: {args.num_frames}; clips are trimmed to 4k+1 frames.
- Hard comp: prediction inside mask + GT outside mask.
- Mask dilation: off.
- Gaussian blur: off.
- VBench: not used.
- Metric backend: `inference/metrics.py`.

## Summary

```json
{summary_json}
```

## Success Candidates

{success_text}

## Tie Candidates

{tie_text}

## Failure Candidates

{failure_text}

## Interpretation

This report should be read together with `reports/videopainter_adapter_gate2000_dpo_diag_summary.md`.
The existing training diagnostics were already flagged as DPO_SATURATED / LOSER_DOMINANT / GRAD_SPIKE_OBSERVED,
so a metric or visual drop would indicate the current DPO objective is not yet stable for VideoPainter.
"""
    for path in [
        PROJECT_ROOT / "reports" / "videopainter_adapter_gate2000_eval_report.md",
        PROJECT_ROOT / "exp14_adapter_videopainter" / "reports" / "eval_report.md",
        Path(args.output_dir) / "report.md",
    ]:
        write_text(path, report)

    write_text(Path(args.output_dir) / "metrics" / "summary.md", report)
    write_text(Path(args.output_dir) / "metrics" / "summary.json", json.dumps(list(summary_rows), indent=2, ensure_ascii=False))
    write_csv(Path(args.output_dir) / "metrics" / "summary.csv", summary_rows)
    write_csv(Path(args.output_dir) / "metrics" / "per_video.csv", per_video_rows)


def write_index(args: argparse.Namespace, samples: Sequence[DavisSample]) -> None:
    links = []
    for sample in samples:
        links.append(
            f'<li><a href="side_by_side/{sample.name}.mp4">{sample.name}</a> '
            f'<img src="contact_sheets/{sample.name}.jpg" style="max-width:100%; display:block; margin:8px 0 24px 0;"></li>'
        )
    html = "<html><body><h1>Exp14 VideoPainter Adapter DAVIS Eval</h1><ul>" + "\n".join(links) + "</ul></body></html>"
    write_text(Path(args.output_dir) / "index.html", html)


def main() -> int:
    args = parse_args()
    args.project_root = str(Path(args.project_root).resolve())
    if args.videopainter_root is None:
        args.videopainter_root = str(Path(args.project_root) / "third_party" / "VideoPainter")
    if args.base_model is None:
        args.base_model = str(Path(args.videopainter_root) / "ckpt" / "CogVideoX-5b-I2V")
    if args.baseline_branch is None:
        args.baseline_branch = str(Path(args.videopainter_root) / "ckpt" / "VideoPainter" / "checkpoints" / "branch")
    if args.adapter_checkpoint is None:
        args.adapter_checkpoint = str(Path(args.project_root) / "exp14_adapter_videopainter" / "runs" / "gate2000" / "last_weights")
    args.output_dir = str((Path(args.project_root) / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir).resolve())

    required = [
        Path(args.videopainter_root),
        Path(args.base_model),
        Path(args.baseline_branch),
        Path(args.adapter_checkpoint),
        Path(args.davis_root),
        PROJECT_ROOT / "inference" / "metrics.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths: " + ", ".join(missing))

    checkpoint_info = checkpoint_audit(args)
    if not checkpoint_info.get("baseline_weight_exists") or not checkpoint_info.get("adapter_weight_exists"):
        raise FileNotFoundError(f"Checkpoint weight missing: {checkpoint_info}")
    if not checkpoint_info.get("weights_different"):
        raise RuntimeError("Adapter weight hash matches baseline or could not be compared; refusing to eval fallback.")

    names = available_videos(Path(args.davis_root))
    if args.video_names.strip():
        requested = [x.strip() for x in args.video_names.split(",") if x.strip()]
        names = [x for x in requested if x in set(names)]
    if args.limit_videos:
        names = names[: args.limit_videos]
    if not names:
        raise ValueError("No DAVIS videos selected")

    ensure_dir(Path(args.output_dir))
    write_reports(args, names, checkpoint_info)
    samples = [load_davis_sample(Path(args.davis_root), name, args.width, args.height, args.num_frames) for name in names]
    build_pair_manifest(args, samples)

    maybe_generate_model(args, "baseline", Path(args.baseline_branch), samples)
    maybe_generate_model(args, "adapter", Path(args.adapter_checkpoint), samples)
    per_video_rows, per_frame_rows = compute_metrics(args, samples)
    summary_rows = summarize_metrics(per_video_rows)
    make_visuals(args, samples)
    write_final_reports(args, summary_rows, per_video_rows)
    write_csv(Path(args.output_dir) / "metrics" / "per_frame.csv", per_frame_rows)
    write_index(args, samples)
    print(json.dumps({"summary": summary_rows, "output_dir": args.output_dir}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
