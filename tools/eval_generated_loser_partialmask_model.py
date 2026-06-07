#!/usr/bin/env python
"""Evaluate DiffuEraser checkpoints on generated-loser partial-mask manifests.

This is intentionally task-specific: it uses the D2 manifest winner video and
the manifest mask_path as the inpainting input, then composites each model
output back onto the winner outside the mask. The resulting metrics answer a
different question from full-mask VBench/qual30: whether an Exp7 partial-mask
checkpoint actually helps on the partial-mask inpainting task it was trained
for.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import imageio
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.metrics import compute_psnr, compute_ssim  # noqa: E402
from tools.generate_diffueraser_fullmask_vbench import build_pipeline  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
PIL_RESAMPLING = getattr(Image, "Resampling", Image)
DIAG_METRICS = [
    "dpo_loss",
    "implicit_acc",
    "winner_gap_reg",
    "win_gap",
    "lose_gap",
    "mse_w_over_ref_mse_w",
    "mse_l_over_ref_mse_l",
    "sigma_term",
    "kl_divergence",
    "loser_dominant_ratio",
]


@dataclass(frozen=True)
class Sample:
    sample_id: str
    row_index: int
    prompt: str
    win_video_path: Path
    mask_path: Path
    d2_loser_path: Optional[Path]


@dataclass(frozen=True)
class ModelSpec:
    label: str
    weights_dir: Path
    is_base: bool = False


@dataclass
class ModelStatus:
    label: str
    weights_dir: str
    status: str
    reason: str = ""


def parse_label_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Expected LABEL=PATH, got {value!r}")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"Empty checkpoint label in {value!r}")
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)
    return safe, Path(path).expanduser()


def jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def first_existing_text(row: Dict[str, Any], keys: Sequence[str], default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def make_sample(row: Dict[str, Any], row_index: int) -> Sample:
    prompt = first_existing_text(row, ["prompt", "caption", "text", "video_caption"], default=f"sample {row_index}")
    win = Path(str(row["win_video_path"]))
    mask = Path(str(row["mask_path"]))
    loser_value = row.get("final_loser_video_path") or row.get("loser_video_path")
    d2_loser = Path(str(loser_value)) if loser_value else None
    sample_name = first_existing_text(row, ["sample_id", "video_id", "id"], default=f"{row_index:06d}")
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in sample_name)[:100]
    sample_id = f"{row_index:06d}_{safe_name}"
    return Sample(sample_id, row_index, prompt, win, mask, d2_loser)


def sample_rows(rows: Sequence[Dict[str, Any]], count: int, seed: int) -> List[Sample]:
    if count <= 0:
        return []
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    selected = sorted(indices[: min(count, len(indices))])
    return [make_sample(rows[i], i) for i in selected]


def image_files(path: Path) -> List[Path]:
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def resolve_ffmpeg() -> str:
    candidates = [
        os.environ.get("FFMPEG_BINARY"),
        shutil.which("ffmpeg"),
    ]
    try:
        import imageio_ffmpeg  # type: ignore

        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    candidates.extend(
        [
            "/mnt/workspace/hongfeng/miniconda3/bin/ffmpeg",
            "/home/ubuntu/.local/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2",
        ]
    )
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise RuntimeError("ffmpeg not found for video decoding")


def read_video_frames(path: Path, num_frames: int, size: Tuple[int, int], is_mask: bool = False) -> List[Image.Image]:
    width, height = size
    frames: List[Image.Image] = []
    resample = PIL_RESAMPLING.NEAREST if is_mask else PIL_RESAMPLING.BILINEAR
    mode = "L" if is_mask else "RGB"

    if path.is_dir():
        files = image_files(path)
        if not files:
            raise FileNotFoundError(f"No image frames found in {path}")
        for frame_path in files[:num_frames]:
            frames.append(Image.open(frame_path).convert(mode).resize((width, height), resample))
    elif path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        ffmpeg = resolve_ffmpeg()
        pix_fmt = "gray" if is_mask else "rgb24"
        channels = 1 if is_mask else 3
        frame_size = width * height * channels
        scale_filter = f"scale={width}:{height}:flags=neighbor" if is_mask else f"scale={width}:{height}"
        cmd = [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            scale_filter,
            "-frames:v",
            str(num_frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            "-",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        count = len(proc.stdout) // frame_size
        if count == 0:
            raise RuntimeError(
                f"ffmpeg decoded zero frames from {path}: {proc.stderr.decode(errors='replace')}"
            )
        for idx in range(min(count, num_frames)):
            chunk = proc.stdout[idx * frame_size : (idx + 1) * frame_size]
            if is_mask:
                arr = np.frombuffer(chunk, dtype=np.uint8).reshape((height, width))
            else:
                arr = np.frombuffer(chunk, dtype=np.uint8).reshape((height, width, channels))
            frames.append(Image.fromarray(arr).convert(mode))
    elif path.is_file() and path.suffix.lower() in IMG_EXTS:
        frames.append(Image.open(path).convert(mode).resize((width, height), resample))
    else:
        raise FileNotFoundError(f"Unsupported video/frame path: {path}")

    if not frames:
        raise RuntimeError(f"No frames loaded from {path}")
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    return frames[:num_frames]


def save_video(frames: Sequence[Image.Image], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in frames]
    try:
        imageio.mimsave(path, arrays, fps=int(fps), codec="libx264", macro_block_size=1)
        return
    except Exception as imageio_error:
        height, width = arrays[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {path}") from imageio_error
        try:
            for frame in arrays:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()


def masked_winner_images(winner_frames: Sequence[Image.Image], mask_frames: Sequence[Image.Image]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for winner, mask in zip(winner_frames, mask_frames):
        win_arr = np.asarray(winner.convert("RGB"), dtype=np.uint8)
        mask_arr = np.asarray(mask.convert("L"), dtype=np.uint8) > 127
        masked = win_arr.copy()
        masked[mask_arr] = 0
        images.append(Image.fromarray(masked, mode="RGB"))
    return images


def composite_inside_mask(
    pred_frames: Sequence[Image.Image],
    winner_frames: Sequence[Image.Image],
    mask_frames: Sequence[Image.Image],
) -> List[Image.Image]:
    comp_frames: List[Image.Image] = []
    for pred, winner, mask in zip(pred_frames, winner_frames, mask_frames):
        pred_arr = np.asarray(pred.convert("RGB"), dtype=np.uint8)
        win_arr = np.asarray(winner.convert("RGB"), dtype=np.uint8)
        mask_arr = np.asarray(mask.convert("L"), dtype=np.uint8) > 127
        if pred_arr.shape != win_arr.shape:
            pred_arr = cv2.resize(pred_arr, (win_arr.shape[1], win_arr.shape[0]), interpolation=cv2.INTER_LINEAR)
        comp = win_arr.copy()
        comp[mask_arr] = pred_arr[mask_arr]
        comp_frames.append(Image.fromarray(comp, mode="RGB"))
    return comp_frames


def mask_overlay_frames(winner_frames: Sequence[Image.Image], mask_frames: Sequence[Image.Image]) -> List[Image.Image]:
    overlays: List[Image.Image] = []
    for winner, mask in zip(winner_frames, mask_frames):
        win_arr = np.asarray(winner.convert("RGB"), dtype=np.float32)
        mask_arr = np.asarray(mask.convert("L"), dtype=np.uint8) > 127
        color = np.zeros_like(win_arr)
        color[..., 0] = 255.0
        color[..., 1] = 40.0
        out = win_arr.copy()
        out[mask_arr] = 0.55 * win_arr[mask_arr] + 0.45 * color[mask_arr]
        overlays.append(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB"))
    return overlays


def stack_columns(columns: Sequence[Sequence[Image.Image]], labels: Sequence[str]) -> List[Image.Image]:
    n = min(len(col) for col in columns)
    out: List[Image.Image] = []
    for frame_idx in range(n):
        arrays = [np.asarray(col[frame_idx].convert("RGB"), dtype=np.uint8) for col in columns]
        labeled = []
        for arr, label in zip(arrays, labels):
            canvas = arr.copy()
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
            cv2.putText(
                canvas,
                label[:36],
                (6, 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            labeled.append(canvas)
        out.append(Image.fromarray(np.concatenate(labeled, axis=1), mode="RGB"))
    return out


def is_direct_diffusers_weights(path: Path) -> bool:
    return (path / "unet_main" / "config.json").is_file() and (path / "brushnet" / "config.json").is_file()


def infer_stage(weights_dir: Path) -> str:
    config_path = weights_dir / "unet_main" / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        cls_name = json.load(f).get("_class_name")
    return "stage2" if cls_name == "UNetMotionModel" else "stage1"


def load_model_pipeline(args: argparse.Namespace, spec: ModelSpec, device: torch.device):
    stage = infer_stage(spec.weights_dir)
    pipe_args = SimpleNamespace(
        base_model_name_or_path=str(args.base_model_name_or_path),
        vae_path=str(args.vae_path),
        weights_path=spec.weights_dir,
        stage=stage,
        torch_dtype=args.torch_dtype,
        vae_dtype=args.vae_dtype,
        revision=args.revision,
        variant=args.variant,
        show_progress=args.show_progress,
    )
    pipeline, weight_dtype = build_pipeline(pipe_args, device)
    return pipeline, weight_dtype


def run_inference_for_sample(
    args: argparse.Namespace,
    pipeline: Any,
    weight_dtype: torch.dtype,
    device: torch.device,
    spec: ModelSpec,
    sample: Sample,
    winner_frames: Sequence[Image.Image],
    mask_frames: Sequence[Image.Image],
    output_path: Path,
) -> Path:
    if args.skip_existing and output_path.exists():
        return output_path

    images = masked_winner_images(winner_frames, mask_frames)
    generator = torch.Generator(device=device).manual_seed(args.seed + sample.row_index)
    autocast_dtype = weight_dtype if device.type == "cuda" and weight_dtype in {torch.float16, torch.bfloat16} else None

    with torch.no_grad():
        if autocast_dtype is None:
            pred_frames = pipeline(
                num_frames=args.frames,
                prompt=sample.prompt,
                images=images,
                masks=list(mask_frames),
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).frames
        else:
            with torch.autocast("cuda", dtype=autocast_dtype):
                pred_frames = pipeline(
                    num_frames=args.frames,
                    prompt=sample.prompt,
                    images=images,
                    masks=list(mask_frames),
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                ).frames

    comp_frames = composite_inside_mask(pred_frames, winner_frames, mask_frames)
    save_video(comp_frames, output_path, args.fps)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output_path


def masked_psnr(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    diff = gt.astype(np.float64) - pred.astype(np.float64)
    diff = diff[mask]
    mse = float(np.mean(diff ** 2))
    return float("inf") if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def bbox_from_mask(mask: np.ndarray, pad: int = 4) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape[:2]
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return None
    return x0, y0, x1, y1


def safe_ssim(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return compute_ssim(a, b)
    except Exception:
        return float("nan")


def read_comp_video(path: Path, frames: int, size: Tuple[int, int]) -> List[Image.Image]:
    return read_video_frames(path, frames, size, is_mask=False)


def compute_video_metrics(
    winner_frames: Sequence[Image.Image],
    comp_frames: Sequence[Image.Image],
    mask_frames: Sequence[Image.Image],
) -> Dict[str, float]:
    whole_psnr: List[float] = []
    whole_ssim: List[float] = []
    mask_psnr: List[float] = []
    mask_ssim: List[float] = []
    boundary_psnr: List[float] = []
    boundary_ssim: List[float] = []
    outside_mean: List[float] = []
    outside_max: List[float] = []

    winner_arrays: List[np.ndarray] = []
    comp_arrays: List[np.ndarray] = []

    for winner, comp, mask in zip(winner_frames, comp_frames, mask_frames):
        gt = np.asarray(winner.convert("RGB"), dtype=np.uint8)
        pred = np.asarray(comp.convert("RGB"), dtype=np.uint8)
        m = np.asarray(mask.convert("L"), dtype=np.uint8) > 127
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        winner_arrays.append(gt)
        comp_arrays.append(pred)

        whole_psnr.append(compute_psnr(gt, pred))
        whole_ssim.append(safe_ssim(gt, pred))
        mask_psnr.append(masked_psnr(gt, pred, np.repeat(m[:, :, None], 3, axis=2)))
        bbox = bbox_from_mask(m)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            mask_ssim.append(safe_ssim(gt[y0:y1, x0:x1], pred[y0:y1, x0:x1]))
        else:
            mask_ssim.append(float("nan"))

        kernel = np.ones((7, 7), np.uint8)
        dil = cv2.dilate(m.astype(np.uint8), kernel, iterations=1).astype(bool)
        ero = cv2.erode(m.astype(np.uint8), kernel, iterations=1).astype(bool)
        boundary = np.logical_xor(dil, ero)
        boundary_psnr.append(masked_psnr(gt, pred, np.repeat(boundary[:, :, None], 3, axis=2)))
        bbox_b = bbox_from_mask(boundary)
        if bbox_b is not None:
            x0, y0, x1, y1 = bbox_b
            boundary_ssim.append(safe_ssim(gt[y0:y1, x0:x1], pred[y0:y1, x0:x1]))
        else:
            boundary_ssim.append(float("nan"))

        outside = ~m
        diff = np.abs(gt.astype(np.float32) - pred.astype(np.float32))
        if np.any(outside):
            outside_diff = diff[np.repeat(outside[:, :, None], 3, axis=2)]
            outside_mean.append(float(np.mean(outside_diff)))
            outside_max.append(float(np.max(outside_diff)))
        else:
            outside_mean.append(float("nan"))
            outside_max.append(float("nan"))

    temporal_diff = []
    temporal_delta = []
    for i in range(1, len(comp_arrays)):
        pred_t = np.abs(comp_arrays[i].astype(np.float32) - comp_arrays[i - 1].astype(np.float32))
        gt_t = np.abs(winner_arrays[i].astype(np.float32) - winner_arrays[i - 1].astype(np.float32))
        temporal_diff.append(float(np.mean(pred_t)))
        temporal_delta.append(float(np.mean(np.abs(pred_t - gt_t))))

    def mean(values: Sequence[float]) -> float:
        finite = [float(v) for v in values if math.isfinite(float(v))]
        return float(np.mean(finite)) if finite else float("nan")

    return {
        "whole_video_psnr": mean(whole_psnr),
        "whole_video_ssim": mean(whole_ssim),
        "mask_region_psnr": mean(mask_psnr),
        "mask_region_ssim": mean(mask_ssim),
        "boundary_psnr": mean(boundary_psnr),
        "boundary_ssim": mean(boundary_ssim),
        "outside_region_diff_mean": mean(outside_mean),
        "outside_region_diff_max": mean(outside_max),
        "temporal_diff": mean(temporal_diff),
        "temporal_diff_delta_vs_gt": mean(temporal_delta),
    }


def finite_values(values: Iterable[Any]) -> List[float]:
    out = []
    for value in values:
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, p))


def summarize_metrics(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(str(row["model_label"]), []).append(row)
    metric_keys = [
        "whole_video_psnr",
        "whole_video_ssim",
        "mask_region_psnr",
        "mask_region_ssim",
        "boundary_psnr",
        "boundary_ssim",
        "outside_region_diff_mean",
        "outside_region_diff_max",
        "temporal_diff",
        "temporal_diff_delta_vs_gt",
    ]
    summary: List[Dict[str, Any]] = []
    for label, label_rows in sorted(by_label.items()):
        item: Dict[str, Any] = {"model_label": label, "sample_count": len(label_rows)}
        for key in metric_keys:
            values = finite_values(row.get(key) for row in label_rows)
            item[f"{key}_mean"] = float(np.mean(values)) if values else float("nan")
            item[f"{key}_median"] = float(np.median(values)) if values else float("nan")
            item[f"{key}_p90"] = percentile(values, 90)
        summary.append(item)
    return summary


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    cols = [
        "model_label",
        "sample_count",
        "mask_region_psnr_mean",
        "mask_region_ssim_mean",
        "whole_video_psnr_mean",
        "whole_video_ssim_mean",
        "outside_region_diff_mean_mean",
        "temporal_diff_delta_vs_gt_mean",
    ]
    lines = ["# Partial-Mask Metric Summary", "", "| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col, "")
            vals.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def diagnostics_table(path: Path) -> Optional[List[Dict[str, Any]]]:
    if not path.is_file():
        return None
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stats_for_rows(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, float]:
    values = finite_values(row.get(key) for row in rows)
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": percentile(values, 90),
        "max": float(np.max(values)),
    }


def fraction(rows: Sequence[Dict[str, Any]], key: str, op: str, threshold: float) -> float:
    values = finite_values(row.get(key) for row in rows)
    if not values:
        return float("nan")
    if op == ">":
        hits = sum(v > threshold for v in values)
    elif op == "<":
        hits = sum(v < threshold for v in values)
    else:
        raise ValueError(op)
    return float(hits) / float(len(values))


def write_dpo_diag_summary(stage1_csv: Optional[Path], stage2_csv: Optional[Path], out_path: Path) -> None:
    stage_paths = {"Stage1": stage1_csv, "Stage2": stage2_csv}
    sections: List[str] = ["# Exp7 Gate1500 DPO Diagnostics Summary", ""]
    all_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for stage, csv_path in stage_paths.items():
        rows = diagnostics_table(csv_path) if csv_path else None
        sections.extend([f"## {stage}", ""])
        if not rows:
            sections.extend([f"No diagnostics CSV found: `{csv_path}`", ""])
            continue
        all_stats[stage] = {metric: stats_for_rows(rows, metric) for metric in DIAG_METRICS}
        sections.append("| metric | mean | median | p90 | max |")
        sections.append("| --- | ---: | ---: | ---: | ---: |")
        for metric in DIAG_METRICS:
            s = all_stats[stage][metric]
            sections.append(
                f"| `{metric}` | {s['mean']:.6g} | {s['median']:.6g} | {s['p90']:.6g} | {s['max']:.6g} |"
            )
        sections.extend(["", "Fractions:", ""])
        frac_specs = [
            ("dpo_loss", "<", 1e-3),
            ("implicit_acc", ">", 0.99),
            ("mse_w_over_ref_mse_w", ">", 5.0),
            ("win_gap", ">", 0.5),
            ("sigma_term", ">", 0.99),
            ("kl_divergence", ">", 1.0),
            ("loser_dominant_ratio", ">", 0.99),
        ]
        sections.append("| condition | fraction |")
        sections.append("| --- | ---: |")
        for key, op, threshold in frac_specs:
            sections.append(f"| `{key} {op} {threshold}` | {fraction(rows, key, op, threshold):.6g} |")
        sections.append("")

    stage2 = all_stats.get("Stage2", {})
    stage1 = all_stats.get("Stage1", {})
    sections.extend(["## Interpretation", ""])
    if stage2:
        win_gap_p90 = stage2.get("win_gap", {}).get("p90", float("nan"))
        loser_ratio_p90 = stage2.get("loser_dominant_ratio", {}).get("p90", float("nan"))
        mse_l_p90 = stage2.get("mse_l_over_ref_mse_l", {}).get("p90", float("nan"))
        sections.append(
            f"- Winner-gap control: Stage2 `win_gap` p90 = {win_gap_p90:.6g}; this is bounded relative to the old collapsed Exp5 mode."
        )
        sections.append(
            f"- Loser degradation: Stage2 `mse_l_over_ref_mse_l` p90 = {mse_l_p90:.6g}; high values mean the loser shortcut is still strong."
        )
        sections.append(
            f"- Loser dominance: Stage2 `loser_dominant_ratio` p90 = {loser_ratio_p90:.6g}; values near 1 mean most steps satisfy DPO by making losers worse."
        )
    if stage1 and stage2:
        l1 = stage1.get("mse_l_over_ref_mse_l", {}).get("p90", float("nan"))
        l2 = stage2.get("mse_l_over_ref_mse_l", {}).get("p90", float("nan"))
        if math.isfinite(l1) and math.isfinite(l2):
            relation = "amplifies" if l2 > l1 else "does not amplify"
            sections.append(f"- Stage2 {relation} loser degradation by p90 ratio: Stage1={l1:.6g}, Stage2={l2:.6g}.")
    sections.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sections), encoding="utf-8")


def write_index(path: Path, side_rows: Sequence[Dict[str, Any]]) -> None:
    rel_root = path.parent
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Exp7 partial-mask eval</title>",
        "<style>body{font-family:sans-serif;margin:24px} video{max-width:100%;height:auto} .item{margin-bottom:28px}</style>",
        "</head><body>",
        "<h1>Exp7 Partial-Mask Eval</h1>",
    ]
    for row in side_rows:
        video_path = Path(str(row["side_by_side_path"]))
        try:
            rel = video_path.relative_to(rel_root)
        except ValueError:
            rel = video_path
        lines.append("<div class='item'>")
        lines.append(f"<h3>{row.get('checkpoint_label', '')} / {row.get('sample_id', '')}</h3>")
        lines.append(f"<p>{row.get('prompt', '')}</p>")
        lines.append(f"<video controls src='{rel.as_posix()}'></video>")
        lines.append("</div>")
    lines.append("</body></html>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def decision_report(
    path: Path,
    output_root: Path,
    model_statuses: Sequence[ModelStatus],
    summary_rows: Sequence[Dict[str, Any]],
    dpo_summary_path: Optional[Path],
) -> None:
    by_label = {str(row["model_label"]): row for row in summary_rows}
    base = by_label.get("DiffuEraser-base")
    candidates = [row for row in summary_rows if row.get("model_label") != "DiffuEraser-base"]

    def score(row: Dict[str, Any]) -> Tuple[float, float]:
        return (
            float(row.get("mask_region_ssim_mean", float("nan"))),
            float(row.get("mask_region_psnr_mean", float("nan"))),
        )

    best = max(candidates, key=score) if candidates else None
    beats_base = False
    if base and best:
        best_psnr = float(best.get("mask_region_psnr_mean", float("nan")))
        base_psnr = float(base.get("mask_region_psnr_mean", float("nan")))
        best_ssim = float(best.get("mask_region_ssim_mean", float("nan")))
        base_ssim = float(base.get("mask_region_ssim_mean", float("nan")))
        beats_base = math.isfinite(best_psnr) and math.isfinite(base_psnr) and best_psnr > base_psnr + 0.1 and best_ssim >= base_ssim - 0.005

    stage1_labels = [row for row in candidates if str(row.get("model_label", "")).startswith("Stage1")]
    stage2_last = by_label.get("Stage2_last")
    best_stage1 = max(stage1_labels, key=score) if stage1_labels else None

    lines = [
        "# Exp7 Gate1500 Partial-Mask Evaluation Report",
        "",
        f"Output root: `{output_root}`",
        "",
        "## Model Status",
        "",
        "| label | status | weights_dir | reason |",
        "| --- | --- | --- | --- |",
    ]
    for status in model_statuses:
        lines.append(f"| `{status.label}` | {status.status} | `{status.weights_dir}` | {status.reason} |")
    lines.extend(["", "## Summary", ""])
    if best:
        lines.append(f"- Best evaluated Exp7 checkpoint by mask-region SSIM/PSNR: `{best['model_label']}`.")
    else:
        lines.append("- No Exp7 checkpoint was directly evaluable.")
    if base and best:
        verdict = "yes" if beats_base else "no / inconclusive"
        lines.append(f"- Does Exp7 beat DiffuEraser-base on true partial-mask eval? {verdict}.")
    stage1_better_than_stage2 = False
    if best_stage1 and stage2_last:
        stage1_score = score(best_stage1)
        stage2_score = score(stage2_last)
        stage1_better_than_stage2 = stage1_score > stage2_score
        better = "Stage1 is better" if stage1_better_than_stage2 else "Stage2_last is better"
        lines.append(f"- Stage1-vs-Stage2: {better} by mask-region SSIM/PSNR.")
    elif stage2_last and not best_stage1:
        lines.append("- Stage1 early checkpoints were not directly evaluable as exported weights, so early-vs-Stage2 cannot be decided.")
    if beats_base and stage1_better_than_stage2:
        lines.append(
            "- Full Exp7 4000+4000 recommendation: do not launch yet; Stage1 beats base on the true partial-mask task, but Stage2 regresses."
        )
        lines.append(
            "- No-lose-gap gate recommendation: recommended next gate if visual review confirms Stage2 loser-degradation artifacts."
        )
    elif beats_base:
        lines.append(
            "- Full Exp7 4000+4000 recommendation: review visuals before launch; metrics beat base but checkpoint dynamics need confirmation."
        )
        lines.append("- No-lose-gap gate recommendation: keep prepared as the fallback.")
    else:
        lines.append(
            "- Full Exp7 4000+4000 recommendation: hold unless this partial-mask report shows a clear visual and metric win over base."
        )
        lines.append(
            "- No-lose-gap gate recommendation: prepare and use if partial-mask eval still shows loser-degradation artifacts."
        )
    if dpo_summary_path:
        lines.append(f"- DPO diagnostics summary: `{dpo_summary_path}`.")
    lines.extend(["", "## Metric Table", ""])
    cols = ["model_label", "mask_region_psnr_mean", "mask_region_ssim_mean", "outside_region_diff_mean_mean", "temporal_diff_delta_vs_gt_mean"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in summary_rows:
        values = []
        for col in cols:
            value = row.get(col, "")
            values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--base_weights_dir", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", type=parse_label_path, default=[])
    parser.add_argument("--base_model_name_or_path", required=True, type=Path)
    parser.add_argument("--vae_path", required=True, type=Path)
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--num_samples_metric", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=12.0)
    parser.add_argument("--torch_dtype", choices=["auto", "fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--vae_dtype", choices=["auto", "fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--no_d2_loser", action="store_true", help="Do not add D2 generated-loser videos as the optional fifth column.")
    parser.add_argument("--num_shards", type=int, default=1, help="Split generation samples into this many disjoint shards.")
    parser.add_argument("--shard_index", type=int, default=0, help="Zero-based shard index for this worker.")
    parser.add_argument("--generate_only", action="store_true", help="Only generate sample videos; skip metrics, side-by-side, and reports.")
    parser.add_argument("--stage1_diag_csv", type=Path, default=None)
    parser.add_argument("--stage2_diag_csv", type=Path, default=None)
    parser.add_argument("--dpo_summary_out", type=Path, default=None)
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")
    if args.num_shards > 1 and not args.generate_only:
        raise ValueError("--num_shards > 1 is for --generate_only workers; run one final unsharded pass to write metrics/report")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output_dir / "samples"
    side_dir = args.output_dir / "side_by_side"
    metrics_dir = args.output_dir / "metrics"
    for d in [samples_dir, side_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    rows = jsonl_rows(args.manifest)
    if not rows:
        raise RuntimeError(f"Empty manifest: {args.manifest}")
    qual_samples = sample_rows(rows, args.num_samples, args.seed)
    metric_samples = sample_rows(rows, args.num_samples_metric, args.seed)
    all_samples = {sample.sample_id: sample for sample in metric_samples}
    for sample in qual_samples:
        all_samples.setdefault(sample.sample_id, sample)
    ordered_samples = list(all_samples.values())
    unsharded_sample_count = len(ordered_samples)
    if args.num_shards > 1:
        ordered_samples = [
            sample
            for sample_pos, sample in enumerate(ordered_samples)
            if sample_pos % args.num_shards == args.shard_index
        ]
        print(
            "[partialmask-eval] shard "
            f"{args.shard_index}/{args.num_shards}: "
            f"{len(ordered_samples)}/{unsharded_sample_count} samples"
        )

    model_statuses: List[ModelStatus] = []
    active_models: List[ModelSpec] = []
    base_spec = ModelSpec("DiffuEraser-base", args.base_weights_dir, is_base=True)
    if is_direct_diffusers_weights(base_spec.weights_dir):
        active_models.append(base_spec)
        model_statuses.append(ModelStatus(base_spec.label, str(base_spec.weights_dir), "active"))
    else:
        model_statuses.append(ModelStatus(base_spec.label, str(base_spec.weights_dir), "skipped", "not exported diffusers weights"))

    for label, weights_dir in args.checkpoint:
        if is_direct_diffusers_weights(weights_dir):
            active_models.append(ModelSpec(label, weights_dir, is_base=False))
            model_statuses.append(ModelStatus(label, str(weights_dir), "active"))
        elif weights_dir.exists():
            model_statuses.append(
                ModelStatus(label, str(weights_dir), "skipped", "checkpoint exists but is not exported as unet_main/ + brushnet/")
            )
        else:
            model_statuses.append(ModelStatus(label, str(weights_dir), "skipped", "path does not exist"))

    if len(active_models) < 2:
        print("[partialmask-eval][warn] fewer than two active models; metrics/report will be limited")

    config = {
        "manifest": str(args.manifest),
        "manifest_rows": len(rows),
        "num_samples": args.num_samples,
        "num_samples_metric": args.num_samples_metric,
        "seed": args.seed,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "generate_only": args.generate_only,
        "unsharded_sample_count": unsharded_sample_count,
        "active_sample_count": len(ordered_samples),
        "base_weights_dir": str(args.base_weights_dir),
        "checkpoints": [{"label": label, "path": str(path)} for label, path in args.checkpoint],
        "model_statuses": [status.__dict__ for status in model_statuses],
    }
    (args.output_dir / "eval_manifest.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_paths: Dict[Tuple[str, str], Path] = {}
    loaded_inputs: Dict[str, Tuple[List[Image.Image], List[Image.Image], Optional[List[Image.Image]]]] = {}

    for spec in active_models:
        print(f"[partialmask-eval] loading {spec.label}: {spec.weights_dir}")
        pipeline, weight_dtype = load_model_pipeline(args, spec, device)
        try:
            for sample in ordered_samples:
                if sample.sample_id not in loaded_inputs:
                    winner = read_video_frames(sample.win_video_path, args.frames, (args.width, args.height), is_mask=False)
                    masks = read_video_frames(sample.mask_path, args.frames, (args.width, args.height), is_mask=True)
                    loser = None
                    if not args.no_d2_loser and sample.d2_loser_path and sample.d2_loser_path.exists():
                        loser = read_video_frames(sample.d2_loser_path, args.frames, (args.width, args.height), is_mask=False)
                    loaded_inputs[sample.sample_id] = (winner, masks, loser)
                winner_frames, mask_frames, _ = loaded_inputs[sample.sample_id]
                out_path = samples_dir / spec.label / f"{sample.sample_id}.mp4"
                output_paths[(spec.label, sample.sample_id)] = run_inference_for_sample(
                    args, pipeline, weight_dtype, device, spec, sample, winner_frames, mask_frames, out_path
                )
        finally:
            del pipeline
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if args.generate_only:
        print(f"[partialmask-eval] generate_only complete output={args.output_dir}")
        return 0

    metric_rows: List[Dict[str, Any]] = []
    for sample in metric_samples:
        winner_frames, mask_frames, _ = loaded_inputs[sample.sample_id]
        for spec in active_models:
            out_path = output_paths.get((spec.label, sample.sample_id))
            if not out_path or not out_path.exists():
                continue
            comp_frames = read_comp_video(out_path, args.frames, (args.width, args.height))
            metrics = compute_video_metrics(winner_frames, comp_frames, mask_frames)
            metric_rows.append(
                {
                    "model_label": spec.label,
                    "sample_id": sample.sample_id,
                    "row_index": sample.row_index,
                    **metrics,
                }
            )

    summary_rows = summarize_metrics(metric_rows)
    write_csv(metrics_dir / "per_sample_metrics.csv", metric_rows)
    write_csv(metrics_dir / "summary.csv", summary_rows)
    (metrics_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8")
    write_summary_md(metrics_dir / "summary.md", summary_rows)

    side_rows: List[Dict[str, Any]] = []
    base_label = "DiffuEraser-base"
    exp_models = [spec for spec in active_models if spec.label != base_label]
    for sample in qual_samples:
        winner_frames, mask_frames, loser_frames = loaded_inputs[sample.sample_id]
        overlay = mask_overlay_frames(winner_frames, mask_frames)
        base_out = output_paths.get((base_label, sample.sample_id))
        if not base_out or not base_out.exists():
            continue
        base_frames = read_comp_video(base_out, args.frames, (args.width, args.height))
        for spec in exp_models:
            exp_out = output_paths.get((spec.label, sample.sample_id))
            if not exp_out or not exp_out.exists():
                continue
            exp_frames = read_comp_video(exp_out, args.frames, (args.width, args.height))
            columns = [winner_frames, overlay, base_frames, exp_frames]
            labels = ["winner/GT", "mask overlay", "DiffuEraser-base", spec.label]
            if loser_frames is not None:
                columns.append(loser_frames)
                labels.append("D2 loser")
            side_frames = stack_columns(columns, labels)
            out_path = side_dir / spec.label / f"{sample.sample_id}.mp4"
            save_video(side_frames, out_path, args.fps)
            side_rows.append(
                {
                    "checkpoint_label": spec.label,
                    "sample_id": sample.sample_id,
                    "prompt": sample.prompt,
                    "win_video_path": str(sample.win_video_path),
                    "mask_path": str(sample.mask_path),
                    "base_output_path": str(base_out),
                    "exp7_output_path": str(exp_out),
                    "d2_loser_path": str(sample.d2_loser_path or ""),
                    "side_by_side_path": str(out_path),
                    "base_weights_dir": str(args.base_weights_dir),
                    "exp7_weights_dir": str(spec.weights_dir),
                }
            )

    write_csv(args.output_dir / "pair_manifest.csv", side_rows)
    write_index(args.output_dir / "index.html", side_rows)

    dpo_summary_path = args.dpo_summary_out
    if dpo_summary_path:
        write_dpo_diag_summary(args.stage1_diag_csv, args.stage2_diag_csv, dpo_summary_path)

    decision_report(
        args.output_dir / "report.md",
        args.output_dir,
        model_statuses,
        summary_rows,
        dpo_summary_path,
    )

    print(f"[partialmask-eval] output={args.output_dir}")
    print(f"[partialmask-eval] side_by_side={side_dir}")
    print(f"[partialmask-eval] metrics={metrics_dir / 'summary.csv'}")
    print(f"[partialmask-eval] report={args.output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
