"""LocalDPO 24-frame DiffuEraser adaptation helpers for Exp27 CLI4.

The helpers are deliberately file-system oriented because the milestone has
two auditable phases: generate real 24F LocalDPO pairs, then run only the
original LocalDPO-style 1/10-step objective on those pairs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageFilter, ImageDraw

from exp27_paper_grounded_preference_study.code.official_parity import (
    install_localdpo_matplotlib_rgb_shim,
    load_localdpo_random_mask_module,
)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class PairGate:
    name: str
    required_pairs: int
    min_technical_valid: int
    min_medium_hard: int


P8_GATE = PairGate("P8", required_pairs=8, min_technical_valid=8, min_medium_hard=6)
P32_GATE = PairGate("P32", required_pairs=32, min_technical_valid=30, min_medium_hard=24)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in path.rglob("*") if x.is_file()):
        h.update(str(p.relative_to(path)).encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(p).encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def list_frames(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    direct = sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS)
    if direct:
        return direct
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS)


def safe_id(text: Any, fallback: str) -> str:
    raw = str(text or fallback)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)[:96]


def load_frames(path: Path, frames: int, size: tuple[int, int], mode: str = "RGB") -> list[Image.Image]:
    files = list_frames(path)
    if not files:
        raise FileNotFoundError(f"no image frames under {path}")
    width, height = size
    resample = Image.Resampling.NEAREST if mode == "L" else Image.Resampling.BILINEAR
    out = [Image.open(p).convert(mode).resize((width, height), resample) for p in files[:frames]]
    while len(out) < frames:
        out.append(out[-1].copy())
    return out[:frames]


def save_frames(frames: list[Image.Image], out_dir: Path, suffix: str = ".png") -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame.save(out_dir / f"{idx:05d}{suffix}")


def mask_array(mask: Image.Image) -> np.ndarray:
    return (np.asarray(mask.convert("L"), dtype=np.float32) >= 128).astype(np.float32)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    err = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    mse = float(err.mean())
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def region_psnr(a: np.ndarray, b: np.ndarray, region: np.ndarray) -> float:
    selector = region.astype(bool)
    if selector.sum() == 0:
        return 99.0
    av = a[selector]
    bv = b[selector]
    return psnr(av, bv)


def official_localdpo_masks(
    *,
    seed: int,
    frames: int,
    height: int,
    width: int,
    connected_components: int = 1,
    require_official: bool = True,
) -> tuple[list[Image.Image], dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    try:
        shim = install_localdpo_matplotlib_rgb_shim()
        module = load_localdpo_random_mask_module()
        if connected_components == 1:
            masks = module.create_random_shape_with_random_motion(
                frames,
                zoomin=0.9,
                zoomout=1.1,
                rotmin=1,
                rotmax=10,
                imageHeight=height,
                imageWidth=width,
            )
        else:
            masks = module.create_random_shape_with_random_motion_multiple_connected_components(
                frames,
                zoomin=0.9,
                zoomout=1.1,
                rotmin=1,
                rotmax=10,
                cc_ratio=max(1, connected_components),
                fix_area=0,
                imageHeight=height,
                imageWidth=width,
            )
        out = [m.convert("L").resize((width, height), Image.Resampling.NEAREST) for m in masks[:frames]]
        return out, {
            "status": "official_localdpo_mask_passed",
            "matplotlib_rgb_shim": shim,
            "source": "Local-DPO random_mask_gen.py",
        }
    except Exception as exc:  # noqa: BLE001 - recorded as an explicit gate failure.
        if require_official:
            raise
        return fallback_bezier_masks(seed=seed, frames=frames, height=height, width=width), {
            "status": "fallback_mask_for_unit_test_only",
            "error": repr(exc),
        }


def fallback_bezier_masks(*, seed: int, frames: int, height: int, width: int) -> list[Image.Image]:
    rng = random.Random(seed)
    cx = rng.randint(width // 4, 3 * width // 4)
    cy = rng.randint(height // 4, 3 * height // 4)
    rx = max(12, width // 10)
    ry = max(10, height // 9)
    out = []
    for t in range(frames):
        dx = int((t - frames / 2) * width / max(frames * 8, 1))
        dy = int(math.sin(t / max(frames - 1, 1) * math.pi) * height / 18)
        pts = []
        for k in range(12):
            ang = 2.0 * math.pi * k / 12.0
            jitter = 0.75 + 0.45 * rng.random()
            pts.append((cx + dx + int(math.cos(ang) * rx * jitter), cy + dy + int(math.sin(ang) * ry * jitter)))
        im = Image.new("L", (width, height), 0)
        ImageDraw.Draw(im).polygon(pts, fill=255)
        out.append(im)
    return out


def make_condition_frames(clean: list[Image.Image], masks: list[Image.Image]) -> list[Image.Image]:
    out: list[Image.Image] = []
    for frame, mask in zip(clean, masks):
        arr = np.asarray(frame.convert("RGB"), dtype=np.uint8).copy()
        m = mask_array(mask).astype(bool)
        arr[m] = 0
        out.append(Image.fromarray(arr, "RGB"))
    return out


def composite_outside_clean(raw_loser: list[Image.Image], clean: list[Image.Image], masks: list[Image.Image]) -> list[Image.Image]:
    out: list[Image.Image] = []
    for loser, winner, mask in zip(raw_loser, clean, masks):
        larr = np.asarray(loser.convert("RGB"), dtype=np.uint8)
        warr = np.asarray(winner.convert("RGB"), dtype=np.uint8)
        m = mask_array(mask).astype(bool)
        comp = warr.copy()
        comp[m] = larr[m]
        out.append(Image.fromarray(comp, "RGB"))
    return out


def controlled_corruption_preview(clean: list[Image.Image], masks: list[Image.Image], seed: int) -> list[Image.Image]:
    rng = np.random.default_rng(seed)
    out: list[Image.Image] = []
    for frame, mask in zip(clean, masks):
        blurred = frame.filter(ImageFilter.GaussianBlur(radius=9))
        arr = np.asarray(frame.convert("RGB"), dtype=np.int16).copy()
        barr = np.asarray(blurred.convert("RGB"), dtype=np.int16)
        noise = rng.normal(0, 18, arr.shape).astype(np.int16)
        corrupt = np.clip((barr + noise), 0, 255).astype(np.uint8)
        m = mask_array(mask).astype(bool)
        arr[m] = corrupt[m]
        out.append(Image.fromarray(arr.astype(np.uint8), "RGB"))
    return out


def save_review_assets(
    *,
    pair_id: str,
    clean: list[Image.Image],
    masks: list[Image.Image],
    loser: list[Image.Image],
    review_dir: Path,
) -> dict[str, str]:
    review_dir.mkdir(parents=True, exist_ok=True)
    strip = temporal_strip(clean, masks, loser, indices=np.linspace(0, len(clean) - 1, min(16, len(clean))).round().astype(int))
    strip_path = review_dir / f"{pair_id}_16frame_strip.jpg"
    strip.save(strip_path, quality=92)
    heat = temporal_difference_heatmap(loser)
    heat_path = review_dir / f"{pair_id}_temporal_diff.jpg"
    heat.save(heat_path, quality=92)
    return {"temporal_strip": str(strip_path), "temporal_diff": str(heat_path)}


def temporal_strip(
    clean: list[Image.Image],
    masks: list[Image.Image],
    loser: list[Image.Image],
    indices: Iterable[int],
) -> Image.Image:
    cells: list[Image.Image] = []
    for idx in indices:
        idx = int(idx)
        c = clean[idx].resize((144, 80), Image.Resampling.BILINEAR)
        m = Image.merge("RGB", [masks[idx].resize((144, 80), Image.Resampling.NEAREST)] * 3)
        l = loser[idx].resize((144, 80), Image.Resampling.BILINEAR)
        cell = Image.new("RGB", (432, 80), "white")
        cell.paste(c, (0, 0))
        cell.paste(m, (144, 0))
        cell.paste(l, (288, 0))
        cells.append(cell)
    canvas = Image.new("RGB", (432, 80 * len(cells)), "white")
    for row, cell in enumerate(cells):
        canvas.paste(cell, (0, row * 80))
    return canvas


def temporal_difference_heatmap(frames: list[Image.Image]) -> Image.Image:
    if len(frames) < 2:
        return Image.new("RGB", frames[0].size, "black")
    diffs = []
    for a, b in zip(frames[:-1], frames[1:]):
        arr_a = np.asarray(a.convert("RGB"), dtype=np.float32)
        arr_b = np.asarray(b.convert("RGB"), dtype=np.float32)
        diffs.append(np.abs(arr_b - arr_a).mean(axis=2))
    mean = np.mean(np.stack(diffs, axis=0), axis=0)
    norm = np.clip(mean / max(float(mean.max()), 1e-6), 0, 1)
    rgb = np.zeros((*norm.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (norm * 255).astype(np.uint8)
    rgb[..., 1] = (np.sqrt(norm) * 128).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def classify_pair(clean: list[Image.Image], masks: list[Image.Image], loser: list[Image.Image]) -> dict[str, Any]:
    inside_vals = []
    outside_vals = []
    global_vals = []
    mask_areas = []
    for c, m, l in zip(clean, masks, loser):
        carr = np.asarray(c.convert("RGB"), dtype=np.uint8)
        larr = np.asarray(l.convert("RGB"), dtype=np.uint8)
        region = mask_array(m)
        mask_areas.append(float(region.mean()))
        inside_vals.append(region_psnr(carr, larr, region))
        outside_vals.append(region_psnr(carr, larr, 1.0 - region))
        global_vals.append(psnr(carr, larr))
    inside = float(np.mean(inside_vals))
    outside = float(np.mean(outside_vals))
    global_psnr = float(np.mean(global_vals))
    mask_area = float(np.mean(mask_areas))
    technical = math.isfinite(inside) and math.isfinite(outside) and 0.005 <= mask_area <= 0.55 and outside >= 34.0
    if not technical:
        bucket = "TECHNICAL_INVALID"
    elif inside >= 45.0:
        bucket = "TOO_CLOSE"
    elif inside < 12.0:
        bucket = "TRIVIAL_BAD"
    elif inside < 20.0:
        bucket = "HARD_BUT_PLAUSIBLE"
    else:
        bucket = "MEDIUM_HARD_ELIGIBLE"
    return {
        "technical_valid": bool(technical),
        "classification": bucket,
        "mask_psnr": inside,
        "outside_psnr": outside,
        "global_psnr": global_psnr,
        "mask_area": mask_area,
        "outside_preservation_passed": bool(outside >= 34.0),
        "global_collapse": bool(global_psnr < 10.0 or outside < 28.0),
    }


def select_manifest_rows(rows: list[dict[str, Any]], count: int, seed: int) -> list[tuple[int, dict[str, Any]]]:
    rng = random.Random(seed)
    candidates = []
    for idx, row in enumerate(rows):
        if row.get("win_video_path") and row.get("mask_path"):
            candidates.append((idx, row))
    rng.shuffle(candidates)
    return sorted(candidates[:count], key=lambda item: item[0])


def run_diffueraser_loser(
    *,
    project_root: Path,
    video_root: Path,
    mask_root: Path,
    output_dir: Path,
    work_dir: Path,
    prompt: str,
    frames: int,
    height: int,
    width: int,
    base_model: Path,
    vae: Path,
    diffueraser_weights: Path,
    propainter_model_dir: Path,
    pcm_weights_path: Path,
) -> None:
    cmd = [
        sys.executable,
        str(project_root / "DPO_finetune" / "infer_diffueraser_candidate.py"),
        "--video_root",
        str(video_root),
        "--mask_root",
        str(mask_root),
        "--output_dir",
        str(output_dir),
        "--work_dir",
        str(work_dir),
        "--project_root",
        str(project_root),
        "--base_model_path",
        str(base_model),
        "--vae_path",
        str(vae),
        "--diffueraser_path",
        str(diffueraser_weights),
        "--propainter_model_dir",
        str(propainter_model_dir),
        "--pcm_weights_path",
        str(pcm_weights_path),
        "--prompt",
        prompt,
        "--num_frames",
        str(frames),
        "--height",
        str(height),
        "--width",
        str(width),
        "--mask_dilation_iter",
        "0",
        "--offload_models",
    ]
    subprocess.run(cmd, cwd=str(project_root), check=True)


def summarize_gate(gate: PairGate, rows: list[dict[str, Any]]) -> dict[str, Any]:
    technical_valid = sum(1 for r in rows if r.get("technical_valid"))
    medium_hard = sum(1 for r in rows if r.get("classification") == "MEDIUM_HARD_ELIGIBLE")
    hard = sum(1 for r in rows if r.get("classification") == "HARD_BUT_PLAUSIBLE")
    trivial = sum(1 for r in rows if r.get("classification") == "TRIVIAL_BAD")
    invalid = sum(1 for r in rows if r.get("classification") == "TECHNICAL_INVALID")
    too_close = sum(1 for r in rows if r.get("classification") == "TOO_CLOSE")
    global_collapse = sum(1 for r in rows if r.get("global_collapse"))
    review_done = sum(1 for r in rows if r.get("review_assets", {}).get("temporal_strip"))
    passed = (
        len(rows) >= gate.required_pairs
        and technical_valid >= gate.min_technical_valid
        and (medium_hard + hard) >= gate.min_medium_hard
        and global_collapse == 0
        and review_done >= gate.required_pairs
        and all(r.get("outside_preservation_passed") for r in rows)
    )
    return {
        "gate": gate.name,
        "status": f"{gate.name}_PASSED" if passed else f"{gate.name}_FAILED_OR_PENDING",
        "pairs": len(rows),
        "technical_valid": technical_valid,
        "medium_hard": medium_hard,
        "hard_plausible": hard,
        "trivial_bad": trivial,
        "too_close": too_close,
        "technical_invalid": invalid,
        "global_collapse": global_collapse,
        "outside_preservation_passed": bool(rows and all(r.get("outside_preservation_passed") for r in rows)),
        "video_review": review_done,
        "thresholds": {
            "technical_valid_min": gate.min_technical_valid,
            "medium_hard_plus_hard_min": gate.min_medium_hard,
            "global_collapse_max": 0,
            "video_review_required": gate.required_pairs,
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row if not isinstance(row.get(k), (dict, list))})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fields} for row in rows])
