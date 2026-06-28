#!/usr/bin/env python3
"""Build Exp40 LocalDPO-v3 PSNR-safe corruption pool.

This script is Exp40-isolated. It reads the Exp25 full VOR metadata index,
locks scene-disjoint train/search/shadow rows, selectively extracts only the
needed VOR members, materializes 17-frame 512px sources, and creates bounded
local corruptions with strict outside preservation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


KNOWN_INVALID_SAMPLE_IDS = {
    "BLENDER_CARTOON006_00001",
    "BLENDER_RIVER007_00001",
    "REAL_ENV044_00004_001_01",
    "REAL_ENV046_00001_001_01",
}

PROFILE_ROTATION = (
    ("C1_mild_object_texture", "C3_boundary_seam_mild", "C5_effect_lighting_mild"),
    ("C2_medium_object_texture", "C4_affected_soft_degradation", "C7_bounded_texture_blur"),
    ("C1_mild_object_texture", "C6_temporal_mild_flicker", "C8_residual_object_mild"),
    ("C3_boundary_seam_mild", "C4_affected_soft_degradation", "C2_medium_object_texture"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-index", type=Path, required=True)
    parser.add_argument("--train-parts-dir", type=Path, required=True)
    parser.add_argument("--mask-parts-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--target-train", type=int, default=96)
    parser.add_argument("--target-search", type=int, default=32)
    parser.add_argument("--target-shadow", type=int, default=32)
    parser.add_argument("--min-train", type=int, default=64)
    parser.add_argument("--min-search", type=int, default=24)
    parser.add_argument("--min-shadow", type=int, default=24)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--heartbeat", type=Path, default=None)
    parser.add_argument("--skip-extract", action="store_true")
    return parser.parse_args()


def heartbeat(path: Path | None, message: str) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{time.time():.0f}\t{message}\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def sha256_tree(path: Path) -> str:
    h = hashlib.sha256()
    for frame in sorted(path.glob("*.png")):
        h.update(frame.name.encode("utf-8"))
        h.update(frame.read_bytes())
    return h.hexdigest()


def source_type(sample_id: str) -> str:
    if sample_id.startswith("REAL_"):
        return "REAL"
    if sample_id.startswith("BLENDER_"):
        return "BLENDER"
    return "UNKNOWN"


def scene_group(row: dict[str, Any]) -> str:
    return str(row.get("scene_group") or row.get("sample_id"))


def split_targets(args: argparse.Namespace) -> dict[str, int]:
    return {"train": args.target_train, "search": args.target_search, "shadow": args.target_shadow}


def split_minimums(args: argparse.Namespace) -> dict[str, int]:
    return {"train": args.min_train, "search": args.min_search, "shadow": args.min_shadow}


def select_sources(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    filtered: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_scenes: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        stype = source_type(sample_id)
        group = scene_group(row)
        reason = ""
        if sample_id in KNOWN_INVALID_SAMPLE_IDS:
            reason = "known_invalid_or_quarantined"
        elif stype not in {"REAL", "BLENDER"}:
            reason = "unknown_source_type"
        elif group in seen_scenes:
            reason = "duplicate_scene_group"
        elif row.get("hard_comp") not in (False, "false", "False", None):
            reason = "hard_comp_not_allowed"
        if reason:
            bad = dict(row)
            bad["reject_reason"] = reason
            rejected.append(bad)
            continue
        seen_scenes.add(group)
        locked = dict(row)
        locked["source_type"] = stype
        locked["scene_group"] = group
        filtered.append(locked)

    by_type: dict[str, list[dict[str, Any]]] = {
        "BLENDER": [r for r in filtered if r["source_type"] == "BLENDER"],
        "REAL": [r for r in filtered if r["source_type"] == "REAL"],
    }
    for key in by_type:
        by_type[key].sort(key=lambda r: (scene_group(r), str(r["sample_id"])))

    selected: dict[str, list[dict[str, Any]]] = {"train": [], "search": [], "shadow": []}
    used_groups: set[str] = set()
    targets = split_targets(args)
    for split, count in targets.items():
        per_type = {"BLENDER": count // 2, "REAL": count - count // 2}
        for stype in ("BLENDER", "REAL"):
            for row in by_type[stype]:
                if len([r for r in selected[split] if r["source_type"] == stype]) >= per_type[stype]:
                    break
                group = scene_group(row)
                if group in used_groups:
                    continue
                locked = dict(row)
                locked["exp40_source_split"] = split
                selected[split].append(locked)
                used_groups.add(group)
        if len(selected[split]) < count:
            for row in filtered:
                if len(selected[split]) >= count:
                    break
                group = scene_group(row)
                if group in used_groups:
                    continue
                locked = dict(row)
                locked["exp40_source_split"] = split
                selected[split].append(locked)
                used_groups.add(group)
    return selected, rejected


def parts(pattern_dir: Path, prefix: str) -> list[Path]:
    files = sorted(pattern_dir.glob(f"{prefix}.tar.gz.part_*"))
    if not files:
        raise FileNotFoundError(f"no tar parts for {prefix} in {pattern_dir}")
    return files


def run_tar_extract(part_files: list[Path], members_file: Path, dest: Path, log_path: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = "cat " + " ".join(str(p) for p in part_files) + f" | tar -xz -C {dest} -T {members_file}"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(command + "\n")
        proc = subprocess.run(command, shell=True, stdout=log, stderr=subprocess.STDOUT, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"tar extract failed rc={proc.returncode}; see {log_path}")


def required_members(selected: dict[str, list[dict[str, Any]]]) -> tuple[list[str], list[str]]:
    train_members: list[str] = []
    mask_members: list[str] = []
    for rows in selected.values():
        for row in rows:
            train_members.append(str(row["condition_member_path"]))
            train_members.append(str(row["winner_member_path"]))
            mask_members.append(str(row["mask_member_path"]))
    return sorted(set(train_members)), sorted(set(mask_members))


def extract_members(selected: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> dict[str, Any]:
    extract_root = args.output_root / "extracted_exact"
    train_members, mask_members = required_members(selected)
    train_list = args.output_root / "member_lists" / "vor_train_members.txt"
    mask_list = args.output_root / "member_lists" / "vor_mask_members.txt"
    train_list.parent.mkdir(parents=True, exist_ok=True)
    train_list.write_text("\n".join(train_members) + "\n", encoding="utf-8")
    mask_list.write_text("\n".join(mask_members) + "\n", encoding="utf-8")
    if not args.skip_extract:
        run_tar_extract(
            parts(args.train_parts_dir, "VOR-Train"),
            train_list,
            extract_root / "VOR-Train",
            args.output_root / "logs" / "extract_vor_train.log",
        )
        run_tar_extract(
            parts(args.mask_parts_dir, "VOR-Train-MASK"),
            mask_list,
            extract_root / "VOR-Train-MASK",
            args.output_root / "logs" / "extract_vor_train_mask.log",
        )
    return {
        "extract_root": str(extract_root),
        "train_members": len(train_members),
        "mask_members": len(mask_members),
        "train_member_list": str(train_list),
        "mask_member_list": str(mask_list),
        "skip_extract": args.skip_extract,
    }


def resolve_mp4(row: dict[str, Any], extract_root: Path, role: str) -> Path:
    if role == "condition":
        return extract_root / "VOR-Train" / str(row["condition_member_path"])
    if role == "winner":
        return extract_root / "VOR-Train" / str(row["winner_member_path"])
    if role == "mask":
        return extract_root / "VOR-Train-MASK" / str(row["mask_member_path"])
    raise ValueError(role)


def read_video_frames(path: Path, num_frames: int, size: int, is_mask: bool = False) -> tuple[list[np.ndarray], dict[str, Any]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[np.ndarray] = []
    for _ in range(num_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if is_mask:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(gray, (size, size), interpolation=cv2.INTER_NEAREST)
            frames.append((frame > 8).astype(np.uint8) * 255)
        else:
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) != num_frames:
        raise RuntimeError(f"{path} decoded {len(frames)} frames, expected {num_frames}")
    return frames, {"source_total_frames": total, "decoded_frames": len(frames)}


def save_rgb_frames(frames: list[np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        cv2.imwrite(str(out_dir / f"{idx:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def save_mask_frames(frames: list[np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        cv2.imwrite(str(out_dir / f"{idx:05d}.png"), frame)


def save_mp4_rgb(frames: list[np.ndarray], path: Path, fps: float = 8.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    if mask is not None:
        weights = mask.astype(np.float32)
        if weights.ndim == 2:
            weights = weights[:, :, None]
        denom = float(weights.sum() * 3.0)
        if denom <= 1e-8:
            return float("nan")
        mse = float((diff * weights).sum() / denom)
    else:
        mse = float(diff.mean())
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    if mask is not None:
        weights = mask.astype(np.float32)
        if weights.ndim == 2:
            weights = weights[:, :, None]
        denom = float(weights.sum() * 3.0)
        if denom <= 1e-8:
            return float("nan")
        return float((diff * weights).sum() / denom)
    return float(diff.mean())


def dilate(mask: np.ndarray, ksize: int) -> np.ndarray:
    return cv2.dilate((mask > 0.05).astype(np.uint8), np.ones((ksize, ksize), dtype=np.uint8), iterations=1).astype(np.float32)


def erode(mask: np.ndarray, ksize: int) -> np.ndarray:
    return cv2.erode((mask > 0.05).astype(np.uint8), np.ones((ksize, ksize), dtype=np.uint8), iterations=1).astype(np.float32)


def soft_blur(mask: np.ndarray, ksize: int) -> np.ndarray:
    return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), 0), 0.0, 1.0)


def profile_region(profile: str, mask: np.ndarray, affected: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    boundary = np.clip(dilate(mask, 7) - erode(mask, 5), 0.0, 1.0)
    affected_soft = soft_blur((affected > 0.07).astype(np.float32), 15)
    if profile == "C1_mild_object_texture":
        return soft_blur(mask, 9), boundary, 0.10, 1.1, 0.82
    if profile == "C2_medium_object_texture":
        return soft_blur(mask, 11), boundary, 0.16, 1.6, 0.82
    if profile == "C3_boundary_seam_mild":
        region = np.clip(np.maximum(soft_blur(boundary, 9), affected_soft * 0.20), 0.0, 1.0)
        return region, boundary, 0.12, 1.0, 0.88
    if profile == "C4_affected_soft_degradation":
        region = np.clip(np.maximum(soft_blur(mask, 9), affected_soft * 0.28), 0.0, 1.0)
        return region, boundary, 0.13, 1.2, 0.86
    if profile == "C5_effect_lighting_mild":
        region = np.clip(np.maximum(soft_blur(mask, 7), affected_soft * 0.24), 0.0, 1.0)
        return region, boundary, 0.10, 0.9, 0.90
    if profile == "C6_temporal_mild_flicker":
        region = soft_blur(mask, 9)
        return region, boundary, 0.11, 1.5, 0.55
    if profile == "C7_bounded_texture_blur":
        return soft_blur(mask, 13), boundary, 0.14, 0.7, 0.92
    if profile == "C8_residual_object_mild":
        region = np.clip(np.maximum(soft_blur(mask, 9), affected_soft * 0.18), 0.0, 1.0)
        return region, boundary, 0.12, 1.1, 0.84
    raise ValueError(profile)


def corrupt_frames(
    winner: list[np.ndarray],
    condition: list[np.ndarray],
    masks: list[np.ndarray],
    profile: str,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    outputs: list[np.ndarray] = []
    regions: list[np.ndarray] = []
    boundaries: list[np.ndarray] = []
    prev_noise: np.ndarray | None = None
    for win, cond, mask255 in zip(winner, condition, masks):
        mask = (mask255 > 8).astype(np.float32)
        affected = np.mean(np.abs(cond.astype(np.float32) - win.astype(np.float32)), axis=2) / 255.0
        region, boundary, alpha, noise_scale, smooth = profile_region(profile, mask, affected)
        region3 = region[:, :, None]
        cond_blur = cv2.GaussianBlur(cond, (5, 5), 0).astype(np.float32)
        win_blur = cv2.GaussianBlur(win, (5, 5), 0).astype(np.float32)
        if profile == "C7_bounded_texture_blur":
            local_target = 0.28 * cond_blur + 0.72 * win_blur
        else:
            local_target = 0.42 * cond_blur + 0.58 * win_blur
        noise = rng.normal(0.0, noise_scale, size=win.shape).astype(np.float32)
        if prev_noise is not None:
            noise = smooth * prev_noise + (1.0 - smooth) * noise
        prev_noise = noise
        candidate = win.astype(np.float32) * (1.0 - alpha * region3) + local_target * (alpha * region3) + noise * region3
        out = np.where(region3 > 1e-4, candidate, win.astype(np.float32))
        outputs.append(np.clip(out, 0, 255).astype(np.uint8))
        regions.append(region)
        boundaries.append(boundary)
    return outputs, regions, boundaries


def metrics_for_candidate(
    winner: list[np.ndarray],
    loser: list[np.ndarray],
    masks: list[np.ndarray],
    boundaries: list[np.ndarray],
    regions: list[np.ndarray],
) -> dict[str, float]:
    vals: dict[str, list[float]] = defaultdict(list)
    for idx, (win, los, mask255, boundary, region) in enumerate(zip(winner, loser, masks, boundaries, regions)):
        mask = (mask255 > 8).astype(np.float32)
        outside = 1.0 - np.clip(region, 0.0, 1.0)
        vals["full_psnr"].append(psnr(win, los))
        vals["mask_psnr"].append(psnr(win, los, mask))
        vals["boundary_psnr"].append(psnr(win, los, boundary))
        vals["affected_psnr"].append(psnr(win, los, region))
        vals["outside_psnr"].append(psnr(win, los, outside))
        vals["mask_mae"].append(mae(win, los, mask))
        vals["boundary_mae"].append(mae(win, los, boundary))
        vals["affected_mae"].append(mae(win, los, region))
        vals["outside_mae"].append(mae(win, los, outside))
        if idx > 0:
            win_dt = np.abs(win.astype(np.float32) - winner[idx - 1].astype(np.float32))
            los_dt = np.abs(los.astype(np.float32) - loser[idx - 1].astype(np.float32))
            vals["temporal_flicker_mae"].append(float(np.abs(los_dt - win_dt).mean()))
    return {k: float(np.nanmean(v)) if v else float("nan") for k, v in vals.items()}


def classify(metrics: dict[str, float]) -> tuple[str, str]:
    outside_ok = metrics["outside_psnr"] >= 55.0 and metrics["outside_mae"] <= 0.18
    boundary_ok = metrics["boundary_psnr"] >= 28.0 and metrics["boundary_mae"] <= 9.0
    if not outside_ok:
        return "TRIVIAL_BAD", "outside damage exceeds Exp40 strict outside bound"
    if not boundary_ok:
        return "TRIVIAL_BAD", "boundary damage exceeds Exp40 strict boundary bound"
    if metrics["mask_mae"] < 1.0 or metrics["mask_psnr"] > 45.0:
        return "TOO_CLOSE", "Exp40 corruption is too close to winner"
    if metrics["mask_psnr"] < 20.0 or metrics["mask_mae"] > 35.0:
        return "TRIVIAL_BAD", "Exp40 corruption is too severe"
    if metrics["mask_psnr"] < 28.0:
        return "HARD_BUT_PLAUSIBLE", "bounded hard local corruption with safe outside/boundary"
    return "MEDIUM_HARD_ELIGIBLE", "PSNR-safe medium-hard local corruption"


def make_review_sheet(condition: list[np.ndarray], winner: list[np.ndarray], loser: list[np.ndarray], masks: list[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    idxs = np.linspace(0, len(winner) - 1, min(8, len(winner))).astype(int).tolist()
    rows = []
    for idx in idxs:
        cond = condition[idx]
        win = winner[idx]
        los = loser[idx]
        diff = np.clip(np.abs(los.astype(np.int16) - win.astype(np.int16)) * 5, 0, 255).astype(np.uint8)
        overlay = win.copy()
        m = masks[idx] > 8
        overlay[m] = (0.58 * overlay[m] + np.array([255, 30, 30]) * 0.42).astype(np.uint8)
        row = np.concatenate([cond, overlay, win, los, diff], axis=1)
        cv2.putText(row, f"f{idx:02d} cond|mask|winner|loser|diffx5", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        rows.append(row)
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def materialize_row(row: dict[str, Any], split: str, args: argparse.Namespace) -> dict[str, Any]:
    extract_root = args.output_root / "extracted_exact"
    sample_id = str(row["sample_id"])
    source_root = args.output_root / "materialized_17f_512" / "sources" / sample_id
    condition_dir = source_root / "condition_frames"
    winner_dir = source_root / "winner_frames"
    mask_dir = source_root / "mask_frames"
    cond_frames, cond_probe = read_video_frames(resolve_mp4(row, extract_root, "condition"), args.num_frames, args.size)
    win_frames, win_probe = read_video_frames(resolve_mp4(row, extract_root, "winner"), args.num_frames, args.size)
    mask_frames, mask_probe = read_video_frames(resolve_mp4(row, extract_root, "mask"), args.num_frames, args.size, is_mask=True)
    save_rgb_frames(cond_frames, condition_dir)
    save_rgb_frames(win_frames, winner_dir)
    save_mask_frames(mask_frames, mask_dir)
    mask_areas = [float((m > 8).mean()) for m in mask_frames]
    affected = []
    affected_out = []
    for cond, win, mask in zip(cond_frames, win_frames, mask_frames):
        diff = np.mean(np.abs(cond.astype(np.float32) - win.astype(np.float32)), axis=2)
        affected.append(float(diff[mask > 8].mean()) if (mask > 8).any() else 0.0)
        affected_out.append(float(diff[mask <= 8].mean()) if (mask <= 8).any() else 0.0)
    enriched = dict(row)
    enriched.update(
        {
            "split": split,
            "condition_path": str(condition_dir),
            "winner_path": str(winner_dir),
            "mask_path": str(mask_dir),
            "condition_frame_dir": str(condition_dir),
            "winner_frame_dir": str(winner_dir),
            "mask_frame_dir": str(mask_dir),
            "condition_decode": cond_probe,
            "winner_decode": win_probe,
            "mask_decode": mask_probe,
            "num_frames": args.num_frames,
            "width": args.size,
            "height": args.size,
            "mask_area_mean": float(np.mean(mask_areas)),
            "mask_area_max": float(np.max(mask_areas)),
            "mask_bucket": "small" if np.mean(mask_areas) < 0.05 else "medium" if np.mean(mask_areas) < 0.2 else "large",
            "affected_mae_mask_mean": float(np.mean(affected)),
            "affected_mae_outside_mean": float(np.mean(affected_out)),
            "source_dataset": "VOR-Train",
            "vor_eval_used": False,
            "hard_comp": False,
            "comp_mode": "none",
            "condition_source_role": "V_obj",
            "winner_source_role": "V_bg",
            "mask_source_role": "foreground_object_mask",
        }
    )
    return enriched


def process_candidate(row: dict[str, Any], split: str, index: int, args: argparse.Namespace) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    sample_id = str(row["sample_id"])
    cond = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in sorted(Path(row["condition_path"]).glob("*.png"))]
    win = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in sorted(Path(row["winner_path"]).glob("*.png"))]
    masks = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in sorted(Path(row["mask_path"]).glob("*.png"))]
    if len(cond) != args.num_frames or len(win) != args.num_frames or len(masks) != args.num_frames:
        raise RuntimeError(f"materialized frame count mismatch for {sample_id}")
    candidates: list[dict[str, Any]] = []
    rotation = PROFILE_ROTATION[index % len(PROFILE_ROTATION)]
    for cand_index, profile in enumerate(rotation):
        rng = np.random.default_rng(args.seed + index * 113 + cand_index * 29)
        loser, regions, boundaries = corrupt_frames(win, cond, masks, profile, rng)
        cand_root = args.output_root / "localdpo_v3" / "outputs" / split / sample_id / profile
        frames_dir = cand_root / "frames"
        evidence_dir = cand_root / "evidence"
        save_rgb_frames(loser, frames_dir)
        raw_mp4 = evidence_dir / "raw_output.mp4"
        review_sheet = evidence_dir / "review_sheet.jpg"
        save_mp4_rgb(loser, raw_mp4)
        make_review_sheet(cond, win, loser, masks, review_sheet)
        metrics = metrics_for_candidate(win, loser, masks, boundaries, regions)
        classification, reason = classify(metrics)
        cand = {
            **row,
            "split": split,
            "profile": profile,
            "candidate_index": cand_index,
            "loser_path": str(frames_dir),
            "raw_output_mp4": str(raw_mp4),
            "review_sheet": str(review_sheet),
            "classification": classification,
            "reason": reason,
            "technical_valid": "yes",
            "outside_reinjection": "pixel_exact_outside_soft_region",
            "temporal_smoothing": "profile_specific_ema",
            "training_eligible": classification in {"MEDIUM_HARD_ELIGIBLE", "HARD_BUT_PLAUSIBLE"},
            "output_sha256": sha256_tree(frames_dir),
            "localdpo_condition_role": "V_obj",
            "localdpo_winner_role": "V_bg",
            "localdpo_loser_role": "locally_corrupted_V_bg",
            "affected_map_definition": "abs(V_obj - V_bg), profile-specific local mask",
            **{k: f"{v:.12f}" for k, v in metrics.items()},
        }
        candidates.append(cand)
    priority = {
        "MEDIUM_HARD_ELIGIBLE": 0,
        "HARD_BUT_PLAUSIBLE": 1,
        "TOO_CLOSE": 2,
        "TRIVIAL_BAD": 3,
        "TECHNICAL_INVALID": 4,
    }
    selected = sorted(candidates, key=lambda c: (priority.get(str(c["classification"]), 99), int(c["candidate_index"])))[0]
    if selected["classification"] not in {"MEDIUM_HARD_ELIGIBLE", "HARD_BUT_PLAUSIBLE"}:
        selected = None
    audit = {
        "sample_id": sample_id,
        "split": split,
        "profiles": ",".join(rotation),
        "selected_profile": selected["profile"] if selected else "",
        "selected_classification": selected["classification"] if selected else "REJECTED_NO_ELIGIBLE_CANDIDATE",
    }
    return selected, candidates, audit


def split_status(selected: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> str:
    mins = split_minimums(args)
    if all(len(selected[k]) >= split_targets(args)[k] for k in selected):
        return "MINIMAX_LOCALDPO_V3_POOL_READY"
    if all(len(selected[k]) >= mins[k] for k in selected):
        return "MINIMAX_LOCALDPO_V3_POOL_READY_MINIMUM"
    return "MINIMAX_LOCALDPO_V3_POOL_INSUFFICIENT"


def counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "")) for row in rows))


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.reports_root.mkdir(parents=True, exist_ok=True)
    args.manifest_root.mkdir(parents=True, exist_ok=True)

    heartbeat(args.heartbeat, "read_metadata")
    metadata_rows = read_jsonl(args.metadata_index)
    sources, source_rejected = select_sources(metadata_rows, args)

    heartbeat(args.heartbeat, "extract_members")
    extraction = extract_members(sources, args)

    materialized: dict[str, list[dict[str, Any]]] = {"train": [], "search": [], "shadow": []}
    materialize_failures: list[dict[str, Any]] = []
    for split, rows in sources.items():
        for idx, row in enumerate(rows):
            heartbeat(args.heartbeat, f"materialize:{split}:{idx + 1}/{len(rows)}:{row['sample_id']}")
            try:
                materialized[split].append(materialize_row(row, split, args))
            except Exception as exc:  # noqa: BLE001
                bad = dict(row)
                bad["failure_stage"] = "materialize"
                bad["failure_error"] = repr(exc)
                materialize_failures.append(bad)

    selected: dict[str, list[dict[str, Any]]] = {"train": [], "search": [], "shadow": []}
    all_candidates: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    candidate_audit: list[dict[str, Any]] = []
    for split, rows in materialized.items():
        for idx, row in enumerate(rows):
            heartbeat(args.heartbeat, f"corrupt:{split}:{idx + 1}/{len(rows)}:{row['sample_id']}")
            try:
                chosen, candidates, audit = process_candidate(row, split, idx, args)
                all_candidates.extend(candidates)
                candidate_audit.append(audit)
                if chosen is not None:
                    selected[split].append(chosen)
                    visual_rows.append(
                        {
                            "sample_id": chosen["sample_id"],
                            "split": split,
                            "profile": chosen["profile"],
                            "raw_output_mp4": chosen["raw_output_mp4"],
                            "review_sheet": chosen["review_sheet"],
                            "condition_path": chosen["condition_path"],
                            "winner_path": chosen["winner_path"],
                            "mask_path": chosen["mask_path"],
                            "classification": chosen["classification"],
                            "outside_damage": "low_by_exact_outside_reinjection_and_metric_gate",
                            "boundary_damage": "low_by_metric_gate",
                            "codex_visual_review": "PENDING_REVIEW_OF_GENERATED_SHEETS",
                            "reason": chosen["reason"],
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                bad = dict(row)
                bad["failure_stage"] = "corrupt"
                bad["failure_error"] = repr(exc)
                materialize_failures.append(bad)

    manifest_names = {
        "train": "exp40_localdpo_v3_train96.jsonl",
        "search": "exp40_localdpo_v3_search32.jsonl",
        "shadow": "exp40_localdpo_v3_shadow32.jsonl",
    }
    for split, name in manifest_names.items():
        write_jsonl(args.manifest_root / name, selected[split])
    rejected = source_rejected + materialize_failures + [c for c in all_candidates if c["classification"] not in {"MEDIUM_HARD_ELIGIBLE", "HARD_BUT_PLAUSIBLE"}]
    write_jsonl(args.manifest_root / "exp40_localdpo_v3_rejected.jsonl", rejected)
    write_csv(args.reports_root / "exp40_localdpo_v3_pool.csv", all_candidates)
    write_csv(args.reports_root / "exp40_localdpo_v3_visual_review.csv", visual_rows)

    status = split_status(selected, args)
    summary = {
        "status": status,
        "metadata_index": str(args.metadata_index),
        "metadata_sha256": sha256_file(args.metadata_index),
        "extraction": extraction,
        "targets": split_targets(args),
        "minimums": split_minimums(args),
        "selected_counts": {k: len(v) for k, v in selected.items()},
        "materialized_counts": {k: len(v) for k, v in materialized.items()},
        "candidate_rows": len(all_candidates),
        "rejected_rows": len(rejected),
        "classification_counts": counts(all_candidates, "classification"),
        "selected_classification_counts": {k: counts(v, "classification") for k, v in selected.items()},
        "selected_source_type_counts": {k: counts(v, "source_type") for k, v in selected.items()},
        "selected_mask_bucket_counts": {k: counts(v, "mask_bucket") for k, v in selected.items()},
        "scene_overlap": {
            "train_search": len({scene_group(r) for r in selected["train"]} & {scene_group(r) for r in selected["search"]}),
            "train_shadow": len({scene_group(r) for r in selected["train"]} & {scene_group(r) for r in selected["shadow"]}),
            "search_shadow": len({scene_group(r) for r in selected["search"]} & {scene_group(r) for r in selected["shadow"]}),
        },
        "vor_eval_used": False,
        "hard_comp_used": False,
        "candidate_rows_per_source_max": 3,
        "visual_review_status": "PENDING_REVIEW_OF_GENERATED_SHEETS",
    }
    write_json(args.reports_root / "exp40_localdpo_v3_summary.json", summary)
    md = [
        "# Exp40 LocalDPO v3 Pool",
        "",
        f"Status: `{status}`",
        "",
        "Source rules:",
        "",
        "- VOR-Train only",
        "- VOR-Eval used: `false`",
        "- hard comp used: `false`",
        "- condition = `V_obj`",
        "- winner = `V_bg`",
        "- loser = locally corrupted `V_bg`",
        "- candidate rows per source <= 3",
        "- outside preservation enforced by exact outside reinjection and strict metric gates",
        "- boundary safety enforced by metric gates",
        "",
        f"Selected counts: `{summary['selected_counts']}`",
        f"Materialized counts: `{summary['materialized_counts']}`",
        f"Candidate rows: `{summary['candidate_rows']}`",
        f"Classification counts: `{summary['classification_counts']}`",
        f"Scene overlap: `{summary['scene_overlap']}`",
        "",
        "Visual review sheets were generated for every selected candidate and",
        "must be opened before any training milestone claims video-review pass.",
    ]
    (args.reports_root / "exp40_localdpo_v3_pool.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    heartbeat(args.heartbeat, "done")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
