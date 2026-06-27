#!/usr/bin/env python3
"""Generate preregistered Exp30 controlled-corruption v3 candidates.

This script intentionally writes a new output tree and never mutates v2
controlled-corruption outputs.  It evaluates all preregistered v3 candidates
and also emits one deterministic primary controlled candidate per source for
the Smoke16 v3 gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

MILD_REPAIR_SOURCES = {
    "BLENDER_FOREST016_00001",
    "BLENDER_FOREST006_00001",
    "BLENDER_FOREST008_00001",
    "BLENDER_FOREST015_00001",
    "BLENDER_FOREST018_00001",
    "REAL_ENV045_00003_001_01",
}

AFFECTED_SOFT_SOURCES = {
    "REAL_ENV045_00003_001_01",
    "REAL_ENV046_00002_001_01",
}


@dataclass(frozen=True)
class Profile:
    profile_id: str
    role: str
    region: str
    noise_sigma: float
    condition_mix: float
    blur_ksize: int
    boundary_feather_px: int
    ema_alpha: float


PROFILES = {
    "CC-v3-A": Profile("CC-v3-A", "mild-object", "object_mask", 2.5, 0.25, 21, 9, 0.75),
    "CC-v3-B": Profile("CC-v3-B", "medium-object", "object_mask", 4.0, 0.35, 17, 9, 0.65),
    "CC-v3-C": Profile("CC-v3-C", "affected-soft", "object_mask_union_soft_affected", 3.0, 0.25, 19, 11, 0.75),
}

CLASS_RANK = {
    "MEDIUM_HARD_ELIGIBLE": 40,
    "HARD_BUT_PLAUSIBLE": 30,
    "TOO_CLOSE": 20,
    "TRIVIAL_BAD": 10,
    "TECHNICAL_INVALID": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--primary-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--label", default="Smoke16 V3")
    parser.add_argument("--min-primary-usable", type=int, default=8)
    parser.add_argument("--max-primary-trivial-bad", type=int, default=6)
    parser.add_argument("--success-status", default="CONTROLLED_CORRUPTION_V3_READY")
    parser.add_argument("--low-yield-status", default="CONTROLLED_CORRUPTION_V3_LOW_YIELD")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def image_files(path: Path) -> list[Path]:
    return sorted([p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS])


def read_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError(f"failed to read {path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def read_gray(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"failed to read {path}")
    return arr


def write_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def write_mp4(path: Path, frames: list[np.ndarray], fps: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def region_values(a: np.ndarray, b: np.ndarray, mask: np.ndarray, inside: bool) -> tuple[np.ndarray, np.ndarray]:
    region = mask > 20
    if not inside:
        region = ~region
    if not region.any():
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    return a[region].astype(np.float32), b[region].astype(np.float32)


def region_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray, inside: bool) -> float:
    av, bv = region_values(a, b, mask, inside)
    if av.size == 0:
        return float("nan")
    return psnr(av, bv)


def region_mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray, inside: bool) -> float:
    av, bv = region_values(a, b, mask, inside)
    if av.size == 0:
        return float("nan")
    return float(np.mean(np.abs(av - bv)))


def temporal_absdiff(frames: list[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    return float(
        np.mean(
            [
                np.mean(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32)))
                for i in range(1, len(frames))
            ]
        )
    )


def overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = frame.copy()
    m = mask > 20
    out[m] = (0.55 * out[m] + 0.45 * np.array([255, 40, 40])).astype(np.uint8)
    return out


def sample_indices(n: int, count: int = 16) -> list[int]:
    if n <= count:
        return list(range(n))
    return sorted({int(round(i * (n - 1) / (count - 1))) for i in range(count)})


def contact_sheet(frames: list[np.ndarray], labels: list[str], tile_w: int = 256, cols: int = 4) -> np.ndarray:
    tiles = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for frame, label_text in zip(frames, labels):
        h, w = frame.shape[:2]
        tile_h = int(round(h * tile_w / w))
        tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, label_text, (8, 24), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    rows = []
    for start in range(0, len(tiles), cols):
        chunk = tiles[start : start + cols]
        while len(chunk) < cols:
            chunk.append(np.zeros_like(tiles[0]))
        rows.append(np.concatenate(chunk, axis=1))
    return np.concatenate(rows, axis=0)


def sha256_tree(path: Path) -> str:
    h = hashlib.sha256()
    for file_path in sorted(path.glob("*.png")):
        h.update(file_path.name.encode())
        h.update(file_path.read_bytes())
    return h.hexdigest()


def feather(mask: np.ndarray, px: int) -> np.ndarray:
    base = (mask > 20).astype(np.float32)
    if px <= 0:
        return base
    k = px * 2 + 1
    return np.clip(cv2.GaussianBlur(base, (k, k), 0), 0.0, 1.0)


def affected_region(cond: np.ndarray, win: np.ndarray, mask: np.ndarray) -> np.ndarray:
    diff = np.mean(np.abs(cond.astype(np.float32) - win.astype(np.float32)), axis=2) / 255.0
    threshold = max(float(np.percentile(diff, 85)), 0.08)
    region = np.logical_or(mask > 20, diff >= threshold).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(region, cv2.MORPH_CLOSE, kernel)


def profile_schedule(sample_id: str) -> list[Profile]:
    profiles = [PROFILES["CC-v3-B"]]
    if sample_id in MILD_REPAIR_SOURCES:
        profiles.append(PROFILES["CC-v3-A"])
    if sample_id in AFFECTED_SOFT_SOURCES:
        profiles.append(PROFILES["CC-v3-C"])
    return profiles


def local_region(profile: Profile, cond: np.ndarray, win: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if profile.region == "object_mask":
        return mask
    if profile.region == "object_mask_union_soft_affected":
        return affected_region(cond, win, mask)
    raise ValueError(f"unknown profile region: {profile.region}")


def corrupt_frames(
    condition: list[np.ndarray],
    winner: list[np.ndarray],
    masks: list[np.ndarray],
    profile: Profile,
    seed: int,
    sample_id: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng_seed = seed + int(hashlib.sha256(f"{sample_id}:{profile.profile_id}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(rng_seed)
    k = profile.blur_ksize if profile.blur_ksize % 2 == 1 else profile.blur_ksize + 1
    out: list[np.ndarray] = []
    regions: list[np.ndarray] = []
    prev_local: np.ndarray | None = None
    for cond, win, mask in zip(condition, winner, masks):
        region = local_region(profile, cond, win, mask)
        soft = feather(region, profile.boundary_feather_px)[:, :, None]
        blurred = cv2.GaussianBlur(win, (k, k), 0)
        noise = rng.normal(0.0, profile.noise_sigma, size=win.shape).astype(np.float32)
        current = (
            profile.condition_mix * cond.astype(np.float32)
            + (1.0 - profile.condition_mix) * blurred.astype(np.float32)
            + noise
        )
        if prev_local is None:
            local = current
        else:
            local = profile.ema_alpha * prev_local + (1.0 - profile.ema_alpha) * current
        prev_local = local
        frame = win.astype(np.float32) * (1.0 - soft) + local * soft
        out.append(np.clip(frame, 0, 255).astype(np.uint8))
        regions.append(region)
    return out, regions


def classify(mask_psnr: float, outside_psnr: float, outside_mae: float, temporal_ratio: float) -> tuple[str, str]:
    if not np.isfinite(mask_psnr) or not np.isfinite(outside_psnr):
        return "TECHNICAL_INVALID", "non-finite region metric"
    if outside_psnr < 40.0 or outside_mae > 2.0:
        return "TRIVIAL_BAD", "outside preservation failed"
    if mask_psnr > 32.0:
        return "TOO_CLOSE", "task region too close to clean winner"
    if mask_psnr < 8.0:
        return "TRIVIAL_BAD", "task region too far from clean winner"
    if temporal_ratio > 2.5:
        return "TRIVIAL_BAD", "temporal instability too high"
    if mask_psnr < 14.0:
        return "HARD_BUT_PLAUSIBLE", "strong local residual with clean outside"
    return "MEDIUM_HARD_ELIGIBLE", "local residual with clean outside"


def primary_sort_key(row: dict) -> tuple[int, float, float, float]:
    target_mask_psnr = 17.0
    return (
        CLASS_RANK.get(str(row["classification"]), 0),
        -abs(float(row["mask_psnr"]) - target_mask_psnr) if np.isfinite(float(row["mask_psnr"])) else -999.0,
        -float(row["temporal_ratio"]) if np.isfinite(float(row["temporal_ratio"])) else -999.0,
        -float(row["outside_mae"]) if np.isfinite(float(row["outside_mae"])) else -999.0,
    )


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)
    review_rows: list[dict] = []
    metrics_rows: list[dict] = []

    for row in rows:
        sample_id = str(row["sample_id"])
        cond_files = image_files(Path(row["condition_frame_dir"]))
        win_files = image_files(Path(row["winner_frame_dir"]))
        mask_files = image_files(Path(row["mask_frame_dir"]))
        n = min(len(cond_files), len(win_files), len(mask_files), int(row["num_frames"]))
        condition = [read_rgb(p) for p in cond_files[:n]]
        winner = [read_rgb(p) for p in win_files[:n]]
        masks = [read_gray(p) for p in mask_files[:n]]

        for profile in profile_schedule(sample_id):
            output, regions = corrupt_frames(condition, winner, masks, profile, args.seed, sample_id)
            sample_root = args.output_root / "controlled_corruption_v3" / profile.profile_id / sample_id
            frame_dir = sample_root / "frames"
            for i, frame in enumerate(output):
                write_rgb(frame_dir / f"{i:05d}.png", frame)

            evidence = sample_root / "evidence"
            raw_mp4 = evidence / "raw_output.mp4"
            comp_mp4 = evidence / "diagnostic_comp.mp4"
            side_mp4 = evidence / "side_by_side.mp4"
            write_mp4(raw_mp4, output)
            write_mp4(comp_mp4, output)
            side_frames = [
                np.concatenate([c, overlay(c, m), w, o], axis=1)
                for c, w, m, o in zip(condition, winner, masks, output)
            ]
            write_mp4(side_mp4, side_frames)

            inds = sample_indices(n, 16)
            strip = contact_sheet([side_frames[i] for i in inds], [f"f{i:03d}" for i in inds], tile_w=384)
            strip_path = evidence / "temporal_strip_16.jpg"
            cv2.imwrite(str(strip_path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
            mid = n // 2
            crop_sheet = contact_sheet(
                [condition[mid], overlay(condition[mid], masks[mid]), winner[mid], output[mid]],
                ["condition_mid", "mask_mid", "winner_mid", f"{profile.profile_id}_mid"],
                tile_w=256,
            )
            crop_path = evidence / "midframe_review_sheet.jpg"
            cv2.imwrite(str(crop_path), cv2.cvtColor(crop_sheet, cv2.COLOR_RGB2BGR))

            full_psnr = float(np.mean([psnr(o, w) for o, w in zip(output, winner)]))
            mask_psnr = float(np.nanmean([region_psnr(o, w, m, True) for o, w, m in zip(output, winner, masks)]))
            outside_psnr = float(np.nanmean([region_psnr(o, w, m, False) for o, w, m in zip(output, winner, masks)]))
            mask_mae = float(np.nanmean([region_mae(o, w, m, True) for o, w, m in zip(output, winner, masks)]))
            outside_mae = float(np.nanmean([region_mae(o, w, m, False) for o, w, m in zip(output, winner, masks)]))
            output_temporal = temporal_absdiff(output)
            winner_temporal = temporal_absdiff(winner)
            temporal_ratio = output_temporal / max(winner_temporal, 1e-6)
            classification, reason = classify(mask_psnr, outside_psnr, outside_mae, temporal_ratio)

            common = {
                "sample_id": sample_id,
                "source_group": row.get("scene_group", ""),
                "source_type": row.get("source_type", ""),
                "model": "controlled_corruption_v3",
                "profile_id": profile.profile_id,
                "profile_role": profile.role,
                "profile_region": profile.region,
                "candidate_source": "controlled_local_condition_mix_blur_noise_temporal_ema",
                "checkpoint": "none",
                "step": 0,
                "condition_path": row.get("condition_frame_dir", ""),
                "winner_path": row.get("winner_frame_dir", ""),
                "loser_path": str(frame_dir),
                "mask_path": row.get("mask_frame_dir", ""),
                "affected_map_path": "derived_from_abs_condition_winner" if profile.profile_id == "CC-v3-C" else "",
                "raw_output_mp4": str(raw_mp4),
                "diagnostic_comp_mp4": str(comp_mp4),
                "side_by_side_mp4": str(side_mp4),
                "temporal_strip_16": str(strip_path),
                "review_sheet": str(crop_path),
                "frames_reviewed": f"0,{mid},{n-1},16-strip",
                "object_removed": "partial_residual_intentional",
                "effect_removed": "partial_residual_intentional",
                "mask_region_quality": "auto_classified_pending_codex_visual_review",
                "boundary_quality": "outside_reinjected_clean_boundary",
                "affected_region_quality": "controlled_residual",
                "outside_damage": "none_expected_outside_winner_reinjected",
                "temporal_flicker": "reduced_by_profile_ema",
                "ghosting": "condition_residual_intentional",
                "color_shift": "local_only",
                "artifact": "controlled_local_residual",
                "classification": classification,
                "reason": reason,
            }
            metrics = dict(common)
            metrics.update(
                {
                    "full_psnr": full_psnr,
                    "mask_psnr": mask_psnr,
                    "boundary_psnr": mask_psnr,
                    "outside_psnr": outside_psnr,
                    "mask_mae": mask_mae,
                    "outside_mae": outside_mae,
                    "tc_proxy_output_temporal_absdiff": output_temporal,
                    "tc_proxy_winner_temporal_absdiff": winner_temporal,
                    "temporal_ratio": temporal_ratio,
                    "output_sha256": sha256_tree(frame_dir),
                    "noise_sigma": profile.noise_sigma,
                    "condition_mix": profile.condition_mix,
                    "blur_ksize": profile.blur_ksize,
                    "boundary_feather_px": profile.boundary_feather_px,
                    "ema_alpha": profile.ema_alpha,
                }
            )
            review_rows.append(common)
            metrics_rows.append(metrics)

    by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in metrics_rows:
        by_sample[str(row["sample_id"])].append(row)
    primary_rows = [sorted(rows_for_sample, key=primary_sort_key, reverse=True)[0] for rows_for_sample in by_sample.values()]
    primary_rows = sorted(primary_rows, key=lambda r: str(r["sample_id"]))

    write_rows(args.review_csv, review_rows)
    write_rows(args.metrics_csv, metrics_rows)
    write_rows(args.primary_csv, primary_rows)

    all_counts = Counter(r["classification"] for r in review_rows)
    primary_counts = Counter(r["classification"] for r in primary_rows)
    primary_usable = primary_counts["MEDIUM_HARD_ELIGIBLE"] + primary_counts["HARD_BUT_PLAUSIBLE"]
    primary_technical_valid = len([r for r in primary_rows if r["classification"] != "TECHNICAL_INVALID"])
    primary_outside_fail = len(
        [r for r in primary_rows if float(r["outside_psnr"]) < 40.0 or float(r["outside_mae"]) > 2.0]
    )
    status = (
        args.success_status
        if primary_technical_valid >= 15
        and primary_usable >= args.min_primary_usable
        and primary_counts["TRIVIAL_BAD"] <= args.max_primary_trivial_bad
        and primary_outside_fail == 0
        else args.low_yield_status
    )
    summary = {
        "status": status,
        "label": args.label,
        "candidate_count": len(review_rows),
        "source_count": len(primary_rows),
        "all_classification_counts": dict(all_counts),
        "primary_classification_counts": dict(primary_counts),
        "primary_usable_count": primary_usable,
        "primary_usable_required": args.min_primary_usable,
        "primary_technical_valid_count": primary_technical_valid,
        "primary_outside_fail_count": primary_outside_fail,
        "primary_trivial_bad_max": args.max_primary_trivial_bad,
        "review_csv": str(args.review_csv),
        "metrics_csv": str(args.metrics_csv),
        "primary_csv": str(args.primary_csv),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(
        f"# Exp30 Controlled-Corruption {args.label}\n\n"
        f"Status: `{status}`\n\n"
        f"- Candidate count: {len(review_rows)}\n"
        f"- Source count: {len(primary_rows)}\n"
        f"- All classification counts: `{dict(all_counts)}`\n"
        f"- Primary classification counts: `{dict(primary_counts)}`\n"
        f"- Primary usable count: {primary_usable} / required {args.min_primary_usable}\n"
        f"- Primary technical-valid count: {primary_technical_valid}\n"
        f"- Primary outside-fail count: {primary_outside_fail}\n\n"
        "The v3 controlled fallback follows the preregistered profile schedule "
        "from `exp30_controlled_corruption_v3_plan.json`: CC-v3-B for all "
        "sources, CC-v3-A on six repair sources, and CC-v3-C on two affected "
        "soft sources.  It is still only one component of Smoke16 v3 and does "
        "not unlock Gate64 or adapter training by itself.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status == args.success_status else 2


if __name__ == "__main__":
    raise SystemExit(main())
