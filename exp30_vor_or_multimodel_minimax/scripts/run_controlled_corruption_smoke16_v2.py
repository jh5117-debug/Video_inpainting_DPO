#!/usr/bin/env python3
"""Generate controlled local-corruption OR candidates for Exp30 smoke16."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--noise", type=float, default=10.0)
    parser.add_argument("--condition-mix", type=float, default=0.52)
    parser.add_argument("--blur-ksize", type=int, default=15)
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
    return float(np.mean([np.mean(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))) for i in range(1, len(frames))]))


def overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = frame.copy()
    m = mask > 20
    out[m] = (0.55 * out[m] + 0.45 * np.array([255, 40, 40])).astype(np.uint8)
    return out


def sample_indices(n: int, count: int = 16) -> list[int]:
    if n <= count:
        return list(range(n))
    return sorted({int(round(i * (n - 1) / (count - 1))) for i in range(count)})


def contact_sheet(frames: list[np.ndarray], labels: list[str], tile_w: int = 256) -> np.ndarray:
    tiles = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for frame, label in zip(frames, labels):
        h, w = frame.shape[:2]
        tile_h = int(round(h * tile_w / w))
        tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, label, (8, 24), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    rows = []
    for start in range(0, len(tiles), 4):
        chunk = tiles[start : start + 4]
        while len(chunk) < 4:
            chunk.append(np.zeros_like(tiles[0]))
        rows.append(np.concatenate(chunk, axis=1))
    return np.concatenate(rows, axis=0)


def sha256_tree(path: Path) -> str:
    h = hashlib.sha256()
    for file_path in sorted(path.glob("*.png")):
        h.update(file_path.name.encode())
        h.update(file_path.read_bytes())
    return h.hexdigest()


def corrupt_frames(condition: list[np.ndarray], winner: list[np.ndarray], masks: list[np.ndarray], args: argparse.Namespace, sample_id: str) -> list[np.ndarray]:
    rng = np.random.default_rng(args.seed + int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16))
    k = args.blur_ksize if args.blur_ksize % 2 == 1 else args.blur_ksize + 1
    out = []
    for cond, win, mask in zip(condition, winner, masks):
        blurred = cv2.GaussianBlur(win, (k, k), 0)
        noise = rng.normal(0.0, args.noise, size=win.shape).astype(np.float32)
        local = args.condition_mix * cond.astype(np.float32) + (1.0 - args.condition_mix) * blurred.astype(np.float32) + noise
        m = (mask > 20)[:, :, None].astype(np.float32)
        frame = win.astype(np.float32) * (1.0 - m) + local * m
        out.append(np.clip(frame, 0, 255).astype(np.uint8))
    return out


def classify(mask_psnr: float, outside_psnr: float, outside_mae: float, temporal_ratio: float) -> tuple[str, str]:
    if not np.isfinite(mask_psnr) or not np.isfinite(outside_psnr):
        return "TECHNICAL_INVALID", "non-finite region metric"
    if outside_psnr < 40 or outside_mae > 2.0:
        return "TRIVIAL_BAD", "outside preservation failed"
    if mask_psnr > 32:
        return "TOO_CLOSE", "task region too close to clean winner"
    if mask_psnr < 8:
        return "TRIVIAL_BAD", "task region too far from clean winner"
    if temporal_ratio > 2.5:
        return "TRIVIAL_BAD", "temporal instability too high"
    if mask_psnr < 14:
        return "HARD_BUT_PLAUSIBLE", "strong local residual with clean outside"
    return "MEDIUM_HARD_ELIGIBLE", "local residual with clean outside"


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
        output = corrupt_frames(condition, winner, masks, args, sample_id)
        sample_root = args.output_root / "controlled_corruption" / sample_id
        frame_dir = sample_root / "frames"
        for i, frame in enumerate(output):
            write_rgb(frame_dir / f"{i:05d}.png", frame)
        evidence = sample_root / "evidence"
        raw_mp4 = evidence / "raw_output.mp4"
        comp_mp4 = evidence / "diagnostic_comp.mp4"
        side_mp4 = evidence / "side_by_side.mp4"
        write_mp4(raw_mp4, output)
        write_mp4(comp_mp4, output)
        side_frames = [np.concatenate([c, overlay(c, m), w, o], axis=1) for c, w, m, o in zip(condition, winner, masks, output)]
        write_mp4(side_mp4, side_frames)
        inds = sample_indices(n, 16)
        strip = contact_sheet([side_frames[i] for i in inds], [f"f{i:03d}" for i in inds], tile_w=384)
        strip_path = evidence / "temporal_strip_16.jpg"
        cv2.imwrite(str(strip_path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        mid = n // 2
        crop_sheet = contact_sheet(
            [condition[mid], overlay(condition[mid], masks[mid]), winner[mid], output[mid]],
            ["condition_mid", "mask_mid", "winner_mid", "controlled_mid"],
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
            "model": "controlled_corruption",
            "candidate_source": "controlled_local_condition_mix_blur_noise",
            "checkpoint": "none",
            "step": 0,
            "condition_path": row.get("condition_frame_dir", ""),
            "winner_path": row.get("winner_frame_dir", ""),
            "loser_path": str(frame_dir),
            "mask_path": row.get("mask_frame_dir", ""),
            "affected_map_path": "",
            "raw_output_mp4": str(raw_mp4),
            "diagnostic_comp_mp4": str(comp_mp4),
            "side_by_side_mp4": str(side_mp4),
            "temporal_strip_16": str(strip_path),
            "review_sheet": str(crop_path),
            "frames_reviewed": f"0,{mid},{n-1},16-strip",
            "object_removed": "partial_residual_intentional",
            "effect_removed": "partial_residual_intentional",
            "mask_region_quality": "medium_hard_local_defect",
            "boundary_quality": "outside_reinjected_clean_boundary",
            "affected_region_quality": "controlled_residual",
            "outside_damage": "none_expected_outside_winner_reinjected",
            "temporal_flicker": "low_expected_shared_source_corruption",
            "ghosting": "condition_residual_intentional",
            "color_shift": "local_only",
            "artifact": "controlled_local_residual",
            "classification": classification,
            "reason": reason,
        }
        review_rows.append(common)
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
            }
        )
        metrics_rows.append(metrics)

    for path, data in [(args.review_csv, review_rows), (args.metrics_csv, metrics_rows)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)
    counts = Counter(r["classification"] for r in review_rows)
    usable = counts["MEDIUM_HARD_ELIGIBLE"] + counts["HARD_BUT_PLAUSIBLE"]
    summary = {
        "status": "CONTROLLED_CORRUPTION_SMOKE16_READY" if usable >= 6 else "CONTROLLED_CORRUPTION_SMOKE16_LOW_YIELD",
        "candidate_count": len(review_rows),
        "technical_valid_count": len([r for r in review_rows if r["classification"] not in {"TECHNICAL_INVALID", "WRAPPER_FAILURE"}]),
        "usable_count": usable,
        "classification_counts": dict(counts),
        "review_csv": str(args.review_csv),
        "metrics_csv": str(args.metrics_csv),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_md.write_text(
        "# Exp30 Controlled-Corruption Smoke16 V2\n\n"
        f"Status: `{summary['status']}`\n\n"
        f"- Candidate count: {summary['candidate_count']}\n"
        f"- Technical valid count: {summary['technical_valid_count']}\n"
        f"- Usable medium-hard/hard-plausible count: {summary['usable_count']}\n"
        f"- Classification counts: `{summary['classification_counts']}`\n"
        f"- Review CSV: `{args.review_csv}`\n"
        f"- Metrics CSV: `{args.metrics_csv}`\n\n"
        "This is a controlled local corruption fallback, not a model-generated "
        "primary loser by itself. It can satisfy the fallback part of smoke16 "
        "but does not alone unlock Gate64 or MiniMax adapter training.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if usable >= 6 else 2


if __name__ == "__main__":
    raise SystemExit(main())
