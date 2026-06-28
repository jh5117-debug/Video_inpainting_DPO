#!/usr/bin/env python3
"""Build an Exp37 LocalDPO-style OR corruption pool for MiniMax."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np


PROFILE_PAIRS = (
    ("L2_object_medium", "L4_boundary_effect"),
    ("L3_object_affected_soft", "L2_object_medium"),
    ("L1_object_mild", "L2_object_medium"),
    ("L4_boundary_effect", "L3_object_affected_soft"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--heldout-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reports-root", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--heartbeat", default="")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


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
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def heartbeat(path: Path | None, message: str) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{time.time():.0f}\t{message}\n", encoding="utf-8")


def frame_paths(path: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(path.glob(pattern))
    return sorted(files)


def load_frames(path: Path) -> list[np.ndarray]:
    frames = []
    for frame_path in frame_paths(path):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"failed to read frame: {frame_path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return frames


def load_masks(path: Path, target_count: int) -> list[np.ndarray]:
    masks = []
    for frame_path in frame_paths(path):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise RuntimeError(f"failed to read mask: {frame_path}")
        masks.append((frame > 8).astype(np.float32))
    if len(masks) != target_count:
        raise RuntimeError(f"mask count mismatch: {path} has {len(masks)}, expected {target_count}")
    return masks


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    if mask is not None:
        weights = mask.astype(np.float32)
        if weights.ndim == 2:
            weights = weights[:, :, None]
        denom = float(weights.sum() * 3.0)
        if denom <= 1e-6:
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
        if denom <= 1e-6:
            return float("nan")
        return float((diff * weights).sum() / denom)
    return float(diff.mean())


def dilate(mask: np.ndarray, ksize: int) -> np.ndarray:
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    return cv2.dilate((mask > 0.05).astype(np.uint8), kernel, iterations=1).astype(np.float32)


def erode(mask: np.ndarray, ksize: int) -> np.ndarray:
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    return cv2.erode((mask > 0.05).astype(np.uint8), kernel, iterations=1).astype(np.float32)


def soft_blur(mask: np.ndarray, ksize: int = 17) -> np.ndarray:
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), 0)
    return np.clip(blurred, 0.0, 1.0)


def profile_region(
    profile: str,
    mask: np.ndarray,
    affected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    boundary = np.clip(dilate(mask, 9) - erode(mask, 7), 0.0, 1.0)
    affected_soft = soft_blur((affected > 0.07).astype(np.float32), 21)
    if profile == "L1_object_mild":
        region = soft_blur(mask, 11)
        return region, boundary, 0.22, 3.5
    if profile == "L2_object_medium":
        region = soft_blur(mask, 15)
        return region, boundary, 0.38, 6.0
    if profile == "L3_object_affected_soft":
        region = np.clip(np.maximum(soft_blur(mask, 13), affected_soft * 0.85), 0.0, 1.0)
        return region, boundary, 0.30, 4.5
    if profile == "L4_boundary_effect":
        region = np.clip(np.maximum(soft_blur(boundary, 11), affected_soft * 0.55), 0.0, 1.0)
        return region, boundary, 0.45, 5.0
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
    for win, cond, mask in zip(winner, condition, masks):
        affected = np.mean(np.abs(cond.astype(np.float32) - win.astype(np.float32)), axis=2) / 255.0
        region, boundary, alpha, noise_scale = profile_region(profile, mask, affected)
        region3 = region[:, :, None]
        cond_blur = cv2.GaussianBlur(cond, (5, 5), 0).astype(np.float32)
        win_blur = cv2.GaussianBlur(win, (5, 5), 0).astype(np.float32)
        local_target = 0.72 * cond_blur + 0.28 * win_blur
        noise = rng.normal(0.0, noise_scale, size=win.shape).astype(np.float32)
        if prev_noise is not None:
            noise = 0.65 * prev_noise + 0.35 * noise
        prev_noise = noise
        candidate = win.astype(np.float32) * (1.0 - alpha * region3) + local_target * (alpha * region3) + noise * region3
        # Strict outside reinjection: only the soft local region can differ.
        out = np.where(region3 > 1e-4, candidate, win.astype(np.float32))
        outputs.append(np.clip(out, 0, 255).astype(np.uint8))
        regions.append(region)
        boundaries.append(boundary)
    return outputs, regions, boundaries


def save_frames(frames: list[np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        cv2.imwrite(str(out_dir / f"{idx:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def save_mp4(frames: list[np.ndarray], path: Path, fps: float = 8.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def make_review_sheet(
    condition: list[np.ndarray],
    winner: list[np.ndarray],
    loser: list[np.ndarray],
    masks: list[np.ndarray],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = sorted(set([0, len(winner) // 4, len(winner) // 2, (3 * len(winner)) // 4, len(winner) - 1]))
    rows = []
    for idx in indices:
        cond = condition[idx]
        win = winner[idx]
        los = loser[idx]
        diff = np.clip(np.abs(los.astype(np.int16) - win.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
        overlay = win.copy()
        mask = masks[idx] > 0.05
        overlay[mask] = (0.6 * overlay[mask] + np.array([255, 40, 40]) * 0.4).astype(np.uint8)
        row = np.concatenate([cond, overlay, win, los, diff], axis=1)
        cv2.putText(row, f"f{idx:02d} cond|mask|winner|loser|diffx4", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        rows.append(row)
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def metrics_for_candidate(
    winner: list[np.ndarray],
    loser: list[np.ndarray],
    masks: list[np.ndarray],
    boundaries: list[np.ndarray],
    regions: list[np.ndarray],
) -> dict[str, float]:
    outside_masks = [1.0 - np.clip(r, 0.0, 1.0) for r in regions]
    vals: dict[str, list[float]] = {
        "full_psnr": [],
        "mask_psnr": [],
        "boundary_psnr": [],
        "outside_psnr": [],
        "affected_psnr": [],
        "mask_mae": [],
        "boundary_mae": [],
        "outside_mae": [],
        "temporal_flicker_mae": [],
    }
    for idx, (win, los, mask, boundary, region, outside) in enumerate(zip(winner, loser, masks, boundaries, regions, outside_masks)):
        vals["full_psnr"].append(psnr(win, los))
        vals["mask_psnr"].append(psnr(win, los, mask))
        vals["boundary_psnr"].append(psnr(win, los, boundary))
        vals["outside_psnr"].append(psnr(win, los, outside))
        vals["affected_psnr"].append(psnr(win, los, region))
        vals["mask_mae"].append(mae(win, los, mask))
        vals["boundary_mae"].append(mae(win, los, boundary))
        vals["outside_mae"].append(mae(win, los, outside))
        if idx > 0:
            win_dt = np.abs(win.astype(np.float32) - winner[idx - 1].astype(np.float32))
            los_dt = np.abs(los.astype(np.float32) - loser[idx - 1].astype(np.float32))
            vals["temporal_flicker_mae"].append(float(np.abs(los_dt - win_dt).mean()))
    return {key: float(np.nanmean(value)) if value else float("nan") for key, value in vals.items()}


def classify(metrics: dict[str, float]) -> tuple[str, str]:
    outside_ok = metrics["outside_psnr"] >= 45.0 and metrics["outside_mae"] <= 1.0
    if not outside_ok:
        return "TRIVIAL_BAD", "outside damage exceeds local-corruption bound"
    if metrics["mask_mae"] < 3.0 or metrics["mask_psnr"] > 36.0:
        return "TOO_CLOSE", "local corruption is too close to winner"
    if metrics["mask_psnr"] < 10.0 or metrics["mask_mae"] > 85.0:
        return "TRIVIAL_BAD", "local corruption is too severe"
    if metrics["mask_psnr"] < 15.0:
        return "HARD_BUT_PLAUSIBLE", "strong local corruption with exact outside preservation"
    return "MEDIUM_HARD_ELIGIBLE", "bounded local corruption with exact outside preservation"


def sha256_tree(path: Path) -> str:
    h = hashlib.sha256()
    for frame_path in frame_paths(path):
        h.update(frame_path.name.encode("utf-8"))
        h.update(frame_path.read_bytes())
    return h.hexdigest()


def process_row(
    row: dict[str, object],
    split: str,
    split_index: int,
    output_root: Path,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    sample_id = str(row["sample_id"])
    condition = load_frames(Path(str(row["condition_path"])))
    winner = load_frames(Path(str(row["winner_path"])))
    masks = load_masks(Path(str(row["mask_path"])), len(winner))
    if len(condition) != len(winner):
        raise RuntimeError(f"condition/winner count mismatch for {sample_id}")
    profile_pair = PROFILE_PAIRS[split_index % len(PROFILE_PAIRS)]
    candidate_rows = []
    for cand_index, profile in enumerate(profile_pair):
        rng = np.random.default_rng(seed + split_index * 101 + cand_index * 17)
        loser, regions, boundaries = corrupt_frames(winner, condition, masks, profile, rng)
        cand_root = output_root / split / sample_id / profile
        frames_dir = cand_root / "frames"
        evidence_dir = cand_root / "evidence"
        save_frames(loser, frames_dir)
        raw_mp4 = evidence_dir / "raw_output.mp4"
        review_sheet = evidence_dir / "review_sheet.jpg"
        save_mp4(loser, raw_mp4)
        make_review_sheet(condition, winner, loser, masks, review_sheet)
        m = metrics_for_candidate(winner, loser, masks, boundaries, regions)
        classification, reason = classify(m)
        candidate_rows.append(
            {
                "split": split,
                "sample_id": sample_id,
                "source_group": row.get("source_group", ""),
                "source_type": row.get("source_type", ""),
                "profile": profile,
                "candidate_index": cand_index,
                "condition_path": row["condition_path"],
                "winner_path": row["winner_path"],
                "mask_path": row["mask_path"],
                "loser_path": str(frames_dir),
                "raw_output_mp4": str(raw_mp4),
                "review_sheet": str(review_sheet),
                "classification": classification,
                "reason": reason,
                "technical_valid": "yes",
                "outside_reinjection": "pixel_exact_outside_soft_region",
                "temporal_smoothing": "ema_noise_0.65",
                "vor_eval_used": False,
                "training_eligible": True,
                "output_sha256": sha256_tree(frames_dir),
                **{key: f"{value:.12f}" for key, value in m.items()},
            }
        )
    priority = {
        "MEDIUM_HARD_ELIGIBLE": 0,
        "HARD_BUT_PLAUSIBLE": 1,
        "TOO_CLOSE": 2,
        "TRIVIAL_BAD": 3,
        "TECHNICAL_INVALID": 4,
    }
    selected = sorted(candidate_rows, key=lambda r: (priority.get(str(r["classification"]), 99), int(r["candidate_index"])))[0]
    selected["selection_role"] = "selected_primary_localdpo_corruption"
    selected["localdpo_condition_role"] = "V_obj"
    selected["localdpo_winner_role"] = "V_bg"
    selected["localdpo_loser_role"] = "locally_corrupted_V_bg"
    selected["affected_map_definition"] = "abs(V_obj - V_bg), used for L3/L4 region construction"
    return selected, candidate_rows


def summarize(selected: list[dict[str, object]], candidates: list[dict[str, object]]) -> dict[str, object]:
    def counts(rows: list[dict[str, object]], key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in rows:
            value = str(row.get(key, ""))
            out[value] = out.get(value, 0) + 1
        return out

    usable = [r for r in selected if r["classification"] in ("MEDIUM_HARD_ELIGIBLE", "HARD_BUT_PLAUSIBLE")]
    outside = [float(r["outside_mae"]) for r in selected]
    return {
        "selected_rows": len(selected),
        "candidate_rows": len(candidates),
        "classification_counts": counts(selected, "classification"),
        "profile_counts": counts(selected, "profile"),
        "usable_selected": len(usable),
        "outside_mae_mean": float(np.mean(outside)) if outside else None,
        "scene_groups": len({str(r.get("source_group", r["sample_id"])) for r in selected}),
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    reports_root = Path(args.reports_root)
    manifest_root = Path(args.manifest_root)
    hb = Path(args.heartbeat) if args.heartbeat else None
    splits = [
        ("train32", read_jsonl(Path(args.train_manifest))),
        ("heldout16", read_jsonl(Path(args.heldout_manifest))),
    ]
    selected_by_split: dict[str, list[dict[str, object]]] = {"train32": [], "heldout16": []}
    all_candidates: list[dict[str, object]] = []
    visual_rows: list[dict[str, object]] = []
    for split, rows in splits:
        for idx, row in enumerate(rows):
            heartbeat(hb, f"{split}:{idx + 1}/{len(rows)}:{row['sample_id']}")
            selected, candidates = process_row(row, split, idx, output_root, args.seed)
            selected_by_split[split].append(selected)
            all_candidates.extend(candidates)
            visual_rows.append(
                {
                    "split": split,
                    "sample_id": selected["sample_id"],
                    "profile": selected["profile"],
                    "review_sheet": selected["review_sheet"],
                    "raw_output_mp4": selected["raw_output_mp4"],
                    "classification": selected["classification"],
                    "codex_visual_review": "PENDING_CODEX_VISUAL_REVIEW",
                    "reason": selected["reason"],
                    "outside_damage": "low_by_exact_outside_reinjection",
                    "temporal_artifact": "pending_visual_review",
                }
            )
    train_path = manifest_root / "localdpo_or_train32.jsonl"
    heldout_path = manifest_root / "localdpo_or_heldout16.jsonl"
    write_jsonl(train_path, selected_by_split["train32"])
    write_jsonl(heldout_path, selected_by_split["heldout16"])
    write_csv(reports_root / "exp37_localdpo_style_or_corruption_pool.csv", all_candidates)
    write_csv(reports_root / "exp37_localdpo_style_visual_review.csv", visual_rows)
    summary = {
        "status": "PENDING_CODEX_VISUAL_REVIEW",
        "train32": summarize(selected_by_split["train32"], [r for r in all_candidates if r["split"] == "train32"]),
        "heldout16": summarize(selected_by_split["heldout16"], [r for r in all_candidates if r["split"] == "heldout16"]),
        "train_manifest": str(train_path),
        "heldout_manifest": str(heldout_path),
        "candidate_rows_per_source_max": 2,
        "vor_eval_used": False,
    }
    write_json(reports_root / "exp37_localdpo_style_or_corruption_pool_summary.json", summary)
    md = [
        "# Exp37 LocalDPO-style OR Corruption Pool",
        "",
        "Status: `PENDING_CODEX_VISUAL_REVIEW`",
        "",
        "Built deterministic local corruptions from VOR-Train style rows only:",
        "",
        "- condition = `V_obj`",
        "- winner = `V_bg`",
        "- loser = locally corrupted `V_bg`",
        "- mask = object mask",
        "- affected map = `abs(V_obj - V_bg)` for profile construction",
        "- candidate rows per source <= 2",
        "- VOR-Eval used = `false`",
        "",
        f"Train selected summary: `{summary['train32']}`",
        f"Heldout selected summary: `{summary['heldout16']}`",
        "",
        "Codex visual review is required before marking the pool ready.",
    ]
    (reports_root / "exp37_localdpo_style_or_corruption_pool.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    heartbeat(hb, "done")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
