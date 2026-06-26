#!/usr/bin/env python3
"""Score and package MiniMax micro candidates for visual review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--heldout-manifest", required=True)
    parser.add_argument("--rejected-manifest", required=True)
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


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def masked_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray, want_inside: bool) -> float:
    region = mask > 20
    if not want_inside:
        region = ~region
    if region.sum() == 0:
        return float("nan")
    av = a[region]
    bv = b[region]
    return psnr(av, bv)


def masked_mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray, want_inside: bool) -> float:
    region = mask > 20
    if not want_inside:
        region = ~region
    if region.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(a[region].astype(np.float32) - b[region].astype(np.float32))))


def temporal_absdiff(frames: list[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    diffs = [np.mean(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))) for i in range(1, len(frames))]
    return float(np.mean(diffs))


def classify(row: dict[str, object]) -> tuple[str, str]:
    mask_psnr = float(row["mask_psnr"])
    outside_psnr = float(row["outside_psnr"])
    outside_mae = float(row["outside_mae"])
    output_temporal = float(row["output_temporal_absdiff"])
    winner_temporal = float(row["winner_temporal_absdiff"])
    temporal_ratio = output_temporal / max(winner_temporal, 1e-6)

    if not np.isfinite(mask_psnr) or not np.isfinite(outside_psnr):
        return "TECHNICAL_INVALID", "non-finite region metric"
    if outside_psnr < 24 or outside_mae > 16:
        return "TRIVIAL_BAD", "systematic outside damage"
    if mask_psnr > 30:
        return "TOO_CLOSE", "candidate too close to clean winner in task mask"
    if mask_psnr < 8:
        return "TRIVIAL_BAD", "task region too far from clean winner"
    if temporal_ratio > 2.5:
        return "TRIVIAL_BAD", "temporal instability much higher than winner"
    if mask_psnr < 14:
        return "HARD_BUT_PLAUSIBLE", "strong local defect with acceptable outside preservation"
    return "MEDIUM_HARD_ELIGIBLE", "finite local defect with acceptable outside preservation"


def sha256_jsonl(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_review_pages(rows: list[dict[str, object]], page_root: Path, rows_per_page: int = 4) -> list[str]:
    page_root.mkdir(parents=True, exist_ok=True)
    pages = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for page_idx, start in enumerate(range(0, len(rows), rows_per_page)):
        strips = []
        for row in rows[start : start + rows_per_page]:
            strip = cv2.imread(str(row["temporal_strip_16"]), cv2.IMREAD_COLOR)
            if strip is None:
                continue
            label = f"{row['sample_id']} seed={row['seed']} {row['classification']} mPSNR={float(row['mask_psnr']):.2f} oPSNR={float(row['outside_psnr']):.2f}"
            cv2.putText(strip, label, (8, 24), font, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            strips.append(strip)
        if not strips:
            continue
        width = max(s.shape[1] for s in strips)
        padded = []
        for strip in strips:
            if strip.shape[1] < width:
                pad = np.zeros((strip.shape[0], width - strip.shape[1], 3), dtype=np.uint8)
                strip = np.concatenate([strip, pad], axis=1)
            padded.append(strip)
        page = np.concatenate(padded, axis=0)
        page_path = page_root / f"minimax_candidate_review_page_{page_idx:02d}.jpg"
        cv2.imwrite(str(page_path), page)
        pages.append(str(page_path))
    return pages


def main() -> None:
    args = parse_args()
    candidate_root = Path(args.candidate_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for done_path in sorted(candidate_root.glob("*/seed_*/DONE.json")):
        meta = json.loads(done_path.read_text(encoding="utf-8"))
        frames_dir = Path(str(meta["output_frame_dir"]))
        winner_dir = Path(str(meta["winner_frame_dir"]))
        mask_dir = Path(str(meta["mask_frame_dir"]))
        out_files = image_files(frames_dir)
        win_files = image_files(winner_dir)
        mask_files = image_files(mask_dir)
        n = min(len(out_files), len(win_files), len(mask_files), int(meta["num_frames"]))
        if n <= 0:
            meta["classification"] = "TECHNICAL_INVALID"
            meta["reason"] = "missing frames"
            rows.append(meta)
            continue
        output_frames = [read_rgb(p) for p in out_files[:n]]
        winner_frames = [read_rgb(p) for p in win_files[:n]]
        masks = [read_gray(p) for p in mask_files[:n]]
        full_psnrs = [psnr(o, w) for o, w in zip(output_frames, winner_frames)]
        mask_psnrs = [masked_psnr(o, w, m, True) for o, w, m in zip(output_frames, winner_frames, masks)]
        outside_psnrs = [masked_psnr(o, w, m, False) for o, w, m in zip(output_frames, winner_frames, masks)]
        mask_maes = [masked_mae(o, w, m, True) for o, w, m in zip(output_frames, winner_frames, masks)]
        outside_maes = [masked_mae(o, w, m, False) for o, w, m in zip(output_frames, winner_frames, masks)]
        meta.update(
            {
                "full_psnr": float(np.nanmean(full_psnrs)),
                "mask_psnr": float(np.nanmean(mask_psnrs)),
                "outside_psnr": float(np.nanmean(outside_psnrs)),
                "mask_mae": float(np.nanmean(mask_maes)),
                "outside_mae": float(np.nanmean(outside_maes)),
                "output_temporal_absdiff": temporal_absdiff(output_frames),
                "winner_temporal_absdiff": temporal_absdiff(winner_frames),
                "review_method": "dense_temporal_strip_plus_metric_preclassification",
                "reviewed_frames": "0-16",
            }
        )
        classification, reason = classify(meta)
        meta["classification"] = classification
        meta["reason"] = reason
        rows.append(meta)

    rows = sorted(rows, key=lambda r: (str(r.get("sample_id")), int(r.get("seed", 0))))
    csv_path = output_root / "exp29_minimax_preference_data_quality.csv"
    save_csv(csv_path, rows)

    eligible = [r for r in rows if r["classification"] in {"MEDIUM_HARD_ELIGIBLE", "HARD_BUT_PLAUSIBLE"}]
    rejected = [r for r in rows if r["classification"] not in {"MEDIUM_HARD_ELIGIBLE", "HARD_BUT_PLAUSIBLE"}]

    # Keep one candidate per scene/sample before splitting, preferring medium-hard
    # over hard-plausible and balanced lower outside damage.
    by_sample: dict[str, list[dict[str, object]]] = {}
    for row in eligible:
        by_sample.setdefault(str(row["sample_id"]), []).append(row)
    ranked_per_sample = []
    for sample_rows in by_sample.values():
        ranked_per_sample.append(
            sorted(
                sample_rows,
                key=lambda r: (
                    0 if r["classification"] == "MEDIUM_HARD_ELIGIBLE" else 1,
                    abs(float(r["mask_psnr"]) - 18.0),
                    float(r["outside_mae"]),
                ),
            )[0]
        )
    ranked_per_sample = sorted(
        ranked_per_sample,
        key=lambda r: (
            str(r.get("source_type")),
            str(r.get("mask_bucket")),
            hashlib.sha256(str(r["sample_id"]).encode("utf-8")).hexdigest(),
        ),
    )
    train = ranked_per_sample[:16]
    heldout = ranked_per_sample[16:32]
    selected_ids = {f"{r['sample_id']}::{r['seed']}" for r in train + heldout}
    rejected.extend([r for r in rows if f"{r['sample_id']}::{r['seed']}" not in selected_ids and r not in rejected])

    write_jsonl(Path(args.train_manifest), train)
    write_jsonl(Path(args.heldout_manifest), heldout)
    write_jsonl(Path(args.rejected_manifest), rejected)
    page_paths = make_review_pages(rows, output_root / "review_pages")

    summary = {
        "candidate_count": len(rows),
        "eligible_count": len(eligible),
        "train_count": len(train),
        "heldout_count": len(heldout),
        "classification_counts": {k: sum(1 for r in rows if r["classification"] == k) for k in sorted({str(r["classification"]) for r in rows})},
        "train_manifest": str(Path(args.train_manifest)),
        "heldout_manifest": str(Path(args.heldout_manifest)),
        "rejected_manifest": str(Path(args.rejected_manifest)),
        "train_sha256": sha256_jsonl(Path(args.train_manifest)) if train else "",
        "heldout_sha256": sha256_jsonl(Path(args.heldout_manifest)) if heldout else "",
        "review_pages": page_paths,
        "status": "MINIMAX_MICRO_DATA_READY" if len(train) == 16 and len(heldout) == 16 and len(eligible) >= 24 else "MINIMAX_DATA_YIELD_INSUFFICIENT",
    }
    (output_root / "exp29_minimax_preference_data_quality_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
