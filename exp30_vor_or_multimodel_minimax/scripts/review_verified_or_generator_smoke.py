#!/usr/bin/env python3
"""Build lightweight metrics and review sheets for Exp30 verified-generator smoke."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--visual-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--models", nargs="+", default=["propainter", "diffueraser"])
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def image_files(path: Path) -> list[Path]:
    return sorted([p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS])


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


def region_mask(mask: np.ndarray, inside: bool) -> np.ndarray:
    reg = mask > 20
    return reg if inside else ~reg


def region_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray, inside: bool) -> float:
    reg = region_mask(mask, inside)
    if not reg.any():
        return float("nan")
    return psnr(a[reg], b[reg])


def region_mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray, inside: bool) -> float:
    reg = region_mask(mask, inside)
    if not reg.any():
        return float("nan")
    return float(np.mean(np.abs(a[reg].astype(np.float32) - b[reg].astype(np.float32))))


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
    reg = mask > 20
    out[reg] = (0.55 * out[reg] + 0.45 * np.array([255, 40, 40])).astype(np.uint8)
    return out


def resize_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    out_h = max(1, int(round(h * width / w)))
    return cv2.resize(img, (width, out_h), interpolation=cv2.INTER_AREA)


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def contact_sheet(images: list[np.ndarray], labels: list[str], tile_width: int = 192, cols: int = 4) -> np.ndarray:
    tiles = [label(resize_width(img, tile_width), lab) for img, lab in zip(images, labels)]
    rows = []
    for start in range(0, len(tiles), cols):
        chunk = tiles[start : start + cols]
        while len(chunk) < cols:
            chunk.append(np.zeros_like(tiles[0]))
        rows.append(np.concatenate(chunk, axis=1))
    return np.concatenate(rows, axis=0)


def sample_indices(n: int, count: int = 8) -> list[int]:
    if n <= count:
        return list(range(n))
    return sorted({int(round(i * (n - 1) / (count - 1))) for i in range(count)})


def classify(mask_psnr: float, outside_psnr: float, outside_mae: float, temporal_ratio: float) -> tuple[str, str]:
    if not np.isfinite(mask_psnr) or not np.isfinite(outside_psnr):
        return "TECHNICAL_INVALID", "non-finite region metric"
    if outside_psnr < 35.0 or outside_mae > 5.0:
        return "TRIVIAL_BAD", "outside damage"
    if mask_psnr > 34.0:
        return "TOO_CLOSE", "task region too close to winner"
    if mask_psnr < 8.0:
        return "TRIVIAL_BAD", "task region too degraded"
    if temporal_ratio > 3.0:
        return "TRIVIAL_BAD", "temporal instability"
    if mask_psnr < 14.0:
        return "HARD_BUT_PLAUSIBLE", "strong local defect with preserved outside"
    return "MEDIUM_HARD_ELIGIBLE", "bounded local defect"


def main() -> int:
    args = parse_args()
    args.asset_root.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.manifest)[: args.limit]
    metrics_rows: list[dict] = []
    visual_rows: list[dict] = []
    review_pages: list[str] = []

    for row in rows:
        sample_id = row["sample_id"]
        condition = [read_rgb(p) for p in image_files(Path(row["condition_frame_dir"]))[: args.num_frames]]
        winner = [read_rgb(p) for p in image_files(Path(row["winner_frame_dir"]))[: args.num_frames]]
        masks = [read_gray(p) for p in image_files(Path(row["mask_frame_dir"]))[: args.num_frames]]
        for model in args.models:
            out_dir = args.output_root / model / "raw_frames" / sample_id
            outputs = [read_rgb(p) for p in image_files(out_dir)[: args.num_frames]] if out_dir.exists() else []
            status = "OK" if len(outputs) == args.num_frames else "FRAME_COUNT_FAIL"
            if status == "OK":
                full_psnr = float(np.mean([psnr(o, w) for o, w in zip(outputs, winner)]))
                mask_psnr = float(np.nanmean([region_psnr(o, w, m, True) for o, w, m in zip(outputs, winner, masks)]))
                outside_psnr = float(
                    np.nanmean([region_psnr(o, w, m, False) for o, w, m in zip(outputs, winner, masks)])
                )
                mask_mae = float(np.nanmean([region_mae(o, w, m, True) for o, w, m in zip(outputs, winner, masks)]))
                outside_mae = float(
                    np.nanmean([region_mae(o, w, m, False) for o, w, m in zip(outputs, winner, masks)])
                )
                output_temporal = temporal_absdiff(outputs)
                winner_temporal = temporal_absdiff(winner)
                temporal_ratio = output_temporal / max(winner_temporal, 1e-6)
                classification, reason = classify(mask_psnr, outside_psnr, outside_mae, temporal_ratio)
                images: list[np.ndarray] = []
                labels: list[str] = []
                for idx in sample_indices(args.num_frames, 8):
                    images.extend([condition[idx], overlay(condition[idx], masks[idx]), winner[idx], outputs[idx]])
                    labels.extend([f"f{idx:02d} cond", f"f{idx:02d} mask", f"f{idx:02d} winner", f"f{idx:02d} {model}"])
                page = contact_sheet(images, labels)
                page_path = args.asset_root / f"{model}_{sample_id}_review.jpg"
                cv2.imwrite(str(page_path), cv2.cvtColor(page, cv2.COLOR_RGB2BGR))
                review_pages.append(str(page_path))
            else:
                full_psnr = mask_psnr = outside_psnr = mask_mae = outside_mae = float("nan")
                output_temporal = winner_temporal = temporal_ratio = float("nan")
                classification, reason = "TECHNICAL_INVALID", f"expected {args.num_frames} frames, found {len(outputs)}"
                page_path = Path("")

            metrics_rows.append(
                {
                    "model": model,
                    "sample_id": sample_id,
                    "status": status,
                    "frames": len(outputs),
                    "full_psnr": full_psnr,
                    "mask_psnr": mask_psnr,
                    "outside_psnr": outside_psnr,
                    "mask_mae": mask_mae,
                    "outside_mae": outside_mae,
                    "output_temporal_absdiff": output_temporal,
                    "winner_temporal_absdiff": winner_temporal,
                    "temporal_ratio": temporal_ratio,
                    "classification_auto": classification,
                    "auto_reason": reason,
                    "output_dir": str(out_dir),
                }
            )
            visual_rows.append(
                {
                    "model": model,
                    "sample_id": sample_id,
                    "review_sheet": str(page_path) if page_path else "",
                    "review_method": "codex_open_review_sheet_required",
                    "classification_auto": classification,
                    "classification_final": "PENDING_CODEX_VISUAL_REVIEW",
                    "reason": reason,
                }
            )

    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)
    with args.visual_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(visual_rows[0].keys()))
        writer.writeheader()
        writer.writerows(visual_rows)

    summary = {
        "status": "PENDING_CODEX_VISUAL_REVIEW",
        "output_root": str(args.output_root),
        "models": args.models,
        "samples": [r["sample_id"] for r in rows],
        "metrics_csv": str(args.metrics_csv),
        "visual_csv": str(args.visual_csv),
        "review_pages": review_pages,
        "auto_counts": {},
    }
    for row in visual_rows:
        key = f"{row['model']}:{row['classification_auto']}"
        summary["auto_counts"][key] = summary["auto_counts"].get(key, 0) + 1
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    args.report_md.write_text(
        "# Exp30 New Generators Smoke2\n\n"
        "Status: `PENDING_CODEX_VISUAL_REVIEW`\n\n"
        "Generated raw/no-comp outputs for ProPainter and DiffuEraser no-PCM on two locked Smoke16 rows. "
        "Metrics and review pages are generated; final classification requires Codex visual readback.\n\n"
        f"Output root: `{args.output_root}`\n\n"
        f"Metrics: `{args.metrics_csv}`\n\n"
        f"Visual review CSV: `{args.visual_csv}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
