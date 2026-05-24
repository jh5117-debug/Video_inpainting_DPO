#!/usr/bin/env python3
"""Cheap candidate scoring for VideoDPO generated-loser calibration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SELECTION_CONFIG = Path("configs/generation/medium_hard_balanced_selection_v1.yaml")


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


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


def _crop_to_mask_bbox(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return a, b, mask
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return a[y0:y1, x0:x1], b[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from inference.metrics import compute_ssim
        return float(compute_ssim(a, b))
    except Exception:
        a_f = a.astype(np.float32)
        b_f = b.astype(np.float32)
        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2
        mu_a, mu_b = float(a_f.mean()), float(b_f.mean())
        var_a, var_b = float(a_f.var()), float(b_f.var())
        cov = float(((a_f - mu_a) * (b_f - mu_b)).mean())
        return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)))


def _clamp01(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))


def _norm_higher(value: float, lo: float, hi: float) -> float:
    return _clamp01((value - lo) / max(1e-8, hi - lo))


def _norm_lower(value: float, lo: float, hi: float) -> float:
    return 1.0 - _norm_higher(value, lo, hi)


def _bucket(quality_score: float, temporal_norm: float, pixel_norm: float, eligible_min: float, eligible_max: float) -> str:
    if quality_score < eligible_min:
        return "too_bad"
    if quality_score > eligible_max:
        return "too_good"
    if temporal_norm < 0.40:
        return "temporal_inconsistent"
    if pixel_norm < 0.40:
        return "texture_or_structure_shift"
    return "balanced_hard"


def _load_selection_config(path: str | Path = DEFAULT_SELECTION_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def compute_candidate_metrics(
    win_dir: str | Path,
    loser_dir: str | Path,
    mask_dir: str | Path,
    selection_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = selection_config or _load_selection_config()
    win_files = image_files(Path(win_dir))
    loser_files = image_files(Path(loser_dir))
    mask_files = image_files(Path(mask_dir))
    n = min(len(win_files), len(loser_files), len(mask_files))
    if n <= 0:
        return {
            "decode_ok": False,
            "frame_count": 0,
            "height": None,
            "width": None,
            "error_message": "empty win/loser/mask frame set",
        }

    psnrs: list[float] = []
    ssims: list[float] = []
    l1s: list[float] = []
    outside_means: list[float] = []
    outside_maxes: list[float] = []
    temporal_diffs: list[float] = []
    prev_win: np.ndarray | None = None
    prev_loser: np.ndarray | None = None
    height = width = None

    for idx in range(n):
        win = read_rgb(win_files[idx])
        loser = read_rgb(loser_files[idx])
        mask = read_gray(mask_files[idx])
        if loser.shape[:2] != win.shape[:2]:
            loser = cv2.resize(loser, (win.shape[1], win.shape[0]), interpolation=cv2.INTER_LINEAR)
        if mask.shape[:2] != win.shape[:2]:
            mask = cv2.resize(mask, (win.shape[1], win.shape[0]), interpolation=cv2.INTER_NEAREST)
        height, width = win.shape[:2]

        inside = mask > 0
        outside = ~inside
        win_crop, loser_crop, mask_crop = _crop_to_mask_bbox(win, loser, mask)
        inside_crop = mask_crop > 0
        if inside_crop.any():
            psnrs.append(_psnr(win_crop[inside_crop], loser_crop[inside_crop]))
            l1s.append(float(np.mean(np.abs(win_crop[inside_crop].astype(np.float32) - loser_crop[inside_crop].astype(np.float32))) / 255.0))
            ssim_win = win_crop.copy()
            ssim_loser = loser_crop.copy()
            ssim_loser[~inside_crop] = ssim_win[~inside_crop]
            ssims.append(_ssim(ssim_win, ssim_loser))
        else:
            psnrs.append(99.0)
            ssims.append(1.0)
            l1s.append(0.0)

        if outside.any():
            outside_diff = np.abs(win[outside].astype(np.float32) - loser[outside].astype(np.float32)) / 255.0
            outside_means.append(float(np.mean(outside_diff)))
            outside_maxes.append(float(np.max(outside_diff)))
        else:
            outside_means.append(0.0)
            outside_maxes.append(0.0)

        if prev_win is not None and prev_loser is not None:
            dw = win.astype(np.float32) - prev_win.astype(np.float32)
            dl = loser.astype(np.float32) - prev_loser.astype(np.float32)
            temporal_diffs.append(float(np.mean(np.abs(dw - dl)) / 255.0))
        prev_win = win
        prev_loser = loser

    inside_psnr = float(np.mean(psnrs))
    inside_ssim = float(np.mean(ssims))
    inside_l1 = float(np.mean(l1s))
    outside_mean = float(np.mean(outside_means))
    outside_max = float(np.max(outside_maxes))
    temporal_diff = float(np.mean(temporal_diffs)) if temporal_diffs else 0.0

    psnr_norm = _norm_higher(inside_psnr, 15.0, 35.0)
    ssim_norm = _norm_higher(inside_ssim, 0.20, 0.95)
    l1_norm = _norm_lower(inside_l1, 0.02, 0.35)
    temporal_norm = _norm_lower(temporal_diff, 0.01, 0.22)
    outside_norm = _norm_lower(outside_mean, 0.0, 0.05)
    weights = cfg["cheap_metrics"]["weights"]
    total_weight = sum(float(v) for v in weights.values())
    quality_score = (
        psnr_norm * float(weights["inside_mask_psnr"])
        + ssim_norm * float(weights["inside_mask_ssim"])
        + l1_norm * float(weights["inside_mask_l1"])
        + temporal_norm * float(weights["temporal_diff"])
        + outside_norm * float(weights["outside_mask_diff_mean"])
    ) / total_weight

    quality_cfg = cfg["quality_score"]
    pixel_norm = float(np.mean([psnr_norm, ssim_norm, l1_norm]))
    return {
        "decode_ok": True,
        "frame_count": n,
        "height": int(height) if height is not None else None,
        "width": int(width) if width is not None else None,
        "inside_mask_psnr": inside_psnr,
        "inside_mask_ssim": inside_ssim,
        "inside_mask_l1": inside_l1,
        "outside_mask_diff_mean": outside_mean,
        "outside_mask_diff_max": outside_max,
        "temporal_diff": temporal_diff,
        "metric_norms": {
            "inside_mask_psnr": psnr_norm,
            "inside_mask_ssim": ssim_norm,
            "inside_mask_l1": l1_norm,
            "temporal_diff": temporal_norm,
            "outside_mask_diff_mean": outside_norm,
        },
        "quality_score": float(_clamp01(quality_score)),
        "quality_band": [
            float(quality_cfg["eligible_min"]),
            float(quality_cfg["eligible_max"]),
        ],
        "quality_target": float(quality_cfg["target"]),
        "defect_bucket": _bucket(
            quality_score,
            temporal_norm,
            pixel_norm,
            float(quality_cfg["eligible_min"]),
            float(quality_cfg["eligible_max"]),
        ),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_manifest_rows(rows: list[dict[str, Any]], selection_config: dict[str, Any]) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        out = dict(row)
        if row.get("status") not in {None, "", "OK"}:
            out.setdefault("candidate_status", row.get("status"))
            out.setdefault("quality_score", 0.0)
            out.setdefault("defect_bucket", "failed")
            scored.append(out)
            continue
        try:
            raw_metrics = compute_candidate_metrics(row["win_video_path"], row["raw_loser_video_path"], row["mask_path"], selection_config)
            comp_path = row.get("comp_loser_video_path") or row.get("raw_loser_video_path")
            comp_metrics = compute_candidate_metrics(row["win_video_path"], comp_path, row["mask_path"], selection_config)
            out["raw_metrics"] = raw_metrics
            out["comp_metrics"] = comp_metrics
            out["quality_score"] = float(comp_metrics.get("quality_score", 0.0))
            out["defect_bucket"] = str(comp_metrics.get("defect_bucket", "unscored"))
            out["status"] = "OK" if comp_metrics.get("decode_ok") else "FAILED"
            out["error_message"] = "" if out["status"] == "OK" else str(comp_metrics.get("error_message", "decode failed"))
        except Exception as exc:
            out["raw_metrics"] = {}
            out["comp_metrics"] = {}
            out["quality_score"] = 0.0
            out["defect_bucket"] = "failed"
            out["status"] = "FAILED"
            out["error_message"] = str(exc)
        scored.append(out)
    return scored


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score VideoDPO loser candidates with cheap metrics.")
    parser.add_argument("--candidates_manifest", required=True)
    parser.add_argument("--selection_config", default=str(DEFAULT_SELECTION_CONFIG))
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--enable_lpips", action="store_true", help="Reserved for calibration subsets; disabled by default.")
    parser.add_argument("--enable_vbench", action="store_true", help="Reserved for calibration subsets; disabled by default.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.enable_lpips:
        raise SystemExit("[error] LPIPS scoring is intentionally off in this cheap scorer; add it only for calibration subsets.")
    if args.enable_vbench:
        raise SystemExit("[error] VBench scoring is intentionally off by default; use the dedicated VBench pipeline for calibration subsets.")
    cfg = _load_selection_config(args.selection_config)
    rows = score_manifest_rows(read_jsonl(Path(args.candidates_manifest)), cfg)
    write_jsonl(Path(args.output_manifest), rows)
    ok = sum(1 for r in rows if r.get("status") == "OK")
    print(json.dumps({"rows": len(rows), "ok": ok, "failed": len(rows) - ok, "output_manifest": args.output_manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
