#!/usr/bin/env python3
"""External validation analysis for Exp26 VideoPainter post-confirmation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
STEPS = ("step0", "step10", "step30", "step50")
VARIANTS = ("raw", "comp")
LOWER_IS_BETTER = {
    "whole_video_lpips",
    "ewarp_mask_region",
    "temporal_diff_delta_vs_gt",
    "outside_diff_mean",
    "outside_diff_max",
    "outside_region_diff_mean",
    "outside_region_diff_max",
}
PRIMARY_METRICS = [
    "whole_video_psnr",
    "whole_video_ssim",
    "whole_video_lpips",
    "strict_mask_pixel_psnr",
    "boundary_pixel_psnr",
    "boundary_psnr",
    "boundary_ssim",
    "outside_diff_mean",
    "outside_diff_max",
    "ewarp_mask_region",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
            f.write(json.dumps(row, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def read_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def read_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(path)
    if shape and arr.shape[:2] != shape:
        arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (arr > 127).astype(np.uint8)


def ensure_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def numeric(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def markdown_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append("nan" if math.isnan(value) else f"{value:.6f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def make_no_first_dir(src: Path, dst: Path) -> Path:
    files = frame_files(src)
    if len(files) < 49:
        raise ValueError(f"expected 49 frames in {src}, got {len(files)}")
    dst.mkdir(parents=True, exist_ok=True)
    for idx, fp in enumerate(files[1:49]):
        ensure_link_or_copy(fp, dst / f"{idx:05d}{fp.suffix.lower() or '.png'}")
    return dst


def build_pair_manifests(args: argparse.Namespace, frame_range: str) -> dict[str, Path]:
    rows = read_jsonl(args.manifest)
    pair_root = args.run_root / "metric_pair_manifests" / frame_range
    derived_root = args.run_root / "derived_no_first_frame" if frame_range == "no_first_frame" else None
    outputs: dict[str, Path] = {}
    for step in STEPS:
        for variant in VARIANTS:
            pair_rows: list[dict[str, Any]] = []
            for row in rows:
                sid = row["sample_id"]
                gt = Path(row["frame_dir"])
                mask = Path(row["mask_dir"])
                pred = args.run_root / step / "official_generation" / f"{variant}_frames" / sid
                if frame_range == "no_first_frame":
                    assert derived_root is not None
                    gt = make_no_first_dir(Path(row["frame_dir"]), derived_root / "gt" / sid)
                    mask = make_no_first_dir(Path(row["mask_dir"]), derived_root / "mask" / sid)
                    pred = make_no_first_dir(pred, derived_root / step / variant / sid)
                pair_rows.append(
                    {
                        "sample_id": sid,
                        "model_label": f"{step}_{variant}_{frame_range}",
                        "gt_video_path": str(gt),
                        "prediction_video_path": str(pred),
                        "mask_path": str(mask),
                        "scene_group": row.get("source_container", row.get("sample_id", "")),
                        "source_dataset": row.get("source_dataset", ""),
                    }
                )
            out = pair_root / f"{step}_{variant}_pairs.jsonl"
            write_jsonl(out, pair_rows)
            outputs[f"{step}_{variant}"] = out
    return outputs


def cmd_status(args: argparse.Namespace) -> int:
    rows = []
    for step in STEPS:
        summary_path = args.run_root / step / "official_generation/gate64_generation_summary.json"
        review_path = args.run_root / step / f"{step}_review/gate64_visual_review_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
        sample_count = len([p for p in (args.run_root / step / "official_generation/raw_frames").iterdir() if p.is_dir()]) if (args.run_root / step / "official_generation/raw_frames").exists() else 0
        rows.append(
            {
                "step": step,
                "generation_status": summary.get("status", "missing"),
                "ok": summary.get("ok", ""),
                "num_rows": summary.get("num_rows", ""),
                "sample_dirs": sample_count,
                "review_status": review.get("status", "missing"),
                "review_samples": review.get("num_samples", ""),
                "output_root": str(args.run_root / step),
            }
        )
    write_csv(args.report_dir / "exp26_external_validation_generation_status.csv", rows)
    status = "EXP26_EXTERNAL_GENERATION_COMPLETE" if all(r["generation_status"] == "passed" and int(r["sample_dirs"]) == 32 for r in rows) else "EXP26_EXTERNAL_GENERATION_INCOMPLETE"
    lines = ["# Exp26 External Validation Generation Status", "", f"Status: `{status}`", "", markdown_table(rows, ["step", "generation_status", "ok", "num_rows", "sample_dirs", "review_status", "review_samples"])]
    (args.report_dir / "exp26_external_validation_generation_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(status)
    return 0 if status == "EXP26_EXTERNAL_GENERATION_COMPLETE" else 2


def equality_fraction(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    eq = np.all(a == b, axis=2)
    if mask is not None:
        if mask.shape != eq.shape:
            mask = cv2.resize(mask, (eq.shape[1], eq.shape[0]), interpolation=cv2.INTER_NEAREST)
        sel = mask > 0
        if not np.any(sel):
            return float("nan")
        eq = eq[sel]
    return float(eq.mean())


def mae_region(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2)
    if mask is not None:
        if mask.shape != diff.shape:
            mask = cv2.resize(mask, (diff.shape[1], diff.shape[0]), interpolation=cv2.INTER_NEAREST)
        sel = mask > 0
        if not np.any(sel):
            return float("nan")
        diff = diff[sel]
    return float(np.mean(diff))


def cmd_leakage(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.manifest)
    out_rows: list[dict[str, Any]] = []
    for step in STEPS:
        for row in rows:
            sid = row["sample_id"]
            gt_files = frame_files(Path(row["frame_dir"]))
            mask_files = frame_files(Path(row["mask_dir"]))
            raw_files = frame_files(args.run_root / step / "official_generation/raw_frames" / sid)
            comp_files = frame_files(args.run_root / step / "official_generation/comp_frames" / sid)
            n = min(len(gt_files), len(mask_files), len(raw_files), len(comp_files), 49)
            if n < 49:
                out_rows.append({"step": step, "sample_id": sid, "status": "MISSING_FRAMES", "frame_count": n})
                continue
            raw_whole_eq = []
            raw_mask_eq = []
            raw_outside_eq = []
            comp_mask_eq = []
            raw_mask_mae = []
            raw_outside_mae = []
            frames1_48_raw_eq = []
            frame0_raw_eq = ""
            for idx in range(49):
                gt = read_rgb(gt_files[idx])
                mask = read_mask(mask_files[idx], shape=gt.shape[:2])
                inv_mask = (mask <= 0).astype(np.uint8)
                raw = read_rgb(raw_files[idx])
                comp = read_rgb(comp_files[idx])
                raw_whole_eq.append(equality_fraction(raw, gt))
                raw_mask_eq.append(equality_fraction(raw, gt, mask))
                raw_outside_eq.append(equality_fraction(raw, gt, inv_mask))
                comp_mask_eq.append(equality_fraction(comp, gt, mask))
                raw_mask_mae.append(mae_region(raw, gt, mask))
                raw_outside_mae.append(mae_region(raw, gt, inv_mask))
                if idx == 0:
                    frame0_raw_eq = equality_fraction(raw, gt)
                else:
                    frames1_48_raw_eq.append(equality_fraction(raw, gt))
            out_rows.append(
                {
                    "step": step,
                    "sample_id": sid,
                    "status": "OK",
                    "frame_count": n,
                    "raw_whole_eq_winner_mean": float(np.nanmean(raw_whole_eq)),
                    "raw_mask_eq_winner_mean": float(np.nanmean(raw_mask_eq)),
                    "raw_outside_eq_winner_mean": float(np.nanmean(raw_outside_eq)),
                    "comp_mask_eq_winner_mean": float(np.nanmean(comp_mask_eq)),
                    "raw_mask_mae_to_winner": float(np.nanmean(raw_mask_mae)),
                    "raw_outside_mae_to_winner": float(np.nanmean(raw_outside_mae)),
                    "frame0_raw_eq_winner": frame0_raw_eq,
                    "frames1_48_raw_eq_winner_mean": float(np.nanmean(frames1_48_raw_eq)),
                    "expected_comp_outside_winner_copy": True,
                    "non_expected_gt_leakage_flag": bool(float(np.nanmean(frames1_48_raw_eq)) > 0.999),
                }
            )
    write_csv(args.report_dir / "exp26_external_validation_leakage_audit.csv", out_rows)
    flagged = [r for r in out_rows if r.get("non_expected_gt_leakage_flag")]
    status = "VALIDATION_BLOCKED_GT_LEAKAGE" if flagged else "NO_UNEXPECTED_WINNER_LEAKAGE_DETECTED"
    (args.report_dir / "exp26_external_validation_leakage_audit.md").write_text(
        "\n".join(
            [
                "# Exp26 External Validation Leakage Audit",
                "",
                f"- status: `{status}`",
                f"- rows: `{len(out_rows)}`",
                f"- flagged: `{len(flagged)}`",
                "",
                "Comp is expected to copy winner outside the mask. Raw frames and comp mask region are checked separately; frame0 is official first-frame GT.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(status)
    return 0 if not flagged else 2


def run_metric_eval(args: argparse.Namespace, pair_manifest: Path, output_dir: Path, max_frames: int) -> None:
    cmd = [
        sys.executable,
        str(args.project_root / "tools/run_inpainting_metric_eval.py"),
        "--pair_manifest",
        str(pair_manifest),
        "--output_dir",
        str(output_dir),
        "--max_frames",
        str(max_frames),
        "--width",
        "720",
        "--height",
        "480",
        "--boundary_pixels",
        "4",
        "--device",
        args.device,
        "--compute_lpips",
        "--compute_ewarp",
        "--strict_missing",
    ]
    subprocess.run(cmd, check=True, cwd=str(args.project_root))


def cmd_metrics(args: argparse.Namespace) -> int:
    for frame_range, max_frames in (("all49", 49), ("no_first_frame", 48)):
        pairs = build_pair_manifests(args, frame_range)
        for key, pair_manifest in pairs.items():
            output_dir = args.run_root / "metrics" / frame_range / key
            summary = output_dir / "metrics/summary.csv"
            if args.skip_existing and summary.exists():
                continue
            run_metric_eval(args, pair_manifest, output_dir, max_frames=max_frames)
    print(args.run_root / "metrics")
    return 0


def per_sample_metrics(run_root: Path, frame_range: str, step: str, variant: str) -> list[dict[str, str]]:
    return read_csv(run_root / "metrics" / frame_range / f"{step}_{variant}" / "metrics/per_sample_metrics.csv")


def summary_metrics(run_root: Path, frame_range: str, step: str, variant: str) -> dict[str, str]:
    rows = read_csv(run_root / "metrics" / frame_range / f"{step}_{variant}" / "metrics/summary.csv")
    return rows[0] if rows else {}


def paired_stats(base_rows: list[dict[str, str]], cand_rows: list[dict[str, str]], metric: str, seed: int = 20260626) -> dict[str, Any]:
    base = {r["sample_id"]: numeric(r.get(metric)) for r in base_rows if r.get("status") == "ok"}
    cand = {r["sample_id"]: numeric(r.get(metric)) for r in cand_rows if r.get("status") == "ok"}
    ids = sorted(set(base) & set(cand))
    deltas = np.array([cand[i] - base[i] for i in ids], dtype=np.float64)
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return {"metric": metric, "n": 0}
    rng = np.random.default_rng(seed)
    boot = np.array([float(rng.choice(deltas, size=deltas.size, replace=True).mean()) for _ in range(10000)], dtype=np.float64)
    higher_good = metric not in LOWER_IS_BETTER
    wins = deltas > 0 if higher_good else deltas < 0
    prob = float((boot > 0).mean()) if higher_good else float((boot < 0).mean())
    loo = [float(np.delete(deltas, i).mean()) for i in range(deltas.size)] if deltas.size > 1 else []
    return {
        "metric": metric,
        "n": int(deltas.size),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "min_delta": float(deltas.min()),
        "max_delta": float(deltas.max()),
        "win_rate": float(wins.mean()),
        "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
        "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
        "probability_improved": prob,
        "leave_one_out_min": float(min(loo)) if loo else "",
        "leave_one_out_max": float(max(loo)) if loo else "",
    }


def first_metric(row: dict[str, str], key: str) -> float:
    return numeric(row.get(f"{key}_mean", row.get(key, "")))


def cmd_stats(args: argparse.Namespace) -> int:
    aggregate_rows: list[dict[str, Any]] = []
    for frame_range in ("all49", "no_first_frame"):
        for step in STEPS:
            for variant in VARIANTS:
                aggregate_rows.append({"frame_range": frame_range, "step": step, "variant": variant, **summary_metrics(args.run_root, frame_range, step, variant)})
    write_csv(args.report_dir / "exp26_external_validation_aggregate_metrics_all49.csv", [r for r in aggregate_rows if r["frame_range"] == "all49"])
    write_csv(args.report_dir / "exp26_external_validation_aggregate_metrics_no_first_frame.csv", [r for r in aggregate_rows if r["frame_range"] == "no_first_frame"])

    per_all: list[dict[str, Any]] = []
    for frame_range in ("all49", "no_first_frame"):
        for step in STEPS:
            for variant in VARIANTS:
                for row in per_sample_metrics(args.run_root, frame_range, step, variant):
                    per_all.append({"frame_range": frame_range, "step": step, "variant": variant, **row})
    write_csv(args.report_dir / "exp26_external_validation_per_video_metrics_all49.csv", [r for r in per_all if r["frame_range"] == "all49"])
    write_csv(args.report_dir / "exp26_external_validation_per_video_metrics_no_first_frame.csv", [r for r in per_all if r["frame_range"] == "no_first_frame"])

    paired_rows: list[dict[str, Any]] = []
    for frame_range in ("all49", "no_first_frame"):
        for variant in VARIANTS:
            base_rows = per_sample_metrics(args.run_root, frame_range, "step0", variant)
            for step in ("step10", "step30", "step50"):
                cand_rows = per_sample_metrics(args.run_root, frame_range, step, variant)
                for metric in PRIMARY_METRICS:
                    paired_rows.append({"frame_range": frame_range, "variant": variant, "comparison": f"{step}-step0", **paired_stats(base_rows, cand_rows, metric)})
    write_csv(args.report_dir / "exp26_external_validation_checkpoint_curve.csv", aggregate_rows)
    write_csv(args.report_dir / "exp26_external_validation_paired_deltas.csv", paired_rows)

    primary = {
        row["metric"]: row
        for row in paired_rows
        if row["frame_range"] == "no_first_frame" and row["variant"] == "comp" and row["comparison"] == "step50-step0"
    }
    step0_comp = next((r for r in aggregate_rows if r["frame_range"] == "no_first_frame" and r["step"] == "step0" and r["variant"] == "comp"), {})
    step50_comp = next((r for r in aggregate_rows if r["frame_range"] == "no_first_frame" and r["step"] == "step50" and r["variant"] == "comp"), {})
    whole_delta = first_metric(step50_comp, "whole_video_psnr") - first_metric(step0_comp, "whole_video_psnr")
    strict = primary.get("strict_mask_pixel_psnr", {})
    boundary = primary.get("boundary_pixel_psnr", {})
    lpips = primary.get("whole_video_lpips", {})
    ewarp = primary.get("ewarp_mask_region", {})
    reasons = []
    if not (strict.get("mean_delta", -999) > 0 and strict.get("probability_improved", 0) >= 0.90 and strict.get("win_rate", 0) >= 0.55):
        reasons.append("strict mask PSNR gate failed")
    if not (boundary.get("mean_delta", -999) > 0):
        reasons.append("boundary PSNR gate failed")
    if whole_delta < -0.02:
        reasons.append(f"whole comp PSNR dropped by {whole_delta:+.6f}")
    if lpips.get("mean_delta", 999) > 0.0005:
        reasons.append("LPIPS worsened beyond tolerance")
    if ewarp.get("mean_delta", 999) > 0.03:
        reasons.append("Ewarp worsened beyond tolerance")
    status = "EXP26_EXTERNAL_VALIDATION_METRIC_PASS" if not reasons else "EXP26_EXTERNAL_VALIDATION_NOT_CONFIRMED"
    summary = {
        "status": status,
        "reasons": reasons,
        "whole_comp_psnr_delta_no_first_frame": whole_delta,
        "primary_strict_mask_psnr": strict,
        "primary_boundary_psnr": boundary,
        "primary_lpips": lpips,
        "primary_ewarp": ewarp,
    }
    (args.report_dir / "exp26_external_validation_paired_statistics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = [
        {"metric": key, **value}
        for key, value in primary.items()
        if key in {"strict_mask_pixel_psnr", "boundary_pixel_psnr", "whole_video_lpips", "ewarp_mask_region"}
    ]
    (args.report_dir / "exp26_external_validation_metrics.md").write_text(
        "\n".join(
            [
                "# Exp26 External Validation Metrics",
                "",
                f"- status: `{status}`",
                f"- whole comp PSNR delta no-first-frame: `{whole_delta:+.6f}`",
                "",
                "## Primary Step50 - Step0 Comp Frame1-48",
                "",
                markdown_table(rows, ["metric", "n", "mean_delta", "median_delta", "win_rate", "bootstrap_ci_low", "bootstrap_ci_high", "probability_improved", "leave_one_out_min", "leave_one_out_max"]),
                "",
                "Step10/Step30 are trajectory diagnostics only and are not used for checkpoint reselection.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _candidate_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if str(path) in ("", "."):
            continue
        if path.exists():
            return path
    return None


def _candidate_openclip_dir(paths: list[Path]) -> Path | None:
    for path in paths:
        if str(path) in ("", "."):
            continue
        if path.is_dir() and (path / "open_clip_pytorch_model.bin").is_file():
            return path
    return None


def _load_frames_for_tc_vfid(path: Path, frame_range: str) -> list[np.ndarray]:
    files = frame_files(path)
    files = files[1:49] if frame_range == "no_first_frame" else files[:49]
    return [read_rgb(fp) for fp in files]


def _compute_tc_batched(tc_model: Any, frames_u8_rgb: list[np.ndarray], batch_size: int = 16) -> float:
    import torch
    import torch.nn.functional as F

    if len(frames_u8_rgb) < 2:
        return 1.0
    feats = []
    for idx in range(0, len(frames_u8_rgb), batch_size):
        batch = torch.stack([tc_model.preprocess(Image.fromarray(frame)) for frame in frames_u8_rgb[idx : idx + batch_size]], dim=0).to(tc_model.device)
        with torch.no_grad():
            feat = tc_model.model.encode_image(batch)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        feats.append(feat.detach().cpu())
    features = torch.cat(feats, dim=0)
    sims = F.cosine_similarity(features[:-1], features[1:], dim=-1)
    return float(sims.mean().item())


def cmd_tc_vfid(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(args.project_root))
    from inference import metrics as metric_backend

    i3d_path = _candidate_path(
        [
            Path(os.environ["I3D_MODEL_PATH"]) if os.environ.get("I3D_MODEL_PATH") else Path(""),
            args.project_root / "weights/i3d_rgb_imagenet.pt",
            Path("/home/hj/runtime_code/H20_Video_inpainting_DPO_exp26_shadowdev_6c3160a/weights/i3d_rgb_imagenet.pt"),
            Path("/mnt/nas/hj/H20_Video_inpainting_DPO/weights/i3d_rgb_imagenet.pt"),
            Path("/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/weights/i3d_rgb_imagenet.pt"),
            Path("/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp20_autoresearch/weights/i3d_rgb_imagenet.pt"),
        ]
    )
    clip_dir = _candidate_openclip_dir(
        [
            Path(os.environ["OPENCLIP_MODEL_DIR"]) if os.environ.get("OPENCLIP_MODEL_DIR") else Path(""),
            args.project_root / "weights/open_clip_vit_h14",
            Path("/home/hj/runtime_code/H20_Video_inpainting_DPO_exp26_shadowdev_6c3160a/weights/open_clip_vit_h14"),
            Path("/home/hj/.tmp/open_clip_vit_h14"),
            Path("/mnt/nas/hj/.tmp/open_clip_vit_h14"),
            Path("/mnt/workspace/hj/.tmp/open_clip_vit_h14"),
            Path("/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp20_autoresearch/weights/open_clip_vit_h14"),
        ]
    )
    if i3d_path is None:
        raise FileNotFoundError("No i3d_rgb_imagenet.pt found for existing VFID backend")
    tc_model = metric_backend.TemporalConsistencyMetric(device=args.device, model_path=str(clip_dir) if clip_dir else None)
    i3d_model = metric_backend.init_i3d_model(str(i3d_path), device=args.device)
    rows = read_jsonl(args.manifest)
    per_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for frame_range in ("all49", "no_first_frame"):
        for step in STEPS:
            for variant in VARIANTS:
                gt_acts: list[np.ndarray] = []
                pred_acts: list[np.ndarray] = []
                tc_vals: list[float] = []
                print(f"[tc-vfid] {frame_range} {step} {variant}", flush=True)
                for row in rows:
                    sid = row["sample_id"]
                    gt_frames = _load_frames_for_tc_vfid(Path(row["frame_dir"]), frame_range)
                    pred_frames = _load_frames_for_tc_vfid(args.run_root / step / "official_generation" / f"{variant}_frames" / sid, frame_range)
                    if len(gt_frames) != len(pred_frames) or not gt_frames:
                        per_rows.append({"frame_range": frame_range, "step": step, "variant": variant, "sample_id": sid, "status": "SKIPPED_FRAME_COUNT", "tc": ""})
                        continue
                    tc = _compute_tc_batched(tc_model, pred_frames)
                    gt_pil = [Image.fromarray(frame) for frame in gt_frames]
                    pred_pil = [Image.fromarray(frame) for frame in pred_frames]
                    gt_act, pred_act = metric_backend.calculate_i3d_activations(gt_pil, pred_pil, i3d_model, args.device)
                    gt_acts.append(gt_act)
                    pred_acts.append(pred_act)
                    tc_vals.append(tc)
                    per_rows.append({"frame_range": frame_range, "step": step, "variant": variant, "sample_id": sid, "status": "ok", "tc": tc})
                vfid = metric_backend.calculate_vfid(np.vstack(gt_acts), np.vstack(pred_acts)) if gt_acts and pred_acts else float("nan")
                summary_rows.append(
                    {
                        "frame_range": frame_range,
                        "step": step,
                        "variant": variant,
                        "rows": len(tc_vals),
                        "tc_mean": float(np.mean(tc_vals)) if tc_vals else float("nan"),
                        "tc_median": float(np.median(tc_vals)) if tc_vals else float("nan"),
                        "vfid": float(vfid),
                        "i3d_model_path": str(i3d_path),
                        "openclip_model_dir": str(clip_dir) if clip_dir else "auto_or_cache",
                    }
                )
    write_csv(args.report_dir / "exp26_external_validation_tc_vfid_per_video.csv", per_rows)
    write_csv(args.report_dir / "exp26_external_validation_tc_vfid_summary.csv", summary_rows)
    (args.report_dir / "exp26_external_validation_tc_vfid.md").write_text(
        "\n".join(
            [
                "# Exp26 External Validation TC/VFID",
                "",
                "TC and VFID are computed through the existing `inference.metrics.py` backend.",
                f"- I3D: `{i3d_path}`",
                f"- OpenCLIP: `{clip_dir if clip_dir else 'auto_or_cache'}`",
                "",
                markdown_table(summary_rows, ["frame_range", "step", "variant", "rows", "tc_mean", "tc_median", "vfid"]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.report_dir / "exp26_external_validation_tc_vfid_summary.csv")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Exp26 external validation analysis")
    parser.add_argument("command", choices=["status", "leakage", "metrics", "stats", "tc-vfid"])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "leakage":
        return cmd_leakage(args)
    if args.command == "metrics":
        return cmd_metrics(args)
    if args.command == "stats":
        return cmd_stats(args)
    if args.command == "tc-vfid":
        return cmd_tc_vfid(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
