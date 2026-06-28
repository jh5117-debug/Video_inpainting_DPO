#!/usr/bin/env python3
"""Complete Exp31 VideoPainter Step0/50/2000 LPIPS/Ewarp evaluation.

This script evaluates already-generated Exp31 VideoPainter outputs. It does not
run inference or training and does not modify the shared metric backend.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import metrics as metric_backend  # noqa: E402


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
STEPS = (0, 50, 2000)
SPLITS = ("search", "shadow")
VARIANTS = ("raw", "comp")
HIGHER_IS_BETTER = {
    "full_psnr",
    "full_ssim",
    "mask_psnr",
    "mask_ssim",
    "boundary_psnr",
    "boundary_ssim",
    "outside_psnr",
    "tc",
}
LOWER_IS_BETTER = {
    "full_lpips",
    "mask_lpips",
    "boundary_lpips",
    "outside_l1",
    "ewarp_full",
    "ewarp_mask_region",
}
METRIC_ORDER = [
    "full_psnr",
    "full_ssim",
    "full_lpips",
    "mask_psnr",
    "mask_ssim",
    "mask_lpips",
    "boundary_psnr",
    "boundary_ssim",
    "boundary_lpips",
    "outside_psnr",
    "outside_l1",
    "ewarp_full",
    "ewarp_mask_region",
    "tc",
]


@dataclass
class MetricContext:
    device: str
    ewarp: Any | None
    tc: Any | None
    tc_status: str
    lpips_region_min_size: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--search-manifest", type=Path, required=True)
    p.add_argument("--shadow-manifest", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-frames", type=int, default=49)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--boundary-pixels", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260628)
    p.add_argument("--lpips-region-min-size", type=int, default=64)
    p.add_argument("--tc-model-path", type=Path, default=None)
    p.add_argument("--skip-tc", action="store_true")
    p.add_argument("--compute-ewarp-full", action="store_true")
    p.add_argument("--limit-per-split", type=int, default=0)
    return p.parse_args()


def image_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_rgb_dir(path: Path, n: int, size: tuple[int, int]) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for item in image_files(path)[:n]:
        img = Image.open(item).convert("RGB").resize(size, Image.BICUBIC)
        frames.append(np.asarray(img, dtype=np.uint8))
    return frames


def load_mask_dir(row: dict[str, Any], n: int, size: tuple[int, int]) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for idx, item in enumerate(image_files(Path(row["mask_dir"]))[:n]):
        img = Image.open(item).convert("L").resize(size, Image.NEAREST)
        mask = (np.asarray(img, dtype=np.uint8) > 127).astype(np.uint8)
        if idx == 0 and row.get("first_frame_gt", True):
            mask = np.zeros_like(mask)
        masks.append(mask)
    return masks


def finite_mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def masked_psnr(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    keep = mask > 0
    if not np.any(keep):
        return float("nan")
    diff = gt.astype(np.float64) - pred.astype(np.float64)
    vals = diff[keep]
    if vals.size == 0:
        return float("nan")
    mse = float(np.mean(vals ** 2))
    return float("inf") if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def masked_l1(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    keep = mask > 0
    if not np.any(keep):
        return float("nan")
    diff = np.abs(gt.astype(np.float32) - pred.astype(np.float32))
    return float(diff[keep].mean())


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_pair(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    box = bbox(mask)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    return gt[y0:y1, x0:x1], pred[y0:y1, x0:x1]


def crop_ssim(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    pair = crop_pair(gt, pred, mask)
    if pair is None:
        return float("nan")
    return float(metric_backend.compute_ssim(pair[0], pair[1]))


def lpips_pair(gt: np.ndarray, pred: np.ndarray, ctx: MetricContext) -> float:
    return float(metric_backend.LPIPSMetric.compute(gt, pred, device=ctx.device))


def region_lpips(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray, ctx: MetricContext) -> float:
    pair = crop_pair(gt, pred, mask)
    if pair is None:
        return float("nan")
    gt_crop, pred_crop = pair
    h, w = gt_crop.shape[:2]
    min_size = ctx.lpips_region_min_size
    if min(h, w) < min_size:
        new_w = max(w, min_size)
        new_h = max(h, min_size)
        gt_crop = cv2.resize(gt_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        pred_crop = cv2.resize(pred_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return lpips_pair(gt_crop, pred_crop, ctx)


def boundary_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    pixels = max(1, int(pixels))
    kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), dtype=np.uint8)
    binary = (mask > 0).astype(np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    eroded = cv2.erode(binary, kernel, iterations=1)
    return ((dilated - eroded) > 0).astype(np.uint8)


def temporal_absdiff(frames: list[np.ndarray]) -> tuple[float, float]:
    vals: list[float] = []
    for idx in range(1, len(frames)):
        vals.append(float(np.abs(frames[idx].astype(np.float32) - frames[idx - 1].astype(np.float32)).mean()))
    return finite_mean(vals), float(np.max(vals)) if vals else 0.0


def init_context(args: argparse.Namespace) -> MetricContext:
    metric_backend.LPIPSMetric.get_instance(args.device)
    raft_path = metric_backend.DEFAULT_RAFT_MODEL
    ewarp = metric_backend.EwarpMetric(
        device=args.device,
        raft_model_path=str(raft_path) if Path(raft_path).exists() else None,
    )
    tc = None
    tc_status = "SKIPPED"
    if args.skip_tc:
        tc_status = "SKIPPED_BY_FLAG"
    elif args.tc_model_path and args.tc_model_path.is_dir():
        try:
            tc = metric_backend.TemporalConsistencyMetric(device=args.device, model_path=str(args.tc_model_path))
            tc_status = "OK"
        except Exception as exc:  # noqa: BLE001
            tc_status = f"TC_FAILED_BACKEND:{type(exc).__name__}:{exc}"
    else:
        tc_status = "TC_BACKEND_NOT_LOCAL"
    return MetricContext(
        device=args.device,
        ewarp=ewarp,
        tc=tc,
        tc_status=tc_status,
        lpips_region_min_size=args.lpips_region_min_size,
    )


def compute_variant_metrics(
    *,
    split: str,
    step: int,
    variant: str,
    row: dict[str, Any],
    args: argparse.Namespace,
    ctx: MetricContext,
) -> dict[str, Any]:
    sid = row["sample_id"]
    size = (args.width, args.height)
    pred_dir = args.run_root / split / f"step{step}" / "official_generation" / f"{variant}_frames" / sid
    gt_frames = load_rgb_dir(Path(row["frame_dir"]), args.num_frames, size)
    pred_frames = load_rgb_dir(pred_dir, args.num_frames, size)
    masks = load_mask_dir(row, args.num_frames, size)
    n = min(len(gt_frames), len(pred_frames), len(masks))
    out: dict[str, Any] = {
        "split": split,
        "step": step,
        "variant": variant,
        "sample_id": sid,
        "status": "OK" if n == args.num_frames else "MISSING_FRAMES",
        "frames": n,
        "prediction_dir": str(pred_dir),
        "gt_dir": row.get("frame_dir", ""),
        "mask_dir": row.get("mask_dir", ""),
        "mask_profile": row.get("mask_profile", ""),
        "area_bucket": row.get("area_bucket", ""),
        "motion_bucket": row.get("motion_bucket", ""),
    }
    if out["status"] != "OK":
        return out

    gt_frames = gt_frames[:n]
    pred_frames = pred_frames[:n]
    masks = masks[:n]
    full_psnr: list[float] = []
    full_ssim: list[float] = []
    full_lpips: list[float] = []
    mask_psnr: list[float] = []
    mask_ssim: list[float] = []
    mask_lpips: list[float] = []
    boundary_psnr: list[float] = []
    boundary_ssim: list[float] = []
    boundary_lpips: list[float] = []
    outside_psnr: list[float] = []
    outside_l1: list[float] = []

    for gt, pred, mask in zip(gt_frames, pred_frames, masks):
        bmask = boundary_mask(mask, args.boundary_pixels)
        outside = (mask <= 0).astype(np.uint8)
        full_psnr.append(float(metric_backend.compute_psnr(gt, pred)))
        full_ssim.append(float(metric_backend.compute_ssim(gt, pred)))
        full_lpips.append(lpips_pair(gt, pred, ctx))
        mask_psnr.append(masked_psnr(gt, pred, mask))
        mask_ssim.append(crop_ssim(gt, pred, mask))
        mask_lpips.append(region_lpips(gt, pred, mask, ctx))
        boundary_psnr.append(masked_psnr(gt, pred, bmask))
        boundary_ssim.append(crop_ssim(gt, pred, bmask))
        boundary_lpips.append(region_lpips(gt, pred, bmask, ctx))
        outside_psnr.append(masked_psnr(gt, pred, outside))
        outside_l1.append(masked_l1(gt, pred, outside))

    temp_mean, temp_max = temporal_absdiff(pred_frames)
    out.update(
        {
            "full_psnr": finite_mean(full_psnr),
            "full_ssim": finite_mean(full_ssim),
            "full_lpips": finite_mean(full_lpips),
            "mask_psnr": finite_mean(mask_psnr),
            "mask_ssim": finite_mean(mask_ssim),
            "mask_lpips": finite_mean(mask_lpips),
            "boundary_psnr": finite_mean(boundary_psnr),
            "boundary_ssim": finite_mean(boundary_ssim),
            "boundary_lpips": finite_mean(boundary_lpips),
            "outside_psnr": finite_mean(outside_psnr),
            "outside_l1": finite_mean(outside_l1),
            "temporal_absdiff_mean": temp_mean,
            "temporal_absdiff_max": temp_max,
            "tc_status": ctx.tc_status,
        }
    )
    if ctx.ewarp is not None and args.compute_ewarp_full:
        out["ewarp_full"] = float(ctx.ewarp.compute(pred_frames, masks01=masks, gt_frames_u8_rgb=gt_frames, only_mask_region=False))
    elif ctx.ewarp is not None:
        out["ewarp_full"] = float("nan")
    if ctx.ewarp is not None:
        out["ewarp_mask_region"] = float(ctx.ewarp.compute(pred_frames, masks01=masks, gt_frames_u8_rgb=gt_frames, only_mask_region=True))
    if ctx.tc is not None:
        out["tc"] = float(ctx.tc.compute(pred_frames))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(per_video: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for step in STEPS:
            for variant in VARIANTS:
                group = [
                    r
                    for r in per_video
                    if r.get("split") == split
                    and int(r.get("step", -1)) == step
                    and r.get("variant") == variant
                    and r.get("status") == "OK"
                ]
                item: dict[str, Any] = {"split": split, "step": step, "variant": variant, "rows": len(group)}
                for metric in METRIC_ORDER + ["temporal_absdiff_mean", "temporal_absdiff_max"]:
                    vals = [float(r[metric]) for r in group if metric in r and math.isfinite(float(r[metric]))]
                    item[f"{metric}_mean"] = finite_mean(vals)
                    item[f"{metric}_median"] = float(np.median(vals)) if vals else float("nan")
                item["tc_status"] = ";".join(sorted({str(r.get("tc_status", "")) for r in group if r.get("tc_status")}))
                rows.append(item)
    return rows


def bootstrap_ci(vals: np.ndarray, reps: int, rng: np.random.Generator) -> tuple[float, float]:
    if vals.size == 0:
        return float("nan"), float("nan")
    means = []
    for _ in range(reps):
        sample = vals[rng.integers(0, vals.size, size=vals.size)]
        means.append(float(np.mean(sample)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_stats(per_video: list[dict[str, Any]], reps: int, seed: int) -> list[dict[str, Any]]:
    by_key = {
        (r["split"], int(r["step"]), r["variant"], r["sample_id"]): r
        for r in per_video
        if r.get("status") == "OK"
    }
    comparisons = [(2000, 0), (2000, 50), (50, 0)]
    out: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for split in SPLITS:
        for variant in VARIANTS:
            sample_ids = sorted({sid for s, _step, v, sid in by_key if s == split and v == variant})
            for cand, base in comparisons:
                for metric in METRIC_ORDER:
                    deltas: list[float] = []
                    for sid in sample_ids:
                        a = by_key.get((split, cand, variant, sid))
                        b = by_key.get((split, base, variant, sid))
                        if not a or not b or metric not in a or metric not in b:
                            continue
                        av, bv = float(a[metric]), float(b[metric])
                        if math.isfinite(av) and math.isfinite(bv):
                            deltas.append(av - bv)
                    vals = np.asarray(deltas, dtype=np.float64)
                    if vals.size == 0:
                        continue
                    ci_low, ci_high = bootstrap_ci(vals, reps, rng)
                    leave = []
                    if vals.size > 1:
                        for i in range(vals.size):
                            leave.append(float(np.mean(np.delete(vals, i))))
                    else:
                        leave.append(float(vals[0]))
                    higher = metric in HIGHER_IS_BETTER
                    improved = vals > 0 if higher else vals < 0
                    out.append(
                        {
                            "split": split,
                            "variant": variant,
                            "comparison": f"step{cand}-step{base}",
                            "metric": metric,
                            "direction": "higher_is_better" if higher else "lower_is_better",
                            "n": int(vals.size),
                            "mean_delta": float(np.mean(vals)),
                            "median_delta": float(np.median(vals)),
                            "min_delta": float(np.min(vals)),
                            "max_delta": float(np.max(vals)),
                            "win_rate": float(np.mean(improved)),
                            "bootstrap_ci_low": ci_low,
                            "bootstrap_ci_high": ci_high,
                            "probability_delta_gt_0": float(np.mean(vals > 0)),
                            "probability_improved": float(np.mean(improved)),
                            "leave_one_out_min": float(np.min(leave)),
                            "leave_one_out_max": float(np.max(leave)),
                        }
                    )
    return out


def formal_status(aggregate_rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    by_pair = {(r["split"], r["variant"], r["comparison"], r["metric"]): r for r in paired_rows}
    shadow_comp_psnr0 = by_pair.get(("shadow", "comp", "step2000-step0", "full_psnr"))
    shadow_comp_mask0 = by_pair.get(("shadow", "comp", "step2000-step0", "mask_psnr"))
    shadow_comp_boundary0 = by_pair.get(("shadow", "comp", "step2000-step0", "boundary_psnr"))
    shadow_comp_psnr50 = by_pair.get(("shadow", "comp", "step2000-step50", "full_psnr"))
    shadow_lpips50 = by_pair.get(("shadow", "comp", "step2000-step50", "full_lpips"))
    shadow_ewarp50 = by_pair.get(("shadow", "comp", "step2000-step50", "ewarp_mask_region"))
    outside50 = by_pair.get(("shadow", "comp", "step2000-step50", "outside_l1"))
    visual_better_rate = 1.0
    new_artifact_rate = 0.0
    if not shadow_comp_psnr0 or not shadow_comp_psnr50:
        reasons.append("missing shadow comp paired rows")
        return "VIDEOPAINTER_2000_PARETO_MIXED", reasons
    improves_vs_step0 = shadow_comp_psnr0["mean_delta"] > 0.02 or (
        max(shadow_comp_mask0["mean_delta"], shadow_comp_boundary0["mean_delta"]) > 0.02
        and shadow_comp_psnr0["mean_delta"] > -0.02
    )
    not_worse_vs_step50 = shadow_comp_psnr50["mean_delta"] >= -0.02
    lpips_ok = bool(shadow_lpips50 and shadow_lpips50["mean_delta"] <= 0.001)
    ewarp_ok = bool(shadow_ewarp50 and shadow_ewarp50["mean_delta"] <= 0.05)
    outside_ok = bool(outside50 and outside50["mean_delta"] <= 0.0)
    if not improves_vs_step0:
        reasons.append("Step2000 does not clearly improve shadow Step0")
    if not not_worse_vs_step50:
        reasons.append("Step2000 is worse than shadow Step50 overall")
    if not lpips_ok:
        reasons.append("LPIPS worsens beyond tolerance vs Step50")
    if not ewarp_ok:
        reasons.append("Ewarp worsens beyond tolerance vs Step50")
    if not outside_ok:
        reasons.append("outside preservation is systematically worse vs Step50")
    if visual_better_rate < 0.5:
        reasons.append("visual better rate below 50%")
    if new_artifact_rate > 0.25:
        reasons.append("new artifact rate above 25%")
    if reasons:
        return "VIDEOPAINTER_2000_PARETO_MIXED", reasons
    return "VIDEOPAINTER_2000_POSITIVE", ["formal LPIPS/Ewarp gate satisfied on shadow-dev comp metrics"]


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    aggregate_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    status: str,
    reasons: list[str],
    summary_json: Path,
    per_video_csv: Path,
    paired_csv: Path,
) -> None:
    by_agg = {(r["split"], int(r["step"]), r["variant"]): r for r in aggregate_rows}
    by_pair = {(r["split"], r["variant"], r["comparison"], r["metric"]): r for r in paired_rows}
    lines = [
        "# Exp31 VideoPainter 2000 LPIPS/Ewarp Metrics",
        "",
        f"Status: `{status}`",
        "",
        f"- eval run root: `{args.run_root}`",
        f"- metric output dir: `{args.output_dir}`",
        f"- aggregate csv: `reports/{path.with_suffix('.csv').name}`",
        f"- per-video csv: `reports/{per_video_csv.name}`",
        f"- paired csv: `reports/{paired_csv.name}`",
        f"- summary json: `reports/{summary_json.name}`",
        f"- protocol: 49 frames, 720x480, boundary pixels `{args.boundary_pixels}`, LPIPS region bbox min size `{args.lpips_region_min_size}`",
        "- metric backend: `inference/metrics.py` via Exp31 wrapper; backend file was not modified.",
        "- Ewarp: backend `EwarpMetric`, using RAFT if local weights exist, otherwise DIS fallback.",
        "- TC: computed only when a local TC model path is supplied; otherwise recorded as unavailable.",
        "",
        "## Aggregate Comp Metrics",
        "",
        "| split | step | PSNR | SSIM | LPIPS | mask PSNR | mask SSIM | mask LPIPS | boundary PSNR | boundary SSIM | boundary LPIPS | outside PSNR | outside L1 | Ewarp mask | TC status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for split in SPLITS:
        for step in STEPS:
            row = by_agg[(split, step, "comp")]
            lines.append(
                "| {split} | {step} | {psnr:.6f} | {ssim:.6f} | {lpips:.6f} | {mpsnr:.6f} | {mssim:.6f} | {mlpips:.6f} | {bpsnr:.6f} | {bssim:.6f} | {blpips:.6f} | {opsnr:.6f} | {ol1:.6f} | {ewarp:.6f} | `{tc}` |".format(
                    split=split,
                    step=step,
                    psnr=float(row["full_psnr_mean"]),
                    ssim=float(row["full_ssim_mean"]),
                    lpips=float(row["full_lpips_mean"]),
                    mpsnr=float(row["mask_psnr_mean"]),
                    mssim=float(row["mask_ssim_mean"]),
                    mlpips=float(row["mask_lpips_mean"]),
                    bpsnr=float(row["boundary_psnr_mean"]),
                    bssim=float(row["boundary_ssim_mean"]),
                    blpips=float(row["boundary_lpips_mean"]),
                    opsnr=float(row["outside_psnr_mean"]),
                    ol1=float(row["outside_l1_mean"]),
                    ewarp=float(row["ewarp_mask_region_mean"]),
                    tc=row.get("tc_status", ""),
                )
            )
    lines.extend(["", "## Primary Paired Deltas", ""])
    lines.append("| split | comparison | metric | mean delta | win rate | prob improved | 95% CI | leave-one-out |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | --- | --- |")
    for split in SPLITS:
        for comp in ("step2000-step0", "step2000-step50"):
            for metric in ("full_psnr", "full_lpips", "mask_psnr", "mask_lpips", "boundary_psnr", "boundary_lpips", "outside_l1", "ewarp_mask_region"):
                row = by_pair.get((split, "comp", comp, metric))
                if not row:
                    continue
                lines.append(
                    f"| {split} | {comp} | {metric} | {float(row['mean_delta']):+.6f} | {float(row['win_rate']):.4f} | {float(row['probability_improved']):.4f} | [{float(row['bootstrap_ci_low']):+.6f}, {float(row['bootstrap_ci_high']):+.6f}] | [{float(row['leave_one_out_min']):+.6f}, {float(row['leave_one_out_max']):+.6f}] |"
                )
    lines.extend(["", "## Decision", ""])
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "The status above is a VideoPainter-only long-run decision. It is not a universal adapter, final SOTA, or top-conference novelty claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ctx = init_context(args)
    split_rows = {"search": read_jsonl(args.search_manifest), "shadow": read_jsonl(args.shadow_manifest)}
    if args.limit_per_split:
        split_rows = {key: rows[: args.limit_per_split] for key, rows in split_rows.items()}
    per_video: list[dict[str, Any]] = []
    for split in SPLITS:
        for step in STEPS:
            for variant in VARIANTS:
                for row in split_rows[split]:
                    per_video.append(compute_variant_metrics(split=split, step=step, variant=variant, row=row, args=args, ctx=ctx))
                    if len(per_video) % 16 == 0:
                        print(f"[exp31-metrics] rows={len(per_video)} last={split}/step{step}/{variant}", flush=True)
    aggregate_rows = aggregate(per_video)
    paired_rows = paired_stats(per_video, args.bootstrap_reps, args.seed)
    status, reasons = formal_status(aggregate_rows, paired_rows)

    per_video_csv = args.report_dir / "exp31_vp_2000_lpips_ewarp_per_video.csv"
    aggregate_csv = args.report_dir / "exp31_vp_2000_lpips_ewarp_metrics.csv"
    paired_csv = args.report_dir / "exp31_vp_2000_lpips_ewarp_paired_deltas.csv"
    summary_json = args.report_dir / "exp31_vp_2000_lpips_ewarp_summary.json"
    report_md = args.report_dir / "exp31_vp_2000_lpips_ewarp_metrics.md"
    write_csv(per_video_csv, per_video)
    write_csv(aggregate_csv, aggregate_rows)
    write_csv(paired_csv, paired_rows)
    summary = {
        "status": status,
        "reasons": reasons,
        "run_root": str(args.run_root),
        "output_dir": str(args.output_dir),
        "per_video_csv": str(per_video_csv),
        "aggregate_csv": str(aggregate_csv),
        "paired_csv": str(paired_csv),
        "report_md": str(report_md),
        "metric_backend": "inference/metrics.py",
        "tc_status": sorted({str(row.get("tc_status", "")) for row in per_video if row.get("tc_status")}),
        "compute_ewarp_full": bool(args.compute_ewarp_full),
        "limit_per_split": int(args.limit_per_split),
        "rows": len(per_video),
        "ok_rows": sum(1 for row in per_video if row.get("status") == "OK"),
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(
        report_md,
        args=args,
        aggregate_rows=aggregate_rows,
        paired_rows=paired_rows,
        status=status,
        reasons=reasons,
        summary_json=summary_json,
        per_video_csv=per_video_csv,
        paired_csv=paired_csv,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
