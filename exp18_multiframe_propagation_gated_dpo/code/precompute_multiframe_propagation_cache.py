#!/usr/bin/env python3
"""Precompute Exp18 multi-frame propagation cache.

Version A (default) uses non-oracle optical-flow propagation and multi-source
agreement. Version B additionally writes oracle confidence maps for diagnostic
upper-bound variants, but those maps are not used by the non-oracle method.
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

from training.dpo.dataset.generated_loser_manifest_dataset import list_image_frames  # noqa: E402


@dataclass
class SampleStats:
    sample_id: str
    status: str
    num_frames: int = 0
    propagation_coverage: float | None = None
    average_confidence: float | None = None
    avg_num_sources_used: float | None = None
    propagated_region_psnr: float | None = None
    full_mask_prop_psnr: float | None = None
    failure_reason: str = ""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_rgb_sequence(path: str | Path, nframes: int, size: tuple[int, int] | None) -> np.ndarray:
    files = list_image_frames(path)
    if len(files) < nframes:
        raise ValueError(f"expected at least {nframes} frames under {path}, found {len(files)}")
    frames = []
    for file in files[:nframes]:
        img = Image.open(file).convert("RGB")
        if size is not None:
            img = img.resize(size, Image.BILINEAR)
        frames.append(np.asarray(img, dtype=np.uint8))
    return np.stack(frames, axis=0)


def load_mask_sequence(path: str | Path, nframes: int, size: tuple[int, int] | None) -> np.ndarray:
    files = list_image_frames(path)
    if len(files) < nframes:
        raise ValueError(f"expected at least {nframes} masks under {path}, found {len(files)}")
    masks = []
    for file in files[:nframes]:
        img = Image.open(file).convert("L")
        if size is not None:
            img = img.resize(size, Image.NEAREST)
        masks.append((np.asarray(img, dtype=np.uint8) > 127).astype(np.uint8))
    return np.stack(masks, axis=0)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_rgb_frames(root: Path, frames: np.ndarray) -> None:
    ensure_dir(root)
    for idx, frame in enumerate(frames):
        Image.fromarray(frame.astype(np.uint8), mode="RGB").save(root / f"{idx:05d}.png")


def save_gray_frames(root: Path, frames: np.ndarray) -> None:
    ensure_dir(root)
    for idx, frame in enumerate(frames):
        arr = np.clip(frame, 0, 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(root / f"{idx:05d}.png")


def psnr_from_mse(mse: float) -> float:
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def calc_flow(gray_a: np.ndarray, gray_b: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        gray_a,
        gray_b,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def remap_array(arr: np.ndarray, flow_t_to_s: np.ndarray, interpolation: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = flow_t_to_s.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + flow_t_to_s[..., 0].astype(np.float32)
    map_y = grid_y + flow_t_to_s[..., 1].astype(np.float32)
    valid = (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
    warped = cv2.remap(arr, map_x, map_y, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, valid


def propagate_one_video(
    frames: np.ndarray,
    masks: np.ndarray,
    source_window: int,
    tau_conf: float,
    write_oracle: bool,
    alpha_oracle: float,
) -> dict[str, np.ndarray | float]:
    n, h, w = frames.shape[:3]
    grays = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames], axis=0)
    propagated = frames.copy()
    confidence = np.zeros((n, h, w), dtype=np.float32)
    source_count = np.zeros((n, h, w), dtype=np.float32)
    source_index = np.full((n, h, w), 255, dtype=np.uint8)

    for t in range(n):
        target_hole = masks[t].astype(bool)
        if not target_hole.any():
            continue
        warped_values = []
        warped_weights = []
        warped_source_ids = []
        source_ids = [s for s in range(max(0, t - source_window), min(n, t + source_window + 1)) if s != t]
        for s in source_ids:
            # Backward map target -> source.
            flow_t_to_s = calc_flow(grays[t], grays[s])
            warped_rgb, in_bounds = remap_array(frames[s], flow_t_to_s, cv2.INTER_LINEAR)
            source_known = (1 - masks[s]).astype(np.float32)
            warped_known, _ = remap_array(source_known, flow_t_to_s, cv2.INTER_NEAREST)

            # Forward-backward consistency: sample source->target at the mapped
            # source location and check f_t2s + f_s2t ~= 0.
            flow_s_to_t = calc_flow(grays[s], grays[t])
            warped_fwd_x, _ = remap_array(flow_s_to_t[..., 0], flow_t_to_s, cv2.INTER_LINEAR)
            warped_fwd_y, _ = remap_array(flow_s_to_t[..., 1], flow_t_to_s, cv2.INTER_LINEAR)
            fb_err = np.sqrt((flow_t_to_s[..., 0] + warped_fwd_x) ** 2 + (flow_t_to_s[..., 1] + warped_fwd_y) ** 2)
            consistency = np.exp(-fb_err / 3.0).astype(np.float32)

            valid = target_hole & in_bounds & (warped_known > 0.5)
            weight = (valid.astype(np.float32) * consistency).clip(0.0, 1.0)
            warped_values.append(warped_rgb.astype(np.float32))
            warped_weights.append(weight)
            warped_source_ids.append(np.full((h, w), s, dtype=np.uint8))

        if not warped_values:
            continue
        values = np.stack(warped_values, axis=0)
        weights = np.stack(warped_weights, axis=0)
        valid_count = (weights > 1e-4).sum(axis=0).astype(np.float32)
        weight_sum = weights.sum(axis=0)
        has_valid = target_hole & (weight_sum > 1e-4)
        if not has_valid.any():
            continue

        weighted = (values * weights[..., None]).sum(axis=0) / np.maximum(weight_sum[..., None], 1e-6)
        median = np.median(np.where(weights[..., None] > 1e-4, values, np.nan), axis=0)
        median = np.where(np.isfinite(median), median, weighted)
        valid_stack = (weights > 1e-4).astype(np.float32)
        denom = np.maximum(valid_stack.sum(axis=0), 1.0)
        mean_rgb = (values * valid_stack[..., None]).sum(axis=0) / denom[..., None]
        var_rgb = (((values - mean_rgb[None, ...]) ** 2) * valid_stack[..., None]).sum(axis=0) / denom[..., None]
        rgb_std = np.sqrt(np.maximum(var_rgb, 0.0)).mean(axis=2)
        rgb_std = np.where(valid_count >= 2, rgb_std, 255.0)
        agreement = np.exp(-rgb_std / 25.0).astype(np.float32)
        count_score = np.clip(valid_count / 2.0, 0.0, 1.0)
        conf = (weight_sum / np.maximum(valid_count, 1.0)) * agreement * count_score
        conf = np.where(has_valid, conf, 0.0).clip(0.0, 1.0)

        use_median = valid_count >= 2
        chosen = np.where(use_median[..., None], median, weighted)
        propagated[t][has_valid] = np.clip(chosen[has_valid], 0, 255).astype(np.uint8)
        confidence[t] = conf
        source_count[t] = valid_count
        best_source = np.argmax(weights, axis=0).astype(np.int32)
        src_ids = np.array(source_ids, dtype=np.uint8)
        source_index[t][has_valid] = src_ids[best_source[has_valid]]

    reliable = ((confidence > tau_conf) & (masks.astype(bool))).astype(np.uint8)
    generate = ((confidence <= tau_conf) & (masks.astype(bool))).astype(np.uint8)

    oracle_conf = None
    if write_oracle:
        err = np.abs(propagated.astype(np.float32) - frames.astype(np.float32)).mean(axis=3)
        norm = np.clip(err / max(float(err[masks.astype(bool)].max()) if masks.astype(bool).any() else float(err.max()), 1.0), 0, 1)
        oracle_conf = np.exp(-alpha_oracle * norm).astype(np.float32) * masks.astype(np.float32)

    hole = masks.astype(bool)
    prop_region = reliable.astype(bool)
    if prop_region.any():
        mse_prop = float(((propagated.astype(np.float32) - frames.astype(np.float32)) ** 2)[prop_region].mean())
        psnr_prop = psnr_from_mse(mse_prop)
    else:
        psnr_prop = float("nan")
    if hole.any():
        mse_full = float(((propagated.astype(np.float32) - frames.astype(np.float32)) ** 2)[hole].mean())
        psnr_full = psnr_from_mse(mse_full)
        coverage = float(reliable.sum() / max(hole.sum(), 1))
        avg_conf = float(confidence[hole].mean())
        avg_sources = float(source_count[hole].mean())
    else:
        psnr_full = float("nan")
        coverage = 0.0
        avg_conf = 0.0
        avg_sources = 0.0

    result: dict[str, np.ndarray | float] = {
        "propagated": propagated,
        "confidence": confidence,
        "source_count": source_count,
        "source_index": source_index,
        "reliable": reliable,
        "generate": generate,
        "propagation_coverage": coverage,
        "average_confidence": avg_conf,
        "avg_num_sources_used": avg_sources,
        "propagated_region_psnr": psnr_prop,
        "full_mask_prop_psnr": psnr_full,
    }
    if oracle_conf is not None:
        result["oracle_confidence"] = oracle_conf
    return result


def process_row(row: dict[str, Any], args: argparse.Namespace, output_root: Path) -> tuple[dict[str, Any] | None, SampleStats]:
    sample_id = str(row.get("sample_id") or row.get("video_name") or row.get("pair_index") or "sample")
    safe_id = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in sample_id)
    sample_root = output_root / "samples" / safe_id
    prop_dir = sample_root / "propagated_frames"
    conf_dir = sample_root / "confidence_maps"
    reliable_dir = sample_root / "reliable_masks"
    generate_dir = sample_root / "generate_masks"
    source_count_dir = sample_root / "source_count_maps"
    source_index_dir = sample_root / "source_index_maps"
    oracle_dir = sample_root / "oracle_confidence_maps"
    done_flag = sample_root / ".done"

    if args.resume and done_flag.exists():
        new_row = dict(row)
        new_row.update(
            propagated_frame_dir=str(prop_dir),
            propagation_confidence_dir=str(conf_dir),
            reliable_mask_dir=str(reliable_dir),
            generate_mask_dir=str(generate_dir),
            source_count_dir=str(source_count_dir),
            source_index_dir=str(source_index_dir),
        )
        if oracle_dir.exists():
            new_row["oracle_confidence_dir"] = str(oracle_dir)
        return new_row, SampleStats(sample_id=sample_id, status="reused")

    size = None if args.width <= 0 or args.height <= 0 else (args.width, args.height)
    frames = load_rgb_sequence(row["win_video_path"], args.nframes, size)
    masks = load_mask_sequence(row["mask_path"], args.nframes, size)
    result = propagate_one_video(
        frames,
        masks,
        source_window=args.source_window,
        tau_conf=args.tau_conf,
        write_oracle=args.write_oracle,
        alpha_oracle=args.oracle_alpha,
    )

    save_rgb_frames(prop_dir, result["propagated"])  # type: ignore[arg-type]
    save_gray_frames(conf_dir, np.asarray(result["confidence"]) * 255.0)
    save_gray_frames(reliable_dir, np.asarray(result["reliable"]) * 255)
    save_gray_frames(generate_dir, np.asarray(result["generate"]) * 255)
    save_gray_frames(source_count_dir, np.clip(np.asarray(result["source_count"]) * 40.0, 0, 255))
    save_gray_frames(source_index_dir, np.asarray(result["source_index"]))
    if "oracle_confidence" in result:
        save_gray_frames(oracle_dir, np.asarray(result["oracle_confidence"]) * 255.0)

    done_flag.write_text("ok\n")
    new_row = dict(row)
    new_row.update(
        propagated_frame_dir=str(prop_dir),
        propagation_confidence_dir=str(conf_dir),
        reliable_mask_dir=str(reliable_dir),
        generate_mask_dir=str(generate_dir),
        source_count_dir=str(source_count_dir),
        source_index_dir=str(source_index_dir),
        propagation_coverage=result["propagation_coverage"],
        average_confidence=result["average_confidence"],
        avg_num_sources_used=result["avg_num_sources_used"],
        propagated_region_psnr=result["propagated_region_psnr"],
        full_mask_prop_psnr=result["full_mask_prop_psnr"],
        propagation_method="farneback_multisource_agreement",
        tau_conf=args.tau_conf,
    )
    if "oracle_confidence" in result:
        new_row["oracle_confidence_dir"] = str(oracle_dir)

    return new_row, SampleStats(
        sample_id=sample_id,
        status="ok",
        num_frames=args.nframes,
        propagation_coverage=float(result["propagation_coverage"]),
        average_confidence=float(result["average_confidence"]),
        avg_num_sources_used=float(result["avg_num_sources_used"]),
        propagated_region_psnr=float(result["propagated_region_psnr"]),
        full_mask_prop_psnr=float(result["full_mask_prop_psnr"]),
    )


def write_reports(output_root: Path, stats: list[SampleStats], args: argparse.Namespace, manifest_path: Path) -> None:
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = reports / "propagation_cache_quality.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "status",
                "num_frames",
                "propagation_coverage",
                "average_confidence",
                "avg_num_sources_used",
                "propagated_region_psnr",
                "full_mask_prop_psnr",
                "failure_reason",
            ],
        )
        writer.writeheader()
        for item in stats:
            writer.writerow(item.__dict__)
    ok = [s for s in stats if s.status in {"ok", "reused"}]
    failed = [s for s in stats if s.status.startswith("failed")]
    def mean(field: str) -> float:
        vals = [getattr(s, field) for s in ok if getattr(s, field) is not None and not math.isnan(float(getattr(s, field)))]
        return float(np.mean(vals)) if vals else float("nan")
    md = reports / "propagation_cache_report.md"
    md.write_text(
        "\n".join(
            [
                "# Exp18 Multi-frame Propagation Cache Report",
                "",
                f"- input_manifest: `{args.input_manifest}`",
                f"- output_manifest: `{manifest_path}`",
                f"- output_root: `{output_root}`",
                f"- limit: `{args.limit}`",
                f"- nframes: `{args.nframes}`",
                f"- method: `farneback_multisource_agreement`",
                f"- source_window: `{args.source_window}`",
                f"- tau_conf: `{args.tau_conf}`",
                f"- write_oracle: `{args.write_oracle}`",
                "",
                "## Summary",
                "",
                f"- total attempted: `{len(stats)}`",
                f"- ok/reused: `{len(ok)}`",
                f"- failed: `{len(failed)}`",
                f"- mean propagation coverage: `{mean('propagation_coverage'):.6f}`",
                f"- mean confidence: `{mean('average_confidence'):.6f}`",
                f"- mean avg_num_sources_used: `{mean('avg_num_sources_used'):.6f}`",
                f"- mean propagated_region_psnr: `{mean('propagated_region_psnr'):.6f}`",
                f"- mean full_mask_prop_psnr: `{mean('full_mask_prop_psnr'):.6f}`",
                "",
                "## Interpretation Rule",
                "",
                "If mean propagation coverage is below 0.05 or propagated-region PSNR is low,",
                "the propagation cache should be treated as not useful and Exp18 training should not start.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--nframes", type=int, default=16)
    parser.add_argument("--width", type=int, default=432)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--source_window", type=int, default=3)
    parser.add_argument("--tau_conf", type=float, default=0.5)
    parser.add_argument("--write_oracle", action="store_true")
    parser.add_argument("--oracle_alpha", type=float, default=5.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    input_manifest = Path(args.input_manifest).expanduser()
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(input_manifest)
    if args.limit > 0:
        rows = rows[: args.limit]
    out_rows: list[dict[str, Any]] = []
    stats: list[SampleStats] = []
    failed_rows = []
    for idx, row in enumerate(rows, start=1):
        sample_id = str(row.get("sample_id") or row.get("video_name") or idx)
        try:
            new_row, stat = process_row(row, args, output_root)
            if new_row is not None:
                out_rows.append(new_row)
            stats.append(stat)
            print(f"[{idx}/{len(rows)}] {sample_id}: {stat.status} coverage={stat.propagation_coverage}")
        except Exception as exc:  # keep cache generation resumable
            msg = str(exc)
            print(f"[{idx}/{len(rows)}] {sample_id}: FAILED {msg}", file=sys.stderr)
            stats.append(SampleStats(sample_id=sample_id, status="failed", failure_reason=msg))
            failed_rows.append({"sample_id": sample_id, "failure_reason": msg})

    manifest_name = "exp18_train_with_multiframe_prop_limit100.jsonl" if args.limit == 100 else f"exp18_train_with_multiframe_prop_limit{args.limit}.jsonl"
    manifest_path = output_root / "manifests" / manifest_name
    write_jsonl(manifest_path, out_rows)
    if failed_rows:
        failed_path = output_root / "failed_cases.csv"
        with failed_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "failure_reason"])
            writer.writeheader()
            writer.writerows(failed_rows)
    write_reports(output_root, stats, args, manifest_path)
    if not out_rows:
        raise SystemExit("No usable Exp18 propagation rows were generated")


if __name__ == "__main__":
    main()
