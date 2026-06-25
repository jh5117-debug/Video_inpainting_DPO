#!/usr/bin/env python3
"""Review Exp25 DiffuEraser OR root-cause matrix outputs.

The generator script writes raw frame directories per stack.  This reviewer
does not run inference.  It builds a pair manifest for the existing
``tools/run_inpainting_metric_eval.py`` adapter, invokes that adapter so core
PSNR/SSIM/LPIPS/Ewarp remain delegated to ``inference/metrics.py``, then adds
root-cause-specific dense evidence and conservative OR loser classifications.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--num-frames", type=int, default=24)
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-lpips", action="store_true")
    p.add_argument("--compute-ewarp", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def list_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def load_rgb(path: Path, limit: int) -> list[np.ndarray]:
    return [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in list_images(path)[:limit]]


def load_mask(path: Path, limit: int, size: tuple[int, int] | None = None) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for p in list_images(path)[:limit]:
        img = Image.open(p).convert("L")
        if size and img.size != size:
            img = img.resize(size, Image.NEAREST)
        out.append((np.asarray(img, dtype=np.uint8) > 127).astype(np.uint8))
    return out


def resize_like(arr: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if arr.shape[:2] == ref.shape[:2]:
        return arr
    return cv2.resize(arr, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)


def overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = frame.astype(np.float32)
    tint = np.zeros_like(out)
    tint[..., 0] = 255
    alpha = (mask > 0).astype(np.float32)[..., None] * 0.45
    return np.clip(out * (1.0 - alpha) + tint * alpha, 0, 255).astype(np.uint8)


def label(arr: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, min(img.width, 540), 22], fill=(0, 0, 0))
    draw.text((5, 5), text, fill=(255, 255, 255))
    return np.asarray(img)


def error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = resize_like(pred, gt).astype(np.float32)
    err = np.abs(pred - gt.astype(np.float32)).mean(axis=2)
    heat = cv2.applyColorMap(np.clip(err * 3.0, 0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def crop_to_mask(arr: np.ndarray, mask: np.ndarray, pad: int = 20) -> np.ndarray:
    if mask.shape != arr.shape[:2]:
        mask = cv2.resize(mask, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return arr
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, arr.shape[1])
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, arr.shape[0])
    return cv2.resize(arr[y0:y1, x0:x1], (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_AREA)


def save_gif(frames: list[np.ndarray], path: Path, size: tuple[int, int] = (320, 180)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return
    pil_frames = [Image.fromarray(cv2.resize(f, size, interpolation=cv2.INTER_AREA)) for f in frames]
    pil_frames[0].save(path, save_all=True, append_images=pil_frames[1:], duration=140, loop=0)


def temporal_scores(frames: list[np.ndarray]) -> list[float]:
    vals = [0.0]
    for idx in range(1, len(frames)):
        vals.append(float(np.abs(frames[idx].astype(np.float32) - resize_like(frames[idx - 1], frames[idx]).astype(np.float32)).mean()))
    return vals


def psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    pred = resize_like(pred, gt).astype(np.float32)
    gt = gt.astype(np.float32)
    if mask is not None:
        if mask.shape != gt.shape[:2]:
            mask = cv2.resize(mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        w = mask.astype(np.float32)[..., None]
        denom = float(w.sum() * 3.0)
        if denom < 1:
            return float("nan")
        mse = float((((pred - gt) ** 2) * w).sum() / denom)
    else:
        mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def finite_mean(vals: list[float]) -> float:
    good = [float(v) for v in vals if math.isfinite(float(v))]
    return float(np.mean(good)) if good else float("nan")


def write_evidence(sample_id: str, stack_id: str, winner: list[np.ndarray], cond: list[np.ndarray], masks: list[np.ndarray], pred: list[np.ndarray], out_dir: Path) -> tuple[str, str]:
    n = min(len(winner), len(cond), len(masks), len(pred))
    if n == 0:
        return "", ""
    mask_area = [float(m.mean()) for m in masks[:n]]
    errs = [float(np.abs(resize_like(pred[i], winner[i]).astype(np.float32) - winner[i].astype(np.float32)).mean()) for i in range(n)]
    tdiff = temporal_scores(pred[:n])
    picks = sorted(set(np.linspace(0, n - 1, num=min(16, n), dtype=int).tolist() + [0, n // 2, n - 1, int(np.argmax(mask_area)), int(np.argmax(errs))] + np.argsort(tdiff)[-3:].tolist()))
    rows = []
    crop_rows = []
    for idx in picks:
        pr = resize_like(pred[idx], winner[idx])
        mask_overlay = overlay(winner[idx], masks[idx])
        err = error_map(pr, winner[idx])
        rows.append(np.concatenate([label(winner[idx], f"winner f{idx}"), label(cond[idx], "condition"), label(mask_overlay, "mask"), label(pr, f"{stack_id} raw"), label(err, "abs error")], axis=1))
        crop_rows.append(np.concatenate([label(crop_to_mask(winner[idx], masks[idx]), f"winner crop f{idx}"), label(crop_to_mask(cond[idx], masks[idx]), "condition crop"), label(crop_to_mask(mask_overlay, masks[idx]), "mask crop"), label(crop_to_mask(pr, masks[idx]), "raw crop"), label(crop_to_mask(err, masks[idx]), "error crop")], axis=1))
    sheet_path = out_dir / "dense_contact_sheets" / stack_id / f"{sample_id}.jpg"
    crop_path = out_dir / "mask_crops" / stack_id / f"{sample_id}.jpg"
    gif_path = out_dir / "animated_gif" / stack_id / f"{sample_id}.gif"
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(sheet_path, quality=92)
    Image.fromarray(np.concatenate(crop_rows, axis=0)).save(crop_path, quality=92)
    save_gif(pred[:n], gif_path)
    return str(sheet_path), str(gif_path)


def classify(item: dict[str, Any]) -> tuple[str, str]:
    if item["status"] != "OK":
        return "TECHNICAL_INVALID", item.get("issue", "missing or failed output")
    if item["black_frame_ratio"] > 0:
        return "TRIVIAL_BAD", "contains black/constant frames"
    if item["mask_psnr_local"] < 12.0 or item["full_psnr_local"] < 10.0:
        return "TRIVIAL_BAD", "very low raw loser PSNR"
    if item["mask_psnr_local"] > 33.0 and item["masked_mae_local"] < 4.0:
        return "TOO_CLOSE", "raw loser is too close to winner"
    if item["mask_psnr_local"] < 18.0 or item["temporal_absdiff_max"] > 55.0 or item["outside_psnr_local"] < 20.0:
        return "HARD_BUT_PLAUSIBLE", "strong finite local/temporal defect"
    return "MEDIUM_HARD_ELIGIBLE", "finite visible defect without global collapse"


def build_metric_manifest(args: argparse.Namespace, stack_status: list[dict[str, str]], source_rows: list[dict[str, Any]], out_dir: Path) -> Path:
    rows: list[dict[str, str]] = []
    ok_stacks = [row for row in stack_status if row.get("status") == "OK"]
    for stack in ok_stacks:
        pred_root = Path(stack["stack_root"]) / "diffueraser" / "raw_frames"
        for src in source_rows:
            rows.append(
                {
                    "sample_id": src["sample_id"],
                    "model_label": stack["stack_id"],
                    "gt_video_path": src["winner_video_path"],
                    "prediction_video_path": str(pred_root / src["sample_id"]),
                    "mask_path": src["mask_path"],
                }
            )
    path = out_dir / "root_cause_metric_pairs.csv"
    write_csv(path, rows)
    return path


def run_metric_adapter(args: argparse.Namespace, metric_manifest: Path, out_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(args.project_root / "tools" / "run_inpainting_metric_eval.py"),
        "--pair_manifest",
        str(metric_manifest),
        "--output_dir",
        str(out_dir),
        "--sample_id_col",
        "sample_id",
        "--model_col",
        "model_label",
        "--gt_col",
        "gt_video_path",
        "--pred_col",
        "prediction_video_path",
        "--mask_col",
        "mask_path",
        "--max_frames",
        str(args.num_frames),
        "--device",
        args.device,
    ]
    if args.compute_lpips:
        cmd.append("--compute_lpips")
    if args.compute_ewarp:
        cmd.append("--compute_ewarp")
    subprocess.run(cmd, cwd=str(args.project_root), check=True)


def main() -> int:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    out_dir = args.run_root / "review_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_rows = read_jsonl(args.source_manifest)
    stack_status = read_csv(args.run_root / "stack_status.csv")
    metric_manifest = build_metric_manifest(args, stack_status, source_rows, out_dir)
    run_metric_adapter(args, metric_manifest, out_dir / "metric_eval")
    metric_rows = read_csv(out_dir / "metric_eval" / "metrics" / "per_sample_metrics.csv")
    metric_by_key = {(r["model_label"], r["sample_id"]): r for r in metric_rows}

    review_rows: list[dict[str, Any]] = []
    for stack in stack_status:
        stack_id = stack["stack_id"]
        if stack.get("status") != "OK":
            review_rows.append(
                {
                    "stack_id": stack_id,
                    "sample_id": "",
                    "status": stack.get("status", ""),
                    "classification": "TECHNICAL_INVALID" if stack.get("status") == "FAILED" else "BLOCKED",
                    "reason": stack.get("block_reason") or f"stack status {stack.get('status')}",
                    "reviewer_pass": False,
                    "review_method": "stack_status_only",
                    "log_path": stack.get("log_path", ""),
                }
            )
            continue
        pred_root = Path(stack["stack_root"]) / "diffueraser" / "raw_frames"
        for src in source_rows:
            sid = src["sample_id"]
            winner = load_rgb(Path(src["winner_video_path"]), args.num_frames)
            cond = load_rgb(Path(src["condition_video_path"]), args.num_frames)
            pred = load_rgb(pred_root / sid, args.num_frames)
            masks = load_mask(Path(src["mask_path"]), args.num_frames, size=winner[0].shape[1::-1] if winner else None)
            n = min(len(winner), len(cond), len(pred), len(masks), args.num_frames)
            item: dict[str, Any] = {
                "stack_id": stack_id,
                "sample_id": sid,
                "source_type": src.get("source_type", ""),
                "gate32_prior_classification": src.get("gate32_classification", ""),
                "status": "OK" if n == args.num_frames else "MISSING_FRAMES",
                "num_frames": n,
                "pred_dir": str(pred_root / sid),
                "winner_dir": src["winner_video_path"],
                "condition_dir": src["condition_video_path"],
                "mask_dir": src["mask_path"],
            }
            if item["status"] == "OK":
                mask_psnr = [psnr(p, g, m) for p, g, m in zip(pred[:n], winner[:n], masks[:n])]
                outside_psnr = [psnr(p, g, 1 - m) for p, g, m in zip(pred[:n], winner[:n], masks[:n])]
                full_vals = [psnr(p, g) for p, g in zip(pred[:n], winner[:n])]
                mae_vals = []
                black = 0
                for p, g, m in zip(pred[:n], winner[:n], masks[:n]):
                    pr = resize_like(p, g).astype(np.float32)
                    w = m.astype(np.float32)[..., None]
                    denom = float(w.sum() * 3.0)
                    mae_vals.append(float((np.abs(pr - g.astype(np.float32)) * w).sum() / max(denom, 1.0)))
                    if float(pr.mean()) < 2.0 or float(pr.std()) < 1.0:
                        black += 1
                tdiff = temporal_scores(pred[:n])
                sheet, gif = write_evidence(sid, stack_id, winner, cond, masks, pred, out_dir)
                item.update(
                    {
                        "full_psnr_local": finite_mean(full_vals),
                        "mask_psnr_local": finite_mean(mask_psnr),
                        "outside_psnr_local": finite_mean(outside_psnr),
                        "masked_mae_local": finite_mean(mae_vals),
                        "black_frame_ratio": float(black / n),
                        "temporal_absdiff_mean": finite_mean(tdiff),
                        "temporal_absdiff_max": max(tdiff) if tdiff else 0.0,
                        "dense_contact_sheet": sheet,
                        "animated_gif": gif,
                    }
                )
            else:
                item.update({"issue": "missing aligned frames", "full_psnr_local": float("nan"), "mask_psnr_local": float("nan"), "outside_psnr_local": float("nan"), "masked_mae_local": float("nan"), "black_frame_ratio": 1.0, "temporal_absdiff_mean": float("nan"), "temporal_absdiff_max": float("nan")})
            cls, reason = classify(item)
            item["classification"] = cls
            item["reason"] = reason
            item["review_method"] = "dense_temporal_evidence_generated"
            item["reviewer_pass"] = True
            metric = metric_by_key.get((stack_id, sid), {})
            for key, value in metric.items():
                if key not in item:
                    item[f"metric_{key}"] = value
            review_rows.append(item)

    csv_path = args.report_dir / "exp25_diffueraser_or_root_cause_visual_review.csv"
    write_csv(csv_path, review_rows)
    summary_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in review_rows:
        grouped[row["stack_id"]].append(row)
    for stack_id, group in sorted(grouped.items()):
        counts = Counter(row["classification"] for row in group)
        ok_rows = [row for row in group if row.get("status") == "OK"]
        summary_rows.append(
            {
                "stack_id": stack_id,
                "rows": len(group),
                "ok_rows": len(ok_rows),
                "medium_hard": counts.get("MEDIUM_HARD_ELIGIBLE", 0),
                "hard_plausible": counts.get("HARD_BUT_PLAUSIBLE", 0),
                "trivial_bad": counts.get("TRIVIAL_BAD", 0),
                "too_close": counts.get("TOO_CLOSE", 0),
                "technical_invalid": counts.get("TECHNICAL_INVALID", 0),
                "blocked": counts.get("BLOCKED", 0),
                "mean_mask_psnr": finite_mean([float(r.get("mask_psnr_local", "nan")) for r in ok_rows]),
                "mean_outside_psnr": finite_mean([float(r.get("outside_psnr_local", "nan")) for r in ok_rows]),
                "mean_temporal_absdiff": finite_mean([float(r.get("temporal_absdiff_mean", "nan")) for r in ok_rows]),
            }
        )
    summary_csv = args.report_dir / "exp25_diffueraser_or_root_cause_matrix_v2.csv"
    write_csv(summary_csv, summary_rows)
    best = max(summary_rows, key=lambda r: (int(r["medium_hard"]) + int(r["hard_plausible"]), -int(r["trivial_bad"])), default={})
    final_decision = "MATRIX_INCONCLUSIVE_ASSET_BLOCKED"
    if best and int(best.get("ok_rows", 0)) >= 11 and int(best.get("medium_hard", 0)) + int(best.get("hard_plausible", 0)) >= 6 and int(best.get("trivial_bad", 99)) <= 2:
        final_decision = "DIFFUSERASER_NATIVE_OR_STACK_USABLE"
    elif any(r.get("ok_rows", 0) for r in summary_rows):
        final_decision = "DIFFUSERASER_OR_SELF_LOSER_YIELD_INSUFFICIENT"

    md = [
        "# Exp25 DiffuEraser OR Root-Cause Matrix v2",
        "",
        f"- run_root: `{args.run_root}`",
        f"- source_manifest: `{args.source_manifest}`",
        f"- metric_pair_manifest: `{metric_manifest}`",
        f"- metric_backend: `tools/run_inpainting_metric_eval.py` -> `inference/metrics.py`",
        f"- final_decision: `{final_decision}`",
        "",
        "## Stack Summary",
        "",
        "| stack | ok | medium-hard | hard-plausible | trivial-bad | too-close | technical-invalid | blocked | mean mask PSNR | mean outside PSNR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        md.append(
            f"| {row['stack_id']} | {row['ok_rows']} | {row['medium_hard']} | {row['hard_plausible']} | {row['trivial_bad']} | {row['too_close']} | {row['technical_invalid']} | {row['blocked']} | {row['mean_mask_psnr']:.4f} | {row['mean_outside_psnr']:.4f} |"
        )
    md.extend(
        [
            "",
            "## Root-Cause Answers",
            "",
            "- 6-step impact: DE-A is the canonical 6-step raw baseline; quality is judged from the per-sample review rows.",
            "- Dilation impact: compare DE-B against DE-A; no model regeneration was hidden in this review.",
            "- ProPainter prior impact: not isolated in this run because verified Exp25 wrapper only exposes the ProPainter-prior path.",
            "- PCM impact: DE-C failed before inference because the active UNetMotionModel lacks `load_lora_adapter`; this is a technical stack incompatibility, not a quality sample.",
            "- Official core vs SFT: official core was not evaluated because no strict-load-identifiable official core checkpoint path was available.",
            "- BR->OR domain shift: can only be inferred from SFT DE-A/DE-B quality; official-core comparison remains blocked.",
            "",
            "Hard-comp outputs were not used as loser quality evidence.",
        ]
    )
    (args.report_dir / "exp25_diffueraser_or_root_cause_matrix_v2.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "review_summary.json").write_text(json.dumps({"summary": summary_rows, "final_decision": final_decision}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary_csv": str(summary_csv), "review_csv": str(csv_path), "final_decision": final_decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
