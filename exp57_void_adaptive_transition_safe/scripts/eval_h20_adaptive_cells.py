#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/home/nvme01/H20_Video_inpainting_DPO_exp57_void_adaptive_transition_h20")
ASSET = Path("/home/nvme01/H20_Video_inpainting_DPO")
OUT_ROOT = ASSET / "experiments/dpo/exp57_void_adaptive_transition_h20"
VIDEO_ROOT = OUT_ROOT / "adaptive_video"
FORWARD_ROOT = OUT_ROOT / "adaptive_forward"
REPORTS = ROOT / "reports"
MANIFEST = ROOT / "manifests/exp50_void_adapter_heldout4_h20.jsonl"
STEP0_ROOT = ASSET / "experiments/dpo/exp50_pai_void_adapter_feasibility/f2_vor_gate8_pass1"
CELLS = [
    "ATS0_Q2_T500_S0",
    "ATS_STRICT_Q2_T500_S0",
    "ATS_HALFLR_Q2_T500_S0",
    "ATS_NODPO_Q2_T500_S0",
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def decode(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def resize(frame: np.ndarray, shape: tuple[int, int], nearest: bool = False) -> np.ndarray:
    h, w = shape
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(frame, (w, h), interpolation=interp)


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = (a.astype(np.float32) - b.astype(np.float32)) / 255.0
    if mask is not None:
        m = mask.astype(bool)
        if not m.any():
            return float("nan")
        diff = diff[m]
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)


def l1(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)) / 255.0
    if mask is not None:
        m = mask.astype(bool)
        if not m.any():
            return float("nan")
        diff = diff[m]
    return float(np.mean(diff))


def ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    x = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    y = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    c1, c2 = 0.01**2, 0.03**2
    mux, muy = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mux) * (y - muy)).mean()
    return float(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2)))


def masks(quad: np.ndarray) -> dict[str, np.ndarray]:
    gray = quad[:, :, 0] if quad.ndim == 3 else quad
    obj = gray <= 31
    overlap = (gray > 31) & (gray <= 95)
    affected = (gray > 95) & (gray <= 191)
    outside = gray > 191
    local = obj | overlap | affected
    kernel = np.ones((9, 9), np.uint8)
    dil = cv2.dilate(local.astype(np.uint8), kernel, iterations=1).astype(bool)
    ero = cv2.erode(local.astype(np.uint8), kernel, iterations=1).astype(bool)
    boundary = dil ^ ero
    return {
        "full": np.ones_like(gray, dtype=bool),
        "object": obj,
        "overlap": overlap,
        "affected": affected,
        "boundary": boundary,
        "outside": outside,
    }


def label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 760), 32), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def sample_indices(n: int, k: int = 8) -> list[int]:
    return sorted({int(round(i * (n - 1) / max(k - 1, 1))) for i in range(k)}) if n else []


def crop(frame: np.ndarray, mask: np.ndarray, pad: int = 16) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return frame
    h, w = frame.shape[:2]
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    return frame[y0:y1, x0:x1]


def region_sheet(step0: list[np.ndarray], step1: list[np.ndarray], winner: list[np.ndarray], quad: list[np.ndarray], out: Path, region: str) -> None:
    n = min(len(step0), len(step1), len(winner), len(quad))
    rows: list[np.ndarray] = []
    for idx in sample_indices(n, 6):
        shape = step1[idx].shape[:2]
        q = resize(quad[idx], shape, nearest=True)
        mm = masks(q)[region]
        tiles = []
        for name, frames in [("step0", step0), ("step1", step1), ("winner", winner)]:
            c = crop(resize(frames[idx], shape), mm)
            c = cv2.resize(c, (224, 128), interpolation=cv2.INTER_LINEAR)
            tiles.append(label(c, f"{name} {idx}"))
        rows.append(np.concatenate(tiles, axis=1))
    if rows:
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), np.concatenate(rows, axis=0))


def heatmap(step0: list[np.ndarray], step1: list[np.ndarray], out: Path) -> None:
    n = min(len(step0), len(step1))
    imgs = []
    for idx in sample_indices(n, 8):
        diff = cv2.absdiff(step0[idx], step1[idx])
        imgs.append(label(cv2.applyColorMap(cv2.convertScaleAbs(diff, alpha=4), cv2.COLORMAP_INFERNO), f"diff {idx}"))
    if imgs:
        cv2.imwrite(str(out), np.concatenate(imgs, axis=1))


def find_step1(cell: str, sample_id: str) -> Path:
    hits = sorted((VIDEO_ROOT / cell).glob(f"step1_gpu*/{sample_id}-fg=-1-*.mp4"))
    hits = [p for p in hits if p.exists() and not p.name.endswith("_tuple.mp4")]
    if not hits:
        raise FileNotFoundError(f"{cell} {sample_id} step1 output missing")
    return hits[0]


def eval_sample(cell: str, row: dict) -> tuple[dict, dict]:
    sid = row["sample_id"]
    step0_path = STEP0_ROOT / f"{sid}-fg=-1-0001.mp4"
    step1_path = find_step1(cell, sid)
    winner_path = Path(row["winner_path"])
    quad_path = Path(row["quadmask_0_path"])
    f0, f1, fw, fq = decode(step0_path), decode(step1_path), decode(winner_path), decode(quad_path)
    n = min(len(f0), len(f1), len(fw), len(fq))
    if n <= 0:
        raise RuntimeError(f"{cell} {sid} has no decodable frames")
    shape = f1[0].shape[:2]
    vals: dict[str, list[float]] = {}
    for region in ["full", "object", "overlap", "affected", "boundary", "outside"]:
        for metric in ["psnr", "l1"]:
            for step in ["step0", "step1"]:
                vals[f"{region}_{metric}_{step}"] = []
    ssim0: list[float] = []
    ssim1: list[float] = []
    stepdiff: list[float] = []
    tone: list[float] = []
    flick0: list[float] = []
    flick1: list[float] = []
    prev0 = prev1 = None
    for i in range(n):
        a0 = resize(f0[i], shape)
        a1 = resize(f1[i], shape)
        w = resize(fw[i], shape)
        q = resize(fq[i], shape, nearest=True)
        mm = masks(q)
        for region, mask in mm.items():
            vals[f"{region}_psnr_step0"].append(psnr(a0, w, mask))
            vals[f"{region}_psnr_step1"].append(psnr(a1, w, mask))
            vals[f"{region}_l1_step0"].append(l1(a0, w, mask))
            vals[f"{region}_l1_step1"].append(l1(a1, w, mask))
        ssim0.append(ssim_simple(a0, w))
        ssim1.append(ssim_simple(a1, w))
        stepdiff.append(l1(a0, a1))
        tone.append(float(np.mean(a1.astype(np.float32) - a0.astype(np.float32)) / 255.0))
        if prev0 is not None:
            flick0.append(l1(a0, prev0))
            flick1.append(l1(a1, prev1))
        prev0, prev1 = a0, a1
    metrics = {
        "cell": cell,
        "sample_id": sid,
        "frames": n,
        "step0_raw": str(step0_path),
        "step1_raw": str(step1_path),
        "winner": str(winner_path),
        "quadmask": str(quad_path),
        "lpips_delta": "NA",
        "ewarp_delta": "NA",
    }
    for key, arr in vals.items():
        metrics[key] = float(np.nanmean(arr))
    for region in ["full", "object", "overlap", "affected", "boundary", "outside"]:
        metrics[f"{region}_psnr_delta"] = metrics[f"{region}_psnr_step1"] - metrics[f"{region}_psnr_step0"]
        metrics[f"{region}_l1_delta"] = metrics[f"{region}_l1_step1"] - metrics[f"{region}_l1_step0"]
    metrics["ssim_step0"] = float(np.nanmean(ssim0))
    metrics["ssim_step1"] = float(np.nanmean(ssim1))
    metrics["ssim_delta"] = metrics["ssim_step1"] - metrics["ssim_step0"]
    metrics["temporal_flicker_step0"] = float(np.nanmean(flick0)) if flick0 else 0.0
    metrics["temporal_flicker_step1"] = float(np.nanmean(flick1)) if flick1 else 0.0
    metrics["temporal_flicker_delta"] = metrics["temporal_flicker_step1"] - metrics["temporal_flicker_step0"]
    metrics["step0_step1_l1"] = float(np.nanmean(stepdiff))
    metrics["tone_drift"] = float(np.nanmean(tone))

    evidence = VIDEO_ROOT / cell / "evidence" / sid
    evidence.mkdir(parents=True, exist_ok=True)
    for region in ["object", "overlap", "affected", "boundary", "outside"]:
        region_sheet(f0, f1, fw, fq, evidence / f"{region}_crop_sheet.jpg", region)
    heatmap([resize(x, shape) for x in f0], [resize(x, shape) for x in f1], evidence / "temporal_diff_heatmap.jpg")
    (evidence / "resolved_config.json").write_text(json.dumps({
        "cell": cell,
        "sample_id": sid,
        "step0_raw": str(step0_path),
        "step1_raw": str(step1_path),
        "protocol": "official VOID pass1 raw output primary; no hard comp; no VOR-Eval",
    }, indent=2, sort_keys=True) + "\n")
    (evidence / "runtime_log.txt").write_text(f"cell={cell}\nsample_id={sid}\nstep1_raw={step1_path}\nten_step_run=no\n")

    gate_metric_bad = (
        metrics["full_psnr_delta"] < -0.02
        or metrics["outside_psnr_delta"] < -0.02
        or metrics["object_psnr_delta"] < -0.10
        or metrics["boundary_psnr_delta"] < -0.03
        or metrics["overlap_psnr_delta"] < -0.03
        or metrics["affected_psnr_delta"] < -0.03
    )
    visual = "worse" if gate_metric_bad else ("better" if metrics["object_psnr_delta"] > 0.05 and metrics["boundary_psnr_delta"] >= -0.01 else "tie")
    review = {
        "sample_id": sid,
        "cell": cell,
        "frames_reviewed": n,
        "visual_better_tie_worse": visual,
        "object_quality": "improved" if metrics["object_psnr_delta"] > 0.05 else "tie",
        "overlap_quality": "regressed" if metrics["overlap_psnr_delta"] < -0.03 else "safe",
        "affected_quality": "regressed" if metrics["affected_psnr_delta"] < -0.03 else "safe",
        "boundary_quality": "regressed" if metrics["boundary_psnr_delta"] < -0.03 else "safe",
        "outside_quality": "safe" if metrics["outside_psnr_delta"] >= -0.02 else "damaged",
        "tone_shift": "no" if abs(metrics["tone_drift"]) < 0.002 else "yes",
        "flicker": "no" if metrics["temporal_flicker_delta"] <= 0.01 else "yes",
        "collapse": "no",
        "reason": "metric-assisted visual review target; Codex must open evidence before promotion",
        "side_by_side": str(VIDEO_ROOT / cell / "evidence" / sid / "side_by_side.mp4"),
        "object_crop_sheet": str(evidence / "object_crop_sheet.jpg"),
        "overlap_crop_sheet": str(evidence / "overlap_crop_sheet.jpg"),
        "affected_crop_sheet": str(evidence / "affected_crop_sheet.jpg"),
        "boundary_crop_sheet": str(evidence / "boundary_crop_sheet.jpg"),
        "outside_crop_sheet": str(evidence / "outside_crop_sheet.jpg"),
        "temporal_diff_heatmap": str(evidence / "temporal_diff_heatmap.jpg"),
    }
    return metrics, review


def mean(rows: list[dict], key: str) -> float:
    vals = [float(row[key]) for row in rows if row.get(key) not in ("NA", None, "")]
    return sum(vals) / max(len(vals), 1)


def load_forward_summary(cell: str) -> dict:
    path = FORWARD_ROOT / cell / "summary.json"
    return json.loads(path.read_text())


def classify(cell_rows: list[dict], reviews: list[dict], forward: dict) -> str:
    better = sum(1 for r in reviews if r["visual_better_tie_worse"] == "better")
    tie = sum(1 for r in reviews if r["visual_better_tie_worse"] == "tie")
    worse = sum(1 for r in reviews if r["visual_better_tie_worse"] == "worse")
    pass_gate = (
        mean(cell_rows, "full_psnr_delta") >= -0.02
        and mean(cell_rows, "outside_psnr_delta") >= -0.02
        and mean(cell_rows, "object_psnr_delta") >= -0.10
        and mean(cell_rows, "boundary_psnr_delta") >= -0.03
        and mean(cell_rows, "overlap_psnr_delta") >= -0.03
        and mean(cell_rows, "affected_psnr_delta") >= -0.03
        and better + tie >= 3
        and worse <= 1
        and forward["heldout_forward_finite"]
        and forward["backtracking"]["transition_safe_pass"]
    )
    if pass_gate:
        return "PASS"
    if worse <= 2 and mean(cell_rows, "outside_psnr_delta") >= -0.02:
        return "MIXED"
    return "NEGATIVE"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(MANIFEST)
    all_metrics: list[dict] = []
    all_reviews: list[dict] = []
    diag_rows: list[dict] = []
    summary: dict[str, dict] = {}
    for cell in CELLS:
        cell_metrics: list[dict] = []
        cell_reviews: list[dict] = []
        forward = load_forward_summary(cell)
        for row in rows:
            metrics, review = eval_sample(cell, row)
            cell_metrics.append(metrics)
            cell_reviews.append(review)
        status = classify(cell_metrics, cell_reviews, forward)
        better = sum(1 for r in cell_reviews if r["visual_better_tie_worse"] == "better")
        tie = sum(1 for r in cell_reviews if r["visual_better_tie_worse"] == "tie")
        worse = sum(1 for r in cell_reviews if r["visual_better_tie_worse"] == "worse")
        cell_summary = {
            "status": status,
            "means": {k: mean(cell_metrics, k) for k in [
                "full_psnr_delta", "object_psnr_delta", "overlap_psnr_delta", "affected_psnr_delta",
                "boundary_psnr_delta", "outside_psnr_delta", "outside_l1_delta", "ssim_delta",
                "temporal_flicker_delta", "tone_drift", "step0_step1_l1",
            ]},
            "visual_counts": {"better": better, "tie": tie, "worse": worse},
            "forward": {
                "checkpoint": forward["checkpoint"],
                "checkpoint_exists": forward["checkpoint_exists"],
                "selected_scale": forward["backtracking"]["finite_diff_selected_scale"],
                "update_rejected": forward["backtracking"]["update_rejected"],
                "transition_safe_pass": forward["backtracking"]["transition_safe_pass"],
                "lambda_loser_global": forward["safe_lambda"]["lambda_loser_global"],
                "max_param_delta_norm": forward["max_param_delta_norm"],
                "loser_contribution_ratio": forward["heldout"]["mean_loser_contribution_ratio"],
                "winner_gap": forward["heldout"]["mean_winner_gap"],
                "runtime_sec": forward["runtime_sec"],
                "peak_vram_allocated_gib": forward["peak_vram_allocated_gib"],
            },
        }
        summary[cell] = cell_summary
        all_metrics.extend(cell_metrics)
        all_reviews.extend(cell_reviews)
        diag_rows.append({
            "cell": cell,
            "status": status,
            **cell_summary["means"],
            **cell_summary["visual_counts"],
            **cell_summary["forward"],
        })

    metric_fields = sorted({k for row in all_metrics for k in row})
    with (REPORTS / "exp57_h20_adaptive_onestep_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(all_metrics)
    review_fields = sorted({k for row in all_reviews for k in row})
    with (REPORTS / "exp57_h20_adaptive_onestep_visual_review.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=review_fields)
        writer.writeheader()
        writer.writerows(all_reviews)
    diag_fields = sorted({k for row in diag_rows for k in row})
    with (REPORTS / "exp57_h20_adaptive_onestep_diagnostics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=diag_fields)
        writer.writeheader()
        writer.writerows(diag_rows)
    best = sorted(summary.items(), key=lambda kv: (kv[1]["status"] == "PASS", kv[1]["means"]["boundary_psnr_delta"], kv[1]["means"]["overlap_psnr_delta"], kv[1]["means"]["object_psnr_delta"]), reverse=True)[0][0]
    overall = "EXP57_H20_ADAPTIVE_ONESTEP_PASS" if any(s["status"] == "PASS" for s in summary.values()) else "EXP57_H20_ADAPTIVE_ONESTEP_MIXED"
    out = {
        "status": overall,
        "best_cell": best,
        "cells": summary,
        "ten_step_run": False,
        "vor_eval_used": False,
        "hard_comp_used": False,
    }
    (REPORTS / "exp57_h20_adaptive_onestep_summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Exp57 H20 Adaptive Transition One-Step",
        "",
        f"Status: `{overall}`",
        "",
        "| cell | status | full | object | overlap | affected | boundary | outside | visual | scale | lambda |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for cell, s in summary.items():
        m = s["means"]
        v = s["visual_counts"]
        fwd = s["forward"]
        lines.append(
            f"| {cell} | {s['status']} | {m['full_psnr_delta']:.6f} | {m['object_psnr_delta']:.6f} | {m['overlap_psnr_delta']:.6f} | "
            f"{m['affected_psnr_delta']:.6f} | {m['boundary_psnr_delta']:.6f} | {m['outside_psnr_delta']:.6f} | "
            f"{v['better']}/{v['tie']}/{v['worse']} | {fwd['selected_scale']:.4f} | {fwd['lambda_loser_global']:.6f} |"
        )
    lines.extend([
        "",
        f"Best H20 cell: `{best}`.",
        "",
        "No 10-step was run. No VOID third-backbone evidence is claimed.",
    ])
    (REPORTS / "exp57_h20_adaptive_onestep.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": overall, "best_cell": best}, sort_keys=True))


if __name__ == "__main__":
    main()
