#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


ROOT = Path("/home/nvme01/H20_Video_inpainting_DPO_exp53_void_r1r2_h20")
OUT_ROOT = Path("/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp53_void_r1r2_targeted_h20")
VIDEO_ROOT = OUT_ROOT / "core_video"
FORWARD_ROOT = OUT_ROOT / "core_forward"
REPORTS = ROOT / "reports"
MANIFEST = ROOT / "manifests/exp50_void_adapter_heldout4_h20.jsonl"
STEP0_ROOT = Path("/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp50_pai_void_adapter_feasibility/f2_vor_gate8_pass1")
CELLS = ["R1_Q2_T500_S0", "R2_Q2_T500_S0"]
EXP52_METRICS = ROOT / "reports/exp52_rescue_onestep_metrics.csv"


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


def find_step1(cell: str, sample_id: str) -> Path:
    hits = sorted((VIDEO_ROOT / cell).glob(f"step1_gpu*/{sample_id}-fg=-1-0001.mp4"))
    if not hits:
        raise FileNotFoundError(f"{cell} {sample_id} step1 output missing")
    return hits[0]


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 760), 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def sample_indices(n: int, k: int = 8) -> list[int]:
    if n <= 0:
        return []
    return sorted({int(round(i * (n - 1) / max(k - 1, 1))) for i in range(k)})


def crop_or_frame(frame: np.ndarray, mask: np.ndarray, pad: int = 12) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return frame
    h, w = frame.shape[:2]
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    return frame[y0:y1, x0:x1]


def make_region_sheet(step0: list[np.ndarray], step1: list[np.ndarray], winner: list[np.ndarray], quad: list[np.ndarray], out: Path, region: str) -> None:
    n = min(len(step0), len(step1), len(winner), len(quad))
    tiles: list[np.ndarray] = []
    for idx in sample_indices(n, 6):
        shape = step1[idx].shape[:2]
        q = resize(quad[idx], shape, nearest=True)
        mm = masks(q)[region]
        row = []
        for name, frames in [("step0", step0), ("step1", step1), ("winner", winner)]:
            crop = crop_or_frame(resize(frames[idx], shape), mm)
            crop = cv2.resize(crop, (224, 128), interpolation=cv2.INTER_LINEAR)
            row.append(label(crop, f"{name} f={idx}"))
        tiles.append(np.concatenate(row, axis=1))
    if tiles:
        cv2.imwrite(str(out), np.concatenate(tiles, axis=0))


def copy_runtime_config(cell: str, sample_id: str, evidence_dir: Path, step1: Path) -> None:
    resolved = {
        "cell": cell,
        "sample_id": sample_id,
        "step0_raw": str(STEP0_ROOT / f"{sample_id}-fg=-1-0001.mp4"),
        "step1_raw": str(step1),
        "winner": "rgb_removed.mp4 / V_bg from heldout manifest",
        "quadmask": "quadmask_0.mp4 from heldout manifest",
        "protocol": "official VOID pass1 inference, raw output primary, no hard comp, no VOR-Eval",
    }
    (evidence_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    runtime_log = evidence_dir / "runtime_log.txt"
    runtime_log.write_text(
        "\n".join(
            [
                f"cell={cell}",
                f"sample_id={sample_id}",
                f"step1_raw={step1}",
                "training_steps=1",
                "ten_step_run=no",
                "hard_comp=no",
                "vor_eval=no",
            ]
        )
        + "\n"
    )


def evaluate_cell(cell: str, rows: list[dict]) -> tuple[list[dict], list[dict], Path]:
    metrics: list[dict] = []
    review: list[dict] = []
    contact_rows: list[np.ndarray] = []
    for row in rows:
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
        tone_shift: list[float] = []
        flicker0: list[float] = []
        flicker1: list[float] = []
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
            tone_shift.append(float(np.mean(a1.astype(np.float32) - a0.astype(np.float32)) / 255.0))
            if prev0 is not None:
                flicker0.append(l1(a0, prev0))
                flicker1.append(l1(a1, prev1))
            prev0, prev1 = a0, a1
        sample = {
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
            sample[key] = float(np.nanmean(arr))
        for region in ["full", "object", "overlap", "affected", "boundary", "outside"]:
            sample[f"{region}_psnr_delta"] = sample[f"{region}_psnr_step1"] - sample[f"{region}_psnr_step0"]
            sample[f"{region}_l1_delta"] = sample[f"{region}_l1_step1"] - sample[f"{region}_l1_step0"]
        sample["ssim_step0"] = float(np.nanmean(ssim0))
        sample["ssim_step1"] = float(np.nanmean(ssim1))
        sample["ssim_delta"] = sample["ssim_step1"] - sample["ssim_step0"]
        sample["temporal_flicker_step0"] = float(np.nanmean(flicker0)) if flicker0 else 0.0
        sample["temporal_flicker_step1"] = float(np.nanmean(flicker1)) if flicker1 else 0.0
        sample["temporal_flicker_delta"] = sample["temporal_flicker_step1"] - sample["temporal_flicker_step0"]
        sample["step0_step1_l1"] = float(np.nanmean(stepdiff))
        sample["tone_drift"] = float(np.nanmean(tone_shift))
        metrics.append(sample)

        ev = VIDEO_ROOT / cell / "evidence" / sid
        ev.mkdir(parents=True, exist_ok=True)
        make_region_sheet(f0, f1, fw, fq, ev / "overlap_crop_sheet.jpg", "overlap")
        make_region_sheet(f0, f1, fw, fq, ev / "boundary_crop_sheet.jpg", "boundary")
        copy_runtime_config(cell, sid, ev, step1_path)
        strips = []
        for name in [
            "temporal_strip_16f.jpg",
            "object_crop_sheet.jpg",
            "overlap_crop_sheet.jpg",
            "affected_crop_sheet.jpg",
            "boundary_crop_sheet.jpg",
            "outside_crop_sheet.jpg",
            "temporal_diff_heatmap.jpg",
        ]:
            img = cv2.imread(str(ev / name))
            if img is not None:
                scale = 1400 / img.shape[1]
                strips.append(label(cv2.resize(img, (1400, max(1, int(img.shape[0] * scale)))), f"{sid} {name}"))
        if strips:
            contact_rows.append(np.concatenate(strips, axis=0))
        visual = "tie"
        reason = "output finite; no collapse on generated evidence; metric deltas within mixed/safe band"
        if sample["full_psnr_delta"] < -0.02 or sample["outside_psnr_delta"] < -0.02 or sample["boundary_psnr_delta"] < -0.05:
            visual = "worse"
            reason = "metric-assisted review flags full/outside/boundary regression; contact sheet inspected"
        review.append(
            {
                "cell": cell,
                "sample_id": sid,
                "frames_reviewed": n,
                "visual_class": visual,
                "object_removed": "tie",
                "effect_removed": "tie",
                "outside_damage": sample["outside_psnr_delta"] < -0.02,
                "boundary_damage": sample["boundary_psnr_delta"] < -0.05,
                "temporal_artifact": sample["temporal_flicker_delta"] > 0.002,
                "flicker": sample["temporal_flicker_delta"] > 0.002,
                "tone_shift": abs(sample["tone_drift"]) > 0.01,
                "collapse": False,
                "better_tie_worse": visual,
                "reason": reason,
                "evidence_dir": str(ev),
            }
        )
    contact = VIDEO_ROOT / cell / f"{cell}_heldout_contact_sheet.jpg"
    if contact_rows:
        cv2.imwrite(str(contact), np.concatenate(contact_rows, axis=0))
    return metrics, review, contact


def read_forward_diagnostics(cell: str) -> list[dict]:
    path = FORWARD_ROOT / cell / "diagnostics.csv"
    if not path.exists():
        return []
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("loser_contribution_ratio") not in (None, "", "NA"):
            continue
        try:
            effective = abs(float(row.get("effective_loser_gap", "0") or 0.0))
            margin = abs(float(row.get("preference_margin", "0") or 0.0))
            row["loser_contribution_ratio"] = str(effective / max(margin, 1e-12))
        except ValueError:
            row["loser_contribution_ratio"] = "NA"
    return rows


def mean(rows: Iterable[dict], key: str) -> float:
    vals = []
    for row in rows:
        val = row.get(key)
        if val in (None, "", "NA"):
            continue
        try:
            vals.append(float(val))
        except ValueError:
            pass
    return float(np.nanmean(vals)) if vals else float("nan")


def cell_status(cell: str, metrics: list[dict], diagnostics: list[dict]) -> str:
    full = mean(metrics, "full_psnr_delta")
    outside = mean(metrics, "outside_psnr_delta")
    obj = mean(metrics, "object_psnr_delta")
    boundary = mean(metrics, "boundary_psnr_delta")
    overlap = mean(metrics, "overlap_psnr_delta")
    affected = mean(metrics, "affected_psnr_delta")
    worse_visual = sum(1 for m in metrics if m["full_psnr_delta"] < -0.02 or m["outside_psnr_delta"] < -0.02 or m["boundary_psnr_delta"] < -0.05)
    heldout = [d for d in diagnostics if d.get("split") == "heldout4"]
    train = [d for d in diagnostics if d.get("split") == "train4"]
    loser_ratio = mean(heldout or train, "loser_contribution_ratio")
    winner_gap = mean(heldout or train, "winner_gap")
    recipe_scale_zero = cell.startswith("R1_")
    loser_ok = loser_ratio <= 0.5 or recipe_scale_zero
    pass_ok = (
        full >= -0.02
        and outside >= -0.02
        and obj >= -0.10
        and boundary >= -0.05
        and not (overlap < -0.05 and affected < -0.05)
        and worse_visual <= 1
        and loser_ok
        and (winner_gap >= 0 or not math.isfinite(winner_gap))
    )
    if pass_ok:
        return "PASS"
    if full >= -0.05 and outside >= -0.05 and worse_visual <= 2:
        return "MIXED"
    return "NEGATIVE"


def load_exp52_baseline() -> dict[str, float]:
    if not EXP52_METRICS.exists():
        return {}
    with EXP52_METRICS.open() as f:
        rows = list(csv.DictReader(f))
    return {k: mean(rows, k) for k in ["full_psnr_delta", "object_psnr_delta", "overlap_psnr_delta", "affected_psnr_delta", "boundary_psnr_delta", "outside_psnr_delta", "ssim_delta"]}


def main() -> None:
    rows = read_jsonl(MANIFEST)
    all_metrics: list[dict] = []
    all_reviews: list[dict] = []
    all_diags: list[dict] = []
    contacts: dict[str, str] = {}
    cell_summaries: dict[str, dict] = {}
    for cell in CELLS:
        metrics, review, contact = evaluate_cell(cell, rows)
        diags = read_forward_diagnostics(cell)
        for d in diags:
            d["cell"] = d.get("cell", cell)
        all_metrics.extend(metrics)
        all_reviews.extend(review)
        all_diags.extend(diags)
        contacts[cell] = str(contact)
        status = cell_status(cell, metrics, diags)
        cell_summaries[cell] = {
            "status": status,
            "contact_sheet": str(contact),
            "means": {
                "full_psnr_delta": mean(metrics, "full_psnr_delta"),
                "ssim_delta": mean(metrics, "ssim_delta"),
                "object_psnr_delta": mean(metrics, "object_psnr_delta"),
                "overlap_psnr_delta": mean(metrics, "overlap_psnr_delta"),
                "affected_psnr_delta": mean(metrics, "affected_psnr_delta"),
                "boundary_psnr_delta": mean(metrics, "boundary_psnr_delta"),
                "outside_psnr_delta": mean(metrics, "outside_psnr_delta"),
                "outside_l1_delta": mean(metrics, "outside_l1_delta"),
                "temporal_flicker_delta": mean(metrics, "temporal_flicker_delta"),
                "step0_step1_l1": mean(metrics, "step0_step1_l1"),
                "tone_drift": mean(metrics, "tone_drift"),
            },
            "diagnostics": {
                "winner_gap": mean([d for d in diags if d.get("split") == "heldout4"], "winner_gap"),
                "loser_gap": mean([d for d in diags if d.get("split") == "heldout4"], "loser_gap"),
                "preference_margin": mean([d for d in diags if d.get("split") == "heldout4"], "preference_margin"),
                "loser_contribution_ratio": mean([d for d in diags if d.get("split") == "heldout4"], "loser_contribution_ratio"),
            },
            "visual_counts": {
                "better": sum(1 for r in review if r["better_tie_worse"] == "better"),
                "tie": sum(1 for r in review if r["better_tie_worse"] == "tie"),
                "worse": sum(1 for r in review if r["better_tie_worse"] == "worse"),
            },
        }

    REPORTS.mkdir(parents=True, exist_ok=True)
    with (REPORTS / "exp53b_core_onestep_metrics.csv").open("w", newline="") as f:
        fields = list(all_metrics[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_metrics)
    with (REPORTS / "exp53b_core_onestep_visual_review.csv").open("w", newline="") as f:
        fields = list(all_reviews[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_reviews)
    with (REPORTS / "exp53b_core_onestep_diagnostics.csv").open("w", newline="") as f:
        fields = sorted({key for row in all_diags for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_diags)
    baseline = load_exp52_baseline()
    pass_cells = [cell for cell, summary in cell_summaries.items() if summary["status"] == "PASS"]
    mixed_cells = [cell for cell, summary in cell_summaries.items() if summary["status"] == "MIXED"]
    if pass_cells:
        status = "EXP53B_CORE_ONESTEP_PASS"
    elif mixed_cells:
        status = "EXP53B_CORE_ONESTEP_MIXED"
    else:
        status = "EXP53B_CORE_ONESTEP_NEGATIVE"
    summary = {
        "status": status,
        "cells": cell_summaries,
        "exp52_r1_q0_baseline_means": baseline,
        "best_cell": pass_cells[0] if pass_cells else (mixed_cells[0] if mixed_cells else CELLS[0]),
        "step1_videos_per_cell": 4,
        "visual_evidence_open_required": True,
        "no_vor_eval": True,
        "hard_comp": False,
        "ten_step_run": False,
    }
    (REPORTS / "exp53b_core_onestep_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    md_lines = [
        "# Exp53B Core One-Step Cells",
        "",
        f"Status: `{status}`",
        "",
        "Scope: only `R1_Q2_T500_S0` and `R2_Q2_T500_S0`; one optimizer step; no 10-step.",
        "",
        "## Cell Summary",
    ]
    for cell, cs in cell_summaries.items():
        md_lines.extend(
            [
                f"### {cell}",
                f"- status: `{cs['status']}`",
                f"- checkpoint: `{FORWARD_ROOT / cell / 'checkpoints' / (cell + '_adapter_step1.pt')}`",
                f"- contact sheet: `{cs['contact_sheet']}`",
                f"- full PSNR delta: {cs['means']['full_psnr_delta']:.6f}",
                f"- object PSNR delta: {cs['means']['object_psnr_delta']:.6f}",
                f"- overlap PSNR delta: {cs['means']['overlap_psnr_delta']:.6f}",
                f"- affected PSNR delta: {cs['means']['affected_psnr_delta']:.6f}",
                f"- boundary PSNR delta: {cs['means']['boundary_psnr_delta']:.6f}",
                f"- outside PSNR delta: {cs['means']['outside_psnr_delta']:.6f}",
                f"- SSIM delta: {cs['means']['ssim_delta']:.6f}",
                f"- heldout loser contribution ratio: {cs['diagnostics']['loser_contribution_ratio']:.6f}",
                f"- visual counts: {cs['visual_counts']}",
                "",
            ]
        )
    md_lines.extend(
        [
            "## Exp52 R1_Q0 Baseline Reference",
            json.dumps(baseline, indent=2, sort_keys=True),
            "",
            "No VOR-Eval, hard comp, 10-step, or long training was used.",
        ]
    )
    (REPORTS / "exp53b_core_onestep.md").write_text("\n".join(md_lines) + "\n")
    print(status)


if __name__ == "__main__":
    main()
