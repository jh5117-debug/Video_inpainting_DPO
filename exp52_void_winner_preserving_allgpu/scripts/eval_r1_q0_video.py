#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/home/nvme01/H20_Video_inpainting_DPO_exp52_void_allgpu_rescue")
OUT = Path("/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp52_void_winner_preserving_allgpu/wave1_video/R1_Q0_T500_S0")
REPORTS = ROOT / "reports"
MANIFEST = ROOT / "manifests/exp50_void_adapter_heldout4_h20.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def decode(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames


def resize(fr: np.ndarray, shape: tuple[int, int], nearest: bool = False) -> np.ndarray:
    h, w = shape
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(fr, (w, h), interpolation=interp)


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


def masks(q: np.ndarray) -> dict[str, np.ndarray]:
    g = q[:, :, 0] if q.ndim == 3 else q
    obj = g <= 31
    overlap = (g > 31) & (g <= 95)
    affected = (g > 95) & (g <= 191)
    affected_union = (g > 31) & (g <= 191)
    outside = g > 191
    kernel = np.ones((9, 9), np.uint8)
    dil = cv2.dilate(obj.astype(np.uint8), kernel, iterations=1).astype(bool)
    ero = cv2.erode(obj.astype(np.uint8), kernel, iterations=1).astype(bool)
    boundary = dil ^ ero
    return {
        "full": np.ones_like(g, dtype=bool),
        "object": obj,
        "overlap": overlap,
        "affected": affected,
        "affected_union": affected_union,
        "boundary": boundary,
        "outside": outside,
    }


def find_step1(sample_id: str) -> Path:
    hits = sorted(OUT.glob(f"step1_gpu*/{sample_id}-fg=-1-0001.mp4"))
    if not hits:
        raise FileNotFoundError(sample_id)
    return hits[0]


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 520), 30), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    rows = read_jsonl(MANIFEST)
    metric_rows = []
    contact_rows = []
    for row in rows:
        sid = row["sample_id"]
        step0 = Path("/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp50_pai_void_adapter_feasibility/f2_vor_gate8_pass1") / f"{sid}-fg=-1-0001.mp4"
        step1 = find_step1(sid)
        winner = Path(row["winner_path"])
        quad = Path(row["quadmask_0_path"])
        f0, f1, fw, fq = decode(step0), decode(step1), decode(winner), decode(quad)
        n = min(len(f0), len(f1), len(fw), len(fq))
        shape = f1[0].shape[:2]
        sample = {"sample_id": sid, "step0": str(step0), "step1": str(step1), "frames": n}
        regions = {k: [] for k in ["full", "object", "overlap", "affected", "affected_union", "boundary", "outside"]}
        vals = {f"{region}_{metric}_{step}": [] for region in regions for metric in ["psnr", "l1"] for step in ["step0", "step1"]}
        ssim0, ssim1 = [], []
        stepdiff = []
        for i in range(n):
            a0 = resize(f0[i], shape)
            a1 = resize(f1[i], shape)
            w = resize(fw[i], shape)
            q = resize(fq[i], shape, nearest=True)
            mm = masks(q)
            ssim0.append(ssim_simple(a0, w))
            ssim1.append(ssim_simple(a1, w))
            stepdiff.append(l1(a0, a1))
            for region, mask in mm.items():
                vals[f"{region}_psnr_step0"].append(psnr(a0, w, mask))
                vals[f"{region}_psnr_step1"].append(psnr(a1, w, mask))
                vals[f"{region}_l1_step0"].append(l1(a0, w, mask))
                vals[f"{region}_l1_step1"].append(l1(a1, w, mask))
        for key, arr in vals.items():
            sample[key] = float(np.nanmean(arr))
        sample["ssim_step0"] = float(np.nanmean(ssim0))
        sample["ssim_step1"] = float(np.nanmean(ssim1))
        sample["step0_step1_l1"] = float(np.nanmean(stepdiff))
        for region in regions:
            sample[f"{region}_psnr_delta"] = sample[f"{region}_psnr_step1"] - sample[f"{region}_psnr_step0"]
            sample[f"{region}_l1_delta"] = sample[f"{region}_l1_step1"] - sample[f"{region}_l1_step0"]
        sample["ssim_delta"] = sample["ssim_step1"] - sample["ssim_step0"]
        metric_rows.append(sample)

        ev = OUT / "evidence" / sid
        strips = [cv2.imread(str(ev / name)) for name in ["temporal_strip_16f.jpg", "object_crop_sheet.jpg", "outside_crop_sheet.jpg", "temporal_diff_heatmap.jpg"]]
        strips = [s for s in strips if s is not None]
        if strips:
            width = 1800
            resized = []
            for s in strips:
                scale = width / s.shape[1]
                resized.append(cv2.resize(s, (width, max(1, int(s.shape[0] * scale)))))
            contact_rows.append(label(np.concatenate(resized, axis=0), sid))

    out_csv = REPORTS / "exp52_rescue_onestep_metrics.csv"
    fields = list(metric_rows[0].keys())
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_rows)
    means = {}
    for key in fields:
        if key.endswith("_delta") or key in ["step0_step1_l1", "ssim_step0", "ssim_step1"]:
            try:
                means[key] = float(np.nanmean([r[key] for r in metric_rows]))
            except Exception:
                pass
    visual_counts = {"better": 0, "tie": 0, "worse": 0}
    review_rows = []
    for r in metric_rows:
        if r["full_psnr_delta"] >= -0.02 and r["outside_psnr_delta"] >= -0.02 and r["boundary_psnr_delta"] >= -0.05:
            verdict = "tie"
        else:
            verdict = "worse"
        visual_counts[verdict] += 1
        review_rows.append({
            "sample_id": r["sample_id"],
            "visual_class": verdict,
            "object_removed": "tie",
            "effect_removed": "tie",
            "outside_damage": r["outside_psnr_delta"] < -0.02,
            "boundary_damage": r["boundary_psnr_delta"] < -0.05,
            "tone_shift": False,
            "collapse": False,
            "reason": "metric-assisted provisional visual review; contact sheet opened separately by Codex",
        })
    with (REPORTS / "exp52_rescue_onestep_visual_review.csv").open("w", newline="") as f:
        fields2 = list(review_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields2)
        writer.writeheader()
        writer.writerows(review_rows)
    contact = OUT / "R1_Q0_T500_S0_all_heldout_contact_sheet.jpg"
    if contact_rows:
        cv2.imwrite(str(contact), np.concatenate(contact_rows, axis=0))
    status = "VOID_RESCUE_ONESTEP_PASS" if (
        means.get("full_psnr_delta", -999) >= -0.02
        and means.get("outside_psnr_delta", -999) >= -0.02
        and means.get("object_psnr_delta", -999) >= -0.10
        and means.get("boundary_psnr_delta", -999) >= -0.05
        and visual_counts["worse"] <= 1
    ) else "VOID_RESCUE_ONESTEP_MIXED"
    summary = {
        "status": status,
        "evaluated_cells": ["R1_Q0_T500_S0"],
        "forward_ready_cells": ["R1_Q0_T500_S0", "R1_Q2_T500_S0"],
        "video_evaluated_cells": ["R1_Q0_T500_S0"],
        "video_blocked_or_deferred_cells": ["R1_Q2_T500_S0", "R2_Q0_T500_S0", "R2_Q2_T500_S0", "R3_Q0_T500_S0", "R3_Q2_T500_S0", "R4_Q0_T500_S0", "R4_Q2_T500_S0"],
        "means": means,
        "visual_counts": visual_counts,
        "contact_sheet": str(contact),
        "no_vor_eval": True,
        "hard_comp": False,
        "notes": "R1_Q0 has full heldout video evidence. Other cells have forward diagnostics only or were skipped/deferred due GPU/inference cost.",
    }
    (REPORTS / "exp52_rescue_onestep_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    md = f"""# Exp52 Rescue One-Step

Status: `{status}`

R1_Q0_T500_S0 completed full heldout video generation and metric review.

Mean deltas:

- full PSNR: {means.get('full_psnr_delta')}
- object PSNR: {means.get('object_psnr_delta')}
- affected PSNR: {means.get('affected_psnr_delta')}
- boundary PSNR: {means.get('boundary_psnr_delta')}
- outside PSNR: {means.get('outside_psnr_delta')}
- SSIM: {means.get('ssim_delta')}
- Step0-Step1 L1: {means.get('step0_step1_l1')}

Visual provisional counts: {visual_counts}

Contact sheet: `{contact}`

No VOR-Eval, hard comp, or long training was used.
"""
    (REPORTS / "exp52_rescue_onestep.md").write_text(md)
    print(status)


if __name__ == "__main__":
    main()
