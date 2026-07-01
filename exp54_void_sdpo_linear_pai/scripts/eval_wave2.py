#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/home/hj/H20_Video_inpainting_DPO_exp54_void_sdpo_linear_pai")
REPORTS = ROOT / "reports"
OUT_BASE = Path("/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp54_void_sdpo_linear_pai/outputs")
VIDEO_BASE = OUT_BASE / "wave2_video"
FORWARD_BASE = OUT_BASE / "wave1_forward"
MANIFEST = ROOT / "manifests/exp50_void_adapter_heldout4.jsonl"
STEP0_BASE = Path("/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp50_pai_void_adapter_feasibility/f2_vor_gate8_pass1")
CELLS = ["R3_Q1_T500_S0", "R4_Q1_T500_S0"]


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
    return cv2.resize(fr, (w, h), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


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
    local = (g <= 191).astype(np.uint8)
    dil = cv2.dilate(local, kernel, iterations=1).astype(bool)
    ero = cv2.erode(local, kernel, iterations=1).astype(bool)
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


def find_step1(cell: str, sample_id: str) -> Path:
    hits = sorted((VIDEO_BASE / cell).glob(f"step1_gpu*/{sample_id}-fg=-1-0001.mp4"))
    if not hits:
        raise FileNotFoundError(f"{cell} {sample_id}")
    return hits[0]


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 420), 30), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def crop_bbox(mask: np.ndarray, shape: tuple[int, int], pad: int = 32) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    h, w = shape
    if len(xs) == 0:
        return 0, 0, w, h
    return max(0, xs.min() - pad), max(0, ys.min() - pad), min(w, xs.max() + pad + 1), min(h, ys.max() + pad + 1)


def write_crop_sheet(path: Path, title: str, frames_by_name: dict[str, list[np.ndarray]], idxs: list[int], bbox: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = bbox
    rows = []
    for name, frames in frames_by_name.items():
        imgs = [label(frames[i][y0:y1, x0:x1], f"{title} {name}") for i in idxs]
        rows.append(np.concatenate(imgs, axis=1))
    cv2.imwrite(str(path), np.concatenate(rows, axis=0))


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    rows = read_jsonl(MANIFEST)
    metric_rows: list[dict] = []
    visual_rows: list[dict] = []
    cell_summaries: dict[str, dict] = {}
    for cell in CELLS:
        contact_parts = []
        for row in rows:
            sid = row["sample_id"]
            step0 = STEP0_BASE / f"{sid}-fg=-1-0001.mp4"
            step1 = find_step1(cell, sid)
            winner = Path(row["winner_path"])
            condition = Path(row["condition_path"])
            quad = Path(row["quadmask_0_path"])
            f0, f1, fw, fc, fq = decode(step0), decode(step1), decode(winner), decode(condition), decode(quad)
            n = min(len(f0), len(f1), len(fw), len(fc), len(fq))
            shape = f1[0].shape[:2]
            vals = {f"{region}_{metric}_{step}": [] for region in ["full", "object", "overlap", "affected", "affected_union", "boundary", "outside"] for metric in ["psnr", "l1"] for step in ["step0", "step1"]}
            ssim0, ssim1, stepdiff = [], [], []
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
            rec = {"cell": cell, "sample_id": sid, "frames": n, "step0_raw": str(step0), "step1_raw": str(step1)}
            for key, arr in vals.items():
                rec[key] = float(np.nanmean(arr))
            rec["ssim_step0"] = float(np.nanmean(ssim0))
            rec["ssim_step1"] = float(np.nanmean(ssim1))
            rec["step0_step1_l1"] = float(np.nanmean(stepdiff))
            for region in ["full", "object", "overlap", "affected", "affected_union", "boundary", "outside"]:
                rec[f"{region}_psnr_delta"] = rec[f"{region}_psnr_step1"] - rec[f"{region}_psnr_step0"]
                rec[f"{region}_l1_delta"] = rec[f"{region}_l1_step1"] - rec[f"{region}_l1_step0"]
            rec["ssim_delta"] = rec["ssim_step1"] - rec["ssim_step0"]
            metric_rows.append(rec)

            ev = VIDEO_BASE / cell / "evidence" / sid
            idxs = [0, n // 2, n - 1]
            resized = {
                "condition": [resize(x, shape) for x in fc],
                "step0": [resize(x, shape) for x in f0],
                "step1": [resize(x, shape) for x in f1],
                "winner": [resize(x, shape) for x in fw],
            }
            qmid = resize(fq[n // 2], shape, nearest=True)
            mmid = masks(qmid)
            write_crop_sheet(ev / "overlap_crop_sheet.jpg", "overlap", resized, idxs, crop_bbox(mmid["overlap"], shape))
            write_crop_sheet(ev / "boundary_crop_sheet.jpg", "boundary", resized, idxs, crop_bbox(mmid["boundary"], shape))
            strips = []
            for name in ["temporal_strip_16f.jpg", "object_crop_sheet.jpg", "overlap_crop_sheet.jpg", "affected_crop_sheet.jpg", "boundary_crop_sheet.jpg", "outside_crop_sheet.jpg", "temporal_diff_heatmap.jpg"]:
                img = cv2.imread(str(ev / name))
                if img is not None:
                    width = 1800
                    scale = width / img.shape[1]
                    strips.append(cv2.resize(img, (width, max(1, int(img.shape[0] * scale)))))
            if strips:
                contact_parts.append(label(np.concatenate(strips, axis=0), f"{cell} {sid}"))

            visually_safe = (
                rec["full_psnr_delta"] >= -0.02
                and rec["outside_psnr_delta"] >= -0.02
                and rec["object_psnr_delta"] >= -0.10
                and rec["boundary_psnr_delta"] >= -0.05
                and not (rec["affected_psnr_delta"] < -0.05 and rec["overlap_psnr_delta"] < -0.05)
            )
            visual_class = "tie" if visually_safe else "worse"
            visual_rows.append({
                "cell": cell,
                "sample_id": sid,
                "visual_class": visual_class,
                "object_removed": "metric_tie",
                "effect_removed": "metric_tie" if visually_safe else "metric_mixed",
                "outside_damage": rec["outside_psnr_delta"] < -0.02,
                "boundary_damage": rec["boundary_psnr_delta"] < -0.05,
                "tone_shift": False,
                "collapse": False,
                "reason": "metric-assisted provisional review; Codex opens contact sheets separately",
                "evidence_dir": str(ev),
            })
        if contact_parts:
            contact = VIDEO_BASE / cell / f"{cell}_heldout_contact_sheet.jpg"
            cv2.imwrite(str(contact), np.concatenate(contact_parts, axis=0))

    fields = list(metric_rows[0].keys())
    with (REPORTS / "exp54_wave2_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_rows)
    with (REPORTS / "exp54_wave2_visual_review.csv").open("w", newline="") as f:
        fields2 = list(visual_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields2)
        writer.writeheader()
        writer.writerows(visual_rows)

    diag_rows: list[dict] = []
    for cell in CELLS:
        summary = json.loads((FORWARD_BASE / cell / "summary.json").read_text())
        diag_path = FORWARD_BASE / cell / "diagnostics.csv"
        with diag_path.open(newline="") as f:
            for d in csv.DictReader(f):
                d["forward_status"] = summary["status"]
                d["checkpoint"] = summary["checkpoint"]
                d["max_param_delta_norm"] = summary["max_param_delta_norm"]
                d["peak_vram_reserved_gib"] = summary["peak_vram_reserved_gib"]
                diag_rows.append(d)
    with (REPORTS / "exp54_wave2_diagnostics.csv").open("w", newline="") as f:
        fields3 = sorted({k for r in diag_rows for k in r.keys()})
        writer = csv.DictWriter(f, fieldnames=fields3)
        writer.writeheader()
        writer.writerows(diag_rows)

    summary = {"cells": {}, "no_vor_eval": True, "hard_comp": False, "ten_step": "not_run"}
    any_pass = False
    any_mixed = False
    for cell in CELLS:
        rows_cell = [r for r in metric_rows if r["cell"] == cell]
        means = {}
        for key in rows_cell[0].keys():
            if key.endswith("_delta") or key in ["step0_step1_l1"]:
                try:
                    means[key] = float(np.nanmean([r[key] for r in rows_cell]))
                except Exception:
                    pass
        vrows = [r for r in visual_rows if r["cell"] == cell]
        vcounts = {"better": 0, "tie": 0, "worse": 0}
        for v in vrows:
            vcounts[v["visual_class"]] += 1
        pass_gate = (
            means.get("full_psnr_delta", -999) >= -0.02
            and means.get("outside_psnr_delta", -999) >= -0.02
            and means.get("object_psnr_delta", -999) >= -0.10
            and means.get("boundary_psnr_delta", -999) >= -0.05
            and not (means.get("affected_psnr_delta", -999) < -0.05 and means.get("overlap_psnr_delta", -999) < -0.05)
            and vcounts["worse"] <= 1
        )
        cell_status = "PASS" if pass_gate else "MIXED"
        any_pass = any_pass or pass_gate
        any_mixed = any_mixed or (not pass_gate)
        summary["cells"][cell] = {
            "status": cell_status,
            "means": means,
            "visual_counts": vcounts,
            "contact_sheet": str(VIDEO_BASE / cell / f"{cell}_heldout_contact_sheet.jpg"),
            "generation_summary": str(REPORTS / f"exp54_{cell}_heldout_generation_summary.json"),
        }
    status = "EXP54_WAVE2_PASS" if any_pass else ("EXP54_WAVE2_MIXED" if any_mixed else "EXP54_WAVE2_NEGATIVE")
    summary["status"] = status
    (REPORTS / "exp54_wave2_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    md = ["# Exp54 SDPO / Linear-DPO One-Step", "", f"Status: `{status}`", ""]
    for cell, csum in summary["cells"].items():
        md.append(f"## {cell}")
        md.append("")
        md.append(f"Status: `{csum['status']}`")
        md.append("")
        for k in ["full_psnr_delta", "object_psnr_delta", "overlap_psnr_delta", "affected_psnr_delta", "boundary_psnr_delta", "outside_psnr_delta", "ssim_delta"]:
            md.append(f"- {k}: {csum['means'].get(k)}")
        md.append(f"- visual provisional: {csum['visual_counts']}")
        md.append(f"- contact sheet: `{csum['contact_sheet']}`")
        md.append("")
    md.append("No VOR-Eval, hard comp, 10-step, or third-backbone claim was used.")
    (REPORTS / "exp54_wave2_decision.md").write_text("\n".join(md) + "\n")
    print(status)


if __name__ == "__main__":
    main()
