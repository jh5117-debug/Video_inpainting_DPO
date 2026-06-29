#!/usr/bin/env python3
"""Audit Exp46 pseudo-success teacher quality without training."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
EXP46_ROOT = Path("/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp46_h20_minimax_pseudosuccess_sft")
STEP0_CACHE = EXP46_ROOT / "sft_ladder" / "_step0_cache"
OUT_REVIEW = REPO / "reports" / "exp47_teacher_review_pages"
FRAME_EXTS = ("*.png", "*.jpg", "*.jpeg")


def list_frames(path: Path) -> List[Path]:
    files: List[Path] = []
    for pat in FRAME_EXTS:
        files.extend(path.glob(pat))
    return sorted(files)


def load_stack(path: Path, rgb: bool = True, indices: np.ndarray | None = None) -> np.ndarray:
    files = list_frames(path)
    if not files:
        raise FileNotFoundError(f"no frames under {path}")
    if indices is not None:
        files = [files[int(i)] for i in indices]
    arrs = []
    for f in files:
        im = Image.open(f).convert("RGB" if rgb else "L")
        arrs.append(np.asarray(im, dtype=np.float32) / 255.0)
    return np.stack(arrs, axis=0)


def mask_stack(path: Path, indices: np.ndarray | None = None) -> np.ndarray:
    return load_stack(path, rgb=False, indices=indices) > 0.5


def shift_bool(x: np.ndarray, dy: int, dx: int, fill: bool) -> np.ndarray:
    out = np.full_like(x, fill)
    src_y0 = max(0, -dy)
    src_y1 = x.shape[-2] - max(0, dy)
    src_x0 = max(0, -dx)
    src_x1 = x.shape[-1] - max(0, dx)
    dst_y0 = max(0, dy)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = max(0, dx)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    out[..., dst_y0:dst_y1, dst_x0:dst_x1] = x[..., src_y0:src_y1, src_x0:src_x1]
    return out


def dilate(x: np.ndarray, radius: int = 4) -> np.ndarray:
    out = np.zeros_like(x, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                out |= shift_bool(x, dy, dx, False)
    return out


def erode(x: np.ndarray, radius: int = 3) -> np.ndarray:
    out = np.ones_like(x, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                out &= shift_bool(x, dy, dx, True)
    return out


def region_values(a: np.ndarray, b: np.ndarray, region: np.ndarray | None) -> Tuple[np.ndarray, np.ndarray]:
    if region is None:
        return a.reshape(-1, 3), b.reshape(-1, 3)
    if not np.any(region):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    return a[region], b[region]


def psnr_from_mse(mse: float) -> float:
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def simple_ssim(a: np.ndarray, b: np.ndarray, region: np.ndarray | None = None) -> float:
    av, bv = region_values(a, b, region)
    if av.size == 0:
        return float("nan")
    x = av.reshape(-1).astype(np.float64)
    y = bv.reshape(-1).astype(np.float64)
    ux, uy = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cxy = ((x - ux) * (y - uy)).mean()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    return float(((2 * ux * uy + c1) * (2 * cxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2)))


def pair_metrics(a: np.ndarray, b: np.ndarray, region: np.ndarray | None = None) -> Dict[str, float]:
    av, bv = region_values(a, b, region)
    if av.size == 0:
        return {"mse": float("nan"), "psnr": float("nan"), "l1": float("nan"), "ssim": float("nan")}
    diff = av - bv
    mse = float(np.mean(diff * diff))
    return {"mse": mse, "psnr": psnr_from_mse(mse), "l1": float(np.mean(np.abs(diff))), "ssim": simple_ssim(a, b, region) if region is None else float("nan")}


def block_average(x: np.ndarray, grid: int = 16) -> np.ndarray:
    t, h, w, c = x.shape
    gh = h // grid
    gw = w // grid
    cropped = x[:, : gh * grid, : gw * grid, :]
    return cropped.reshape(t, grid, gh, grid, gw, c).mean(axis=(2, 4))


def hist_distance(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    total = 0.0
    for ch in range(3):
        ha, _ = np.histogram(a[..., ch], bins=bins, range=(0.0, 1.0), density=True)
        hb, _ = np.histogram(b[..., ch], bins=bins, range=(0.0, 1.0), density=True)
        ha = ha / (ha.sum() + 1e-12)
        hb = hb / (hb.sum() + 1e-12)
        total += 0.5 * float(np.abs(ha - hb).sum())
    return total / 3.0


def temporal_flicker_delta(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(a, axis=0) - np.diff(b, axis=0))))


def add_prefixed(out: Dict[str, object], prefix: str, vals: Dict[str, float]) -> None:
    for k, v in vals.items():
        out[f"{prefix}_{k}"] = v


def classify(metrics: Dict[str, object]) -> str:
    outside_psnr = float(metrics["target_gt_outside_psnr"])
    outside_l1 = float(metrics["target_gt_outside_l1"])
    boundary_psnr = float(metrics["target_gt_boundary_psnr"])
    mask_gain = float(metrics["mask_removal_psnr_gain"])
    brightness = abs(float(metrics["global_brightness_delta"]))
    contrast = abs(float(metrics["global_contrast_delta"]))
    hist = float(metrics["color_hist_distance"])
    lowfreq = float(metrics["lowfreq_l1"])
    if outside_psnr < 29.0 or outside_l1 > 0.040:
        return "PSEUDO_TARGET_OUTSIDE_BAD"
    if boundary_psnr < 23.0:
        return "PSEUDO_TARGET_BOUNDARY_BAD"
    if outside_psnr < 33.0 or outside_l1 > 0.028 or brightness > 0.020 or contrast > 0.020 or hist > 0.16 or lowfreq > 0.020:
        if mask_gain > 0.0:
            return "PSEUDO_TARGET_GLOBAL_DRIFT"
        return "PSEUDO_TARGET_BAD_TEACHER"
    if float(metrics["target_condition_full_psnr"]) > 42.0 and mask_gain < 0.5:
        return "PSEUDO_TARGET_TOO_CLOSE"
    if outside_psnr >= 36.0 and outside_l1 <= 0.018 and brightness <= 0.012 and hist <= 0.10 and lowfreq <= 0.012 and boundary_psnr >= 27.0 and mask_gain >= 0.5:
        return "PSEUDO_TARGET_CLEAN_STRICT"
    if mask_gain > 0.0 and outside_psnr >= 32.0:
        return "PSEUDO_TARGET_USABLE_LOCAL_ONLY"
    return "PSEUDO_TARGET_QUALITY_INSUFFICIENT"


def load_manifest(split: str) -> List[Dict[str, str]]:
    path = REPO / "manifests" / f"exp46_runner_pseudosuccess_{split}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def make_sheet(row: Dict[str, str], cond: np.ndarray, gt: np.ndarray, pseudo: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    idxs = sorted(set([0, cond.shape[0] // 2, cond.shape[0] - 1]))
    thumb_w = 160
    thumb_h = 160
    label_h = 24
    title_h = 34
    stacks = [("condition", cond), ("V_bg", gt), ("pseudo_success", pseudo), ("mask", np.repeat(mask[..., None], 3, axis=-1).astype(np.float32))]
    canvas = Image.new("RGB", (len(idxs) * thumb_w, title_h + len(stacks) * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 6), f"{row['split']} {row['sample_id']} {row['source_group']}"[:120], fill=(0, 0, 0))
    y = title_h
    for label, stack in stacks:
        draw.text((4, y + 4), label, fill=(0, 0, 0))
        yy = y + label_h
        for col, idx in enumerate(idxs):
            im = Image.fromarray(np.clip(stack[idx] * 255.0, 0, 255).astype(np.uint8)).resize((thumb_w, thumb_h), Image.BILINEAR)
            canvas.paste(im, (col * thumb_w, yy))
            draw.text((col * thumb_w + 4, yy + 4), f"f{idx:02d}", fill=(255, 255, 0))
        y += thumb_h + label_h
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def make_montage(items: List[Tuple[Dict[str, str], Path, str]], out_path: Path) -> None:
    if not items:
        return
    sheets = [Image.open(p).convert("RGB") for _, p, _ in items]
    w = max(im.width for im in sheets)
    h = max(im.height for im in sheets)
    page = Image.new("RGB", (w * 2, h * math.ceil(len(sheets) / 2)), "white")
    draw = ImageDraw.Draw(page)
    for i, ((row, _p, label), im) in enumerate(zip(items, sheets)):
        x = (i % 2) * w
        y = (i // 2) * h
        page.paste(im, (x, y))
        color = (180, 0, 0) if "DRIFT" in label or "BAD" in label else (0, 120, 0)
        draw.text((x + 4, y + h - 18), label, fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(out_path)


def audit_row(row: Dict[str, str]) -> Tuple[Dict[str, object], Path]:
    condition_path = Path(row["condition_path"])
    gt_path = Path(row["loser_path"])
    pseudo_path = Path(row["winner_path"])
    mask_path = Path(row["mask_path"])
    step0_path = STEP0_CACHE / row["split"] / row["sample_id"] / "frames"
    n = min(len(list_frames(condition_path)), len(list_frames(gt_path)), len(list_frames(pseudo_path)), len(list_frames(mask_path)), len(list_frames(step0_path)))
    sample_idx = np.array(sorted(set([0, n // 4, n // 2, (3 * n) // 4, n - 1])), dtype=np.int64)
    cond = load_stack(condition_path, indices=sample_idx)[:, ::4, ::4, :]
    gt = load_stack(gt_path, indices=sample_idx)[:, ::4, ::4, :]
    pseudo = load_stack(pseudo_path, indices=sample_idx)[:, ::4, ::4, :]
    mask = mask_stack(mask_path, indices=sample_idx)[:, ::4, ::4]
    step0 = load_stack(step0_path, indices=sample_idx)[:, ::4, ::4, :]
    n = int(cond.shape[0])
    boundary = dilate(mask, 1) & ~erode(mask, 1)
    affected = (np.mean(np.abs(cond - gt), axis=-1) > 0.035) | mask
    outside = ~dilate(mask | affected, 2)
    out: Dict[str, object] = {
        "split": row["split"], "row_id": row.get("row_id", ""), "sample_id": row["sample_id"],
        "source_group": row["source_group"], "source_id": row["source_id"],
        "condition_path": row["condition_path"], "gt_winner_path": row["loser_path"],
        "pseudo_success_path": row["winner_path"], "mask_path": row["mask_path"], "step0_path": str(step0_path),
        "num_frames": n, "metric_sampling": "frames_0_quarter_mid_3quarter_last_spatial_stride4", "mask_area_frac": float(mask.mean()), "boundary_area_frac": float(boundary.mean()),
        "affected_area_frac": float(affected.mean()), "outside_area_frac": float(outside.mean()),
        "lpips_status": "not_available_on_h20_env", "ewarp_status": "proxy_temporal_flicker_delta_reported",
    }
    for name, region in {"full": None, "mask": mask, "boundary": boundary, "affected": affected, "outside": outside}.items():
        add_prefixed(out, f"target_gt_{name}", pair_metrics(pseudo, gt, region))
        add_prefixed(out, f"target_condition_{name}", pair_metrics(pseudo, cond, region))
        add_prefixed(out, f"condition_gt_{name}", pair_metrics(cond, gt, region))
        add_prefixed(out, f"step0_gt_{name}", pair_metrics(step0, gt, region))
    out["global_rgb_mean_delta_r"] = float((pseudo[..., 0] - gt[..., 0]).mean())
    out["global_rgb_mean_delta_g"] = float((pseudo[..., 1] - gt[..., 1]).mean())
    out["global_rgb_mean_delta_b"] = float((pseudo[..., 2] - gt[..., 2]).mean())
    out["global_brightness_delta"] = float(pseudo.mean() - gt.mean())
    out["global_contrast_delta"] = float(pseudo.std() - gt.std())
    out["color_hist_distance"] = hist_distance(pseudo, gt)
    out["lowfreq_l1"] = float(np.mean(np.abs(block_average(pseudo) - block_average(gt))))
    out["temporal_flicker_delta"] = temporal_flicker_delta(pseudo, gt)
    out["outside_identity_score"] = out["target_gt_outside_psnr"]
    out["boundary_identity_score"] = out["target_gt_boundary_psnr"]
    out["mask_removal_psnr_gain"] = float(out["target_gt_mask_psnr"] - out["condition_gt_mask_psnr"])
    out["mask_removal_l1_gain"] = float(out["condition_gt_mask_l1"] - out["target_gt_mask_l1"])
    out["teacher_label"] = classify(out)
    out["codex_review"] = "pending_contact_sheet_inspection"
    sheet = OUT_REVIEW / row["split"] / f"{row['sample_id']}.png"
    make_sheet(row, cond, gt, pseudo, mask, sheet)
    out["review_sheet"] = str(sheet)
    return out, sheet


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    labels = Counter(str(r["teacher_label"]) for r in rows)
    by_split: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_split[str(r["split"])][str(r["teacher_label"])] += 1
    def mean(key: str) -> float:
        vals = [float(r[key]) for r in rows if key in r and not math.isnan(float(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")
    clean = labels.get("PSEUDO_TARGET_CLEAN_STRICT", 0)
    drift = labels.get("PSEUDO_TARGET_GLOBAL_DRIFT", 0) + labels.get("PSEUDO_TARGET_OUTSIDE_BAD", 0)
    local = labels.get("PSEUDO_TARGET_USABLE_LOCAL_ONLY", 0)
    if clean >= 32:
        status = "EXP47_TEACHER_CLEAN_ENOUGH"
    elif drift >= len(rows) // 2:
        status = "EXP47_TEACHER_GLOBAL_DRIFT_CONFIRMED"
    elif local + drift >= len(rows) // 2:
        status = "EXP47_TEACHER_LOCAL_ONLY"
    else:
        status = "EXP47_TEACHER_QUALITY_INSUFFICIENT"
    return {
        "status": status,
        "rows": len(rows),
        "label_counts": dict(labels),
        "label_counts_by_split": {k: dict(v) for k, v in by_split.items()},
        "means": {
            "target_gt_full_psnr": mean("target_gt_full_psnr"),
            "target_gt_mask_psnr": mean("target_gt_mask_psnr"),
            "target_gt_boundary_psnr": mean("target_gt_boundary_psnr"),
            "target_gt_outside_psnr": mean("target_gt_outside_psnr"),
            "target_gt_outside_l1": mean("target_gt_outside_l1"),
            "global_brightness_delta_abs": float(np.mean([abs(float(r["global_brightness_delta"])) for r in rows])),
            "global_contrast_delta_abs": float(np.mean([abs(float(r["global_contrast_delta"])) for r in rows])),
            "color_hist_distance": mean("color_hist_distance"),
            "lowfreq_l1": mean("lowfreq_l1"),
            "temporal_flicker_delta": mean("temporal_flicker_delta"),
            "mask_removal_psnr_gain": mean("mask_removal_psnr_gain"),
            "step0_gt_full_psnr": mean("step0_gt_full_psnr"),
        },
    }


def write_outputs(rows: List[Dict[str, object]], summary: Dict[str, object]) -> None:
    reports = REPO / "reports"
    csv_path = reports / "exp47_pseudosuccess_teacher_quality_audit.csv"
    visual_csv = reports / "exp47_pseudosuccess_teacher_visual_review.csv"
    json_path = reports / "exp47_pseudosuccess_teacher_quality_summary.json"
    md_path = reports / "exp47_pseudosuccess_teacher_quality_audit.md"
    keys = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    with visual_csv.open("w", newline="") as f:
        fields = ["split", "sample_id", "source_group", "teacher_label", "review_sheet", "contact_page", "codex_review", "visual_note"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields} | {"visual_note": "generated for blind/informed inspection; class uses aligned frame metrics"})
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    label_lines = "\n".join(f"- `{k}`: `{v}`" for k, v in sorted(summary["label_counts"].items()))
    means = summary["means"]
    md_path.write_text(f"""# Exp47 Pseudo-Success Teacher Quality Audit

Status: `{summary['status']}`

Rows audited: `{summary['rows']}` search/shadow rows (`24/24`). Metrics use start/quarter/mid/three-quarter/end frames with spatial stride 4 for forensic speed; this audit is read-only and frame-space only; it does not train, run DPO, run GT-only SFT, or perform an optimizer step.

## Label Counts

{label_lines}

## Mean Metrics

- pseudo target vs V_bg full PSNR: `{means['target_gt_full_psnr']:.6f}`
- pseudo target vs V_bg mask PSNR: `{means['target_gt_mask_psnr']:.6f}`
- pseudo target vs V_bg boundary PSNR: `{means['target_gt_boundary_psnr']:.6f}`
- pseudo target vs V_bg outside PSNR: `{means['target_gt_outside_psnr']:.6f}`
- pseudo target vs V_bg outside L1: `{means['target_gt_outside_l1']:.6f}`
- absolute brightness delta: `{means['global_brightness_delta_abs']:.6f}`
- absolute contrast delta: `{means['global_contrast_delta_abs']:.6f}`
- color histogram distance: `{means['color_hist_distance']:.6f}`
- low-frequency L1 drift proxy: `{means['lowfreq_l1']:.6f}`
- temporal flicker delta proxy: `{means['temporal_flicker_delta']:.6f}`
- mask removal PSNR gain over condition: `{means['mask_removal_psnr_gain']:.6f}`
- Step0 vs V_bg full PSNR mean: `{means['step0_gt_full_psnr']:.6f}`

LPIPS was not available in the H20 Python environment. Ewarp is represented here by a temporal flicker delta proxy; Exp46 official Ewarp deltas remain recorded in Exp46 reports.

## Interpretation

Rows labelled `PSEUDO_TARGET_GLOBAL_DRIFT` or `PSEUDO_TARGET_OUTSIDE_BAD` are unsafe for full-video global SFT because they can teach tone/outside drift even when local removal succeeds. Rows labelled `PSEUDO_TARGET_USABLE_LOCAL_ONLY` may still be useful for localized pseudo-success targets or same-source preference, but not as global SFT targets without localization.

Outputs:

- `reports/exp47_pseudosuccess_teacher_quality_audit.csv`
- `reports/exp47_pseudosuccess_teacher_visual_review.csv`
- `reports/exp47_pseudosuccess_teacher_quality_summary.json`
- `reports/exp47_teacher_review_pages/`
""")


def main() -> None:
    OUT_REVIEW.mkdir(parents=True, exist_ok=True)
    all_rows = load_manifest("search") + load_manifest("shadow")
    audited: List[Dict[str, object]] = []
    split_items: Dict[str, List[Tuple[Dict[str, str], Path, str]]] = defaultdict(list)
    for idx, row in enumerate(all_rows, 1):
        print(f"teacher_audit {idx}/{len(all_rows)} {row['split']} {row['sample_id']}", flush=True)
        rec, sheet = audit_row(row)
        audited.append(rec)
        split_items[row["split"]].append((row, sheet, str(rec["teacher_label"])))
    contact_dir = OUT_REVIEW / "contact_pages"
    for split, items in split_items.items():
        for page_idx in range(0, len(items), 12):
            page = contact_dir / f"{split}_page_{page_idx // 12:02d}.png"
            page_items = items[page_idx:page_idx + 12]
            make_montage(page_items, page)
            sample_ids = {row["sample_id"] for row, _sheet, _label in page_items}
            for rec in audited:
                if rec["sample_id"] in sample_ids:
                    rec["contact_page"] = str(page)
    summary = summarize(audited)
    summary["review_pages"] = sorted(str(p) for p in contact_dir.glob("*.png"))
    write_outputs(audited, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
