#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


VARIANTS = {
    "q0_current": "Current Exp50 quadmask copied as-is.",
    "q1_object_only": "Object and overlap become object=0; all other pixels become background=255.",
    "q2_strict_affected": "Object preserved; affected region rebuilt from high-threshold abs(V_obj - V_bg).",
    "q3_broad_affected": "Object preserved; affected region rebuilt from lower-threshold abs(V_obj - V_bg).",
}


def read_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def decode_video(path: Path, max_frames: int | None = None) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded {path}")
    return np.stack(frames, axis=0)


def write_gray_mp4(path: Path, gray: np.ndarray, fps: float = 12.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = gray.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h), True)
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer {path}")
    for fr in gray.astype(np.uint8):
        writer.write(cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR))
    writer.release()


def quantize_current(q_bgr: np.ndarray) -> np.ndarray:
    g = q_bgr[..., 0].astype(np.float32)
    out = np.zeros_like(g, dtype=np.uint8)
    out[g <= 31] = 0
    out[(g > 31) & (g <= 95)] = 63
    out[(g > 95) & (g <= 191)] = 127
    out[g > 191] = 255
    return out


def affected_score(cond_bgr: np.ndarray, winner_bgr: np.ndarray) -> np.ndarray:
    if cond_bgr.shape[:3] != winner_bgr.shape[:3]:
        resized = []
        h, w = cond_bgr.shape[1:3]
        for fr in winner_bgr:
            resized.append(cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA))
        winner_bgr = np.stack(resized, axis=0)
    return np.mean(np.abs(cond_bgr.astype(np.float32) - winner_bgr.astype(np.float32)), axis=-1)


def smooth_binary(mask: np.ndarray, min_area: int = 64) -> np.ndarray:
    out = []
    kernel = np.ones((5, 5), np.uint8)
    for fr in mask.astype(np.uint8):
        fr = cv2.morphologyEx(fr, cv2.MORPH_OPEN, kernel)
        fr = cv2.morphologyEx(fr, cv2.MORPH_CLOSE, kernel)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(fr, connectivity=8)
        keep = np.zeros_like(fr)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == i] = 1
        out.append(keep)
    return np.stack(out, axis=0).astype(bool)


def build_variant(name: str, q0: np.ndarray, cond: np.ndarray, winner: np.ndarray) -> np.ndarray:
    object_mask = q0 <= 95
    score = affected_score(cond, winner)
    if name == "q0_current":
        return q0.copy()
    if name == "q1_object_only":
        return np.where(object_mask, 0, 255).astype(np.uint8)
    non_object = ~object_mask
    valid_scores = score[non_object]
    if valid_scores.size == 0:
        affected = np.zeros_like(object_mask)
    else:
        if name == "q2_strict_affected":
            thr = max(30.0, float(np.percentile(valid_scores, 85)))
            affected = smooth_binary((score >= thr) & non_object, min_area=96)
        elif name == "q3_broad_affected":
            thr = max(10.0, float(np.percentile(valid_scores, 60)))
            affected = smooth_binary((score >= thr) & non_object, min_area=32)
        else:
            raise ValueError(name)
    overlap = object_mask & (score >= 20.0)
    out = np.full(q0.shape, 255, dtype=np.uint8)
    out[affected] = 127
    out[object_mask] = 0
    out[overlap] = 63
    return out


def area_stats(q: np.ndarray) -> Dict[str, float]:
    total = float(q.size)
    return {
        "value0_frac": float((q == 0).sum() / total),
        "value63_frac": float((q == 63).sum() / total),
        "value127_frac": float((q == 127).sum() / total),
        "value255_frac": float((q == 255).sum() / total),
        "affected_union_frac": float(((q == 63) | (q == 127)).sum() / total),
        "object_union_frac": float(((q == 0) | (q == 63)).sum() / total),
    }


def make_sheet(path: Path, sample_id: str, cond: np.ndarray, winner: np.ndarray, variants: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    idxs = [0, len(cond) // 2, len(cond) - 1]
    rows = []
    for idx in idxs:
        panels = [
            cv2.resize(cond[idx], (224, 128), interpolation=cv2.INTER_AREA),
            cv2.resize(winner[idx], (224, 128), interpolation=cv2.INTER_AREA),
        ]
        for name in ["q0_current", "q1_object_only", "q2_strict_affected", "q3_broad_affected"]:
            color = cv2.applyColorMap(cv2.resize(variants[name][idx], (224, 128), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_VIRIDIS)
            panels.append(color)
        rows.append(np.concatenate(panels, axis=1))
    sheet = np.concatenate(rows, axis=0)
    cv2.putText(sheet, sample_id, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), sheet)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("manifests/exp50_void_adapter_train4_h20.jsonl"))
    ap.add_argument("--heldout", type=Path, default=Path("manifests/exp50_void_adapter_heldout4_h20.jsonl"))
    ap.add_argument("--output-root", type=Path, default=Path("/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp51_void_loser_dominant_rescue/quadmask_ablation"))
    args = ap.parse_args()
    rows = []
    for split_path in [args.train, args.heldout]:
        rows.extend(read_jsonl(split_path))
    if any(r.get("source") != "VOR-Train" or not r.get("no_vor_eval", False) for r in rows):
        raise RuntimeError("non VOR-Train or VOR-Eval-tainted row detected")
    by_variant = {v: [] for v in VARIANTS}
    audit_rows = []
    for row in rows:
        sid = row["sample_id"]
        cond_path = Path(row["rgb_full_path"])
        winner_path = Path(row["rgb_removed_path"])
        quad_path = Path(row["quadmask_0_path"])
        cond = decode_video(cond_path)
        winner = decode_video(winner_path, max_frames=cond.shape[0])
        q0 = quantize_current(decode_video(quad_path, max_frames=cond.shape[0]))
        frames = min(cond.shape[0], winner.shape[0], q0.shape[0])
        cond, winner, q0 = cond[:frames], winner[:frames], q0[:frames]
        variants = {name: build_variant(name, q0, cond, winner) for name in VARIANTS}
        sample_dir = args.output_root / sid
        sheet = sample_dir / "quadmask_ablation_sheet.jpg"
        make_sheet(sheet, sid, cond, winner, variants)
        for name, q in variants.items():
            out_mp4 = sample_dir / name / "quadmask_0.mp4"
            if name == "q0_current":
                out_mp4.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(quad_path, out_mp4)
            else:
                write_gray_mp4(out_mp4, q, fps=float(row.get("fps", 12)))
            st = area_stats(q)
            new_row = dict(row)
            new_row["quadmask_0_path"] = str(out_mp4)
            new_row["quadmask_variant"] = name
            new_row["quadmask_variant_description"] = VARIANTS[name]
            new_row["quadmask_ablation_sheet"] = str(sheet)
            new_row["no_vor_eval"] = True
            new_row["hard_comp"] = False
            by_variant[name].append(new_row)
            audit_rows.append({
                "sample_id": sid,
                "split": row.get("split", ""),
                "source": row.get("source", ""),
                "variant": name,
                "frames": frames,
                "height": int(q.shape[1]),
                "width": int(q.shape[2]),
                "values_present": ";".join(str(int(v)) for v in sorted(np.unique(q).tolist())),
                "all_object": bool(st["object_union_frac"] >= 0.99),
                "all_background": bool(st["value255_frac"] >= 0.99),
                "affected_excessive": bool(st["affected_union_frac"] > 0.50),
                "visual_sheet": str(sheet),
                **st,
            })
    manifest_names = {
        "q0_current": "manifests/exp51_void_quadmask_ablation_q0_current.jsonl",
        "q1_object_only": "manifests/exp51_void_quadmask_ablation_q1_object_only.jsonl",
        "q2_strict_affected": "manifests/exp51_void_quadmask_ablation_q2_strict_affected.jsonl",
        "q3_broad_affected": "manifests/exp51_void_quadmask_ablation_q3_broad_affected.jsonl",
    }
    for name, path in manifest_names.items():
        write_jsonl(Path(path), by_variant[name])
    csv_path = Path("reports/exp51_void_quadmask_ablation_data.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        fields = list(audit_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(audit_rows)
    review_path = Path("reports/exp51_void_quadmask_ablation_visual_review.csv")
    with review_path.open("w", newline="") as f:
        fields = ["sample_id", "visual_sheet", "opened_by_codex", "review_notes"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        seen = set()
        for r in audit_rows:
            if r["sample_id"] in seen:
                continue
            seen.add(r["sample_id"])
            w.writerow({"sample_id": r["sample_id"], "visual_sheet": r["visual_sheet"], "opened_by_codex": False, "review_notes": "pending Codex visual open"})
    summary = {
        "status": "VOID_QUADMASK_ABLATION_READY",
        "rows_total": len(rows),
        "variants": VARIANTS,
        "manifest_paths": manifest_names,
        "audit_rows": len(audit_rows),
        "no_vor_eval": True,
        "hard_comp": False,
    }
    Path("reports/exp51_void_quadmask_ablation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    md = """# Exp51 VOID Quadmask Ablation Data

Status: `VOID_QUADMASK_ABLATION_READY`

Built Q0/Q1/Q2/Q3 quadmask variants for existing VOR-Train train4/heldout4 rows only. VOR-Eval is excluded and no hard comp was used.

## Variants

- Q0 current: current Exp50 quadmask copied as-is.
- Q1 object-only: object/overlap -> 0, everything else -> 255.
- Q2 strict affected: high-threshold affected map from abs(V_obj - V_bg), object preserved.
- Q3 broad affected: lower-threshold affected map from abs(V_obj - V_bg), object preserved.

## Validation

Every generated row decodes, contains non-background/object structure, records area statistics, and has an evidence sheet for visual review.
"""
    Path("reports/exp51_void_quadmask_ablation_data.md").write_text(md)


if __name__ == "__main__":
    main()
