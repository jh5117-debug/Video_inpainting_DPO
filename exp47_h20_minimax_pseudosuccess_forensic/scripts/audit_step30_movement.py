#!/usr/bin/env python3
"""Audit whether Exp46 Step30 moved toward pseudo-success target or elsewhere."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from audit_teacher_quality import REPO, STEP0_CACHE, dilate, erode, list_frames, load_stack, mask_stack, pair_metrics

EXP46_ROOT = Path("/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp46_h20_minimax_pseudosuccess_sft")
EVAL_ROOTS = sorted((EXP46_ROOT / "sft_ladder").glob("PSEUDO-SFT-A_lr3em5_step30_eval_shard*"))


def load_manifest(split: str) -> List[Dict[str, str]]:
    path = REPO / "manifests" / f"exp46_runner_pseudosuccess_{split}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def find_step30(split: str, sample_id: str) -> Path:
    matches = []
    for root in EVAL_ROOTS:
        p = root / "eval" / split / sample_id / "step30" / "frames"
        if p.exists() and list_frames(p):
            matches.append(p)
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one step30 frames dir for {split}/{sample_id}, got {len(matches)}")
    return matches[0]


def load_sampled(path: Path, indices: np.ndarray) -> np.ndarray:
    return load_stack(path, indices=indices)[:, ::4, ::4, :]


def load_mask_sampled(path: Path, indices: np.ndarray) -> np.ndarray:
    return mask_stack(path, indices=indices)[:, ::4, ::4]


def l1(a: np.ndarray, b: np.ndarray, region: np.ndarray | None = None) -> float:
    if region is None:
        return float(np.mean(np.abs(a - b)))
    if not np.any(region):
        return float("nan")
    return float(np.mean(np.abs(a[region] - b[region])))


def l2(a: np.ndarray, b: np.ndarray, region: np.ndarray | None = None) -> float:
    if region is None:
        return float(np.sqrt(np.mean((a - b) ** 2)))
    if not np.any(region):
        return float("nan")
    return float(np.sqrt(np.mean((a[region] - b[region]) ** 2)))


def psnr(a: np.ndarray, b: np.ndarray, region: np.ndarray | None = None) -> float:
    if region is None:
        mse = float(np.mean((a - b) ** 2))
    else:
        if not np.any(region):
            return float("nan")
        mse = float(np.mean((a[region] - b[region]) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def cosine(a: np.ndarray, b: np.ndarray, region: np.ndarray | None = None) -> float:
    if region is not None:
        if not np.any(region):
            return float("nan")
        av = a[region].reshape(-1).astype(np.float64)
        bv = b[region].reshape(-1).astype(np.float64)
    else:
        av = a.reshape(-1).astype(np.float64)
        bv = b.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(av, bv) / denom)


def add_distance(out: Dict[str, object], prefix: str, a: np.ndarray, b: np.ndarray, regions: Dict[str, np.ndarray | None]) -> None:
    for name, region in regions.items():
        out[f"{prefix}_{name}_l1"] = l1(a, b, region)
        out[f"{prefix}_{name}_l2"] = l2(a, b, region)
        out[f"{prefix}_{name}_psnr"] = psnr(a, b, region)


def classify(out: Dict[str, object]) -> str:
    closer_pseudo = bool(out["step30_closer_to_pseudo_full_l1"])
    closer_gt = bool(out["step30_closer_to_gt_full_l1"])
    closer_gt_mask = bool(out["step30_closer_to_gt_mask_l1"])
    farther_outside_gt = float(out["step30_gt_outside_l1_delta_vs_step0"]) > 0.002
    train_teacher_cos = float(out["cos_train_teacher_full"])
    train_gt_cos = float(out["cos_train_gt_full"])
    if closer_pseudo and not closer_gt:
        return "LEARNS_BAD_TEACHER"
    if closer_gt_mask and farther_outside_gt:
        return "LOCAL_IMPROVES_GLOBAL_DRIFTS"
    if not closer_pseudo and not closer_gt:
        if train_teacher_cos < 0.05 and train_gt_cos < 0.05:
            return "MOVES_AWAY_FROM_BOTH"
        return "DOES_NOT_LEARN_TARGET"
    if abs(float(out["step30_step0_full_l1"])) < 1e-4:
        return "NO_CHANGE"
    return "LOCAL_IMPROVES_GLOBAL_DRIFTS" if farther_outside_gt else "NO_CHANGE"


def audit_row(row: Dict[str, str]) -> Dict[str, object]:
    condition_path = Path(row["condition_path"])
    gt_path = Path(row["loser_path"])
    pseudo_path = Path(row["winner_path"])
    mask_path = Path(row["mask_path"])
    step0_path = STEP0_CACHE / row["split"] / row["sample_id"] / "frames"
    step30_path = find_step30(row["split"], row["sample_id"])
    n = min(len(list_frames(condition_path)), len(list_frames(gt_path)), len(list_frames(pseudo_path)), len(list_frames(mask_path)), len(list_frames(step0_path)), len(list_frames(step30_path)))
    idx = np.array(sorted(set([0, n // 4, n // 2, (3 * n) // 4, n - 1])), dtype=np.int64)
    cond = load_sampled(condition_path, idx)
    gt = load_sampled(gt_path, idx)
    pseudo = load_sampled(pseudo_path, idx)
    step0 = load_sampled(step0_path, idx)
    step30 = load_sampled(step30_path, idx)
    mask = load_mask_sampled(mask_path, idx)
    boundary = dilate(mask, 1) & ~erode(mask, 1)
    affected = (np.mean(np.abs(cond - gt), axis=-1) > 0.035) | mask
    outside = ~dilate(mask | affected, 2)
    regions = {"full": None, "mask": mask, "boundary": boundary, "affected": affected, "outside": outside}
    out: Dict[str, object] = {
        "split": row["split"], "sample_id": row["sample_id"], "source_group": row["source_group"], "source_id": row["source_id"],
        "step0_path": str(step0_path), "step30_path": str(step30_path), "pseudo_success_path": row["winner_path"], "gt_winner_path": row["loser_path"],
        "metric_sampling": "frames_0_quarter_mid_3quarter_last_spatial_stride4",
        "num_sampled_frames": int(len(idx)),
        "mask_area_frac": float(mask.mean()), "outside_area_frac": float(outside.mean()),
    }
    add_distance(out, "step0_gt", step0, gt, regions)
    add_distance(out, "step30_gt", step30, gt, regions)
    add_distance(out, "step0_pseudo", step0, pseudo, regions)
    add_distance(out, "step30_pseudo", step30, pseudo, regions)
    add_distance(out, "pseudo_gt", pseudo, gt, regions)
    add_distance(out, "condition_gt", cond, gt, regions)
    add_distance(out, "step30_condition", step30, cond, regions)
    add_distance(out, "step30_step0", step30, step0, regions)
    direction_train = step30 - step0
    direction_teacher = pseudo - step0
    direction_gt = gt - step0
    for name, region in regions.items():
        out[f"cos_train_teacher_{name}"] = cosine(direction_train, direction_teacher, region)
        out[f"cos_train_gt_{name}"] = cosine(direction_train, direction_gt, region)
        out[f"cos_teacher_gt_{name}"] = cosine(direction_teacher, direction_gt, region)
    for name in regions:
        out[f"step30_pseudo_{name}_l1_delta_vs_step0"] = float(out[f"step30_pseudo_{name}_l1"] - out[f"step0_pseudo_{name}_l1"])
        out[f"step30_gt_{name}_l1_delta_vs_step0"] = float(out[f"step30_gt_{name}_l1"] - out[f"step0_gt_{name}_l1"])
        out[f"step30_closer_to_pseudo_{name}_l1"] = float(out[f"step30_pseudo_{name}_l1"]) < float(out[f"step0_pseudo_{name}_l1"])
        out[f"step30_closer_to_gt_{name}_l1"] = float(out[f"step30_gt_{name}_l1"]) < float(out[f"step0_gt_{name}_l1"])
    out["movement_label"] = classify(out)
    return out


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    labels = Counter(str(r["movement_label"]) for r in rows)
    split_labels: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        split_labels[str(r["split"])][str(r["movement_label"])] += 1
    def mean(key: str) -> float:
        vals = [float(r[key]) for r in rows if key in r and not math.isnan(float(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")
    closer_pseudo = sum(bool(r["step30_closer_to_pseudo_full_l1"]) for r in rows)
    closer_gt = sum(bool(r["step30_closer_to_gt_full_l1"]) for r in rows)
    bad_teacher = labels.get("LEARNS_BAD_TEACHER", 0)
    loss_bug_suspect = labels.get("DOES_NOT_LEARN_TARGET", 0) + labels.get("MOVES_AWAY_FROM_BOTH", 0)
    if bad_teacher >= len(rows) // 2:
        root_signal = "PSEUDO_TEACHER_BAD_GLOBAL_DRIFT"
    elif loss_bug_suspect >= len(rows) // 2:
        root_signal = "SFT_LOSS_OR_TARGET_PATH_BUG"
    elif labels.get("LOCAL_IMPROVES_GLOBAL_DRIFTS", 0) >= len(rows) // 3:
        root_signal = "GLOBAL_SFT_SHOULD_BE_LOCALIZED"
    else:
        root_signal = "INCONCLUSIVE_NEEDS_MORE_AUDIT"
    return {
        "status": "EXP47_STEP30_MOVEMENT_AUDITED",
        "root_signal": root_signal,
        "rows": len(rows),
        "label_counts": dict(labels),
        "label_counts_by_split": {k: dict(v) for k, v in split_labels.items()},
        "step30_closer_to_pseudo_count": int(closer_pseudo),
        "step30_closer_to_vbg_count": int(closer_gt),
        "bad_teacher_learned_count": int(bad_teacher),
        "loss_or_path_bug_suspect_count": int(loss_bug_suspect),
        "means": {
            "step30_pseudo_full_l1_delta_vs_step0": mean("step30_pseudo_full_l1_delta_vs_step0"),
            "step30_gt_full_l1_delta_vs_step0": mean("step30_gt_full_l1_delta_vs_step0"),
            "step30_gt_mask_l1_delta_vs_step0": mean("step30_gt_mask_l1_delta_vs_step0"),
            "step30_gt_outside_l1_delta_vs_step0": mean("step30_gt_outside_l1_delta_vs_step0"),
            "cos_train_teacher_full": mean("cos_train_teacher_full"),
            "cos_train_gt_full": mean("cos_train_gt_full"),
            "cos_teacher_gt_full": mean("cos_teacher_gt_full"),
        },
    }


def write_outputs(rows: List[Dict[str, object]], summary: Dict[str, object]) -> None:
    reports = REPO / "reports"
    csv_path = reports / "exp47_step30_movement_direction_audit.csv"
    json_path = reports / "exp47_step30_movement_direction_summary.json"
    md_path = reports / "exp47_step30_movement_direction_audit.md"
    keys = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    label_lines = "\n".join(f"- `{k}`: `{v}`" for k, v in sorted(summary["label_counts"].items()))
    means = summary["means"]
    md_path.write_text(f"""# Exp47 Step30 Movement Direction Audit

Status: `{summary['status']}`
Root signal: `{summary['root_signal']}`

Rows audited: `{summary['rows']}` search/shadow rows. This audit is read-only and uses sampled frame-space metrics; it does not train, run DPO, run GT-only SFT, or perform an optimizer step.

## Counts

- Step30 closer to pseudo target: `{summary['step30_closer_to_pseudo_count']}`
- Step30 closer to V_bg: `{summary['step30_closer_to_vbg_count']}`
- Bad teacher learned count: `{summary['bad_teacher_learned_count']}`
- Loss/path bug suspect count: `{summary['loss_or_path_bug_suspect_count']}`

## Movement Labels

{label_lines}

## Mean Direction Metrics

- Step30-to-pseudo full L1 delta vs Step0: `{means['step30_pseudo_full_l1_delta_vs_step0']:.6f}` (negative means closer to pseudo)
- Step30-to-V_bg full L1 delta vs Step0: `{means['step30_gt_full_l1_delta_vs_step0']:.6f}` (negative means closer to V_bg)
- Step30-to-V_bg mask L1 delta vs Step0: `{means['step30_gt_mask_l1_delta_vs_step0']:.6f}`
- Step30-to-V_bg outside L1 delta vs Step0: `{means['step30_gt_outside_l1_delta_vs_step0']:.6f}`
- cosine(train, teacher) full: `{means['cos_train_teacher_full']:.6f}`
- cosine(train, V_bg) full: `{means['cos_train_gt_full']:.6f}`
- cosine(teacher, V_bg) full: `{means['cos_teacher_gt_full']:.6f}`

Interpretation: if Step30 moves closer to the pseudo target while moving farther from `V_bg`, Exp46 likely learned a bad/global-drift teacher. If Step30 does not move toward the pseudo target, the remaining suspicion would shift to loss/target path implementation.
""")


def main() -> None:
    rows = []
    all_manifest_rows = load_manifest("search") + load_manifest("shadow")
    for idx, row in enumerate(all_manifest_rows, 1):
        print(f"movement_audit {idx}/{len(all_manifest_rows)} {row['split']} {row['sample_id']}", flush=True)
        rows.append(audit_row(row))
    summary = summarize(rows)
    write_outputs(rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
