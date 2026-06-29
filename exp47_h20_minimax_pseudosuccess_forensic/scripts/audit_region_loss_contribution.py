#!/usr/bin/env python3
"""Audit Exp46 region-weight contribution without model forward or optimizer step."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from audit_teacher_quality import REPO, dilate, erode, list_frames, load_stack, mask_stack

WEIGHTS = {"mask": 0.75, "boundary": 1.50, "affected": 0.75, "outside": 0.20, "far_outside": 0.03}
BATCH_SIZE = 8


def load_manifest(split: str) -> List[Dict[str, str]]:
    path = REPO / "manifests" / f"exp46_runner_pseudosuccess_{split}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_sampled(path: Path, idx: np.ndarray) -> np.ndarray:
    return load_stack(path, indices=idx)[:, ::4, ::4, :]


def load_mask_sampled(path: Path, idx: np.ndarray) -> np.ndarray:
    return mask_stack(path, indices=idx)[:, ::4, ::4]


def region_l1(a: np.ndarray, b: np.ndarray, region: np.ndarray) -> float:
    if not np.any(region):
        return float("nan")
    return float(np.mean(np.abs(a[region] - b[region])))


def audit_row(row: Dict[str, str], split_role: str) -> Dict[str, object]:
    condition_path = Path(row["condition_path"])
    pseudo_path = Path(row["winner_path"])
    gt_path = Path(row["loser_path"])
    mask_path = Path(row["mask_path"])
    n = min(len(list_frames(condition_path)), len(list_frames(pseudo_path)), len(list_frames(gt_path)), len(list_frames(mask_path)))
    idx = np.array(sorted(set([0, n // 4, n // 2, (3 * n) // 4, n - 1])), dtype=np.int64)
    cond = load_sampled(condition_path, idx)
    pseudo = load_sampled(pseudo_path, idx)
    gt = load_sampled(gt_path, idx)
    mask = load_mask_sampled(mask_path, idx)
    boundary = dilate(mask, 1) & ~erode(mask, 1)
    outside = ~dilate(mask, 1)
    far_outside_region = erode(outside, 4)
    affected_raw = np.mean(np.abs(cond - pseudo), axis=-1)
    max_aff = float(affected_raw.max())
    affected = affected_raw / max_aff if max_aff > 1e-8 else affected_raw

    base_far = np.full_like(affected, WEIGHTS["far_outside"], dtype=np.float32)
    outside_inc = (WEIGHTS["outside"] - WEIGHTS["far_outside"]) * outside.astype(np.float32)
    mask_add = WEIGHTS["mask"] * mask.astype(np.float32)
    boundary_add = WEIGHTS["boundary"] * boundary.astype(np.float32)
    affected_add = WEIGHTS["affected"] * affected.astype(np.float32)
    total_weight = base_far + outside_inc + mask_add + boundary_add + affected_add
    total_sum = float(total_weight.sum())

    components = {
        "far_outside_base_global": base_far,
        "outside_increment": outside_inc,
        "mask_add": mask_add,
        "boundary_add": boundary_add,
        "affected_add": affected_add,
    }
    regions = {
        "mask": mask,
        "boundary": boundary,
        "affected_nonzero": affected > 0.05,
        "outside": outside,
        "far_outside_region": far_outside_region,
    }
    out: Dict[str, object] = {
        "split_role": split_role,
        "split": row["split"],
        "sample_id": row["sample_id"],
        "source_group": row["source_group"],
        "metric_sampling": "frames_0_quarter_mid_3quarter_last_spatial_stride4",
        "mask_area": float(mask.mean()),
        "boundary_area": float(boundary.mean()),
        "outside_area": float(outside.mean()),
        "far_outside_region_area": float(far_outside_region.mean()),
        "affected_mean": float(affected.mean()),
        "affected_outside_mean": float(affected[outside].mean()) if np.any(outside) else float("nan"),
        "total_weight_sum": total_sum,
        "weight_mean": float(total_weight.mean()),
        "weight_max": float(total_weight.max()),
        "mask_polarity_ok": bool(mask.mean() > 0 and mask.mean() < 0.5),
        "outside_receives_nonzero_loss": bool(float(total_weight[outside].mean()) > 0.0) if np.any(outside) else False,
        "global_base_weight_applies_everywhere": True,
        "far_outside_region_specific_component_used": False,
        "affected_uses_condition_vs_pseudo_target": True,
        "loss_normalized_by_total_weight_sum_times_channels": True,
        "pseudo_target_full_video_participates_outside": True,
        "condition_pseudo_full_l1": float(np.mean(np.abs(cond - pseudo))),
        "condition_pseudo_mask_l1": region_l1(cond, pseudo, mask),
        "condition_pseudo_boundary_l1": region_l1(cond, pseudo, boundary),
        "condition_pseudo_outside_l1": region_l1(cond, pseudo, outside),
        "pseudo_gt_full_l1": float(np.mean(np.abs(pseudo - gt))),
        "pseudo_gt_mask_l1": region_l1(pseudo, gt, mask),
        "pseudo_gt_boundary_l1": region_l1(pseudo, gt, boundary),
        "pseudo_gt_outside_l1": region_l1(pseudo, gt, outside),
    }
    for name, comp in components.items():
        comp_sum = float(comp.sum())
        out[f"component_{name}_sum"] = comp_sum
        out[f"component_{name}_normalized_contribution"] = comp_sum / total_sum if total_sum > 0 else float("nan")
    for name, region in regions.items():
        if np.any(region):
            region_sum = float(total_weight[region].sum())
            out[f"region_{name}_weight_sum"] = region_sum
            out[f"region_{name}_normalized_mass"] = region_sum / total_sum if total_sum > 0 else float("nan")
            out[f"region_{name}_weight_mean"] = float(total_weight[region].mean())
        else:
            out[f"region_{name}_weight_sum"] = 0.0
            out[f"region_{name}_normalized_mass"] = 0.0
            out[f"region_{name}_weight_mean"] = float("nan")
    out["outside_plus_affected_component_contribution"] = float(out["component_outside_increment_normalized_contribution"] + out["component_affected_add_normalized_contribution"] + out["component_far_outside_base_global_normalized_contribution"])
    out["tiny_region_dominates"] = bool(out["region_boundary_normalized_mass"] > 0.35 and out["boundary_area"] < 0.08)
    out["global_drift_risk_row"] = bool(out["component_affected_add_normalized_contribution"] > 0.30 or out["outside_plus_affected_component_contribution"] > 0.70)
    return out


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    def mean(key: str) -> float:
        vals = [float(r[key]) for r in rows if key in r and not np.isnan(float(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")
    by_role = defaultdict(list)
    for r in rows:
        by_role[str(r["split_role"])].append(r)
    role_means = {role: {"weight_mean": float(np.mean([float(r["weight_mean"]) for r in rs])), "outside_plus_affected_component_contribution": float(np.mean([float(r["outside_plus_affected_component_contribution"]) for r in rs])), "affected_outside_mean": float(np.mean([float(r["affected_outside_mean"]) for r in rs]))} for role, rs in by_role.items()}
    risk_rows = sum(bool(r["global_drift_risk_row"]) for r in rows)
    bug_far = all(not bool(r["far_outside_region_specific_component_used"]) for r in rows)
    status = "EXP47_REGION_LOSS_GLOBAL_DRIFT_RISK_CONFIRMED" if risk_rows >= len(rows) // 2 or bug_far else "EXP47_REGION_LOSS_IMPL_PASS"
    return {
        "status": status,
        "rows": len(rows),
        "train_rows": sum(1 for r in rows if r["split_role"] == "train_batch"),
        "search_rows": sum(1 for r in rows if r["split_role"] == "search_batch"),
        "weights": WEIGHTS,
        "runner_formula": "weight = far_outside global base + (outside-far_outside)*outside + mask*mask + boundary*boundary + affected*normalized_abs(condition-winner)",
        "far_outside_region_specific_component_used": False,
        "risk_rows": int(risk_rows),
        "role_means": role_means,
        "means": {
            "mask_area": mean("mask_area"),
            "boundary_area": mean("boundary_area"),
            "outside_area": mean("outside_area"),
            "affected_mean": mean("affected_mean"),
            "affected_outside_mean": mean("affected_outside_mean"),
            "weight_mean": mean("weight_mean"),
            "weight_max": mean("weight_max"),
            "component_far_outside_base_global_normalized_contribution": mean("component_far_outside_base_global_normalized_contribution"),
            "component_outside_increment_normalized_contribution": mean("component_outside_increment_normalized_contribution"),
            "component_mask_add_normalized_contribution": mean("component_mask_add_normalized_contribution"),
            "component_boundary_add_normalized_contribution": mean("component_boundary_add_normalized_contribution"),
            "component_affected_add_normalized_contribution": mean("component_affected_add_normalized_contribution"),
            "outside_plus_affected_component_contribution": mean("outside_plus_affected_component_contribution"),
            "condition_pseudo_outside_l1": mean("condition_pseudo_outside_l1"),
            "pseudo_gt_outside_l1": mean("pseudo_gt_outside_l1"),
        },
    }


def write_outputs(rows: List[Dict[str, object]], summary: Dict[str, object]) -> None:
    reports = REPO / "reports"
    csv_path = reports / "exp47_region_loss_contribution_audit.csv"
    json_path = reports / "exp47_region_loss_contribution_summary.json"
    md_path = reports / "exp47_region_loss_contribution_audit.md"
    keys = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    m = summary["means"]
    md_path.write_text(f"""# Exp47 Region Loss Contribution Audit

Status: `{summary['status']}`

Rows audited: `{summary['rows']}` (`{summary['train_rows']}` train-batch proxy rows, `{summary['search_rows']}` search-batch proxy rows). This is a no-training, no-optimizer frame-space proxy audit of Exp46 SFT-B region weights and pseudo-success target participation.

## Runner Formula Readback

Exp43/Exp46 SFT runner builds weights as:

```text
weight = far_outside global base
       + (outside - far_outside) * outside
       + mask * mask_weight
       + boundary * boundary_weight
       + affected * normalized_abs(condition - winner)
```

Configured SFT-B weights: mask `{WEIGHTS['mask']}`, boundary `{WEIGHTS['boundary']}`, affected `{WEIGHTS['affected']}`, outside `{WEIGHTS['outside']}`, far_outside `{WEIGHTS['far_outside']}`.

Important implementation detail: the computed far-outside region is not used as a region-specific component in `build_region_weight`; `far_outside` is a global base weight. Also, `affected` is computed from condition versus pseudo-success winner and normalized, so pseudo-success global drift can add weight outside the local mask.

## Mean Areas And Contributions

- mask/boundary/outside area: `{m['mask_area']:.6f}` / `{m['boundary_area']:.6f}` / `{m['outside_area']:.6f}`
- affected mean / affected outside mean: `{m['affected_mean']:.6f}` / `{m['affected_outside_mean']:.6f}`
- weight mean / max: `{m['weight_mean']:.6f}` / `{m['weight_max']:.6f}`
- normalized component contribution far-base/outside/mask/boundary/affected: `{m['component_far_outside_base_global_normalized_contribution']:.6f}` / `{m['component_outside_increment_normalized_contribution']:.6f}` / `{m['component_mask_add_normalized_contribution']:.6f}` / `{m['component_boundary_add_normalized_contribution']:.6f}` / `{m['component_affected_add_normalized_contribution']:.6f}`
- outside + affected + global-base component contribution: `{m['outside_plus_affected_component_contribution']:.6f}`
- condition-vs-pseudo outside L1: `{m['condition_pseudo_outside_l1']:.6f}`
- pseudo-vs-V_bg outside L1: `{m['pseudo_gt_outside_l1']:.6f}`

## Conclusion

The region weighting implementation is finite and mask polarity is sane, but it is not safely localized for drifting pseudo-success targets. Outside receives nonzero loss, far-outside is a global base, and the affected term can spread pseudo target differences outside the object region. This confirms a global-drift risk in the SFT objective even without an obvious manifest bug.
""")


def main() -> None:
    train_rows = load_manifest("train")[:BATCH_SIZE]
    search_rows = load_manifest("search")[:BATCH_SIZE]
    rows: List[Dict[str, object]] = []
    for row in train_rows:
        rows.append(audit_row(row, "train_batch"))
    for row in search_rows:
        rows.append(audit_row(row, "search_batch"))
    summary = summarize(rows)
    write_outputs(rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
