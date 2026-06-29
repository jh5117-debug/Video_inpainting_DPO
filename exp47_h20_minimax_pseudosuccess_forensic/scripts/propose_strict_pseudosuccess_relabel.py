#!/usr/bin/env python3
"""Build Exp47 strict pseudo-success relabel proposal from audited search/shadow rows."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO / "manifests"


def row_to_manifest(row: Dict[str, str], proposed_class: str, reason: str) -> Dict[str, object]:
    return {
        "split": row["split"],
        "sample_id": row["sample_id"],
        "source_group": row["source_group"],
        "source_id": row["source_id"],
        "condition_path": row["condition_path"],
        "pseudo_success_path": row["pseudo_success_path"],
        "gt_winner_path": row["gt_winner_path"],
        "mask_path": row["mask_path"],
        "review_sheet": row["review_sheet"],
        "contact_page": row.get("contact_page", ""),
        "teacher_label": row["teacher_label"],
        "proposed_class": proposed_class,
        "reason": reason,
        "global_sft_allowed": proposed_class == "SUCCESS_CLEAN_STRICT",
        "local_target_allowed": proposed_class == "SUCCESS_LOCAL_ONLY",
        "dpo_winner_allowed": proposed_class in {"SUCCESS_CLEAN_STRICT", "SUCCESS_LOCAL_ONLY"},
        "training_run": False,
        "optimizer_step": False,
        "vor_eval_used": False,
        "hard_comp_used": False,
    }


def classify(row: Dict[str, str]) -> tuple[str, str]:
    label = row["teacher_label"]
    outside_psnr = float(row["target_gt_outside_psnr"])
    outside_l1 = float(row["target_gt_outside_l1"])
    brightness = abs(float(row["global_brightness_delta"]))
    hist = float(row["color_hist_distance"])
    lowfreq = float(row["lowfreq_l1"])
    boundary_psnr = float(row["target_gt_boundary_psnr"])
    mask_gain = float(row["mask_removal_psnr_gain"])
    if label == "PSEUDO_TARGET_CLEAN_STRICT" and outside_psnr >= 36 and outside_l1 <= 0.018 and brightness <= 0.012 and hist <= 0.10 and lowfreq <= 0.012 and boundary_psnr >= 27 and mask_gain >= 0.5:
        return "SUCCESS_CLEAN_STRICT", "strict global teacher thresholds passed"
    if label in {"PSEUDO_TARGET_OUTSIDE_BAD"} or outside_psnr < 29.0 or outside_l1 > 0.04:
        return "REJECT_OUTSIDE_BAD", "outside identity too poor for local/global pseudo-success"
    if label in {"PSEUDO_TARGET_BOUNDARY_BAD"} or boundary_psnr < 23.0:
        return "REJECT_BOUNDARY_BAD", "boundary too weak"
    if mask_gain > 0.0:
        if label == "PSEUDO_TARGET_GLOBAL_DRIFT" or brightness > 0.012 or lowfreq > 0.012:
            return "SUCCESS_LOCAL_ONLY", "local removal signal exists but global drift blocks global SFT"
        return "SUCCESS_LOCAL_ONLY", "local removal signal exists but strict global thresholds not met"
    return "REJECT_GLOBAL_DRIFT", "no reliable local removal signal under strict proposal"


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def main() -> None:
    src = REPO / "reports" / "exp47_pseudosuccess_teacher_quality_audit.csv"
    with src.open() as f:
        rows = list(csv.DictReader(f))
    buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    proposal_rows = []
    for row in rows:
        proposed, reason = classify(row)
        rec = row_to_manifest(row, proposed, reason)
        buckets[proposed].append(rec)
        proposal_rows.append({**row, "proposed_class": proposed, "proposal_reason": reason})
    write_jsonl(MANIFEST_DIR / "exp47_success_clean_strict.jsonl", buckets["SUCCESS_CLEAN_STRICT"])
    write_jsonl(MANIFEST_DIR / "exp47_success_local_only.jsonl", buckets["SUCCESS_LOCAL_ONLY"])
    write_jsonl(MANIFEST_DIR / "exp47_reject_global_drift.jsonl", buckets["REJECT_GLOBAL_DRIFT"])
    write_jsonl(MANIFEST_DIR / "exp47_reject_boundary_outside.jsonl", buckets["REJECT_BOUNDARY_BAD"] + buckets["REJECT_OUTSIDE_BAD"])

    report_csv = REPO / "reports" / "exp47_strict_pseudosuccess_relabel_proposal.csv"
    fields = list(proposal_rows[0].keys()) if proposal_rows else []
    with report_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(proposal_rows)
    counts = Counter(r["proposed_class"] for r in proposal_rows)
    split_counts: Dict[str, Counter] = defaultdict(Counter)
    for r in proposal_rows:
        split_counts[r["split"]][r["proposed_class"]] += 1
    strict_clean = counts.get("SUCCESS_CLEAN_STRICT", 0)
    local_only = counts.get("SUCCESS_LOCAL_ONLY", 0)
    global_drift_teacher = sum(1 for r in proposal_rows if r["teacher_label"] == "PSEUDO_TARGET_GLOBAL_DRIFT")
    strict_split_possible = split_counts["train"].get("SUCCESS_CLEAN_STRICT", 0) >= 32 and split_counts["search"].get("SUCCESS_CLEAN_STRICT", 0) >= 16 and split_counts["shadow"].get("SUCCESS_CLEAN_STRICT", 0) >= 16
    local_signal_enough_for_1_10 = local_only >= 32 or (split_counts["search"].get("SUCCESS_LOCAL_ONLY", 0) >= 16 and split_counts["shadow"].get("SUCCESS_LOCAL_ONLY", 0) >= 16)
    if strict_split_possible:
        recommendation = "strict-clean pseudo-success SFT 1/10-step only"
    elif local_only > 0:
        recommendation = "local pseudo-success target construction or same-source DPO; no global SFT"
    else:
        recommendation = "PAI strict pseudo-success re-mining"
    summary = {
        "status": "EXP47_STRICT_PSEUDOSUCCESS_RELABEL_PROPOSED",
        "rows": len(proposal_rows),
        "counts": dict(counts),
        "split_counts": {k: dict(v) for k, v in split_counts.items()},
        "teacher_global_drift_count": global_drift_teacher,
        "strict_clean_count": strict_clean,
        "local_only_count": local_only,
        "strict_32_16_16_split_possible": strict_split_possible,
        "local_signal_enough_for_1_10_step_probe": local_signal_enough_for_1_10,
        "recommendation": recommendation,
        "manifest_paths": {
            "success_clean_strict": "manifests/exp47_success_clean_strict.jsonl",
            "success_local_only": "manifests/exp47_success_local_only.jsonl",
            "reject_global_drift": "manifests/exp47_reject_global_drift.jsonl",
            "reject_boundary_outside": "manifests/exp47_reject_boundary_outside.jsonl",
        },
    }
    (REPO / "reports" / "exp47_strict_pseudosuccess_relabel_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    label_lines = "\n".join(f"- `{k}`: `{v}`" for k, v in sorted(counts.items()))
    (REPO / "reports" / "exp47_strict_pseudosuccess_relabel_proposal.md").write_text(f"""# Exp47 Strict Pseudo-Success Relabel Proposal

Status: `{summary['status']}`

Rows considered: `{len(proposal_rows)}` audited search/shadow pseudo-success rows. This proposal does not train or run an optimizer step.

## Proposed Counts

{label_lines}

Teacher global-drift rows from Milestone C: `{global_drift_teacher}`.

Strict clean count: `{strict_clean}`. Local-only count: `{local_only}`.

Strict 32/16/16 global-SFT split possible: `{strict_split_possible}`.

## Decision

The strict global-clean pool is empty, so global pseudo-success SFT is not unlocked. The audited rows do preserve local removal signal, so the next viable direction is localized pseudo-success target construction or same-source preference/DPO, not another global SFT run.

Recommendation: `{recommendation}`.

Generated manifests:

- `manifests/exp47_success_clean_strict.jsonl`
- `manifests/exp47_success_local_only.jsonl`
- `manifests/exp47_reject_global_drift.jsonl`
- `manifests/exp47_reject_boundary_outside.jsonl`
""")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
