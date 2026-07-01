#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("/home/nvme01/H20_Video_inpainting_DPO_exp52_void_allgpu_rescue")
REPORTS = ROOT / "reports"
WAVE1 = Path("/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp52_void_winner_preserving_allgpu/wave1_forward")
VIDEO = Path("/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp52_void_winner_preserving_allgpu/wave1_video/R1_Q0_T500_S0")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    diagnostics = []
    forward_summaries = []
    for summary_path in sorted(WAVE1.glob("*/summary.json")):
        summary = read_json(summary_path)
        cell = summary["cell"]
        train = summary.get("train", {})
        heldout = summary.get("heldout", {})
        forward_summaries.append(summary)
        diagnostics.append({
            "cell": cell,
            "recipe": summary.get("recipe"),
            "variant": summary.get("variant"),
            "timestep": summary.get("timestep"),
            "scope": summary.get("scope"),
            "gpu": summary.get("gpu"),
            "forward_status": summary.get("status"),
            "runtime_sec": summary.get("runtime_sec"),
            "checkpoint": summary.get("checkpoint"),
            "checkpoint_exists": summary.get("checkpoint_exists"),
            "reload_ok": summary.get("reload_ok"),
            "optimizer_steps": summary.get("optimizer_steps"),
            "max_param_delta_norm": summary.get("max_param_delta_norm"),
            "grad_norm_before_clip": summary.get("grad_norm_before_clip"),
            "peak_vram_reserved_gib": summary.get("peak_vram_reserved_gib"),
            "train_winner_gap": train.get("mean_winner_gap"),
            "train_loser_gap": train.get("mean_loser_gap"),
            "train_effective_loser_gap": train.get("mean_effective_loser_gap"),
            "train_loser_contribution_ratio": train.get("mean_loser_contribution_ratio"),
            "heldout_winner_gap": heldout.get("mean_winner_gap"),
            "heldout_loser_gap": heldout.get("mean_loser_gap"),
            "heldout_effective_loser_gap": heldout.get("mean_effective_loser_gap"),
            "heldout_loser_contribution_ratio": heldout.get("mean_loser_contribution_ratio"),
            "winner_ok": summary.get("winner_ok"),
            "loser_controlled": summary.get("loser_controlled"),
            "vor_eval_used": summary.get("vor_eval_used"),
            "hard_comp_used": summary.get("hard_comp_used"),
        })

    if diagnostics:
        write_csv(REPORTS / "exp52_rescue_onestep_diagnostics.csv", diagnostics)

    metrics = read_csv(REPORTS / "exp52_rescue_onestep_metrics.csv")
    means = read_json(REPORTS / "exp52_rescue_onestep_summary.json").get("means", {})

    manual_review = [
        {
            "sample_id": "BLENDER_CON001_00742",
            "visual_class": "tie",
            "object_removed": "tie_or_slight_better",
            "effect_removed": "mixed_affected_worse",
            "outside_damage": "false",
            "boundary_damage": "false",
            "tone_shift": "false",
            "collapse": "false",
            "reason": "Step1 is visually close to Step0; object/core is slightly cleaner by metric, but affected/overlap texture regresses.",
        },
        {
            "sample_id": "BLENDER_CON001_00744",
            "visual_class": "worse",
            "object_removed": "tie",
            "effect_removed": "mixed_affected_worse",
            "outside_damage": "minor_metric_regression",
            "boundary_damage": "false",
            "tone_shift": "false",
            "collapse": "false",
            "reason": "Small full/outside regression and affected degradation; no collapse, but not safe enough for PASS.",
        },
        {
            "sample_id": "REAL_ENV102_00001_002_02",
            "visual_class": "tie",
            "object_removed": "tie",
            "effect_removed": "mixed_affected_worse",
            "outside_damage": "false",
            "boundary_damage": "false",
            "tone_shift": "false",
            "collapse": "false",
            "reason": "Outside improves and visual appearance remains stable; affected/object metrics are mixed.",
        },
        {
            "sample_id": "REAL_ENV200_00001_006_02",
            "visual_class": "tie",
            "object_removed": "better_metric",
            "effect_removed": "tie",
            "outside_damage": "minor_metric_regression",
            "boundary_damage": "false",
            "tone_shift": "false",
            "collapse": "false",
            "reason": "Object and boundary improve strongly; full/outside slightly regress, with no visible collapse.",
        },
    ]
    write_csv(REPORTS / "exp52_rescue_onestep_visual_review.csv", manual_review)

    visual_counts = {"better": 0, "tie": 0, "worse": 0}
    for row in manual_review:
        visual_counts[row["visual_class"]] += 1

    forward_ready = [s["cell"] for s in forward_summaries if s.get("status") == "FORWARD_READY"]
    forward_mixed = [s["cell"] for s in forward_summaries if s.get("status") == "FORWARD_MIXED"]
    skipped = ["R4_Q2_T500_S0"]

    # Conservative gate: R1_Q0 controls loser dominance and improves full/object/boundary/outside,
    # but affected/overlap regress and visual evidence is tie-heavy, so do not unlock 10-step.
    status = "VOID_RESCUE_ONESTEP_MIXED"
    summary = {
        "status": status,
        "wave1_preregistered_cells": [
            "R1_Q0_T500_S0",
            "R1_Q2_T500_S0",
            "R2_Q0_T500_S0",
            "R2_Q2_T500_S0",
            "R3_Q0_T500_S0",
            "R3_Q2_T500_S0",
            "R4_Q0_T500_S0",
            "R4_Q2_T500_S0",
        ],
        "forward_ready_cells": forward_ready,
        "forward_mixed_cells": forward_mixed,
        "skipped_cells": skipped,
        "video_evaluated_cells": ["R1_Q0_T500_S0"],
        "video_opened_by_codex": True,
        "visual_counts": visual_counts,
        "means": means,
        "contact_sheet": str(VIDEO / "R1_Q0_T500_S0_all_heldout_contact_sheet.jpg"),
        "decision": "10-step not unlocked because affected/overlap metrics regress and visual evidence is not positive.",
        "no_vor_eval": True,
        "hard_comp": False,
        "long_training": False,
    }
    (REPORTS / "exp52_rescue_onestep_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    md = f"""# Exp52 VOID Rescue One-Step Grid

Status: `{status}`

Wave 1 ran the preregistered all-GPU one-step forward/checkpoint grid where GPUs were available. Seven cells completed a one-step checkpoint; `R4_Q2_T500_S0` was skipped because GPU7 had an unrelated external process and was not killed.

Full heldout video evidence was generated for `R1_Q0_T500_S0`, because it was the clearest forward-ready cell and has a valid Step0 baseline. Codex opened the contact sheet and inspected all four heldout samples.

## R1_Q0_T500_S0 Mean Step1 - Step0

- full PSNR: {means.get('full_psnr_delta')}
- object PSNR: {means.get('object_psnr_delta')}
- overlap PSNR: {means.get('overlap_psnr_delta')}
- affected PSNR: {means.get('affected_psnr_delta')}
- affected-union PSNR: {means.get('affected_union_psnr_delta')}
- boundary PSNR: {means.get('boundary_psnr_delta')}
- outside PSNR: {means.get('outside_psnr_delta')}
- SSIM: {means.get('ssim_delta')}
- Step0-Step1 L1: {means.get('step0_step1_l1')}

## Interpretation

R1 reduces the Exp50 loser-dominant failure mode in the intended direction: object, boundary, outside, and full PSNR are safe or positive, with no collapse and no systematic outside damage. However, affected and overlap regions regress, and the visual review is tie-heavy rather than positive. This is not enough to unlock 10-step training under Exp52 gates.

Visual review: {visual_counts}

No VOR-Eval, hard comp, or long training was used.
"""
    (REPORTS / "exp52_rescue_onestep.md").write_text(md)

    print(status)


if __name__ == "__main__":
    main()
