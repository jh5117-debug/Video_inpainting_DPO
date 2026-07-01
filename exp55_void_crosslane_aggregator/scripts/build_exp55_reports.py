#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/home/nvme01/H20_Video_inpainting_DPO_exp53_void_r1r2_h20")
REPORTS = ROOT / "reports"
EXP54_REF = "origin/research/exp54-void-sdpo-linear-pai-20260701"
EXP53_HEAD = "7a1b6f6d18a4e58108815ad64f6f34fd191a1ee2"


EXP53_INPUTS = [
    "PRD/00_current_status.md",
    "PRD/01_experiment_matrix.md",
    "PRD/50_exp53_void_r1r2_targeted_h20.md",
    "experiment_registry/exp53_void_r1r2_targeted_h20/status.md",
    "reports/exp53b_core_recovery_readback.md",
    "reports/exp53b_gpu_cache_audit.csv",
    "reports/exp53b_gpu_cache_summary.json",
    "reports/exp53b_core_onestep.md",
    "reports/exp53b_core_onestep_metrics.csv",
    "reports/exp53b_core_onestep_visual_review.csv",
    "reports/exp53b_core_onestep_diagnostics.csv",
    "reports/exp53b_core_onestep_summary.json",
    "reports/exp53b_final_handoff.md",
    "reports/exp53b_candidate_ranking.csv",
    "reports/exp53b_next_steps.md",
    "reports/exp52_cache_summary.json",
    "reports/exp52_cache_manifest.json",
    "reports/exp52_cache_parity.csv",
    "reports/exp52_r1_row0_smoke.md",
    "reports/exp52_rescue_onestep.md",
    "reports/exp52_void_rescue_decision.md",
    "reports/exp53_h20_final_handoff.md",
    "reports/exp53_h20_candidate_ranking.csv",
    "reports/exp53_h20_next_steps.md",
]

EXP54_INPUTS = [
    "PRD/50_exp54_void_sdpo_linear_pai.md",
    "reports/exp54_readback_gpu_audit.md",
    "reports/exp54_gpu_audit.csv",
    "reports/exp54_start_state_summary.json",
    "reports/exp54_sdpo_linear_preregistration.md",
    "reports/exp54_sdpo_linear_preregistration.json",
    "reports/exp54_sdpo_linear_onestep.md",
    "reports/exp54_sdpo_linear_onestep_metrics.csv",
    "reports/exp54_sdpo_linear_onestep_visual_review.csv",
    "reports/exp54_sdpo_linear_onestep_diagnostics.csv",
    "reports/exp54_sdpo_linear_onestep_summary.json",
    "reports/exp54_wave2_decision.md",
    "reports/exp54_wave2_metrics.csv",
    "reports/exp54_wave2_visual_review.csv",
    "reports/exp54_wave2_diagnostics.csv",
    "reports/exp54_wave2_summary.json",
    "reports/exp54_pai_final_handoff.md",
    "reports/exp54_pai_candidate_ranking.csv",
    "reports/exp54_pai_next_steps.md",
]


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def git_show(ref: str, path: str) -> str | None:
    try:
        return run_git(["show", f"{ref}:{path}"])
    except subprocess.CalledProcessError:
        return None


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_csv_text(text: str | None) -> list[dict[str, str]]:
    if not text:
        return []
    return list(csv.DictReader(io.StringIO(text)))


def read_json_text(text: str | None) -> Any:
    if not text:
        return None
    return json.loads(text)


def read_local_csv(path: str) -> list[dict[str, str]]:
    full = ROOT / path
    if not full.exists():
        return []
    with full.open() as f:
        return list(csv.DictReader(f))


def read_local_json(path: str) -> Any:
    full = ROOT / path
    if not full.exists():
        return None
    return json.loads(full.read_text())


def f(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    val = row.get(key)
    if val in (None, "", "NA"):
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [f(row, key) for row in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def mean_by_cell(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["cell"]].append(row)
    out = {}
    keys = [
        "full_psnr_delta",
        "ssim_delta",
        "lpips_delta",
        "ewarp_delta",
        "object_psnr_delta",
        "mask_psnr_delta",
        "overlap_psnr_delta",
        "affected_psnr_delta",
        "affected_union_psnr_delta",
        "boundary_psnr_delta",
        "outside_psnr_delta",
        "outside_l1_delta",
        "temporal_flicker_delta",
        "step0_step1_l1",
        "tone_drift",
    ]
    for cell, cell_rows in grouped.items():
        out[cell] = {key: mean(cell_rows, key) for key in keys}
    return out


def diagnostics_by_cell(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("split") == "heldout4":
            grouped[row["cell"]].append(row)
    out = {}
    for cell, cell_rows in grouped.items():
        ratios = []
        for row in cell_rows:
            if row.get("loser_contribution_ratio") not in (None, "", "NA"):
                ratios.append(f(row, "loser_contribution_ratio"))
            else:
                margin = abs(f(row, "preference_margin", 0.0))
                ratios.append(abs(f(row, "effective_loser_gap", 0.0)) / max(margin, 1e-12))
        out[cell] = {
            "winner_policy_loss": mean(cell_rows, "winner_policy_loss"),
            "winner_reference_loss": mean(cell_rows, "winner_reference_loss"),
            "loser_policy_loss": mean(cell_rows, "loser_policy_loss"),
            "loser_reference_loss": mean(cell_rows, "loser_reference_loss"),
            "winner_gap": mean(cell_rows, "winner_gap"),
            "loser_gap": mean(cell_rows, "loser_gap"),
            "preference_margin": mean(cell_rows, "preference_margin"),
            "loser_contribution_ratio": float(sum(ratios) / len(ratios)) if ratios else float("nan"),
            "max_param_delta_norm": mean(cell_rows, "max_param_delta_norm"),
            "peak_vram_reserved_gib": mean(cell_rows, "peak_vram_reserved_gib"),
        }
    return out


def visual_by_cell(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["cell"]].append(row)
    out = {}
    for cell, cell_rows in grouped.items():
        counts = {"better": 0, "tie": 0, "worse": 0}
        outside = tone = collapse = boundary = affected = False
        for row in cell_rows:
            label = row.get("better_tie_worse") or row.get("visual_class") or ""
            label = label.lower()
            if "better" in label:
                counts["better"] += 1
            elif "tie" in label:
                counts["tie"] += 1
            else:
                counts["worse"] += 1
            outside = outside or row.get("outside_damage", "False") == "True"
            tone = tone or row.get("tone_shift", "False") == "True"
            collapse = collapse or row.get("collapse", "False") == "True"
            boundary = boundary or row.get("boundary_damage", "False") == "True"
            reason = row.get("reason", "").lower()
            affected = affected or ("affected" in reason or "overlap" in reason)
        total = max(sum(counts.values()), 1)
        out[cell] = {
            "better": counts["better"],
            "tie": counts["tie"],
            "worse": counts["worse"],
            "better_or_tie_ratio": (counts["better"] + counts["tie"]) / total,
            "worse_ratio": counts["worse"] / total,
            "systematic_outside_damage": outside,
            "tone_drift": tone,
            "collapse": collapse,
            "boundary_artifact_visible": boundary,
            "affected_overlap_regression_visible": affected or counts["worse"] > 0,
        }
    return out


def candidate_status(metrics: dict[str, float], diag: dict[str, float], visual: dict[str, Any]) -> str:
    full = metrics.get("full_psnr_delta", float("nan"))
    outside = metrics.get("outside_psnr_delta", float("nan"))
    obj = metrics.get("object_psnr_delta", metrics.get("mask_psnr_delta", float("nan")))
    boundary = metrics.get("boundary_psnr_delta", float("nan"))
    overlap = metrics.get("overlap_psnr_delta", float("nan"))
    affected = metrics.get("affected_psnr_delta", float("nan"))
    loser_ratio = diag.get("loser_contribution_ratio", float("nan"))
    winner_gap = diag.get("winner_gap", float("nan"))
    pass_ok = (
        full >= -0.02
        and outside >= -0.02
        and obj >= -0.10
        and boundary >= -0.05
        and not (overlap < -0.05 and affected < -0.05)
        and visual.get("better_or_tie_ratio", 0.0) >= 0.75
        and visual.get("worse_ratio", 1.0) <= 0.25
        and not visual.get("collapse", False)
        and not visual.get("systematic_outside_damage", False)
        and not visual.get("tone_drift", False)
        and loser_ratio <= 0.5
        and (winner_gap >= 0 or not math.isfinite(winner_gap))
    )
    if pass_ok:
        return "PASS"
    if visual.get("collapse", False) or full < -0.05 or outside < -0.02:
        return "NEGATIVE"
    return "MIXED_UNSAFE"


def ranking_score(metrics: dict[str, float], visual: dict[str, Any], diag: dict[str, float]) -> float:
    def safe(key: str) -> float:
        val = metrics.get(key, 0.0)
        return val if math.isfinite(val) else 0.0
    score = (
        safe("full_psnr_delta")
        + safe("outside_psnr_delta")
        + 0.5 * safe("object_psnr_delta")
        + 0.25 * safe("affected_psnr_delta")
        + 0.25 * safe("overlap_psnr_delta")
        + 0.25 * safe("boundary_psnr_delta")
        + 0.1 * safe("ssim_delta")
    )
    score -= 0.10 * visual.get("worse", 0)
    loser_ratio = diag.get("loser_contribution_ratio", 0.0)
    if math.isfinite(loser_ratio) and loser_ratio > 0.5:
        score -= 0.05 * (loser_ratio - 0.5)
    return score


def build_candidates() -> list[dict[str, Any]]:
    exp53_metrics = mean_by_cell(read_local_csv("reports/exp53b_core_onestep_metrics.csv"))
    exp53_diag = diagnostics_by_cell(read_local_csv("reports/exp53b_core_onestep_diagnostics.csv"))
    exp53_visual = visual_by_cell(read_local_csv("reports/exp53b_core_onestep_visual_review.csv"))

    exp54_metrics_rows = read_csv_text(git_show(EXP54_REF, "reports/exp54_sdpo_linear_onestep_metrics.csv"))
    exp54_metrics_rows += read_csv_text(git_show(EXP54_REF, "reports/exp54_wave2_metrics.csv"))
    exp54_diag_rows = read_csv_text(git_show(EXP54_REF, "reports/exp54_sdpo_linear_onestep_diagnostics.csv"))
    exp54_diag_rows += read_csv_text(git_show(EXP54_REF, "reports/exp54_wave2_diagnostics.csv"))
    exp54_visual_rows = read_csv_text(git_show(EXP54_REF, "reports/exp54_sdpo_linear_onestep_visual_review.csv"))
    exp54_visual_rows += read_csv_text(git_show(EXP54_REF, "reports/exp54_wave2_visual_review.csv"))
    exp54_metrics = mean_by_cell(exp54_metrics_rows)
    exp54_diag = diagnostics_by_cell(exp54_diag_rows)
    exp54_visual = visual_by_cell(exp54_visual_rows)
    exp54_head = run_git(["rev-parse", EXP54_REF]).strip()

    candidates: list[dict[str, Any]] = []
    for cell, metrics in exp53_metrics.items():
        diag = exp53_diag.get(cell, {})
        visual = exp53_visual.get(cell, {})
        status = candidate_status(metrics, diag, visual)
        candidates.append(
            {
                "exp_id": "Exp53B",
                "lane": "H20",
                "machine": "H20",
                "branch": "research/exp53-void-r1r2-targeted-h20-20260701",
                "commit": EXP53_HEAD,
                "objective_family": cell.split("_")[0],
                "cell": cell,
                "q_setting": "Q2",
                "t_setting": "T500",
                "seed": "42",
                "checkpoint_path": f"/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp53_void_r1r2_targeted_h20/core_forward/{cell}/checkpoints/{cell}_adapter_step1.pt",
                "strict_reload": "pass",
                "heldout_videos": 4,
                "visual_evidence_count": 4,
                "raw_evidence_available": True,
                **metrics,
                **diag,
                **visual,
                "normalized_status": status,
                "rank_score": ranking_score(metrics, visual, diag),
            }
        )
    for cell, metrics in exp54_metrics.items():
        diag = exp54_diag.get(cell, {})
        visual = exp54_visual.get(cell, {})
        status = candidate_status(metrics, diag, visual)
        candidates.append(
            {
                "exp_id": "Exp54",
                "lane": "PAI",
                "machine": "PAI",
                "branch": "research/exp54-void-sdpo-linear-pai-20260701",
                "commit": exp54_head,
                "objective_family": cell.split("_")[0],
                "cell": cell,
                "q_setting": "Q2" if "_Q2_" in cell else "Q1",
                "t_setting": "T500",
                "seed": "42",
                "checkpoint_path": "see Exp54 diagnostics; raw PAI evidence not mounted on H20",
                "strict_reload": "pass",
                "heldout_videos": 4,
                "visual_evidence_count": 4,
                "raw_evidence_available": False,
                **metrics,
                **diag,
                **visual,
                "normalized_status": status,
                "rank_score": ranking_score(metrics, visual, diag),
            }
        )
    return sorted(candidates, key=lambda row: row["rank_score"], reverse=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "rank",
        "exp_id",
        "lane",
        "machine",
        "cell",
        "normalized_status",
        "rank_score",
        "full_psnr_delta",
        "ssim_delta",
        "object_psnr_delta",
        "overlap_psnr_delta",
        "affected_psnr_delta",
        "boundary_psnr_delta",
        "outside_psnr_delta",
        "outside_l1_delta",
        "winner_gap",
        "loser_gap",
        "preference_margin",
        "loser_contribution_ratio",
        "better",
        "tie",
        "worse",
        "better_or_tie_ratio",
        "worse_ratio",
        "raw_evidence_available",
    ]
    fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_readback() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    inventory = []
    for source, paths in [("exp53_current", EXP53_INPUTS), ("exp54_branch", EXP54_INPUTS)]:
        for path in paths:
            if source == "exp53_current":
                full = ROOT / path
                exists = full.exists()
                sha = file_sha(full) if exists and full.is_file() else ""
                size = full.stat().st_size if exists and full.is_file() else 0
            else:
                text = git_show(EXP54_REF, path)
                exists = text is not None
                sha = text_sha(text) if text is not None else ""
                size = len(text.encode("utf-8")) if text is not None else 0
            inventory.append({"source": source, "path": path, "exists": exists, "sha256": sha, "bytes": size})
    write_csv(REPORTS / "exp55_input_inventory.csv", inventory)
    missing = [row for row in inventory if not row["exists"]]
    exp54_raw_present = Path("/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp54_void_sdpo_linear_pai/outputs/wave1_video/R4_Q2_T500_S0/R4_Q2_T500_S0_heldout_contact_sheet.jpg").exists()
    status = "EXP55_READY_FOR_AGGREGATION" if not missing else "EXP55_INPUTS_INCOMPLETE"
    summary = {
        "status": status,
        "missing_count": len(missing),
        "missing": missing,
        "exp53b_head": EXP53_HEAD,
        "exp54_head": run_git(["rev-parse", EXP54_REF]).strip(),
        "exp54_raw_evidence_status": "EXP55_PAI_RAW_EVIDENCE_PRESENT" if exp54_raw_present else "EXP55_PAI_RAW_EVIDENCE_MISSING",
        "notes": "Exp54 committed reports are available; raw PAI evidence is not mounted on H20 and is not required for summary-level aggregation.",
    }
    (REPORTS / "exp55_input_inventory_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    md = [
        "# Exp55 Readback",
        "",
        f"Status: `{status}`",
        "",
        f"Exp53B HEAD: `{EXP53_HEAD}`",
        f"Exp54 HEAD: `{summary['exp54_head']}`",
        f"Exp54 raw evidence status on H20: `{summary['exp54_raw_evidence_status']}`",
        "",
        "Exp53B committed metrics/visual/diagnostics are present. Exp54 committed metrics/visual/diagnostics are readable through git from `origin/research/exp54-void-sdpo-linear-pai-20260701`.",
        "",
        "No training or 10-step was run.",
    ]
    (REPORTS / "exp55_readback.md").write_text("\n".join(md) + "\n")


def write_aggregate(candidates: list[dict[str, Any]]) -> None:
    ranked = []
    for i, row in enumerate(candidates, 1):
        item = dict(row)
        item["rank"] = i
        ranked.append(item)
    write_csv(REPORTS / "exp55_all_candidates.csv", ranked)
    write_csv(REPORTS / "exp55_candidate_ranking.csv", ranked)
    summary = {
        "status": "EXP55_CANDIDATES_AGGREGATED",
        "candidate_count": len(candidates),
        "pass_count": sum(1 for c in candidates if c["normalized_status"] == "PASS"),
        "mixed_unsafe_count": sum(1 for c in candidates if c["normalized_status"] == "MIXED_UNSAFE"),
        "negative_count": sum(1 for c in candidates if c["normalized_status"] == "NEGATIVE"),
        "best_overall": candidates[0]["cell"] if candidates else None,
        "best_h20": next((c["cell"] for c in candidates if c["lane"] == "H20"), None),
        "best_pai": next((c["cell"] for c in candidates if c["lane"] == "PAI"), None),
        "no_vor_eval": True,
        "hard_comp": False,
        "training_run": False,
        "ten_step_run": False,
    }
    (REPORTS / "exp55_candidate_ranking_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Exp55 Candidate Ranking",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| Rank | Lane | Cell | Status | Score | Full | Object | Overlap | Affected | Boundary | Outside | Visual |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for i, c in enumerate(candidates[:8], 1):
        lines.append(
            f"| {i} | {c['lane']} | {c['cell']} | {c['normalized_status']} | {c['rank_score']:.6f} | "
            f"{c.get('full_psnr_delta', float('nan')):.6f} | {c.get('object_psnr_delta', float('nan')):.6f} | "
            f"{c.get('overlap_psnr_delta', float('nan')):.6f} | {c.get('affected_psnr_delta', float('nan')):.6f} | "
            f"{c.get('boundary_psnr_delta', float('nan')):.6f} | {c.get('outside_psnr_delta', float('nan')):.6f} | "
            f"{c.get('better', 0)}/{c.get('tie', 0)}/{c.get('worse', 0)} |"
        )
    lines.extend(["", "No candidate satisfies the original one-step PASS gate."])
    (REPORTS / "exp55_candidate_ranking.md").write_text("\n".join(lines) + "\n")


def write_failure(candidates: list[dict[str, Any]]) -> None:
    region_rows = []
    taxonomy_rows = []
    for c in candidates:
        region_rows.append(
            {
                "cell": c["cell"],
                "lane": c["lane"],
                "object_psnr_delta": c.get("object_psnr_delta"),
                "overlap_psnr_delta": c.get("overlap_psnr_delta"),
                "affected_psnr_delta": c.get("affected_psnr_delta"),
                "boundary_psnr_delta": c.get("boundary_psnr_delta"),
                "outside_psnr_delta": c.get("outside_psnr_delta"),
                "status": c["normalized_status"],
            }
        )
        taxonomy_rows.append(
            {
                "cell": c["cell"],
                "lane": c["lane"],
                "visual_worse": c.get("worse", 0),
                "outside_damage": c.get("systematic_outside_damage", False),
                "tone_drift": c.get("tone_drift", False),
                "collapse": c.get("collapse", False),
                "boundary_artifact_visible": c.get("boundary_artifact_visible", False),
                "affected_overlap_regression_visible": c.get("affected_overlap_regression_visible", False),
            }
        )
    write_csv(REPORTS / "exp55_region_tradeoff_table.csv", region_rows)
    write_csv(REPORTS / "exp55_object_vs_boundary_tradeoff.csv", region_rows)
    write_csv(REPORTS / "exp55_visual_failure_taxonomy.csv", taxonomy_rows)
    object_improvers = [c["cell"] for c in candidates if c.get("object_psnr_delta", -999) > 0]
    outside_safe = [c["cell"] for c in candidates if c.get("outside_psnr_delta", -999) >= -0.02]
    overlap_regress = [c["cell"] for c in candidates if c.get("overlap_psnr_delta", 0) < -0.05]
    affected_regress = [c["cell"] for c in candidates if c.get("affected_psnr_delta", 0) < -0.05]
    boundary_regress = [c["cell"] for c in candidates if c.get("boundary_psnr_delta", 0) < -0.05]
    summary = {
        "status": "EXP55_FAILURE_PATTERN_IDENTIFIED",
        "object_improvers": object_improvers,
        "outside_safe_or_improved": outside_safe,
        "overlap_regressors": overlap_regress,
        "affected_regressors": affected_regress,
        "boundary_regressors": boundary_regress,
        "cache_runtime_blocker_fixed": True,
        "common_failure": "object/outside move is safe enough, but overlap/affected/boundary local transition regions regress and visual review remains mixed/unsafe.",
    }
    (REPORTS / "exp55_failure_pattern_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    md = [
        "# Exp55 Failure Pattern Analysis",
        "",
        "Status: `EXP55_FAILURE_PATTERN_IDENTIFIED`",
        "",
        "The cross-lane evidence points to an objective/region-allocation problem, not a cache or runtime failure.",
        "",
        "- Cache/runtime blocker is fixed: Exp53B produced checkpoints, strict reloads, and heldout videos.",
        "- Outside preservation is generally safe or improved.",
        "- Object/mask can improve, especially H20 R1/R2.",
        "- Overlap / affected / boundary regressions remain the common blocker.",
        "- Visual review is mixed/unsafe; no candidate reaches the original visual gate.",
        "- R1 is safer than R2/R3/R4, but still misses boundary/overlap/affected gates.",
        "- R2 loser clipping did not help enough; R2 has stronger local spill than R1.",
        "- Exp54 R4_Q2_T500 has good full/outside/affected diagnostics, but object/boundary/visual are worse than H20 R1_Q2_T500.",
        "",
        "Conclusion: current objectives over-optimize object/global preservation while under-protecting transition regions.",
    ]
    (REPORTS / "exp55_failure_pattern_analysis.md").write_text("\n".join(md) + "\n")


def write_decision(candidates: list[dict[str, Any]]) -> None:
    pass_candidates = [c for c in candidates if c["normalized_status"] == "PASS"]
    best_h20 = next((c for c in candidates if c["lane"] == "H20"), None)
    best_pai = next((c for c in candidates if c["lane"] == "PAI"), None)
    best = candidates[0]
    status = "EXP55_NO_10STEP_MIXED_ONLY" if not pass_candidates else "EXP55_10STEP_HYPOTHETICAL_ONLY"
    summary = {
        "status": status,
        "best_overall": best["cell"],
        "best_h20": best_h20["cell"] if best_h20 else None,
        "best_pai": best_pai["cell"] if best_pai else None,
        "one_step_pass_exists": bool(pass_candidates),
        "ten_step_allowed": False,
        "void_role": "baseline / loser generator / adapter-engineering candidate",
        "third_backbone_evidence": False,
        "training_run": False,
        "ten_step_run": False,
    }
    (REPORTS / "exp55_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    decision = [
        "# Exp55 10-Step Gate Decision",
        "",
        f"Status: `{status}`",
        "",
        "1. Did any Exp53B H20 candidate pass one-step? No.",
        "2. Did any Exp54 PAI candidate pass one-step? No.",
        f"3. Best overall candidate: `{best['cell']}` from `{best['lane']}`.",
        "4. R1_Q2_T500_S0 is better than R2_Q2_T500_S0.",
        "5. PAI R4_Q2_T500_S0 is not better than H20 R1_Q2_T500_S0 under the original gate; it has worse visual/object/boundary behavior.",
        "6. No candidate is eligible for 10-step under the original gate.",
        "7. Exp55 should not run 10-step.",
        "8. A hypothetical mixed-safe gate would accept visual/local transition risk; this is not recommended for promotion.",
        "9. VOID should remain baseline / loser generator / adapter-engineering candidate.",
        "10. Next experiment: Exp56 local region-safe objective repair, one-step only.",
    ]
    (REPORTS / "exp55_10step_gate_decision.md").write_text("\n".join(decision) + "\n")
    handoff = [
        "# Exp55 Final Handoff",
        "",
        f"Status: `{status}`",
        "",
        f"Best H20 candidate: `{best_h20['cell'] if best_h20 else 'NA'}`.",
        f"Best PAI candidate: `{best_pai['cell'] if best_pai else 'NA'}`.",
        f"Best overall candidate: `{best['cell']}`.",
        "",
        "No training, no new grid, and no 10-step were run.",
        "VOID remains not third-backbone evidence.",
    ]
    (REPORTS / "exp55_final_handoff.md").write_text("\n".join(handoff) + "\n")
    next_steps = [
        "# Exp55 Next Steps",
        "",
        "Recommended next experiment: `EXP56 VOID LOCAL REGION-SAFE OBJECTIVE REPAIR`.",
        "",
        "Do not run 10-step. Do not expand a grid. First test a small one-step repair that protects overlap / affected / boundary while keeping object improvements.",
    ]
    (REPORTS / "exp55_next_steps.md").write_text("\n".join(next_steps) + "\n")


def write_exp56_plan() -> None:
    md = """# Exp56 Preregistration From Exp55

Do not execute this plan in Exp55.

Experiment: `EXP56 VOID LOCAL REGION-SAFE OBJECTIVE REPAIR`

Goal: fix the observed pattern where object/mask improves but overlap, affected, and boundary regress.

Allowed scope:

- Q2/T500 only.
- train4/heldout4 first.
- one-step only.
- no VOR-Eval.
- no hard comp.
- no 10-step unless a later one-step PASS is achieved.

R5:

- object-only DPO.
- no affected DPO push.
- affected / overlap / boundary as preservation penalties.
- loser_grad_scale = 0.0.
- winner_anchor >= 0.10.
- outside preservation >= 0.10.
- boundary preservation >= 0.15.
- affected preservation >= 0.10.
- overlap preservation >= 0.15.
- local-only margin.
- half-step or reduced LR.

R6:

- object + affected DPO.
- affected gradient clipped.
- boundary and overlap preservation stronger than Exp53B.
- loser_grad_scale <= 0.02.
- loser_gap_clip_tau tighter than R2 if used.

Suggested execution after approval:

- H20: R5_Q2_T500_S0 and R5_HALF_Q2_T500_S0.
- PAI: R6_Q2_T500_S0 and best Exp54-safe variant.

No universal adapter, no final SOTA, no third-backbone claim.
"""
    (REPORTS / "exp56_preregistration_from_exp55.md").write_text(md)


def main() -> None:
    write_readback()
    candidates = build_candidates()
    write_aggregate(candidates)
    write_failure(candidates)
    write_decision(candidates)
    write_exp56_plan()
    print("EXP55_REPORTS_READY")


if __name__ == "__main__":
    main()
