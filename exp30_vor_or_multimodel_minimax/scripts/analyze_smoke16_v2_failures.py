#!/usr/bin/env python3
"""Analyze Exp30 smoke16 v2 per-candidate failure modes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


USABLE = {"MEDIUM_HARD_ELIGIBLE", "HARD_BUT_PLAUSIBLE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, default=Path("reports/exp30_multimodel_or_smoke16_metrics_v2.csv"))
    parser.add_argument("--report-md", type=Path, default=Path("reports/exp30_smoke16_v2_failure_analysis.md"))
    parser.add_argument("--report-csv", type=Path, default=Path("reports/exp30_smoke16_v2_failure_analysis.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("reports/exp30_smoke16_v2_failure_summary.json"))
    return parser.parse_args()


def f(row: dict, key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def controlled_failure(row: dict) -> tuple[str, str, str]:
    cls = row["classification"]
    reason = row["reason"]
    mask_psnr = f(row, "mask_psnr")
    outside_psnr = f(row, "outside_psnr")
    outside_mae = f(row, "outside_mae")
    temporal_ratio = f(row, "temporal_ratio")
    if row.get("technical_valid") != "yes":
        return "technical bug", "TECHNICAL_INVALID", "Do not reuse until source/output decoding is repaired."
    if cls in USABLE:
        if cls == "HARD_BUT_PLAUSIBLE":
            return "local residual too sharp", cls, "Keep as hard-plausible reference; v3 should soften this profile."
        return "bounded local texture mismatch", cls, "Keep as positive reference for v3."
    if outside_psnr < 40.0 or outside_mae > 2.0:
        return "outside preservation issue", cls, "Bug because controlled corruption should reinject outside winner."
    if temporal_ratio > 2.5:
        return "temporal discontinuity", cls, "Reduce noise/condition mix and add temporal smoothing."
    if mask_psnr > 32.0:
        return "corruption too weak / too-close", cls, "Increase local defect only if still temporally coherent."
    if mask_psnr < 8.0:
        return "corruption too strong", cls, "Lower strength and feather region."
    if "temporal" in reason.lower():
        return "temporal discontinuity", cls, "Reduce frame-independent noise."
    return "texture mismatch", cls, "Use softer object/affected-region profile."


def minimax_failure(row: dict) -> tuple[str, str, str]:
    cls = row["classification"]
    reason = row["reason"].lower()
    mask_psnr = f(row, "mask_psnr")
    outside_psnr = f(row, "outside_psnr")
    temporal_ratio = f(row, "temporal_ratio")
    if row.get("technical_valid") != "yes":
        return "decode/frame issue", "TECHNICAL_INVALID", "Do not reuse until wrapper/source issue is fixed."
    if cls in USABLE:
        if cls == "HARD_BUT_PLAUSIBLE":
            return "strong but plausible local defect", cls, "Keep as usable MiniMax candidate."
        return "bounded residual / medium-hard", cls, "Keep as usable MiniMax candidate."
    if "outside" in reason or outside_psnr < 24.0:
        return "outside damage", cls, "Do not use raw candidate unless outside preservation improves."
    if temporal_ratio > 10.0 or "temporal" in reason:
        return "temporal flicker / instability", cls, "Try fewer iterations or smoother mask/seed only under preregistration."
    if mask_psnr < 8.0 or "too far" in reason:
        return "too bad", cls, "Reject; local result too far from clean winner."
    if mask_psnr > 30.0:
        return "too-close / residual object", cls, "Reject or use only as too-close diagnostic."
    if "black" in row.get("human_visual_note", "").lower() or "smudge" in row.get("human_visual_note", "").lower():
        return "black blob / smearing", cls, "Reject unless future stack removes smearing."
    return "object/effect residual or texture hallucination", cls, "Reject for training; keep as failure mode."


def analyze(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        if row["model"] == "controlled_corruption":
            failure, refined, recommendation = controlled_failure(row)
        elif row["model"] == "minimax_official":
            failure, refined, recommendation = minimax_failure(row)
        else:
            failure, refined, recommendation = "unknown generator", row["classification"], "Audit generator."
        out.append(
            {
                "model": row["model"],
                "sample_id": row["sample_id"],
                "source_type": row.get("source_type", ""),
                "classification": row["classification"],
                "usable": row["usable"],
                "failure_category": failure,
                "refined_label": refined,
                "original_reason": row["reason"],
                "mask_psnr": row["mask_psnr"],
                "outside_psnr": row["outside_psnr"],
                "outside_mae": row["outside_mae"],
                "temporal_ratio": row["temporal_ratio"],
                "technical_valid": row["technical_valid"],
                "raw_output_mp4": row["raw_output_mp4"],
                "side_by_side_mp4": row["side_by_side_mp4"],
                "temporal_strip_16": row["temporal_strip_16"],
                "review_basis": "v2 metrics csv + continuation v3 opened 4 controlled overview pages and 4 MiniMax review pages",
                "v3_recommendation": recommendation,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "sample_id",
        "source_type",
        "classification",
        "usable",
        "failure_category",
        "refined_label",
        "original_reason",
        "mask_psnr",
        "outside_psnr",
        "outside_mae",
        "temporal_ratio",
        "technical_valid",
        "raw_output_mp4",
        "side_by_side_mp4",
        "temporal_strip_16",
        "review_basis",
        "v3_recommendation",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict]) -> dict:
    by_model = {}
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        by_model[model] = {
            "candidate_count": len(sub),
            "technical_valid": sum(1 for r in sub if r["technical_valid"] == "yes"),
            "usable": sum(1 for r in sub if r["classification"] in USABLE),
            "classification_counts": dict(Counter(r["classification"] for r in sub)),
            "failure_counts": dict(Counter(r["failure_category"] for r in sub)),
        }
    summary = {
        "status": "SMOKE16_V2_FAILURE_ANALYZED",
        "candidate_count": len(rows),
        "technical_valid": sum(1 for r in rows if r["technical_valid"] == "yes"),
        "usable": sum(1 for r in rows if r["classification"] in USABLE),
        "classification_counts": dict(Counter(r["classification"] for r in rows)),
        "by_model": by_model,
        "v3_preregistered_fix_items": [
            "controlled corruption must reduce frame-independent noise and add temporal smoothing",
            "controlled corruption must include mild/medium object and affected-soft profiles, not one aggressive profile",
            "MiniMax should remain one family in a multi-model pool, not the sole source of training losers",
            "DiffuEraser and ProPainter may be enabled only after verified stack smoke",
            "EffectErase remains diagnostic-only and excluded from smoke promotion/training pairs",
        ],
        "continuation_v3_visual_evidence_opened": {
            "controlled_corruption_overview_pages": 4,
            "minimax_review_pages": 4,
            "candidate_coverage": 32,
        },
        "gate64_unlocked": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def write_md(path: Path, rows: list[dict], summary: dict) -> None:
    cc = summary["by_model"].get("controlled_corruption", {})
    mm = summary["by_model"].get("minimax_official", {})
    lines = [
        "# Exp30 Smoke16 V2 Failure Analysis",
        "",
        "Status: `SMOKE16_V2_FAILURE_ANALYZED`",
        "",
        "No new generation, GPU inference, Gate64, adapter gate, or training was run for this milestone.",
        "",
        "## Overall",
        "",
        f"- Candidates analyzed: {summary['candidate_count']}.",
        f"- Technical-valid candidates: {summary['technical_valid']}.",
        f"- Usable candidates: {summary['usable']}.",
        f"- Classification counts: `{summary['classification_counts']}`.",
        "- Smoke16 v2 remains blocked because controlled corruption produced only 5/16 usable fallback candidates, below the preregistered >=6/16 threshold.",
        "- Continuation v3 visual readback opened 4 controlled-corruption overview pages and 4 MiniMax review pages, covering all 32 v2 candidates.",
        "",
        "## Controlled Corruption",
        "",
        f"- Technical-valid: {cc.get('technical_valid', 0)}/16.",
        f"- Usable: {cc.get('usable', 0)}/16.",
        f"- Failure counts: `{cc.get('failure_counts', {})}`.",
        "- Main failure: excessive temporal discontinuity from a single aggressive frame-wise corruption profile.",
        "- Outside preservation was not the issue for controlled corruption; outside pixels were effectively reinjected.",
        "",
        "## MiniMax Official",
        "",
        f"- Technical-valid: {mm.get('technical_valid', 0)}/16.",
        f"- Usable: {mm.get('usable', 0)}/16.",
        f"- Failure counts: `{mm.get('failure_counts', {})}`.",
        "- Main failures: outside damage, temporal flicker/instability, too-bad local outputs, residual object/effect, and occasional smudged/black artifacts.",
        "- MiniMax should remain part of a multi-model candidate pool; it is not sufficient as the only loser generator.",
        "",
        "## Samples Suitable For V3 Planning",
        "",
    ]
    for model in ("controlled_corruption", "minimax_official"):
        usable = [r for r in rows if r["model"] == model and r["classification"] in USABLE]
        lines.append(f"- `{model}` usable references: " + ", ".join(r["sample_id"] for r in usable))
    lines.extend(
        [
            "",
            "## Samples To Retain But Not Tune Against",
            "",
            "The same repaired smoke16 source rows should remain locked for smoke16 v3, but the v2 per-candidate outputs must not be used to cherry-pick source replacements. The v2 failures guide preregistered generator changes only.",
            "",
            "## Preregistered Fix Items For Smoke16 V3",
            "",
        ]
    )
    for item in summary["v3_preregistered_fix_items"]:
        lines.append(f"- {item}.")
    lines.extend(
        [
            "",
            "## Per-Candidate Failure Table",
            "",
            "| model | sample_id | classification | failure_category | v3_recommendation |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['sample_id']} | {r['classification']} | {r['failure_category']} | {r['v3_recommendation']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    with args.metrics_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    analyzed = analyze(rows)
    write_csv(args.report_csv, analyzed)
    summary = write_summary(args.summary_json, analyzed)
    write_md(args.report_md, analyzed, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
