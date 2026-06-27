#!/usr/bin/env python3
"""Aggregate Exp30 Smoke32 V3 multi-model OR candidates."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


USABLE = {"MEDIUM_HARD_ELIGIBLE", "HARD_BUT_PLAUSIBLE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controlled-primary-csv", type=Path, required=True)
    parser.add_argument("--verified-visual-csv", type=Path, required=True)
    parser.add_argument("--verified-metrics-csv", type=Path, required=True)
    parser.add_argument("--minimax-csv", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--best-per-source-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object) -> float:
    try:
        if value in {None, ""}:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def normalize_class(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = (row.get(key) or "").strip()
        if value and value != "PENDING_CODEX_VISUAL_REVIEW":
            return value
    return "TECHNICAL_INVALID"


def candidate(
    *,
    sample_id: str,
    model: str,
    profile_or_seed: str,
    classification: str,
    source: str,
    mask_psnr: object = "",
    outside_psnr: object = "",
    temporal_ratio: object = "",
    evidence: str = "",
    visual_reason: str = "",
) -> dict[str, object]:
    technical_valid = classification != "TECHNICAL_INVALID"
    usable = classification in USABLE
    return {
        "sample_id": sample_id,
        "model": model,
        "profile_or_seed": profile_or_seed,
        "classification_final": classification,
        "technical_valid": "yes" if technical_valid else "no",
        "usable": "yes" if usable else "no",
        "source": source,
        "mask_psnr": mask_psnr,
        "outside_psnr": outside_psnr,
        "temporal_ratio": temporal_ratio,
        "evidence": evidence,
        "visual_reason": visual_reason,
    }


def load_controlled(path: Path) -> list[dict[str, object]]:
    rows = []
    for row in read_csv(path):
        rows.append(
            candidate(
                sample_id=row["sample_id"],
                model="controlled_corruption_v3",
                profile_or_seed=row.get("profile_id", ""),
                classification=normalize_class(row, "classification"),
                source="controlled_v3_primary",
                mask_psnr=row.get("mask_psnr", ""),
                outside_psnr=row.get("outside_psnr", ""),
                temporal_ratio=row.get("temporal_ratio", ""),
                evidence=row.get("temporal_strip_16", ""),
                visual_reason=row.get("reason", ""),
            )
        )
    return rows


def load_verified(visual_path: Path, metrics_path: Path) -> list[dict[str, object]]:
    metrics = {(r["model"], r["sample_id"]): r for r in read_csv(metrics_path)}
    rows = []
    for row in read_csv(visual_path):
        key = (row["model"], row["sample_id"])
        metric = metrics.get(key, {})
        rows.append(
            candidate(
                sample_id=row["sample_id"],
                model=row["model"],
                profile_or_seed="official_full16",
                classification=normalize_class(row, "classification_final", "classification_auto"),
                source="verified_generator_full16",
                mask_psnr=metric.get("mask_psnr", ""),
                outside_psnr=metric.get("outside_psnr", ""),
                temporal_ratio=metric.get("temporal_ratio", ""),
                evidence=row.get("review_sheet") or metric.get("output_dir", ""),
                visual_reason=row.get("reason", ""),
            )
        )
    return rows


def load_minimax(path: Path) -> list[dict[str, object]]:
    rows = []
    for row in read_csv(path):
        rows.append(
            candidate(
                sample_id=row["sample_id"],
                model="minimax_official_v3",
                profile_or_seed=f"seed{row.get('seed', '')}",
                classification=normalize_class(row, "classification"),
                source="minimax_v3_seed20260627",
                mask_psnr=row.get("mask_psnr", ""),
                outside_psnr=row.get("outside_psnr", ""),
                temporal_ratio=(
                    as_float(row.get("output_temporal_absdiff"))
                    / max(as_float(row.get("winner_temporal_absdiff")), 1e-6)
                ),
                evidence=row.get("temporal_strip_16", ""),
                visual_reason=row.get("reason", ""),
            )
        )
    return rows


def best_key(row: dict[str, object]) -> tuple[int, float, float]:
    cls = str(row["classification_final"])
    if cls == "MEDIUM_HARD_ELIGIBLE":
        rank = 0
    elif cls == "HARD_BUT_PLAUSIBLE":
        rank = 1
    elif cls == "TOO_CLOSE":
        rank = 2
    elif cls == "TRIVIAL_BAD":
        rank = 3
    else:
        rank = 4
    mask_psnr = as_float(row.get("mask_psnr"))
    outside_psnr = as_float(row.get("outside_psnr"))
    return (rank, abs(mask_psnr - 18.0), -outside_psnr)


def main() -> int:
    args = parse_args()
    candidates = []
    candidates.extend(load_controlled(args.controlled_primary_csv))
    candidates.extend(load_verified(args.verified_visual_csv, args.verified_metrics_csv))
    candidates.extend(load_minimax(args.minimax_csv))
    candidates = sorted(candidates, key=lambda r: (str(r["sample_id"]), str(r["model"]), str(r["profile_or_seed"])))

    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        by_source[str(row["sample_id"])].append(row)
    best = [sorted(rows, key=best_key)[0] for rows in by_source.values()]
    best = sorted(best, key=lambda r: str(r["sample_id"]))

    candidate_count = len(candidates)
    source_count = len(by_source)
    technical_valid = sum(1 for r in candidates if r["technical_valid"] == "yes")
    total_usable = sum(1 for r in candidates if r["usable"] == "yes")
    source_usable = sum(1 for r in best if r["usable"] == "yes")
    controlled_usable_sources = {
        str(r["sample_id"])
        for r in candidates
        if r["model"] == "controlled_corruption_v3" and r["usable"] == "yes"
    }
    usable_families = sorted({str(r["model"]) for r in candidates if r["usable"] == "yes"})
    model_counts = Counter(f"{r['model']}:{r['classification_final']}" for r in candidates)
    class_counts = Counter(str(r["classification_final"]) for r in candidates)
    best_counts = Counter(str(r["classification_final"]) for r in best)

    tech_rate = technical_valid / candidate_count if candidate_count else 0.0
    pass_gate = (
        candidate_count >= 64
        and source_count == 16
        and tech_rate >= 0.90
        and total_usable >= 14
        and source_usable >= 10
        and len(controlled_usable_sources) >= 7
        and len(usable_families) >= 2
    )
    status = "MULTIMODEL_OR_SMOKE32_V3_PASS" if pass_gate else "MULTIMODEL_OR_SMOKE32_V3_BLOCKED"

    summary = {
        "status": status,
        "candidate_count": candidate_count,
        "source_count": source_count,
        "technical_valid_candidates": technical_valid,
        "technical_valid_rate": tech_rate,
        "total_usable_candidates": total_usable,
        "best_per_source_usable": source_usable,
        "controlled_v3_usable_source_count": len(controlled_usable_sources),
        "usable_generator_families": usable_families,
        "classification_counts": dict(sorted(class_counts.items())),
        "model_classification_counts": dict(sorted(model_counts.items())),
        "best_per_source_counts": dict(sorted(best_counts.items())),
        "gate_rule": (
            "pass if candidate_count>=64, source_count=16, technical-valid>=90%, "
            "usable>=14, best-per-source usable>=10/16, controlled usable sources>=7/16, "
            "and usable candidates from at least two generator families"
        ),
    }

    write_csv(args.candidates_csv, candidates)
    write_csv(args.best_per_source_csv, best)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(
        "\n".join(
            [
                "# Exp30 Multi-Model OR Smoke32 V3 Aggregate",
                "",
                f"Status: `{status}`",
                "",
                f"- Candidate rows: {candidate_count}",
                f"- Source rows: {source_count}",
                f"- Technical-valid candidates: {technical_valid}/{candidate_count} ({tech_rate:.3f})",
                f"- Total usable candidates: {total_usable}",
                f"- Best-per-source usable: {source_usable}/16",
                f"- Controlled usable source coverage: {len(controlled_usable_sources)}/16",
                f"- Usable generator families: `{usable_families}`",
                f"- Best-per-source counts: `{dict(sorted(best_counts.items()))}`",
                f"- Model classification counts: `{dict(sorted(model_counts.items()))}`",
                "",
                "Smoke32 does not launch Gate64 or training by itself; it only decides whether the limited Gate64 pool may be prepared next.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if pass_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
