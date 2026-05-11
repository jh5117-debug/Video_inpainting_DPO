#!/usr/bin/env python
"""Summarize VBench JSON outputs into compact JSON/CSV tables.

Besides raw per-dimension values, this computes the official VBench
leaderboard aggregate scores: quality, semantic, and total.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DIM_WEIGHT = {
    "subject consistency": 1,
    "background consistency": 1,
    "temporal flickering": 1,
    "motion smoothness": 1,
    "aesthetic quality": 1,
    "imaging quality": 1,
    "dynamic degree": 0.5,
    "object class": 1,
    "multiple objects": 1,
    "human action": 1,
    "color": 1,
    "spatial relationship": 1,
    "scene": 1,
    "appearance style": 1,
    "temporal style": 1,
    "overall consistency": 1,
}

NORMALIZE_DIC = {
    "subject consistency": {"Min": 0.1462, "Max": 1.0},
    "background consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal flickering": {"Min": 0.6293, "Max": 1.0},
    "motion smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic quality": {"Min": 0.0, "Max": 1.0},
    "imaging quality": {"Min": 0.0, "Max": 1.0},
    "object class": {"Min": 0.0, "Max": 1.0},
    "multiple objects": {"Min": 0.0, "Max": 1.0},
    "human action": {"Min": 0.0, "Max": 1.0},
    "color": {"Min": 0.0, "Max": 1.0},
    "spatial relationship": {"Min": 0.0, "Max": 1.0},
    "scene": {"Min": 0.0, "Max": 0.8222},
    "appearance style": {"Min": 0.0009, "Max": 0.2855},
    "temporal style": {"Min": 0.0, "Max": 0.364},
    "overall consistency": {"Min": 0.0, "Max": 0.364},
}

QUALITY_LIST = [
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "aesthetic quality",
    "imaging quality",
    "dynamic degree",
]

SEMANTIC_LIST = [
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency",
]

QUALITY_WEIGHT = 4
SEMANTIC_WEIGHT = 1


def extract_average(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)) and value:
        return extract_average(value[0])
    if isinstance(value, dict):
        for key in ("avg", "average", "mean", "score"):
            if key in value:
                return extract_average(value[key])
    return None


def normalize_dim_name(dim: str) -> str:
    return dim.replace("_", " ").strip().lower()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def weighted_average(values: dict[str, float | None], dims: list[str]) -> float | None:
    total = 0.0
    weight_total = 0.0
    for dim in dims:
        score = values.get(dim)
        if score is None:
            return None
        weight = DIM_WEIGHT[dim]
        total += score * weight
        weight_total += weight
    return total / weight_total if weight_total else None


def official_vbench_scores(raw_scores: dict[str, float | None]) -> dict[str, Any]:
    normalized: dict[str, float | None] = {}
    for dim, limits in NORMALIZE_DIC.items():
        raw = raw_scores.get(dim)
        if raw is None:
            normalized[dim] = None
            continue
        normalized[dim] = clamp01((raw - limits["Min"]) / (limits["Max"] - limits["Min"]))

    quality = weighted_average(normalized, QUALITY_LIST)
    semantic = weighted_average(normalized, SEMANTIC_LIST)
    total = None
    if quality is not None and semantic is not None:
        total = (quality * QUALITY_WEIGHT + semantic * SEMANTIC_WEIGHT) / (
            QUALITY_WEIGHT + SEMANTIC_WEIGHT
        )
    return {
        "quality_score": quality,
        "semantic_score": semantic,
        "total_score": total,
        "quality_score_percent": quality * 100 if quality is not None else None,
        "semantic_score_percent": semantic * 100 if semantic is not None else None,
        "total_score_percent": total * 100 if total is not None else None,
        "normalized_dimensions": normalized,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--output_csv", type=Path, default=None)
    args = parser.parse_args()

    with args.result_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    raw_scores: dict[str, float | None] = {}
    for dim, value in sorted(raw.items()):
        avg = extract_average(value)
        canonical = normalize_dim_name(dim)
        raw_scores[canonical] = avg
        rows.append({"dimension": dim, "canonical_dimension": canonical, "score": avg})

    valid = [r["score"] for r in rows if r["score"] is not None]
    official_scores = official_vbench_scores(raw_scores)
    summary = {
        "source": str(args.result_json),
        "num_dimensions": len(rows),
        "num_valid": len(valid),
        "mean_score": sum(valid) / len(valid) if valid else None,
        **official_scores,
        "dimensions": rows,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["dimension", "canonical_dimension", "score", "normalized_score"]
            )
            writer.writeheader()
            normalized = official_scores["normalized_dimensions"]
            for row in rows:
                row = dict(row)
                row["normalized_score"] = normalized.get(row["canonical_dimension"])
                writer.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
