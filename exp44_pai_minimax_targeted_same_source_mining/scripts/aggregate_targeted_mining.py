#!/usr/bin/env python3
"""Aggregate Exp44 targeted MiniMax mining worker outputs.

This script is CPU-only. It merges worker JSONL/CSV outputs produced by
``mine_targeted_candidates.py`` into the milestone-level artifacts. Automatic
labels remain provisional; strict visual relabeling is a later milestone gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-manifest-root", required=True)
    parser.add_argument("--target-source-manifest", required=True)
    parser.add_argument("--output-manifest-root", required=True)
    parser.add_argument("--reports-root", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def numeric_summary(rows: list[dict[str, object]], key: str) -> dict[str, float | int | None]:
    vals = [finite_float(row.get(key)) for row in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {"count": len(vals), "mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}


def load_worker_rows(worker_root: Path) -> tuple[list[dict[str, object]], list[Path]]:
    paths = sorted(worker_root.glob("exp44_targeted_candidates_all_worker*.jsonl"))
    if not paths:
        paths = sorted(worker_root.glob("exp44_targeted_candidates_all.jsonl"))
    rows: list[dict[str, object]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows, paths


def source_group(row: dict[str, object]) -> str:
    return str(row.get("scene_group") or row.get("source_group") or str(row.get("sample_id", "")).rsplit("_", 1)[0])


def classify_group_yield(rows: list[dict[str, object]]) -> dict[str, object]:
    by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_group[source_group(row)].append(row)

    group_rows: list[dict[str, object]] = []
    auto_pair_count = 0
    overlap_groups = 0
    for group, items in sorted(by_group.items()):
        success = [row for row in items if row.get("auto_classification") == "SUCCESSFUL_REMOVAL_CANDIDATE"]
        failure = [row for row in items if row.get("auto_classification") == "MEDIUM_HARD_REMOVAL"]
        pair_count = min(len(success), len(failure))
        auto_pair_count += pair_count
        if pair_count:
            overlap_groups += 1
        group_rows.append(
            {
                "source_group": group,
                "num_candidates": len(items),
                "num_auto_success": len(success),
                "num_auto_medium_hard_failure": len(failure),
                "auto_same_source_pair_capacity": pair_count,
                "requires_visual_relabel": True,
            }
        )

    return {
        "group_rows": group_rows,
        "auto_same_source_pair_capacity": auto_pair_count,
        "auto_overlap_groups": overlap_groups,
    }


def main() -> None:
    args = parse_args()
    worker_root = Path(args.worker_manifest_root).resolve()
    target_source_manifest = Path(args.target_source_manifest).resolve()
    output_manifest_root = Path(args.output_manifest_root).resolve()
    reports_root = Path(args.reports_root).resolve()
    output_manifest_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    rows, worker_paths = load_worker_rows(worker_root)
    rows.sort(key=lambda row: (source_group(row), str(row.get("sample_id", "")), int(row.get("seed_index", 0) or 0)))
    if not rows:
        raise RuntimeError(f"no worker manifests found under {worker_root}")

    status = "MINIMAX_TARGETED_MINING_COMPLETED"
    if any(row.get("auto_classification") == "TECHNICAL_INVALID" for row in rows):
        status = "MINIMAX_TARGETED_MINING_PARTIAL"

    success_rows = [row for row in rows if row.get("auto_classification") == "SUCCESSFUL_REMOVAL_CANDIDATE"]
    failure_rows = [row for row in rows if row.get("auto_classification") == "MEDIUM_HARD_REMOVAL"]
    group_yield = classify_group_yield(rows)

    all_manifest = output_manifest_root / "exp44_targeted_candidates_all.jsonl"
    success_manifest = output_manifest_root / "exp44_targeted_success_auto.jsonl"
    failure_manifest = output_manifest_root / "exp44_targeted_failure_auto.jsonl"
    group_csv = reports_root / "exp44_targeted_mining_group_yield.csv"
    metrics_csv = reports_root / "exp44_targeted_mining_metrics.csv"
    summary_json = reports_root / "exp44_targeted_mining_summary.json"
    report_md = reports_root / "exp44_targeted_mining.md"

    write_jsonl(all_manifest, rows)
    write_jsonl(success_manifest, success_rows)
    write_jsonl(failure_manifest, failure_rows)
    write_csv(metrics_csv, rows)
    write_csv(group_csv, group_yield["group_rows"])

    class_counts = Counter(str(row.get("auto_classification", "")) for row in rows)
    summary = {
        "status": status,
        "target_source_manifest": str(target_source_manifest),
        "target_source_manifest_sha256": sha256_file(target_source_manifest),
        "worker_manifest_root": str(worker_root),
        "worker_manifests": [str(path) for path in worker_paths],
        "num_candidates": len(rows),
        "classification_counts": dict(sorted(class_counts.items())),
        "num_auto_success": len(success_rows),
        "num_auto_medium_hard_failure": len(failure_rows),
        "auto_same_source_pair_capacity": group_yield["auto_same_source_pair_capacity"],
        "auto_overlap_groups": group_yield["auto_overlap_groups"],
        "all_manifest": str(all_manifest),
        "all_manifest_sha256": sha256_file(all_manifest),
        "success_manifest": str(success_manifest),
        "success_manifest_sha256": sha256_file(success_manifest),
        "failure_manifest": str(failure_manifest),
        "failure_manifest_sha256": sha256_file(failure_manifest),
        "metrics": {
            key: numeric_summary(rows, key)
            for key in ("full_psnr", "mask_psnr", "boundary_psnr", "outside_psnr", "outside_mae", "temporal_diff_mae")
        },
        "visual_relabel_required": True,
        "training_run": False,
        "vor_eval_used": False,
        "hard_comp_used": False,
    }
    write_json(summary_json, summary)
    report = [
        "# Exp44 Targeted MiniMax Same-Source Mining",
        "",
        f"Status: `{status}`",
        "",
        "This milestone aggregates official MiniMax raw inference candidates from",
        "the targeted second-pass workers. Automatic labels are provisional; the",
        "next milestone must perform strict visual relabeling before any same-source",
        "pair or Stage2 handoff manifest is trusted.",
        "",
        "## Counts",
        "",
        f"- Candidates: `{len(rows)}`",
        f"- Auto successful-removal candidates: `{len(success_rows)}`",
        f"- Auto medium-hard failure candidates: `{len(failure_rows)}`",
        f"- Auto same-source pair capacity: `{summary['auto_same_source_pair_capacity']}`",
        f"- Auto overlap groups: `{summary['auto_overlap_groups']}`",
        "",
        "## Guardrails",
        "",
        "- Training run: `false`",
        "- VOR-Eval used: `false`",
        "- Hard comp used: `false`",
        "- Raw output primary: `true`",
        "",
        "## Artifacts",
        "",
        f"- All candidates: `{all_manifest}`",
        f"- Auto success: `{success_manifest}`",
        f"- Auto failure: `{failure_manifest}`",
        f"- Metrics CSV: `{metrics_csv}`",
        f"- Group yield CSV: `{group_csv}`",
        "",
    ]
    report_md.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
