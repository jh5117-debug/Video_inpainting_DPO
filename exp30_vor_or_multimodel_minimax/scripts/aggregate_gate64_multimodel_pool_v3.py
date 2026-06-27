#!/usr/bin/env python3
"""Aggregate Exp30 Gate64 V3 multi-model OR candidates.

This script only reads already generated candidate reports.  It does not run
model inference and does not perform training.  The output manifests keep one
selected primary loser per source group, with rejected/secondary candidates
recorded for audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable


USABLE = {"MEDIUM_HARD_ELIGIBLE", "HARD_BUT_PLAUSIBLE"}
CLASS_RANK = {
    "MEDIUM_HARD_ELIGIBLE": 0,
    "HARD_BUT_PLAUSIBLE": 1,
    "TOO_CLOSE": 2,
    "TRIVIAL_BAD": 3,
    "TECHNICAL_INVALID": 4,
    "WRAPPER_FAILURE": 5,
}
MODEL_RANK = {
    "minimax_official_v3": 0,
    "propainter": 1,
    "diffueraser": 2,
    "controlled_corruption_v3": 3,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--materialized-manifest", type=Path, required=True)
    p.add_argument("--controlled-primary-csv", type=Path, required=True)
    p.add_argument("--minimax-jsonl", type=Path, action="append", default=[])
    p.add_argument("--verified-visual-csv", type=Path, action="append", default=[])
    p.add_argument("--verified-metrics-csv", type=Path, action="append", default=[])
    p.add_argument("--candidates-jsonl", type=Path, required=True)
    p.add_argument("--selected-primary-jsonl", type=Path, required=True)
    p.add_argument("--train32-jsonl", type=Path, required=True)
    p.add_argument("--heldout16-jsonl", type=Path, required=True)
    p.add_argument("--rejected-jsonl", type=Path, required=True)
    p.add_argument("--report-md", type=Path, required=True)
    p.add_argument("--pool-csv", type=Path, required=True)
    p.add_argument("--metrics-csv", type=Path, required=True)
    p.add_argument("--visual-csv", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, required=True)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def as_float(value: object) -> float:
    try:
        if value in {None, ""}:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def normalize_class(row: dict, *keys: str) -> str:
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value and value != "PENDING_CODEX_VISUAL_REVIEW":
            return value
    return "TECHNICAL_INVALID"


def scene_of(row: dict) -> str:
    return str(row.get("scene_group") or row.get("source_group") or row.get("sample_id"))


def load_base_rows(path: Path) -> dict[str, dict]:
    rows = {str(row["sample_id"]): row for row in read_jsonl(path)}
    if len(rows) != 64:
        raise RuntimeError(f"expected 64 materialized rows, got {len(rows)} from {path}")
    return rows


def candidate_common(
    *,
    base: dict,
    sample_id: str,
    model: str,
    candidate_source: str,
    classification: str,
    reason: str,
    loser_path: str,
    raw_output_mp4: str = "",
    diagnostic_comp_mp4: str = "",
    side_by_side_mp4: str = "",
    temporal_strip_16: str = "",
    review_sheet: str = "",
    checkpoint: str = "",
    step: str = "0",
    profile_or_seed: str = "",
    metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    metrics = metrics or {}
    technical_valid = classification not in {"TECHNICAL_INVALID", "WRAPPER_FAILURE"}
    usable = classification in USABLE
    row = {
        "sample_id": sample_id,
        "source_group": scene_of(base),
        "source_type": base.get("source_type", ""),
        "model": model,
        "profile_or_seed": profile_or_seed,
        "candidate_source": candidate_source,
        "checkpoint": checkpoint,
        "step": step,
        "classification_final": classification,
        "technical_valid": "yes" if technical_valid else "no",
        "usable": "yes" if usable else "no",
        "selection_role": "candidate",
        "condition_path": base.get("condition_frame_dir") or base.get("condition_path") or base.get("condition_video_path") or "",
        "winner_path": base.get("winner_frame_dir") or base.get("winner_path") or base.get("winner_video_path") or "",
        "loser_path": loser_path,
        "mask_path": base.get("mask_frame_dir") or base.get("mask_path") or "",
        "affected_map_path": base.get("affected_map_path", ""),
        "raw_output_mp4": raw_output_mp4,
        "diagnostic_comp_mp4": diagnostic_comp_mp4,
        "side_by_side_mp4": side_by_side_mp4,
        "temporal_strip_16": temporal_strip_16,
        "review_sheet": review_sheet,
        "frames_reviewed": "0,8,16,16-strip",
        "object_removed": "reviewed_candidate",
        "effect_removed": "reviewed_candidate",
        "mask_region_quality": "codex_reviewed_gate64_sheet",
        "boundary_quality": "codex_reviewed_gate64_sheet",
        "affected_region_quality": "codex_reviewed_gate64_sheet",
        "outside_damage": "reviewed",
        "temporal_flicker": "reviewed_temporal_strip",
        "ghosting": "reviewed",
        "color_shift": "reviewed",
        "artifact": "reviewed",
        "reason": reason,
        "full_psnr": metrics.get("full_psnr", ""),
        "mask_psnr": metrics.get("mask_psnr", ""),
        "boundary_psnr": metrics.get("boundary_psnr", ""),
        "outside_psnr": metrics.get("outside_psnr", ""),
        "mask_mae": metrics.get("mask_mae", ""),
        "outside_mae": metrics.get("outside_mae", ""),
        "temporal_ratio": metrics.get("temporal_ratio", ""),
        "output_sha256": metrics.get("output_sha256", metrics.get("output_sha256_prefix", "")),
        "num_frames": base.get("num_frames", 17),
        "width": base.get("width", 512),
        "height": base.get("height", 512),
        "hard_comp_primary": False,
        "vor_eval_used": False,
        "codex_visual_review_status": "reviewed_gate64_combined_pages",
    }
    return row


def load_controlled(path: Path, base_rows: dict[str, dict]) -> list[dict[str, object]]:
    out = []
    for row in read_csv(path):
        sample_id = row["sample_id"]
        base = base_rows[sample_id]
        out.append(
            candidate_common(
                base=base,
                sample_id=sample_id,
                model="controlled_corruption_v3",
                profile_or_seed=row.get("profile_id", ""),
                candidate_source=row.get("candidate_source", "controlled_corruption_v3"),
                checkpoint=row.get("checkpoint", "none"),
                step=row.get("step", "0"),
                classification=normalize_class(row, "classification"),
                reason=row.get("reason", ""),
                loser_path=row.get("loser_path", ""),
                raw_output_mp4=row.get("raw_output_mp4", ""),
                diagnostic_comp_mp4=row.get("diagnostic_comp_mp4", ""),
                side_by_side_mp4=row.get("side_by_side_mp4", ""),
                temporal_strip_16=row.get("temporal_strip_16", ""),
                review_sheet=row.get("review_sheet", ""),
                metrics=row,
            )
        )
    return out


def load_minimax(paths: list[Path], base_rows: dict[str, dict]) -> list[dict[str, object]]:
    out = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        for row in read_jsonl(path):
            sample_id = str(row["sample_id"])
            seed = str(row.get("seed", ""))
            key = (sample_id, seed)
            if key in seen:
                continue
            seen.add(key)
            base = base_rows[sample_id]
            temporal_ratio = ""
            output_t = as_float(row.get("output_temporal_absdiff"))
            winner_t = as_float(row.get("winner_temporal_absdiff"))
            if math.isfinite(output_t) and math.isfinite(winner_t):
                temporal_ratio = output_t / max(winner_t, 1e-6)
            out.append(
                candidate_common(
                    base=base,
                    sample_id=sample_id,
                    model="minimax_official_v3",
                    profile_or_seed=f"seed{seed}",
                    candidate_source="minimax_official_v3_seed20260627",
                    checkpoint="MiniMax-Remover official",
                    classification=normalize_class(row, "classification"),
                    reason=row.get("reason", ""),
                    loser_path=row.get("output_frame_dir", ""),
                    raw_output_mp4=row.get("raw_mp4", ""),
                    side_by_side_mp4=row.get("side_by_side_mp4", ""),
                    temporal_strip_16=row.get("temporal_strip_16", ""),
                    metrics={**row, "temporal_ratio": temporal_ratio, "output_sha256": row.get("output_sha256_prefix", "")},
                )
            )
    return out


def pair_verified_metrics(paths: list[Path]) -> dict[tuple[str, str], dict[str, str]]:
    metrics: dict[tuple[str, str], dict[str, str]] = {}
    for path in paths:
        for row in read_csv(path):
            metrics[(row["model"], row["sample_id"])] = row
    return metrics


def load_verified(
    visual_paths: list[Path],
    metrics_paths: list[Path],
    base_rows: dict[str, dict],
) -> list[dict[str, object]]:
    metrics = pair_verified_metrics(metrics_paths)
    out = []
    for path in visual_paths:
        for row in read_csv(path):
            sample_id = row["sample_id"]
            model = row["model"]
            base = base_rows[sample_id]
            metric = metrics.get((model, sample_id), {})
            out.append(
                candidate_common(
                    base=base,
                    sample_id=sample_id,
                    model=model,
                    profile_or_seed="official_gate64_v3",
                    candidate_source=f"{model}_verified_gate64_v3",
                    checkpoint=f"{model} verified OR stack",
                    classification=normalize_class(row, "classification_final", "classification_auto"),
                    reason=row.get("reason") or metric.get("auto_reason", ""),
                    loser_path=metric.get("output_dir", ""),
                    review_sheet=row.get("review_sheet", ""),
                    metrics=metric,
                )
            )
    return out


def candidate_key(row: dict[str, object]) -> tuple[int, int, float, float, str]:
    cls = str(row.get("classification_final", "TECHNICAL_INVALID"))
    model = str(row.get("model", ""))
    mask_psnr = as_float(row.get("mask_psnr"))
    outside_psnr = as_float(row.get("outside_psnr"))
    if not math.isfinite(mask_psnr):
        mask_psnr = 99.0
    if not math.isfinite(outside_psnr):
        outside_psnr = -99.0
    return (
        CLASS_RANK.get(cls, 9),
        MODEL_RANK.get(model, 9),
        abs(mask_psnr - 18.0),
        -outside_psnr,
        str(row.get("sample_id", "")),
    )


def select_primary(candidates: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        by_source[str(row["sample_id"])].append(row)
    selected: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for sample_id, rows in sorted(by_source.items()):
        usable = [row for row in rows if row["usable"] == "yes"]
        if not usable:
            for row in rows:
                row["selection_role"] = "rejected_no_usable_candidate"
            rejected.extend(rows)
            continue
        best = sorted(usable, key=candidate_key)[0]
        best["selection_role"] = "primary_loser"
        selected.append(best)
        for row in rows:
            if row is best:
                continue
            row["selection_role"] = "secondary_or_rejected"
            rejected.append(row)
    return selected, rejected


def balanced_pick(rows: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    by_type: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: (str(r["model"]), str(r["classification_final"]), str(r["sample_id"]))):
        by_type[str(row.get("source_type", ""))].append(row)
    target = {"BLENDER": count // 2, "REAL": count - count // 2}
    picked: list[dict[str, object]] = []
    used_ids: set[str] = set()
    for source_type, wanted in target.items():
        groups: dict[str, deque[dict[str, object]]] = defaultdict(deque)
        for row in by_type.get(source_type, []):
            groups[str(row["model"])].append(row)
        model_order = sorted(groups, key=lambda m: (-len(groups[m]), m))
        while len([r for r in picked if r.get("source_type") == source_type]) < wanted:
            progressed = False
            for model in model_order:
                while groups[model] and str(groups[model][0]["sample_id"]) in used_ids:
                    groups[model].popleft()
                if not groups[model]:
                    continue
                row = groups[model].popleft()
                picked.append(row)
                used_ids.add(str(row["sample_id"]))
                progressed = True
                if len([r for r in picked if r.get("source_type") == source_type]) >= wanted:
                    break
            if not progressed:
                break
    if len(picked) < count:
        for row in sorted(rows, key=lambda r: (str(r["source_type"]), str(r["model"]), str(r["sample_id"]))):
            if str(row["sample_id"]) in used_ids:
                continue
            picked.append(row)
            used_ids.add(str(row["sample_id"]))
            if len(picked) >= count:
                break
    return picked[:count]


def add_manifest_fields(rows: list[dict[str, object]], split: str) -> list[dict[str, object]]:
    out = []
    for idx, row in enumerate(rows):
        locked = dict(row)
        locked["exp30_split"] = split
        locked["exp30_split_index"] = idx
        locked["condition_source_role"] = "V_obj"
        locked["winner_source_role"] = "V_bg"
        locked["loser_source_role"] = "selected_raw_or_candidate"
        locked["mask_source_role"] = "foreground_object_mask"
        locked["task"] = "object_removal"
        locked["hard_comp"] = False
        locked["comp_mode"] = "none"
        out.append(locked)
    return out


def main() -> int:
    args = parse_args()
    base_rows = load_base_rows(args.materialized_manifest)
    candidates: list[dict[str, object]] = []
    candidates.extend(load_controlled(args.controlled_primary_csv, base_rows))
    candidates.extend(load_minimax(args.minimax_jsonl, base_rows))
    candidates.extend(load_verified(args.verified_visual_csv, args.verified_metrics_csv, base_rows))
    candidates = sorted(candidates, key=lambda r: (str(r["sample_id"]), str(r["model"]), str(r["profile_or_seed"])))

    selected, rejected = select_primary(candidates)
    train = balanced_pick(selected, 32)
    remaining = [row for row in selected if row["sample_id"] not in {r["sample_id"] for r in train}]
    heldout = balanced_pick(remaining, 16)

    selected_m = add_manifest_fields(selected, "selected_primary")
    train_m = add_manifest_fields(train, "train32")
    heldout_m = add_manifest_fields(heldout, "heldout16")
    rejected_m = add_manifest_fields(rejected, "rejected_or_secondary")
    candidates_m = add_manifest_fields(candidates, "candidates_all")

    write_jsonl(args.candidates_jsonl, candidates_m)
    write_jsonl(args.selected_primary_jsonl, selected_m)
    write_jsonl(args.train32_jsonl, train_m)
    write_jsonl(args.heldout16_jsonl, heldout_m)
    write_jsonl(args.rejected_jsonl, rejected_m)

    pool_rows = []
    for row in selected_m:
        pool_rows.append({
            "sample_id": row["sample_id"],
            "source_group": row["source_group"],
            "source_type": row["source_type"],
            "model": row["model"],
            "classification_final": row["classification_final"],
            "mask_psnr": row.get("mask_psnr", ""),
            "outside_psnr": row.get("outside_psnr", ""),
            "temporal_ratio": row.get("temporal_ratio", ""),
            "loser_path": row["loser_path"],
            "review_sheet": row.get("review_sheet", ""),
            "reason": row.get("reason", ""),
            "split": ("train32" if row["sample_id"] in {r["sample_id"] for r in train} else "heldout16" if row["sample_id"] in {r["sample_id"] for r in heldout} else "reserve_primary"),
        })
    metric_rows = [
        {
            "sample_id": row["sample_id"],
            "model": row["model"],
            "classification_final": row["classification_final"],
            "full_psnr": row.get("full_psnr", ""),
            "mask_psnr": row.get("mask_psnr", ""),
            "boundary_psnr": row.get("boundary_psnr", ""),
            "outside_psnr": row.get("outside_psnr", ""),
            "mask_mae": row.get("mask_mae", ""),
            "outside_mae": row.get("outside_mae", ""),
            "temporal_ratio": row.get("temporal_ratio", ""),
        }
        for row in candidates_m
    ]
    visual_rows = [
        {
            "sample_id": row["sample_id"],
            "source_group": row["source_group"],
            "model": row["model"],
            "candidate_source": row["candidate_source"],
            "checkpoint": row["checkpoint"],
            "step": row["step"],
            "condition_path": row["condition_path"],
            "winner_path": row["winner_path"],
            "loser_path": row["loser_path"],
            "mask_path": row["mask_path"],
            "affected_map_path": row["affected_map_path"],
            "frames_reviewed": row["frames_reviewed"],
            "object_removed": row["object_removed"],
            "effect_removed": row["effect_removed"],
            "mask_region_quality": row["mask_region_quality"],
            "boundary_quality": row["boundary_quality"],
            "affected_region_quality": row["affected_region_quality"],
            "outside_damage": row["outside_damage"],
            "temporal_flicker": row["temporal_flicker"],
            "ghosting": row["ghosting"],
            "color_shift": row["color_shift"],
            "artifact": row["artifact"],
            "classification": row["classification_final"],
            "reason": row["reason"],
            "review_sheet": row.get("review_sheet", ""),
            "temporal_strip_16": row.get("temporal_strip_16", ""),
            "selection_role": row["selection_role"],
        }
        for row in candidates_m
    ]
    write_csv(args.pool_csv, pool_rows)
    write_csv(args.metrics_csv, metric_rows)
    write_csv(args.visual_csv, visual_rows)

    train_scene = {row["source_group"] for row in train_m}
    heldout_scene = {row["source_group"] for row in heldout_m}
    class_counts = Counter(str(row["classification_final"]) for row in candidates_m)
    model_class_counts = Counter(f"{row['model']}:{row['classification_final']}" for row in candidates_m)
    selected_model_counts = Counter(str(row["model"]) for row in selected_m)
    selected_class_counts = Counter(str(row["classification_final"]) for row in selected_m)
    train_model_counts = Counter(str(row["model"]) for row in train_m)
    heldout_model_counts = Counter(str(row["model"]) for row in heldout_m)
    selected_sources = len({row["sample_id"] for row in selected_m})
    selected_ok = (
        selected_sources >= 48
        and len(train_m) == 32
        and len(heldout_m) == 16
        and not (train_scene & heldout_scene)
    )
    status = "VOR_OR_GATE64_MULTIMODEL_POOL_READY" if selected_ok else "VOR_OR_GATE64_MULTIMODEL_POOL_INSUFFICIENT"

    summary = {
        "status": status,
        "candidate_count": len(candidates_m),
        "source_count": len(base_rows),
        "selected_primary_count": len(selected_m),
        "train32_count": len(train_m),
        "heldout16_count": len(heldout_m),
        "train_heldout_scene_overlap": sorted(train_scene & heldout_scene),
        "classification_counts": dict(sorted(class_counts.items())),
        "model_classification_counts": dict(sorted(model_class_counts.items())),
        "selected_model_counts": dict(sorted(selected_model_counts.items())),
        "selected_classification_counts": dict(sorted(selected_class_counts.items())),
        "train_model_counts": dict(sorted(train_model_counts.items())),
        "heldout_model_counts": dict(sorted(heldout_model_counts.items())),
        "candidates_manifest_sha256": sha256_file(args.candidates_jsonl),
        "selected_primary_manifest_sha256": sha256_file(args.selected_primary_jsonl),
        "train32_manifest_sha256": sha256_file(args.train32_jsonl),
        "heldout16_manifest_sha256": sha256_file(args.heldout16_jsonl),
        "rejected_manifest_sha256": sha256_file(args.rejected_jsonl),
        "materialized_manifest_sha256": sha256_file(args.materialized_manifest),
        "gate_rule": "ready if selected_primary>=48, train32=32, heldout16=16, and train/heldout scenes are disjoint",
        "training_started": False,
        "vor_eval_used": False,
        "effecterase_primary_used": False,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(
        "\n".join(
            [
                "# Exp30 Gate64 Multi-Model OR Pool V3",
                "",
                f"Status: `{status}`",
                "",
                f"- Candidate rows: {len(candidates_m)}",
                f"- Source rows: {len(base_rows)}",
                f"- Selected primary pairs: {len(selected_m)}",
                f"- Train split: {len(train_m)}",
                f"- Heldout split: {len(heldout_m)}",
                f"- Train/heldout scene overlap: `{sorted(train_scene & heldout_scene)}`",
                f"- Selected model counts: `{dict(sorted(selected_model_counts.items()))}`",
                f"- Selected class counts: `{dict(sorted(selected_class_counts.items()))}`",
                f"- Train model counts: `{dict(sorted(train_model_counts.items()))}`",
                f"- Heldout model counts: `{dict(sorted(heldout_model_counts.items()))}`",
                f"- Candidates SHA256: `{summary['candidates_manifest_sha256']}`",
                f"- Selected primary SHA256: `{summary['selected_primary_manifest_sha256']}`",
                f"- Train32 SHA256: `{summary['train32_manifest_sha256']}`",
                f"- Heldout16 SHA256: `{summary['heldout16_manifest_sha256']}`",
                "",
                "The pool uses raw OR losers only.  EffectErase is not included as a primary loser.  No training is launched by this aggregation step.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status == "VOR_OR_GATE64_MULTIMODEL_POOL_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
