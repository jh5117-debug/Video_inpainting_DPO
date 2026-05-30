#!/usr/bin/env python3
"""Select medium-hard generated loser candidates for VideoDPO manifests."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

DEFAULT_SELECTION_CONFIG = Path("configs/generation/medium_hard_balanced_selection_v1.yaml")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_config(path: str | Path = DEFAULT_SELECTION_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _quality(row: dict[str, Any]) -> float:
    try:
        value = float(row.get("quality_score", 0.0))
        return value if math.isfinite(value) else 0.0
    except Exception:
        return 0.0


def _ok(row: dict[str, Any]) -> bool:
    return str(row.get("status", "")).upper() == "OK"


def _model(row: dict[str, Any]) -> str:
    return str(row.get("generation_model") or row.get("model") or "unknown")


def _bucket(row: dict[str, Any]) -> str:
    return str(row.get("defect_bucket") or "unscored")


def _in_band(row: dict[str, Any], qmin: float, qmax: float) -> bool:
    q = _quality(row)
    return qmin <= q <= qmax


def _source_pressure(row: dict[str, Any], source_counts: dict[str, int], weights: dict[str, float]) -> float:
    model = _model(row)
    return source_counts.get(model, 0) / max(0.01, float(weights.get(model, 1.0)))


def _candidate_id(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("sample_id", "")),
        str(row.get("mask_id", "")),
        _model(row),
    ]
    return "::".join(parts)


def select_for_sample(
    candidates: list[dict[str, Any]],
    source_counts: dict[str, int],
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    qcfg = config["quality_score"]
    qmin = float(qcfg["eligible_min"])
    qmax = float(qcfg["eligible_max"])
    target = float(qcfg["target"])
    weights = {str(k): float(v) for k, v in config.get("source_weights", {}).items()}
    before = dict(source_counts)
    ok = [r for r in candidates if _ok(r)]
    ordered_all = sorted(ok, key=lambda r: (_quality(r), _model(r), str(r.get("mask_id", ""))))
    if not ok:
        return None, None, {
            "selection_policy": config["policy_name"],
            "fallback_reason": "no_successful_candidates",
            "source_counts_before": before,
            "source_counts_after": dict(source_counts),
            "source_weights": weights,
            "quality_band": [qmin, qmax],
            "quality_target": target,
            "candidate_quality_order": [],
        }

    eligible = [r for r in ok if _in_band(r, qmin, qmax)]
    fallback_reason = ""
    pool = eligible
    if not pool:
        pool = ok
        fallback_reason = "no_candidate_in_quality_band"

    ranked = sorted(
        pool,
        key=lambda r: (
            0 if _in_band(r, qmin, qmax) else 1,
            _source_pressure(r, source_counts, weights),
            abs(_quality(r) - target),
            _quality(r),
            _model(r),
            str(r.get("mask_id", "")),
        ),
    )
    primary = ranked[0]
    primary_model = _model(primary)
    primary_bucket = _bucket(primary)

    secondary_pool = [r for r in ranked if _candidate_id(r) != _candidate_id(primary)]
    if not secondary_pool:
        secondary = primary
        fallback_reason = fallback_reason or "only_one_successful_candidate"
    else:
        secondary = sorted(
            secondary_pool,
            key=lambda r: (
                _model(r) == primary_model,
                _bucket(r) == primary_bucket,
                0 if _in_band(r, qmin, qmax) else 1,
                _source_pressure(r, source_counts, weights),
                abs(_quality(r) - target),
                _quality(r),
            ),
        )[0]

    source_counts[primary_model] = source_counts.get(primary_model, 0) + 1
    source_counts[_model(secondary)] = source_counts.get(_model(secondary), 0) + 1
    order = [
        {
            "candidate_id": _candidate_id(r),
            "generation_model": _model(r),
            "mask_id": r.get("mask_id"),
            "quality_score": _quality(r),
            "defect_bucket": _bucket(r),
            "in_quality_band": _in_band(r, qmin, qmax),
        }
        for r in ordered_all
    ]
    meta = {
        "selection_policy": config["policy_name"],
        "source_counts_before": before,
        "source_counts_after": dict(source_counts),
        "source_weights": weights,
        "quality_band": [qmin, qmax],
        "quality_target": target,
        "candidate_quality_order": order,
        "selected_primary": _candidate_id(primary),
        "selected_secondary": _candidate_id(secondary),
        "fallback_reason": fallback_reason,
    }
    return primary, secondary, meta


def selected_manifest_row(candidate: dict[str, Any], role: str, final_type: str, selection_meta: dict[str, Any], rank: int) -> dict[str, Any]:
    final_path = candidate.get("comp_loser_video_path") if final_type == "comp" else candidate.get("raw_loser_video_path")
    return {
        "sample_id": candidate.get("sample_id"),
        "source_video_id": candidate.get("source_video_id"),
        "pair_index": candidate.get("pair_index"),
        "prompt": candidate.get("prompt", ""),
        "win_video_path": candidate.get("win_video_path"),
        "final_loser_video_path": final_path,
        "raw_loser_video_path": candidate.get("raw_loser_video_path"),
        "comp_loser_video_path": candidate.get("comp_loser_video_path"),
        "final_loser_type": final_type,
        "selected_role": role,
        "mask_id": candidate.get("mask_id"),
        "mask_path": candidate.get("mask_path"),
        "mask_policy": candidate.get("mask_policy"),
        "generation_model": _model(candidate),
        "quality_score": _quality(candidate),
        "defect_bucket": _bucket(candidate),
        "selection_policy": selection_meta["selection_policy"],
        "selection_reason": selection_meta.get("fallback_reason") or "medium_hard_balanced",
        "candidate_rank": rank,
        "source_counts_snapshot": selection_meta.get("source_counts_after", {}),
        "selection_meta": selection_meta,
        "seed": candidate.get("seed"),
        "fps": candidate.get("fps"),
        "num_frames": candidate.get("num_frames"),
        "height": candidate.get("height"),
        "width": candidate.get("width"),
        "status": candidate.get("status"),
    }


def group_by_sample(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("sample_id"))].append(row)
    return grouped


def select_manifests(rows: list[dict[str, Any]], config: dict[str, Any], mode: str) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    source_counts: dict[str, int] = {}
    manifests = {
        "selected_primary_comp": [],
        "selected_primary_nocomp": [],
        "selected_secondary_comp": [],
        "selected_secondary_nocomp": [],
        "selected_primary_fullmask": [],
        "selected_secondary_fullmask": [],
    }
    selection_events = []
    for sample_id, candidates in sorted(group_by_sample(rows).items()):
        primary, secondary, meta = select_for_sample(candidates, source_counts, config)
        meta["sample_id"] = sample_id
        selection_events.append(meta)
        if primary is None:
            continue
        secondary = secondary or primary
        if mode == "full":
            manifests["selected_primary_fullmask"].append(selected_manifest_row(primary, "primary", "raw", meta, 1))
            manifests["selected_secondary_fullmask"].append(selected_manifest_row(secondary, "secondary", "raw", meta, 2))
        else:
            manifests["selected_primary_comp"].append(selected_manifest_row(primary, "primary", "comp", meta, 1))
            manifests["selected_primary_nocomp"].append(selected_manifest_row(primary, "primary", "raw", meta, 1))
            manifests["selected_secondary_comp"].append(selected_manifest_row(secondary, "secondary", "comp", meta, 2))
            manifests["selected_secondary_nocomp"].append(selected_manifest_row(secondary, "secondary", "raw", meta, 2))
    return manifests, selection_events


def histogram(values: list[float], bins: list[float]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        counts[f"{lo:.2f}-{hi:.2f}"] = sum(1 for v in values if lo <= v < hi)
    counts[f">={bins[-1]:.2f}"] = sum(1 for v in values if v >= bins[-1])
    return counts


def write_calibration_report(
    path: Path,
    rows: list[dict[str, Any]],
    manifests: dict[str, list[dict[str, Any]]],
    selection_events: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    ok_rows = [r for r in rows if _ok(r)]
    failed_rows = [r for r in rows if not _ok(r)]
    qs = [_quality(r) for r in ok_rows]
    qcfg = config["quality_score"]
    qmin = float(qcfg["eligible_min"])
    qmax = float(qcfg["eligible_max"])
    too_bad = sum(1 for q in qs if q < qmin)
    eligible = sum(1 for q in qs if qmin <= q <= qmax)
    too_good = sum(1 for q in qs if q > qmax)
    primary = manifests.get("selected_primary_comp") or manifests.get("selected_primary_fullmask") or []
    secondary = manifests.get("selected_secondary_comp") or manifests.get("selected_secondary_fullmask") or []
    mask_areas = [float(r.get("mask_area_ratio", 0.0)) for r in rows if r.get("mask_area_ratio") is not None]
    outside_max = [
        float((r.get("comp_metrics") or {}).get("outside_mask_diff_max", 0.0))
        for r in ok_rows
        if isinstance(r.get("comp_metrics"), dict)
    ]
    lines = [
        "# Generated Loser Calibration Report",
        "",
        "## Summary",
        "",
        f"- candidate_count: {len(rows)}",
        f"- successful_candidate_count: {len(ok_rows)}",
        f"- failed_candidate_count: {len(failed_rows)}",
        f"- fail_count_by_model: {dict(Counter(_model(r) for r in failed_rows))}",
        f"- quality_score_histogram: {histogram(qs, [0.0, 0.15, 0.30, 0.45, 0.65, 0.80, 1.0])}",
        f"- too_bad_ratio: {(too_bad / len(qs)) if qs else 0.0:.4f}",
        f"- eligible_ratio: {(eligible / len(qs)) if qs else 0.0:.4f}",
        f"- too_good_ratio: {(too_good / len(qs)) if qs else 0.0:.4f}",
        f"- selected_primary_distribution: {dict(Counter(_model(r) for r in primary))}",
        f"- selected_secondary_distribution: {dict(Counter(_model(r) for r in secondary))}",
        f"- mask_area_mean: {(float(np.mean(mask_areas)) if mask_areas else 0.0):.6f}",
        f"- mask_area_min: {(float(np.min(mask_areas)) if mask_areas else 0.0):.6f}",
        f"- mask_area_max: {(float(np.max(mask_areas)) if mask_areas else 0.0):.6f}",
        f"- mask_motion_distribution: {dict(Counter(str(r.get('mask_motion_type') or r.get('motion_type') or '') for r in rows))}",
        f"- comp_outside_diff_max: {(float(np.max(outside_max)) if outside_max else 0.0):.6f}",
        "",
        "## Selection Events",
        "",
    ]
    for event in selection_events[:50]:
        lines.append(
            f"- sample_id={event.get('sample_id')} primary={event.get('selected_primary')} "
            f"secondary={event.get('selected_secondary')} fallback={event.get('fallback_reason') or 'none'}"
        )
    lines.extend([
        "",
        "## Preview Paths",
        "",
    ])
    previews = [r.get("preview_path") or r.get("visual_preview_path") for r in ok_rows if r.get("preview_path") or r.get("visual_preview_path")]
    for preview in previews[:50]:
        lines.append(f"- `{preview}`")
    if not previews:
        lines.append("- no preview paths recorded in candidates manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select VideoDPO generated-loser candidates.")
    parser.add_argument("--candidates_manifest", required=True)
    parser.add_argument("--selection_config", default=str(DEFAULT_SELECTION_CONFIG))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=["partial", "full"], default="partial")
    parser.add_argument("--calibration_report", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = read_jsonl(Path(args.candidates_manifest))
    config = load_config(args.selection_config)
    manifests, selection_events = select_manifests(rows, config, args.mode)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, manifest_rows in manifests.items():
        if manifest_rows:
            write_jsonl(output_dir / f"{name}.jsonl", manifest_rows)
    (output_dir / "selection_events.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in selection_events),
        encoding="utf-8",
    )
    if args.calibration_report:
        write_calibration_report(Path(args.calibration_report), rows, manifests, selection_events, config)
    print(json.dumps({"samples": len(group_by_sample(rows)), "rows": len(rows), "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
