#!/usr/bin/env python3
"""Build the locked Exp26 Gate64 mixed-mask protocol.

This is a source/protocol builder only. It does not extract videos, generate
masks, run VideoPainter, or create preference pairs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


MASK_PROFILES = [
    ("irregular_freeform", 16),
    ("object_like_polygon", 16),
    ("soft_blob", 8),
    ("edge_touch_freeform", 8),
    ("ellipse_circle_subset", 8),
    ("thin_structure_freeform", 8),
]

AREA_CYCLE = [
    "small",
    "medium",
    "medium",
    "large",
    "small",
    "medium",
    "large",
    "medium",
]
MOTION_CYCLE = ["low", "medium", "high", "medium", "low", "medium", "high", "medium"]
DEFORM_CYCLE = ["slow", "moderate", "slow", "moderate"]


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash(*parts: object) -> str:
    return sha256_bytes("|".join(str(p) for p in parts).encode("utf-8"))


def choose_gate64_sources(rows: list[dict], seed: int) -> list[dict]:
    if len({r["scene_group"] for r in rows}) != len(rows):
        raise ValueError("train source manifest is not scene-disjoint")
    blender = [r for r in rows if str(r.get("source_sample_id", "")).startswith("BLENDER_")]
    real = [r for r in rows if not str(r.get("source_sample_id", "")).startswith("BLENDER_")]
    rng = random.Random(seed)
    blender = sorted(blender, key=lambda r: r["sample_id"])
    real = sorted(real, key=lambda r: r["sample_id"])
    rng.shuffle(real)
    selected = blender + real[: 64 - len(blender)]
    if len(selected) != 64:
        raise RuntimeError(f"expected 64 Gate64 sources, got {len(selected)}")
    rng.shuffle(selected)
    return selected


def expand_profiles() -> list[str]:
    profiles: list[str] = []
    for name, count in MASK_PROFILES:
        profiles.extend([name] * count)
    if len(profiles) != 64:
        raise AssertionError("MASK_PROFILES must sum to 64")
    return profiles


def build_config(seed: int, source_manifest: Path, historical_audit: Path) -> dict:
    return {
        "name": "vp2_mixed_br_mask_v1",
        "status": "LOCKED_GATE64_PROTOCOL_PENDING_GENERATION",
        "seed": seed,
        "formal_num_frames": 49,
        "first_frame_gt": True,
        "source_manifest": str(source_manifest),
        "historical_mask_audit": str(historical_audit),
        "source_of_truth": {
            "historical_br": "YouTube-VOS D2 partial-mask K4 / Exp10-Exp11 style masks, sampled through selected_primary_comp.repaired.pai_paths.jsonl",
            "gate16": "VideoPainter official 49F Gate16 used ellipse/circle masks and passed with one retained rejection",
            "decision": "Gate64 must be mixed-mask and cannot be ellipse/circle-only.",
        },
        "mask_families": {
            "irregular_freeform": {"count": 16, "description": "multi-lobe free-form brush-like regions"},
            "object_like_polygon": {"count": 16, "description": "compact non-elliptic object-like polygon masks"},
            "soft_blob": {"count": 8, "description": "smooth but non-circular blob masks"},
            "edge_touch_freeform": {"count": 8, "description": "free-form regions intentionally touching one frame edge"},
            "ellipse_circle_subset": {"count": 8, "description": "small controlled subset retained for continuity with Gate16"},
            "thin_structure_freeform": {"count": 8, "description": "elongated or thin free-form regions"},
        },
        "area_buckets": {
            "small": [0.08, 0.14],
            "medium": [0.18, 0.27],
            "large": [0.28, 0.36],
        },
        "motion_buckets": {
            "low": {"centroid_motion_fraction": [0.02, 0.06]},
            "medium": {"centroid_motion_fraction": [0.08, 0.16]},
            "high": {"centroid_motion_fraction": [0.18, 0.28]},
        },
        "deformation_buckets": {
            "slow": {"scale_oscillation": [0.02, 0.06], "rotation_degrees": [0, 12]},
            "moderate": {"scale_oscillation": [0.06, 0.12], "rotation_degrees": [12, 35]},
        },
        "hard_constraints": [
            "exactly 49 real frames per source",
            "one mask per source",
            "one seed per source",
            "scene-group disjoint sources",
            "first frame GT: frame0 mask area must be zero",
            "no VOR-Eval, no Exp26 search-dev/shadow-dev, no Exp25 search/shadow rows",
            "formal Gate64 generation must use this config without changing mask distribution",
        ],
    }


def build_manifest(rows: list[dict], seed: int) -> list[dict]:
    profiles = expand_profiles()
    out: list[dict] = []
    for idx, (row, profile) in enumerate(zip(rows, profiles)):
        area_bucket = AREA_CYCLE[idx % len(AREA_CYCLE)]
        motion_bucket = MOTION_CYCLE[(idx + (1 if profile.startswith("edge") else 0)) % len(MOTION_CYCLE)]
        deformation_bucket = DEFORM_CYCLE[idx % len(DEFORM_CYCLE)]
        edge_touch = profile == "edge_touch_freeform" or (idx % 11 == 0 and profile != "ellipse_circle_subset")
        sample_seed = int(stable_hash(seed, row["sample_id"], profile)[:8], 16)
        gate_id = f"vp2_gate64_{idx:03d}_{row['source_sample_id']}"
        new_row = dict(row)
        new_row.update(
            {
                "sample_id": gate_id,
                "gate64_source_sample_id": row["sample_id"],
                "gate64_index": idx,
                "gate64_protocol": "vp2_mixed_br_mask_v1",
                "mask_profile": profile,
                "area_bucket": area_bucket,
                "motion_bucket": motion_bucket,
                "deformation_bucket": deformation_bucket,
                "edge_touch_target": edge_touch,
                "mask_generator_seed": sample_seed,
                "first_frame_gt": True,
                "formal_49f": True,
                "plumbing_only_13f": False,
                "num_frames": 49,
                "status": "GATE64_SOURCE_LOCKED_PENDING_EXTRACTION_MASK_GENERATION_INFERENCE",
                "identity_hash": stable_hash(gate_id, row["scene_group"], row["winner_member_path"], profile, sample_seed),
            }
        )
        out.append(new_row)
    return out


def write_audit_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "sample_id",
        "gate64_source_sample_id",
        "source_sample_id",
        "scene_group",
        "winner_member_path",
        "mask_profile",
        "area_bucket",
        "motion_bucket",
        "deformation_bucket",
        "edge_touch_target",
        "mask_generator_seed",
        "identity_hash",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fields} for row in rows])


def summarize_rows(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "unique_scene_groups": len({row["scene_group"] for row in rows}),
        "mask_profile": dict(sorted(Counter(row["mask_profile"] for row in rows).items())),
        "area_bucket": dict(sorted(Counter(row["area_bucket"] for row in rows).items())),
        "motion_bucket": dict(sorted(Counter(row["motion_bucket"] for row in rows).items())),
        "deformation_bucket": dict(sorted(Counter(row["deformation_bucket"] for row in rows).items())),
        "source_kind": dict(sorted(Counter("BLENDER" if row["source_sample_id"].startswith("BLENDER_") else "REAL" for row in rows).items())),
    }


def write_report(path: Path, rows: list[dict], config: dict, source_sha: str, output_sha: str, pai_status: str) -> None:
    counts = {
        "mask_profile": Counter(row["mask_profile"] for row in rows),
        "area_bucket": Counter(row["area_bucket"] for row in rows),
        "motion_bucket": Counter(row["motion_bucket"] for row in rows),
        "deformation_bucket": Counter(row["deformation_bucket"] for row in rows),
        "source_kind": Counter("BLENDER" if row["source_sample_id"].startswith("BLENDER_") else "REAL" for row in rows),
    }
    lines = [
        "# Exp26 Gate64 Readback and Mixed-Mask Protocol",
        "",
        "Status: `GATE64_PROTOCOL_LOCKED_PENDING_PAI_GENERATION`",
        "",
        "## Source State",
        "",
        f"- train source manifest SHA256: `{source_sha}`",
        f"- Gate64 manifest SHA256: `{output_sha}`",
        f"- Gate64 rows: `{len(rows)}`",
        f"- unique scene groups: `{len({row['scene_group'] for row in rows})}`",
        f"- PAI status during this milestone: `{pai_status}`",
        "",
        "## Distribution",
    ]
    for name, counter in counts.items():
        lines.append(f"### {name}")
        for key, value in sorted(counter.items()):
            lines.append(f"- `{key}`: {value}")
        lines.append("")
    lines += [
        "## Mask Source Audit",
        "",
        "- Historical BR masks from YouTube-VOS K4 / Exp10-Exp11 style are not ellipse-only: area mean about 0.254, bbox height p50 about 0.791, edge-touch ratio about 0.223.",
        "- Probe4/Gate16 masks were synthetic ellipse/circle masks; Gate16 passed, but that protocol is too narrow for Gate64.",
        "- `vp2_mixed_br_mask_v1` therefore includes irregular free-form, object-like polygon, soft blob, edge-touch, thin-structure, and a small ellipse/circle subset.",
        "",
        "## Locked Config",
        "",
        f"- config: `exp26_videopainter_dpo_v2/configs/vp2_mixed_br_mask_v1.json`",
        f"- manifest: `exp26_videopainter_dpo_v2/manifests/vp2_gate64_source_manifest.jsonl`",
        "",
        "## Banned Actions",
        "",
        "- Do not rerun Gate16 or replace the retained failed Gate16 row.",
        "- Do not start VideoPainter DPO until Gate64 generation, full metrics, and full video review pass.",
        "- Do not adjust this mask distribution after Gate64 generation starts.",
        "",
        "## Next Milestone",
        "",
        "After a fresh readback and PAI SSH restoration, run Gate64 official VideoPainter generation from this locked manifest/config.",
        "",
        "## Config Summary",
        "",
        "```json",
        json.dumps(config, indent=2, sort_keys=True),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source-manifest", type=Path, default=Path("exp26_videopainter_dpo_v2/manifests/vp2_vor_bg_train_source_128.jsonl"))
    parser.add_argument("--historical-audit-csv", type=Path, default=Path("reports/exp26_br_mask_distribution_audit_fast512.csv"))
    parser.add_argument("--output-manifest", type=Path, default=Path("exp26_videopainter_dpo_v2/manifests/vp2_gate64_source_manifest.jsonl"))
    parser.add_argument("--output-config", type=Path, default=Path("exp26_videopainter_dpo_v2/configs/vp2_mixed_br_mask_v1.json"))
    parser.add_argument("--audit-csv", type=Path, default=Path("reports/exp26_br_mask_source_audit.csv"))
    parser.add_argument("--audit-md", type=Path, default=Path("reports/exp26_br_mask_source_audit.md"))
    parser.add_argument("--gate64-readback-md", type=Path, default=Path("reports/exp26_gate64_readback.md"))
    parser.add_argument("--summary-json", type=Path, default=Path("reports/exp26_gate64_protocol_summary.json"))
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--pai-status", default="not_checked")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_bytes = args.train_source_manifest.read_bytes()
    source_sha = sha256_bytes(source_bytes)
    rows = choose_gate64_sources(read_jsonl(args.train_source_manifest), args.seed)
    manifest_rows = build_manifest(rows, args.seed)
    config = build_config(args.seed, args.train_source_manifest, args.historical_audit_csv)
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(args.output_manifest, manifest_rows)
    manifest_sha = sha256_bytes(args.output_manifest.read_bytes())
    write_audit_csv(args.audit_csv, manifest_rows)
    write_report(args.audit_md, manifest_rows, config, source_sha, manifest_sha, args.pai_status)
    summary = summarize_rows(manifest_rows)
    summary.update(
        {
            "status": "GATE64_PROTOCOL_LOCKED_PENDING_PAI_GENERATION",
            "train_source_manifest": str(args.train_source_manifest),
            "train_source_manifest_sha256": source_sha,
            "gate64_manifest": str(args.output_manifest),
            "gate64_manifest_sha256": manifest_sha,
            "config": str(args.output_config),
            "pai_status": args.pai_status,
            "banned_next_actions": ["no_gate16_rerun", "no_failed_row_replacement", "no_dpo_training_before_gate64_review"],
        }
    )
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readback = [
        "# Exp26 Gate64 Milestone Readback",
        "",
        "Status: `GATE64_READBACK_COMPLETED_PROTOCOL_LOCKED`",
        "",
        f"- train source manifest: `{args.train_source_manifest}`",
        f"- train source SHA256: `{source_sha}`",
        f"- Gate64 manifest: `{args.output_manifest}`",
        f"- Gate64 manifest SHA256: `{manifest_sha}`",
        f"- config: `{args.output_config}`",
        f"- PAI status: `{args.pai_status}`",
        "",
        "Files read include PRD/00, PRD/01, PRD/48, registry status, Gate16 final review, Probe4 mask audit, 49F sampler parity, historical BR mask audit, source split statistics, and Exp26 source/mask/generation code.",
        "",
        "Already completed: L0-L4, Probe4, Gate16 final video review.",
        "Pending: Gate64 extraction, mask generation, official inference, metrics, dense video review, and only then DPO micro-training.",
        "",
        "Banned repeats: no Gate16 rerun, no failed-row replacement, no Gate64 generation before this locked protocol, no DPO training.",
    ]
    args.gate64_readback_md.write_text("\n".join(readback) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(manifest_rows), "manifest_sha256": manifest_sha, "config": str(args.output_config)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
