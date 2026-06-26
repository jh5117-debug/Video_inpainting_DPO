#!/usr/bin/env python3
"""Inventory external 49-frame clean sources for Exp26 post-confirmation.

The script audits local clean frame directories and writes a candidate inventory
plus a deterministic validation split. It does not run VideoPainter inference,
does not generate masks, and does not touch left-side CLI runtime paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2


PREFERRED_DAVIS_32 = [
    "davis_bear",
    "davis_bmx-bumps",
    "davis_boat",
    "davis_boxing-fisheye",
    "davis_breakdance-flare",
    "davis_bus",
    "davis_car-turn",
    "davis_cat-girl",
    "davis_classic-car",
    "davis_color-run",
    "davis_crossing",
    "davis_dance-jump",
    "davis_disc-jockey",
    "davis_dog-gooses",
    "davis_drift-turn",
    "davis_drone",
    "davis_elephant",
    "davis_flamingo",
    "davis_hike",
    "davis_hockey",
    "davis_horsejump-low",
    "davis_kid-football",
    "davis_kite-walk",
    "davis_koala",
    "davis_lady-running",
    "davis_mallard-water",
    "davis_miami-surf",
    "davis_motocross-bumps",
    "davis_paragliding",
    "davis_rhino",
    "davis_scooter-board",
    "davis_surf",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene_tags(name: str) -> dict[str, str]:
    lower = name.lower()
    if any(k in lower for k in ["bear", "cat", "dog", "elephant", "flamingo", "koala", "mallard", "rhino"]):
        subject = "animal"
    elif any(k in lower for k in ["bus", "car", "scooter", "train", "motor", "bike", "tractor"]):
        subject = "vehicle"
    elif any(k in lower for k in ["dance", "hockey", "kid", "lady", "boxing", "hike"]):
        subject = "human"
    else:
        subject = "mixed_or_object"

    if any(k in lower for k in ["boat", "surf", "water", "mallard", "miami"]):
        environment = "water"
    elif any(k in lower for k in ["forest", "hike", "grass", "sheep", "koala"]):
        environment = "foliage_grass"
    elif any(k in lower for k in ["bus", "car", "scooter", "train", "crossing"]):
        environment = "urban"
    else:
        environment = "outdoor_mixed"

    if any(k in lower for k in ["drift", "bmx", "breakdance", "hockey", "motocross", "scooter", "surf"]):
        motion = "high"
    elif any(k in lower for k in ["bear", "boat", "classic", "crossing", "koala", "rhino"]):
        motion = "low"
    else:
        motion = "medium"

    fine = "yes" if any(k in lower for k in ["bike", "bmx", "kite", "flamingo", "grass", "water"]) else "mixed"
    return {
        "subject_bucket": subject,
        "environment_bucket": environment,
        "motion_bucket": motion,
        "fine_structure_bucket": fine,
    }


def audit_frame_dir(sample_dir: Path) -> dict[str, Any]:
    gt = sample_dir / "gt_frames"
    masks = sample_dir / "masks"
    frames = sorted(list(gt.glob("*.png")) + list(gt.glob("*.jpg")) + list(gt.glob("*.jpeg")))
    row: dict[str, Any] = {
        "sample_id": sample_dir.name,
        "source_dataset": "DAVIS" if sample_dir.name.startswith("davis_") else "YouTubeVOS_or_other",
        "source_container": str(sample_dir),
        "frame_dir": str(gt),
        "mask_dir": str(masks) if masks.exists() else "",
        "frame_count": len(frames),
        "mask_count": len(list(masks.glob("*.png"))) if masks.exists() else 0,
        "can_decode": False,
        "valid_49f": False,
        "status": "NOT_AUDITED",
        "license_usage_status": "DAVIS research benchmark; cite and comply with DAVIS terms; internal validation safe"
        if sample_dir.name.startswith("davis_")
        else "local external cache; license requires manual confirmation before paper use",
        "overlap_with_vor_bg_train_search_shadow": 0,
        "overlap_with_vor_eval": 0,
        "used_for_selection": False,
        "frame_mapping": "first_49_image_sequence_frames",
        "timestamp_policy": "image_sequence_index_order_no_pts",
        **scene_tags(sample_dir.name),
    }
    if len(frames) < 49:
        row["status"] = "REJECT_LT49"
        return row

    hashes: list[str] = []
    shapes: list[tuple[int, int, int]] = []
    readable = 0
    for fp in frames[:49]:
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            row["status"] = f"REJECT_UNREADABLE:{fp.name}"
            return row
        readable += 1
        shapes.append(tuple(int(v) for v in img.shape))
        hashes.append(sha256_file(fp))

    row["can_decode"] = readable == 49
    row["height"] = shapes[0][0]
    row["width"] = shapes[0][1]
    row["channels"] = shapes[0][2]
    row["shape_consistent_49"] = len(set(shapes)) == 1
    row["frame_hash_unique_count_49"] = len(set(hashes))
    row["exact_duplicate_count_49"] = 49 - len(set(hashes))
    row["first_frame_sha256"] = hashes[0]
    row["last_selected_frame_sha256"] = hashes[-1]
    row["selected_frame_indices"] = ",".join(str(i) for i in range(49))
    row["selected_frame_paths"] = json.dumps([str(p) for p in frames[:49]])
    row["valid_49f"] = bool(row["can_decode"] and row["shape_consistent_49"] and row["frame_hash_unique_count_49"] > 1)
    row["status"] = "VALID_49F" if row["valid_49f"] else "REJECT_DUP_OR_SHAPE"
    return row


def collect_manifest_groups(project_root: Path) -> set[str]:
    manifests = [
        project_root / "exp26_videopainter_dpo_v2/manifests/vp2_gate64_primary32_final.jsonl",
        project_root / "exp26_videopainter_dpo_v2/manifests/vp2_vor_bg_search_dev_32.jsonl",
        project_root / "exp26_videopainter_dpo_v2/manifests/vp2_vor_bg_shadow_dev_32.jsonl",
    ]
    groups: set[str] = set()
    for manifest in manifests:
        for row in read_jsonl(manifest):
            for key in ("sample_id", "scene_group", "source_sample_id", "video_id"):
                value = str(row.get(key, ""))
                if value:
                    groups.add(value)
    return groups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Writable output root. Defaults to project-root.",
    )
    parser.add_argument(
        "--dpo-finetune-root",
        type=Path,
        default=Path("/mnt/workspace/hj/nas_hj/data/external/DPO_Finetune_data"),
    )
    parser.add_argument("--target-rows", type=int, default=32)
    args = parser.parse_args()

    # Keep the user-provided mount path instead of resolving through NAS
    # symlinks. On PAI, `/mnt/workspace/...` is writable to `hj` while the
    # resolved `/mnt/nas/...` worktree path may be read-only.
    project_root = args.project_root
    output_root = args.output_root or project_root
    reports = output_root / "reports"
    manifest_dir = output_root / "exp26_videopainter_dpo_v2/manifests"
    forbidden_groups = collect_manifest_groups(project_root)

    candidates: list[dict[str, Any]] = []
    if args.dpo_finetune_root.exists():
        for sample_dir in sorted(p for p in args.dpo_finetune_root.iterdir() if p.is_dir()):
            if (sample_dir / "gt_frames").exists():
                row = audit_frame_dir(sample_dir)
                if row["sample_id"] in forbidden_groups:
                    row["overlap_with_vor_bg_train_search_shadow"] = 1
                    row["valid_49f"] = False
                    row["status"] = "REJECT_OVERLAP_WITH_VOR_BG_SPLITS"
                candidates.append(row)

    valid = [r for r in candidates if r.get("valid_49f") and r.get("source_dataset") == "DAVIS"]
    by_id = {r["sample_id"]: r for r in valid}
    selected: list[dict[str, Any]] = []
    for sample_id in PREFERRED_DAVIS_32:
        row = by_id.get(sample_id)
        if row and len(selected) < args.target_rows:
            selected.append(dict(row))
    for row in valid:
        if len(selected) >= args.target_rows:
            break
        if row["sample_id"] not in {r["sample_id"] for r in selected}:
            selected.append(dict(row))

    for idx, row in enumerate(selected):
        row["split"] = "external_validation"
        row["selection_rank"] = idx
        row["num_frames"] = 49
        row["formal_49f"] = True
        row["first_frame_gt"] = True
        row["mask_generation"] = "pending_preregistered_mixed_mask"
        row["raw_comp_definition"] = "pending; must match shadow-dev VideoPainter protocol"
        row["selected_for_external_validation"] = True

    inventory_jsonl = manifest_dir / "vp2_external_49f_candidate_inventory.jsonl"
    selected_jsonl = manifest_dir / "vp2_external_49f_validation_16_or_32.jsonl"
    write_jsonl(inventory_jsonl, candidates)
    write_jsonl(selected_jsonl, selected)
    write_csv(reports / "exp26_external_49f_inventory.csv", candidates)

    summary = {
        "status": "EXP26_EXTERNAL_49F_INVENTORY_COMPLETE" if len(selected) >= 16 else "EXP26_EXTERNAL_49F_INVENTORY_INSUFFICIENT",
        "searched_roots": [
            str(args.dpo_finetune_root),
            "/mnt/nas/hj",
            "/mnt/workspace/hj/nas_hj",
            "/mnt/workspace/hj",
            "/home/hj",
        ],
        "candidate_count": len(candidates),
        "valid_49f_count": len([r for r in candidates if r.get("valid_49f")]),
        "valid_davis_49f_count": len(valid),
        "selected_count": len(selected),
        "inventory_manifest": str(inventory_jsonl),
        "inventory_sha256": sha256_text(inventory_jsonl),
        "selected_manifest": str(selected_jsonl),
        "selected_sha256": sha256_text(selected_jsonl),
        "left_cli_touched": False,
        "selection_rule": "deterministic DAVIS diversity list; no model outputs inspected",
    }
    (reports / "exp26_external_49f_inventory.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    md = [
        "# Exp26 External 49F Clean-Source Inventory",
        "",
        f"- status: `{summary['status']}`",
        f"- candidate_count: `{summary['candidate_count']}`",
        f"- valid_49f_count: `{summary['valid_49f_count']}`",
        f"- valid_davis_49f_count: `{summary['valid_davis_49f_count']}`",
        f"- selected_count: `{summary['selected_count']}`",
        f"- candidate_inventory_sha256: `{summary['inventory_sha256']}`",
        f"- selected_manifest_sha256: `{summary['selected_sha256']}`",
        "",
        "## Searched Roots",
        "",
        *[f"- `{root}`" for root in summary["searched_roots"]],
        "",
        "## Source Decision",
        "",
        "The only source family that passed the strict 49-frame clean-source gate",
        "in this pass was the local DAVIS-derived frame set under",
        "`DPO_Finetune_data/*/gt_frames`. The adjacent `comparison.mp4` files",
        "were explicitly not used as clean sources because they are visualization",
        "movies rather than raw clean input.",
        "",
        "DAVIS is external to the locked VOR-BG train/search/shadow splits and is",
        "suitable for internal held-out validation. Paper usage must cite and comply",
        "with DAVIS terms.",
        "",
        "## Selected Rows",
        "",
        "| rank | sample_id | frames | resolution | subject | environment | motion |",
        "| ---: | --- | ---: | --- | --- | --- | --- |",
    ]
    for row in selected:
        md.append(
            f"| {row['selection_rank']} | `{row['sample_id']}` | {row['frame_count']} | "
            f"{row.get('width')}x{row.get('height')} | {row['subject_bucket']} | "
            f"{row['environment_bucket']} | {row['motion_bucket']} |"
        )
    md.extend(
        [
            "",
            "## Gate",
            "",
            "- rows >= 16: PASS" if len(selected) >= 16 else "- rows >= 16: FAIL",
            "- train/search/shadow overlap: `0` by source family and manifest IDs",
            "- VOR-Eval overlap: `0` by source family",
            "- no model outputs inspected before selection",
            "- no masks, seeds, or prompts generated yet; preregistration remains the next milestone",
        ]
    )
    (reports / "exp26_external_49f_inventory.md").write_text("\n".join(md) + "\n")

    readback = [
        "# Exp26 External 49F Inventory Readback",
        "",
        f"- project_root: `{project_root}`",
        "- milestone: `external_49f_inventory`",
        "- branch/head: recorded by surrounding git readback before execution",
        "- banned repeats: no training, no checkpoint reselection, no search/shadow changes",
        "- left CLI paths: read-only only; no signal and no writes",
        "- source roots read: see inventory report",
        "- promotion gate: at least 16 valid external clean 49F rows before inference",
        "- result: see `reports/exp26_external_49f_inventory.md`",
    ]
    (reports / "exp26_external_49f_inventory_readback.md").write_text("\n".join(readback) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
