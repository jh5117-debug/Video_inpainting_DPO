#!/usr/bin/env python3
"""Build an Exp30 VOR-OR source pool from existing exact extraction caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def infer_source_type(sample_id: str) -> str:
    if sample_id.startswith("REAL_"):
        return "REAL"
    if sample_id.startswith("BLENDER_"):
        return "BLENDER"
    return "UNKNOWN"


def infer_scene_group(sample_id: str) -> str:
    parts = sample_id.split("_")
    if sample_id.startswith("REAL_") and len(parts) >= 3:
        return "_".join(parts[:3])
    if sample_id.startswith("BLENDER_") and len(parts) >= 2:
        return "_".join(parts[:2])
    return sample_id


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_exclusions(paths: list[Path]) -> tuple[set[str], set[str]]:
    sample_ids: set[str] = set()
    scene_groups: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            sample_id = str(row.get("sample_id", ""))
            scene_group = str(row.get("scene_group", "")) or infer_scene_group(sample_id)
            if sample_id:
                sample_ids.add(sample_id)
            if scene_group:
                scene_groups.add(scene_group)
    return sample_ids, scene_groups


def probe_video(path: Path, max_frames: int = 24) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"ok": False, "frames": 0, "width": 0, "height": 0}
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    decoded = 0
    for _ in range(max_frames):
        ok, _frame = cap.read()
        if not ok:
            break
        decoded += 1
    cap.release()
    return {"ok": decoded > 0, "frames": frames, "decoded_probe": decoded, "width": width, "height": height}


def read_even_frames(path: Path, count: int = 4, size: tuple[int, int] = (160, 90)) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        positions = [0]
    else:
        positions = np.linspace(0, max(total - 1, 0), count).astype(int).tolist()
    frames: list[np.ndarray] = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frame = cv2.resize(frame, size)
        frames.append(frame)
    cap.release()
    return frames


def mask_stats(path: Path, sample_frames: int = 24) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    areas: list[float] = []
    decoded = 0
    while decoded < sample_frames:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        areas.append(float((gray > 10).mean()))
        decoded += 1
    cap.release()
    if not areas:
        return {"decoded": 0, "area_mean": 0.0, "area_max": 0.0, "bucket": "invalid"}
    mean = float(np.mean(areas))
    bucket = "small" if mean < 0.05 else "medium" if mean < 0.2 else "large"
    return {"decoded": decoded, "area_mean": mean, "area_max": float(np.max(areas)), "bucket": bucket}


def affected_stats(condition: Path, winner: Path, mask: Path, sample_frames: int = 12) -> dict[str, Any]:
    ccap = cv2.VideoCapture(str(condition))
    wcap = cv2.VideoCapture(str(winner))
    mcap = cv2.VideoCapture(str(mask))
    diffs: list[float] = []
    outside: list[float] = []
    decoded = 0
    while decoded < sample_frames:
        okc, cf = ccap.read()
        okw, wf = wcap.read()
        okm, mf = mcap.read()
        if not (okc and okw and okm):
            break
        diff = np.mean(np.abs(cf.astype(np.float32) - wf.astype(np.float32)), axis=2) / 255.0
        mask_bin = cv2.cvtColor(mf, cv2.COLOR_BGR2GRAY) > 10
        diffs.append(float(diff.mean()))
        if (~mask_bin).any():
            outside.append(float(diff[~mask_bin].mean()))
        decoded += 1
    for cap in (ccap, wcap, mcap):
        cap.release()
    if not diffs:
        return {"decoded": 0, "affected_mean": 0.0, "affected_outside_mean": 0.0}
    return {
        "decoded": decoded,
        "affected_mean": float(np.mean(diffs)),
        "affected_outside_mean": float(np.mean(outside)) if outside else 0.0,
    }


def discover_triplets(root: Path) -> list[dict[str, Any]]:
    fg_dir = root / "VOR-Train" / "VOR-Train" / "FG_BG"
    bg_dir = root / "VOR-Train" / "VOR-Train" / "BG"
    mask_dir = root / "VOR-Train-MASK" / "MASK"
    rows: list[dict[str, Any]] = []
    if not fg_dir.exists() or not bg_dir.exists() or not mask_dir.exists():
        return rows
    for condition in sorted(fg_dir.glob("*.mp4")):
        sample_id = condition.stem
        winner = bg_dir / f"{sample_id}.mp4"
        mask = mask_dir / f"{sample_id}.mp4"
        if winner.exists() and mask.exists():
            rows.append(
                {
                    "sample_id": sample_id,
                    "scene_group": infer_scene_group(sample_id),
                    "source_type": infer_source_type(sample_id),
                    "condition_path": str(condition),
                    "winner_path": str(winner),
                    "mask_path": str(mask),
                    "cache_root": str(root),
                }
            )
    return rows


def choose(rows: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    by_type = {"REAL": [], "BLENDER": []}
    for row in rows:
        if row["source_type"] in by_type:
            by_type[row["source_type"]].append(row)
    for source_type in by_type:
        by_type[source_type] = sorted(by_type[source_type], key=lambda r: (r["scene_group"], r["sample_id"]))
    out: list[dict[str, Any]] = []
    per_type = target // 2
    for source_type in ("REAL", "BLENDER"):
        out.extend(by_type[source_type][:per_type])
    if len(out) < target:
        used = {r["scene_group"] for r in out}
        rest = [r for source_type in ("REAL", "BLENDER") for r in by_type[source_type] if r["scene_group"] not in used]
        out.extend(rest[: target - len(out)])
    return out


def add_audit_fields(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rank, row in enumerate(rows):
        cprobe = probe_video(Path(row["condition_path"]))
        wprobe = probe_video(Path(row["winner_path"]))
        mprobe = probe_video(Path(row["mask_path"]))
        mstats = mask_stats(Path(row["mask_path"]))
        astats = affected_stats(Path(row["condition_path"]), Path(row["winner_path"]), Path(row["mask_path"]))
        valid = (
            cprobe["ok"]
            and wprobe["ok"]
            and mprobe["ok"]
            and mstats["area_max"] > 0
            and astats["affected_mean"] > 0
        )
        enriched = dict(row)
        enriched.update(
            {
                "selection_rank": rank,
                "split": split,
                "condition_role": "V_obj",
                "winner_role": "V_bg",
                "mask_role": "foreground_object_mask",
                "source_dataset": "VOR-Train",
                "vor_eval_used": False,
                "eligible_for_generation": valid,
                "eligible_for_training": False,
                "condition_frames": cprobe["frames"],
                "winner_frames": wprobe["frames"],
                "mask_frames": mprobe["frames"],
                "width": cprobe["width"],
                "height": cprobe["height"],
                "mask_area_mean": mstats["area_mean"],
                "mask_area_max": mstats["area_max"],
                "mask_bucket": mstats["bucket"],
                "affected_mean": astats["affected_mean"],
                "affected_outside_mean": astats["affected_outside_mean"],
                "decode_verified": bool(cprobe["ok"] and wprobe["ok"] and mprobe["ok"]),
                "mask_nonempty_verified": mstats["area_max"] > 0,
                "affected_nonempty_verified": astats["affected_mean"] > 0,
                "preview_generated": True,
                "selection_rule": "existing_exact_cache_scene_unique_balanced_source_type",
            }
        )
        out.append(enriched)
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_batch_previews(rows: list[dict[str, Any]], out_dir: Path, batch_size: int = 8) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for batch_idx in range(0, len(rows), batch_size):
        batch = rows[batch_idx : batch_idx + batch_size]
        sample_imgs: list[np.ndarray] = []
        for row in batch:
            cframes = read_even_frames(Path(row["condition_path"]), 4)
            wframes = read_even_frames(Path(row["winner_path"]), 4)
            mframes = read_even_frames(Path(row["mask_path"]), 4)
            strips = [np.concatenate(frames, axis=1) for frames in (cframes, wframes, mframes)]
            panel = np.concatenate(strips, axis=0)
            title = f"{row['sample_id']} {row['source_type']} {row['mask_bucket']} m={row['mask_area_mean']:.3f} aff={row['affected_mean']:.3f}"
            cv2.putText(panel, title[:120], (5, 18), font, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            sample_imgs.append(panel)
        width = max(img.shape[1] for img in sample_imgs)
        padded = []
        for img in sample_imgs:
            if img.shape[1] < width:
                pad = np.zeros((img.shape[0], width - img.shape[1], 3), dtype=np.uint8)
                img = np.concatenate([img, pad], axis=1)
            padded.append(img)
        page = np.concatenate(padded, axis=0)
        path = out_dir / f"source_pool_batch_{batch_idx // batch_size:03d}.jpg"
        cv2.imwrite(str(path), page)
        paths.append(str(path))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", action="append", type=Path, required=True)
    parser.add_argument("--exclude-jsonl", action="append", type=Path, default=[])
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--source-target", type=int, default=128)
    parser.add_argument("--reserve-target", type=int, default=128)
    args = parser.parse_args()

    exclude_samples, exclude_groups = load_exclusions(args.exclude_jsonl)
    discovered: list[dict[str, Any]] = []
    for root in args.cache_root:
        discovered.extend(discover_triplets(root))

    seen_groups: set[str] = set()
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in sorted(discovered, key=lambda r: (r["source_type"], r["scene_group"], r["sample_id"])):
        reason = ""
        if row["sample_id"] in exclude_samples or row["scene_group"] in exclude_groups:
            reason = "excluded_previous_exp29_or_effecterase_scene"
        elif row["scene_group"] in seen_groups:
            reason = "duplicate_scene_group"
        elif row["source_type"] not in {"REAL", "BLENDER"}:
            reason = "unknown_source_type"
        if reason:
            rej = dict(row)
            rej["reject_reason"] = reason
            rejected.append(rej)
            continue
        seen_groups.add(row["scene_group"])
        candidates.append(row)

    source_raw = choose(candidates, args.source_target)
    used_groups = {r["scene_group"] for r in source_raw}
    reserve_candidates = [r for r in candidates if r["scene_group"] not in used_groups]
    reserve_raw = choose(reserve_candidates, args.reserve_target)

    source_rows = add_audit_fields(source_raw, "source_pool128")
    reserve_rows = add_audit_fields(reserve_raw, "reserve")

    manifest_dir = args.out_root / "manifests"
    report_dir = args.out_root / "reports"
    preview_dir = report_dir / "exp30_vor_or_source_pool_previews"
    source_path = manifest_dir / "vor_or_pool128_sources.jsonl"
    reserve_path = manifest_dir / "vor_or_pool128_reserve.jsonl"
    rejected_path = manifest_dir / "vor_or_source_pool_rejected.jsonl"
    write_jsonl(source_path, source_rows)
    write_jsonl(reserve_path, reserve_rows)
    write_jsonl(rejected_path, rejected)
    write_csv(report_dir / "exp30_vor_or_source_pool_audit.csv", source_rows + reserve_rows)
    preview_paths = make_batch_previews(source_rows, preview_dir)

    summary = {
        "status": "VOR_OR_SOURCE_POOL_READY" if len(source_rows) == args.source_target and len(reserve_rows) == args.reserve_target else "VOR_OR_SOURCE_POOL_RESERVE_PARTIAL",
        "cache_roots": [str(p) for p in args.cache_root],
        "discovered_triplets": len(discovered),
        "candidate_scene_groups_after_exclusion": len(candidates),
        "source_rows": len(source_rows),
        "reserve_rows": len(reserve_rows),
        "source_sha256": sha256_file(source_path),
        "reserve_sha256": sha256_file(reserve_path),
        "source_type_counts": Counter(r["source_type"] for r in source_rows),
        "mask_bucket_counts": Counter(r["mask_bucket"] for r in source_rows),
        "preview_pages": preview_paths,
        "preview_pages_count": len(preview_paths),
        "rejected_count": len(rejected),
    }
    summary = json.loads(json.dumps(summary, default=lambda x: dict(x) if isinstance(x, Counter) else str(x)))
    (report_dir / "exp30_vor_or_source_pool_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    md = [
        "# Exp30 VOR-OR Source Pool Audit",
        "",
        f"Status: `{summary['status']}`",
        "",
        "## Counts",
        "",
        f"- Discovered triplets: {summary['discovered_triplets']}",
        f"- Candidate scene groups after exclusions: {summary['candidate_scene_groups_after_exclusion']}",
        f"- Source rows: {summary['source_rows']}",
        f"- Reserve rows: {summary['reserve_rows']}",
        f"- Rejected rows: {summary['rejected_count']}",
        "",
        "## Identity",
        "",
        f"- Source manifest SHA256: `{summary['source_sha256']}`",
        f"- Reserve manifest SHA256: `{summary['reserve_sha256']}`",
        "",
        "## Balance",
        "",
        f"- Source type counts: `{summary['source_type_counts']}`",
        f"- Mask bucket counts: `{summary['mask_bucket_counts']}`",
        "",
        "## Preview",
        "",
        f"- Batch preview pages: {summary['preview_pages_count']}",
        "- Each source preview page row contains condition, winner, and mask strips.",
        "",
        "## Caveat",
        "",
        "This milestone uses existing exact extraction caches and does not rescan VOR tar archives. If reserve rows are below 128, the source pool is usable for initial smoke but reserve capacity is partial.",
    ]
    (report_dir / "exp30_vor_or_source_pool_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

