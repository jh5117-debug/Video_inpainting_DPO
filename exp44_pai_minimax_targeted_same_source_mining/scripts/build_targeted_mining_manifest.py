#!/usr/bin/env python3
"""Build the locked Exp44 source/seed manifest for targeted MiniMax mining."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-group-plan", required=True)
    parser.add_argument("--exp42-all", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--include-fallback", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed_start(source_group: str) -> int:
    digest = hashlib.sha256(source_group.encode("utf-8")).hexdigest()
    return 2026062900 + (int(digest[:8], 16) % 100000)


def seed_list(source_group: str, count: int) -> list[int]:
    start = stable_seed_start(source_group)
    return [start + idx for idx in range(count)]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def source_group(row: dict[str, object]) -> str:
    return str(row.get("scene_group") or row.get("source_group") or str(row.get("sample_id", "")).rsplit("_", 1)[0])


def load_original_rows(exp42_rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    manifest_cache: dict[Path, list[dict[str, object]]] = {}
    sample_to_origin: dict[str, dict[str, object]] = {}
    for row in exp42_rows:
        sample_id = str(row.get("sample_id", ""))
        manifest_path = row.get("source_manifest_path")
        if not sample_id or not manifest_path:
            continue
        path = Path(str(manifest_path))
        if path not in manifest_cache:
            if not path.exists():
                continue
            manifest_cache[path] = read_jsonl(path)
        for original in manifest_cache[path]:
            if str(original.get("sample_id", "")) == sample_id:
                sample_to_origin.setdefault(sample_id, original)
                break
    return sample_to_origin


def main() -> None:
    args = parse_args()
    plan_path = Path(args.source_group_plan)
    exp42_all_path = Path(args.exp42_all)
    plan_rows = load_csv(plan_path)
    exp42_rows = read_jsonl(exp42_all_path)
    original_by_sample = load_original_rows(exp42_rows)

    exp42_by_sample: dict[str, dict[str, object]] = {}
    exp42_by_group: dict[str, list[dict[str, object]]] = {}
    for row in exp42_rows:
        sample_id = str(row.get("sample_id", ""))
        group = source_group(row)
        if sample_id:
            exp42_by_sample.setdefault(sample_id, row)
        exp42_by_group.setdefault(group, []).append(row)

    out_rows: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    for plan in plan_rows:
        bucket = plan["bucket"]
        if bucket.startswith("D_") and not args.include_fallback:
            continue
        sample_id = plan.get("existing_samples", "").split(";")[0].strip()
        group = plan["source_group"]
        seed_count = int(plan["initial_new_candidate_budget"])
        exp42_ref = exp42_by_sample.get(sample_id)
        if exp42_ref is None and exp42_by_group.get(group):
            exp42_ref = exp42_by_group[group][0]
            sample_id = str(exp42_ref.get("sample_id", sample_id))
        original = original_by_sample.get(sample_id)
        if original is None:
            missing.append({"source_group": group, "sample_id": sample_id, "reason": "original source row not found"})
            continue

        row = dict(original)
        row["condition_path"] = row.get("condition_path") or row.get("condition_frame_dir")
        row["winner_path"] = row.get("winner_path") or row.get("winner_frame_dir")
        row["mask_path"] = row.get("mask_path") or row.get("mask_frame_dir")
        row["loser_path"] = row.get("loser_path") or row.get("winner_path")
        row["scene_group"] = group
        row["sample_id"] = sample_id
        row["exp44_bucket"] = bucket
        row["exp44_priority"] = int(plan["priority"])
        row["exp44_goal"] = plan["goal"]
        row["exp44_existing_success_rows"] = int(plan["existing_success_rows"])
        row["exp44_existing_failure_rows"] = int(plan["existing_failure_rows"])
        row["exp44_seed_rule"] = plan["seed_rule"]
        row["exp44_timestep_noise_rule"] = plan["timestep_noise_rule"]
        row["exp44_seed_start"] = stable_seed_start(group)
        row["exp44_seeds"] = seed_list(group, seed_count)
        row["exp44_initial_new_candidate_budget"] = seed_count
        row["exp44_max_new_candidate_budget"] = int(plan["max_new_candidate_budget"])
        row["exp44_from_exp42_candidate"] = exp42_ref.get("candidate_id") if exp42_ref else ""
        row["vor_eval_used"] = False
        out_rows.append(row)

    out_rows.sort(key=lambda r: (int(r["exp44_priority"]), str(r["scene_group"]), str(r["sample_id"])))
    output_manifest = Path(args.output_manifest)
    write_jsonl(output_manifest, out_rows)
    summary = {
        "status": "EXP44_TARGETED_SOURCE_MANIFEST_READY" if out_rows else "EXP44_TARGETED_SOURCE_MANIFEST_BLOCKED",
        "source_group_plan": str(plan_path),
        "source_group_plan_sha256": sha256_file(plan_path),
        "exp42_all": str(exp42_all_path),
        "exp42_all_sha256": sha256_file(exp42_all_path),
        "output_manifest": str(output_manifest),
        "output_manifest_sha256": sha256_file(output_manifest) if output_manifest.exists() else "",
        "rows": len(out_rows),
        "missing_rows": missing,
        "include_fallback": bool(args.include_fallback),
        "candidate_budget": sum(len(row["exp44_seeds"]) for row in out_rows),
        "bucket_counts": {bucket: sum(1 for row in out_rows if row["exp44_bucket"] == bucket) for bucket in sorted({str(row["exp44_bucket"]) for row in out_rows})},
    }
    write_json(Path(args.output_summary), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
