#!/usr/bin/env python3
"""Plan and download deterministic Exp60C VPData replacement rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from exp60b_videopainter_vpdata_d3mask.vpdata_subset import (
    TEST_CSV,
    TRAIN_CSV,
    SourceRow,
    classify_path,
    download_url,
    ffprobe,
    iter_vpdata_csv,
    read_pexels_url_map,
    sha256_file,
    source_row_from_csv,
    write_jsonl,
)


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_failed(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            split, sample_id, reason, url = line.rstrip("\n").split("\t", 3)
            rows.append({"split": split, "sample_id": sample_id, "reason": reason, "url": url})
    return rows


def stable_key(seed: int, failed_id: str, candidate: SourceRow) -> str:
    text = "|".join([
        str(seed),
        failed_id,
        candidate.split,
        candidate.path,
        candidate.source_video_id,
        candidate.native_mask_id,
    ])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def eligible_candidates(
    *,
    split: str,
    output_root: Path,
    pexels_map: dict[int, tuple[str, str]],
    exclude_source_ids: set[str],
    exclude_paths: set[str],
    exclude_urls: set[str],
) -> list[SourceRow]:
    url = TRAIN_CSV if split == "train" else TEST_CSV
    out: list[SourceRow] = []
    seen_paths: set[str] = set()
    for raw in iter_vpdata_csv(url):
        kind, _, source_id = classify_path(raw["path"])
        if kind != "pexels":
            continue
        if raw["path"] in seen_paths:
            continue
        seen_paths.add(raw["path"])
        try:
            row = source_row_from_csv(raw, split=split, output_root=output_root, pexels_map=pexels_map)
        except Exception:
            continue
        if row.source_video_id in exclude_source_ids:
            continue
        if row.path in exclude_paths:
            continue
        if row.source_url in exclude_urls:
            continue
        out.append(row)
    return out


def write_plan(args: argparse.Namespace) -> int:
    manifest_dir = Path(args.manifest_dir)
    reports_dir = Path(args.reports_dir)
    output_root = Path(args.output_root)
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(Path(args.train_manifest))
    test_rows = load_jsonl(Path(args.test_manifest))
    failed = load_failed(Path(args.failed_urls))
    failed_ids = {r["sample_id"] for r in failed}
    failed_urls = {r["url"] for r in failed}

    kept_train = [r for r in train_rows if r["vpdata_id"] not in failed_ids]
    kept_test = [r for r in test_rows if r["vpdata_id"] not in failed_ids]
    original_all = train_rows + test_rows

    exclude_source_ids = {str(r.get("source_video_id")) for r in original_all if r.get("source_video_id")}
    exclude_paths = {str(r.get("path")) for r in original_all if r.get("path")}
    exclude_urls = {str(r.get("source_url")) for r in original_all if r.get("source_url")} | failed_urls

    pexels_map = read_pexels_url_map()
    pools = {
        "train": eligible_candidates(
            split="train",
            output_root=output_root,
            pexels_map=pexels_map,
            exclude_source_ids=exclude_source_ids,
            exclude_paths=exclude_paths,
            exclude_urls=exclude_urls,
        ),
        "test": eligible_candidates(
            split="test",
            output_root=output_root,
            pexels_map=pexels_map,
            exclude_source_ids=exclude_source_ids,
            exclude_paths=exclude_paths,
            exclude_urls=exclude_urls,
        ),
    }

    used_source_ids: set[str] = set()
    used_urls: set[str] = set()
    plan = []
    flat_csv = []
    for fail in sorted(failed, key=lambda r: (r["split"], r["sample_id"])):
        ranked = sorted(pools[fail["split"]], key=lambda c: stable_key(args.seed, fail["sample_id"], c))
        candidates = []
        for cand in ranked:
            if cand.source_video_id in used_source_ids or cand.source_url in used_urls:
                continue
            candidates.append(cand)
            used_source_ids.add(cand.source_video_id)
            if cand.source_url:
                used_urls.add(cand.source_url)
            if len(candidates) >= args.backups_per_failed:
                break
        if len(candidates) < args.backups_per_failed:
            raise SystemExit(f"not enough replacement candidates for {fail['sample_id']}")
        record = {
            "failed_split": fail["split"],
            "failed_vpdata_id": fail["sample_id"],
            "failed_url": fail["url"],
            "failed_reason": fail["reason"],
            "candidate_count": len(candidates),
            "candidates": [asdict(c) for c in candidates],
        }
        plan.append(record)
        for rank, cand in enumerate(candidates, 1):
            flat_csv.append({
                "failed_split": fail["split"],
                "failed_vpdata_id": fail["sample_id"],
                "rank": rank,
                "replacement_vpdata_id": cand.vpdata_id,
                "source_video_id": cand.source_video_id,
                "path": cand.path,
                "source_url": cand.source_url,
                "native_mask_id": cand.native_mask_id,
                "reason": "same-split deterministic Pexels-only replacement; original URL HTTP 403",
            })

    selected_by_failed = {p["failed_vpdata_id"]: p["candidates"][0] for p in plan}
    repaired_train = kept_train + [selected_by_failed[p["failed_vpdata_id"]] for p in plan if p["failed_split"] == "train"]
    repaired_test = kept_test + [selected_by_failed[p["failed_vpdata_id"]] for p in plan if p["failed_split"] == "test"]
    repaired_train = sorted(repaired_train, key=lambda r: r["vpdata_id"])
    repaired_test = sorted(repaired_test, key=lambda r: r["vpdata_id"])

    train_ids = {r["source_video_id"] for r in repaired_train}
    test_ids = {r["source_video_id"] for r in repaired_test}
    if train_ids & test_ids:
        raise SystemExit("replacement plan created train/test source_video_id overlap")

    write_jsonl(manifest_dir / "exp60c_vpdata_replacement_plan.jsonl", plan)
    write_jsonl(manifest_dir / "exp60c_vpdata_train1000_sources_h20_repaired.jsonl", repaired_train)
    write_jsonl(manifest_dir / "exp60c_vpdata_test100_sources_h20_repaired.jsonl", repaired_test)

    with (reports_dir / "exp60c_replacement_plan.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "failed_split",
            "failed_vpdata_id",
            "rank",
            "replacement_vpdata_id",
            "source_video_id",
            "path",
            "source_url",
            "native_mask_id",
            "reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_csv)

    summary = {
        "status": "EXP60C_REPLACEMENT_PLAN_READY",
        "seed": args.seed,
        "failed_rows": len(failed),
        "failed_train": sum(1 for r in failed if r["split"] == "train"),
        "failed_test": sum(1 for r in failed if r["split"] == "test"),
        "backups_per_failed": args.backups_per_failed,
        "replacement_candidates": len(flat_csv),
        "repaired_train_count": len(repaired_train),
        "repaired_test_count": len(repaired_test),
        "train_test_overlap_count": 0,
        "source_filter": "pexels_only",
        "full_vpdata_downloaded": False,
    }
    (reports_dir / "exp60c_replacement_plan_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    md = f"""# Exp60C Deterministic Replacement Plan

Status: `EXP60C_REPLACEMENT_PLAN_READY`

- Seed: `{args.seed}`
- Failed rows: `{len(failed)}`
- Train replacements needed: `{summary['failed_train']}`
- Test replacements needed: `{summary['failed_test']}`
- Backup candidates per failed row: `{args.backups_per_failed}`
- Candidate pool: same split, Pexels-only
- Excluded: existing 1,089 successful rows, original 11 failed rows, duplicate source ids, duplicate URLs, cross-split overlap
- Repaired train manifest count: `{len(repaired_train)}`
- Repaired test manifest count: `{len(repaired_test)}`

No videos were downloaded in this planning milestone.
"""
    (reports_dir / "exp60c_replacement_plan.md").write_text(md)
    print(json.dumps(summary, indent=2))
    return 0


def download_replacements(args: argparse.Namespace) -> int:
    manifest_dir = Path(args.manifest_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plan = load_jsonl(manifest_dir / "exp60c_vpdata_replacement_plan.jsonl")
    actual = []
    attempts = []
    failed_lines = []
    sha_lines = []
    for record in plan:
        chosen = None
        for rank, cand in enumerate(record["candidates"], 1):
            row = dict(cand)
            path = Path(row["planned_video_path"])
            status = "NOT_RUN"
            detail = ""
            try:
                if not path.exists():
                    download_url(row["source_url"], path)
                row["file_size_bytes"] = path.stat().st_size
                row["sha256"] = sha256_file(path)
                decode_status, frames, resolution, duration = ffprobe(path)
                row["decode_status"] = decode_status
                row["frame_count"] = frames
                row["resolution"] = resolution
                row["duration_sec"] = duration
                row["download_status"] = "DOWNLOADED"
                status = "DOWNLOADED"
                chosen = row
                sha_lines.append(f"{row['sha256']}  {row['planned_video_path']}")
            except Exception as exc:
                status = "FAILED"
                detail = f"{type(exc).__name__}:{exc}"
                row["download_status"] = f"DOWNLOAD_FAILED:{detail}"
                failed_lines.append(f"{record['failed_split']}\t{record['failed_vpdata_id']}\t{row['vpdata_id']}\t{row['download_status']}\t{row['source_url']}")
            attempts.append({
                "failed_split": record["failed_split"],
                "failed_vpdata_id": record["failed_vpdata_id"],
                "rank": rank,
                "replacement_vpdata_id": row["vpdata_id"],
                "source_video_id": row["source_video_id"],
                "source_url": row["source_url"],
                "status": status,
                "detail": detail,
                "planned_video_path": row["planned_video_path"],
                "file_size_bytes": row.get("file_size_bytes"),
                "sha256": row.get("sha256"),
                "decode_status": row.get("decode_status"),
            })
            if chosen is not None:
                break
        if chosen is None:
            actual.append({
                "failed_split": record["failed_split"],
                "failed_vpdata_id": record["failed_vpdata_id"],
                "replacement_status": "BLOCKED",
            })
        else:
            chosen["replacement_for_vpdata_id"] = record["failed_vpdata_id"]
            chosen["replacement_for_url"] = record["failed_url"]
            actual.append(chosen)

    write_jsonl(manifest_dir / "exp60c_vpdata_replacement_actual.jsonl", actual)
    with (reports_dir / "exp60c_replacement_download.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "failed_split",
            "failed_vpdata_id",
            "rank",
            "replacement_vpdata_id",
            "source_video_id",
            "source_url",
            "status",
            "detail",
            "planned_video_path",
            "file_size_bytes",
            "sha256",
            "decode_status",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(attempts)
    (reports_dir / "exp60c_replacement_failed_urls.txt").write_text("\n".join(failed_lines) + ("\n" if failed_lines else ""))
    (reports_dir / "exp60c_replacement_sha256.txt").write_text("\n".join(sha_lines) + ("\n" if sha_lines else ""))
    success = sum(1 for r in actual if r.get("download_status") == "DOWNLOADED")
    summary = {
        "status": "EXP60C_REPLACEMENT_DOWNLOAD_READY" if success == len(plan) else ("EXP60C_REPLACEMENT_DOWNLOAD_PARTIAL" if success else "EXP60C_REPLACEMENT_DOWNLOAD_BLOCKED"),
        "failed_rows": len(plan),
        "replacement_downloaded": success,
        "attempts": len(attempts),
        "failed_attempts": sum(1 for a in attempts if a["status"] != "DOWNLOADED"),
        "full_vpdata_downloaded": False,
    }
    (reports_dir / "exp60c_replacement_download_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    md = f"""# Exp60C Replacement Download

Status: `{summary['status']}`

- Replacement rows needed: `{len(plan)}`
- Replacement rows downloaded: `{success}`
- Attempts: `{len(attempts)}`
- Failed attempts: `{summary['failed_attempts']}`
- Full VPData downloaded: no
- Existing 1,089 videos redownloaded: no
"""
    (reports_dir / "exp60c_replacement_download.md").write_text(md)
    print(json.dumps(summary, indent=2))
    return 0 if success == len(plan) else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "download"])
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--manifest_dir", required=True)
    parser.add_argument("--reports_dir", required=True)
    parser.add_argument("--train_manifest", default="manifests/exp60b_vpdata_train1000_sources_h20.jsonl")
    parser.add_argument("--test_manifest", default="manifests/exp60b_vpdata_test100_sources_h20.jsonl")
    parser.add_argument("--failed_urls", default="reports/exp60b_h20_vpdata_subset_proxy_failed_urls.txt")
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--backups_per_failed", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "plan":
        return write_plan(args)
    return download_replacements(args)


if __name__ == "__main__":
    raise SystemExit(main())
