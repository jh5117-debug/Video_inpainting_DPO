#!/usr/bin/env python3
"""H20-local D3 selected-primary-nocomp readiness check for Exp9."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tools.d2_training_readiness_check as d2_ready


D3_ROOT = (
    "/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/"
    "official_videodpo_diffueraser_youtubevos_partialmask_loser_k4"
)
FIELDS = ["win_video_path", "raw_loser_video_path", "final_loser_video_path", "mask_path"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default=D3_ROOT)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--canonical_num_frames", type=int, default=16)
    parser.add_argument("--canonical_height", type=int, default=320)
    parser.add_argument("--canonical_width", type=int, default=512)
    parser.add_argument("--report", default="")
    return parser.parse_args()


def sample_audit_nocomp(rows: list[dict], args: argparse.Namespace) -> dict:
    import random
    from collections import Counter

    rng = random.Random(args.seed)
    sampled = rng.sample(rows, min(args.sample_size, len(rows))) if rows else []
    expected_sizes = {f"{args.canonical_width}x{args.canonical_height}", f"{args.canonical_height}x{args.canonical_width}"}
    frame_counter = Counter()
    size_counter = Counter()
    issues = []
    for row in sampled:
        sid = row.get("sample_id", "<unknown>")
        mask_id = row.get("mask_id", "<unknown>")
        for field in FIELDS:
            info = d2_ready.frame_dir_info(str(row.get(field, "") or ""))
            if info["frames"]:
                frame_counter[f"{field}:{info['frames']}"] += 1
            if info["size"]:
                size_counter[f"{field}:{info['size']}"] += 1
            if (
                not info["exists"]
                or not info["readable"]
                or info["frames"] != args.canonical_num_frames
                or info["size"] not in expected_sizes
            ):
                issues.append(f"{sid}/{mask_id} {field}: {info}")
        final_path = str(row.get("final_loser_video_path", "") or "")
        raw_path = str(row.get("raw_loser_video_path", "") or "")
        if final_path != raw_path:
            issues.append(f"{sid}/{mask_id} final_loser_video_path != raw_loser_video_path")
    return {
        "sampled": len(sampled),
        "frame_counter": dict(frame_counter),
        "size_counter": dict(size_counter),
        "issues": issues[:200],
        "issue_count": len(issues),
    }


def write_report(path: Path, payload: dict) -> None:
    lines = [
        "# D3 H20 Primary-Nocomp Gate Readiness Report",
        "",
        f"- output_root: `{payload['output_root']}`",
        f"- manifest: `{payload['manifest']}`",
        f"- rows: {payload['rows']}",
        f"- ready_h20_primary_nocomp_gate: `{payload['ready_h20_primary_nocomp_gate']}`",
        "",
        "## Path Audit",
        "",
        "```json",
        json.dumps(payload["path_audit"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Sample Audit",
        "",
        "```json",
        json.dumps(payload["sample_audit"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Issues",
        "",
    ]
    if payload["issues"]:
        lines.extend(f"- {issue}" for issue in payload["issues"][:100])
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    manifests_dir = output_root / "manifests"
    repaired = manifests_dir / "selected_primary_nocomp.repaired.jsonl"
    original = manifests_dir / "selected_primary_nocomp.jsonl"
    manifest = Path(args.manifest).expanduser() if args.manifest else (repaired if repaired.exists() else original)
    report = Path(args.report).expanduser() if args.report else output_root / "reports" / "d3_h20_primary_nocomp_gate_readiness_report.md"

    rows, parse_errors = d2_ready.read_jsonl(manifest)
    issues = []
    if parse_errors:
        issues.extend(f"parse: {err}" for err in parse_errors[:20])
    if not rows:
        issues.append("manifest has zero rows")
    for row in rows:
        for value in row.values():
            if isinstance(value, str) and value.startswith("/mnt/workspace/"):
                issues.append("manifest contains PAI /mnt/workspace paths")
                break
        if issues and issues[-1] == "manifest contains PAI /mnt/workspace paths":
            break

    path_audit = d2_ready.path_audit(rows, FIELDS) if rows else {}
    if path_audit.get("missing"):
        issues.append(f"missing paths: {path_audit['missing']}")
    sample_audit = sample_audit_nocomp(rows, args) if rows else {"sampled": 0, "issue_count": 1, "issues": ["no rows"]}
    if sample_audit.get("issue_count", 0):
        issues.append(f"sample issues: {sample_audit['issue_count']}")

    ready = not issues
    payload = {
        "output_root": str(output_root),
        "manifest": str(manifest),
        "rows": len(rows),
        "path_audit": path_audit,
        "sample_audit": sample_audit,
        "issues": issues,
        "ready_h20_primary_nocomp_gate": ready,
    }
    write_report(report, payload)
    print(f"[d3-h20-nocomp-ready] output_root={output_root}")
    print(f"[d3-h20-nocomp-ready] manifest={manifest}")
    print(f"[d3-h20-nocomp-ready] report={report}")
    print(f"[d3-h20-nocomp-ready] rows={len(rows)}")
    print(f"[d3-h20-nocomp-ready] sampled={sample_audit.get('sampled')} sample_issues={sample_audit.get('issue_count')}")
    print(f"[d3-h20-nocomp-ready] ready_h20_primary_nocomp_gate={str(ready).lower()}")
    if issues:
        for issue in issues[:20]:
            print(f"[d3-h20-nocomp-ready][issue] {issue}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
