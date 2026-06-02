#!/usr/bin/env python3
"""Primary-comp gate readiness check for Exp9 D3 Stage1 training.

The full D3 readiness script checks primary and secondary manifests. The first
Exp9 gate only consumes selected-primary-comp, so this read-only checker
separates full D3 readiness from primary-comp gate readiness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tools.d2_training_readiness_check as d2_ready


D3_ROOT = (
    "/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/"
    "official_videodpo_diffueraser_youtubevos_partialmask_loser_k4"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default=D3_ROOT)
    parser.add_argument(
        "--manifest",
        default="",
        help="Defaults to <output_root>/manifests/selected_primary_comp.repaired.jsonl.",
    )
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--canonical_num_frames", type=int, default=16)
    parser.add_argument("--canonical_height", type=int, default=320)
    parser.add_argument("--canonical_width", type=int, default=512)
    parser.add_argument("--outside_mean_abs_threshold", type=float, default=0.5)
    parser.add_argument("--outside_max_abs_threshold", type=float, default=2.0)
    parser.add_argument("--h20_prefix", default="/home/nvme01/H20_Video_inpainting_DPO")
    parser.add_argument("--report", default="")
    return parser


def write_report(path: Path, payload: dict) -> None:
    lines = [
        "# D3 Primary-Comp Gate Readiness Report",
        "",
        f"- output_root: `{payload['output_root']}`",
        f"- manifest: `{payload['manifest']}`",
        f"- d3_full_readiness: `{payload['d3_full_readiness']}`",
        f"- ready_primary_comp_gate: `{payload['ready_primary_comp_gate']}`",
        "- note: secondary manifests are not required for the first Exp9 Stage1 gate.",
        "",
        "## Counts",
        "",
        f"- rows: {payload['rows']}",
        f"- sampled: {payload['sample_audit']['sampled']}",
        f"- sample_issues: {payload['sample_audit']['issue_count']}",
        "",
        "## Path Audit",
        "",
        "```json",
        json.dumps(payload["path_audit"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Metadata",
        "",
        "```json",
        json.dumps(payload["metadata"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Sample Decode And Comp Check",
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
    lines.extend(
        [
            "",
            "## Exp9 Gate Contract",
            "",
            "- `TRAIN_MASK_MODE=partial`",
            "- `MASK_FROM_MANIFEST=true`",
            "- `LOSS_REGION_MODE=full`",
            "- Stage1 DPO only; no DPO Stage2.",
            "- Use target-domain inpainting metrics from `inference/metrics.py`; no VBench.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    manifests_dir = output_root / "manifests"
    manifest = Path(args.manifest).expanduser() if args.manifest else manifests_dir / "selected_primary_comp.repaired.jsonl"
    report = Path(args.report).expanduser() if args.report else output_root / "reports" / "d3_primary_comp_gate_readiness_report.md"

    rows, parse_errors = d2_ready.read_jsonl(manifest)
    issues: list[str] = []
    if parse_errors:
        issues.extend(f"parse: {error}" for error in parse_errors[:20])
    if not rows:
        issues.append("manifest has zero rows")

    h20_path_count = 0
    for row in rows:
        for value in row.values():
            if isinstance(value, str) and value.startswith(args.h20_prefix):
                h20_path_count += 1
    if h20_path_count:
        issues.append(f"manifest contains H20-only paths: {h20_path_count}")

    path_audit = d2_ready.path_audit(rows, d2_ready.TRAINING_PATH_FIELDS) if rows else {}
    if path_audit.get("missing"):
        issues.append(f"missing paths: {path_audit['missing']}")
    if path_audit.get("h20_only"):
        issues.append(f"H20-only paths: {path_audit['h20_only']}")

    metadata = d2_ready.metadata_audit(rows) if rows else {}
    sample_audit = d2_ready.sample_audit(rows, args) if rows else {"sampled": 0, "issue_count": 1, "issues": ["no rows"]}
    if sample_audit.get("issue_count", 0):
        issues.append(f"sample issues: {sample_audit['issue_count']}")

    ready = len(issues) == 0
    payload = {
        "output_root": str(output_root),
        "manifest": str(manifest),
        "rows": len(rows),
        "d3_full_readiness": False,
        "ready_primary_comp_gate": ready,
        "path_audit": path_audit,
        "metadata": metadata,
        "sample_audit": sample_audit,
        "issues": issues,
    }
    write_report(report, payload)

    print(f"[d3-primary-ready] output_root={output_root}")
    print(f"[d3-primary-ready] manifest={manifest}")
    print(f"[d3-primary-ready] report={report}")
    print(f"[d3-primary-ready] rows={len(rows)}")
    print(f"[d3-primary-ready] sampled={sample_audit.get('sampled')} sample_issues={sample_audit.get('issue_count')}")
    print("[d3-primary-ready] d3_full_readiness=false")
    print(f"[d3-primary-ready] ready_primary_comp_gate={str(ready).lower()}")
    if issues:
        for issue in issues[:20]:
            print(f"[d3-primary-ready][issue] {issue}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
