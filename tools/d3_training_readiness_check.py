#!/usr/bin/env python3
"""Final training-readiness checks for D3 repaired manifests.

Run this after D3 has been synced to PAI and after
`tools/d3_post_generation_audit_and_repair.py` has created repaired manifests.
The script is read-only for generated frames and writes optional PAI-path
rewritten manifests next to the repaired manifests.
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
    parser.add_argument("--expected_selected_rows", type=int, default=3327)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--canonical_num_frames", type=int, default=16)
    parser.add_argument("--canonical_height", type=int, default=320)
    parser.add_argument("--canonical_width", type=int, default=512)
    parser.add_argument("--outside_mean_abs_threshold", type=float, default=0.5)
    parser.add_argument("--outside_max_abs_threshold", type=float, default=2.0)
    parser.add_argument("--h20_prefix", default="/home/nvme01/H20_Video_inpainting_DPO")
    parser.add_argument("--pai_prefix", default="/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO")
    parser.add_argument("--report", default="")
    return parser


def write_report(path: Path, payload: dict) -> None:
    lines = [
        "# D3 Training Readiness Report",
        "",
        f"- output_root: `{payload['output_root']}`",
        f"- ready_for_exp9_stage1_gate: `{payload['ready_for_exp9_stage1_gate']}`",
        "- note: this report checks D3 target-domain generated-loser data only; it does not approve launching Exp9 training.",
        "",
        "## Repaired Manifest Counts",
        "",
    ]
    for name, count in payload["manifest_counts"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Metadata", "", "```json"])
    lines.append(json.dumps(payload["metadata"], ensure_ascii=False, indent=2, sort_keys=True))
    lines.extend(["```", "", "## Path Audit", "", "```json"])
    lines.append(json.dumps(payload["path_audit"], ensure_ascii=False, indent=2, sort_keys=True))
    lines.extend(["```", "", "## Consistency", "", "```json"])
    lines.append(json.dumps(payload["consistency"], ensure_ascii=False, indent=2, sort_keys=True))
    lines.extend(["```", "", "## Random Sample Decode And Comp Check", "", "```json"])
    lines.append(json.dumps(payload["sample_audit"], ensure_ascii=False, indent=2, sort_keys=True))
    lines.extend(["```", "", "## PAI Path Rewrites", ""])
    if payload["rewritten_manifests"]:
        for item in payload["rewritten_manifests"]:
            lines.append(f"- `{item}`")
    else:
        lines.append("- none needed")
    lines.extend(["", "## JSONL Parse Errors", ""])
    if any(payload["parse_errors"].values()):
        for name, errors in payload["parse_errors"].items():
            for error in errors[:50]:
                lines.append(f"- `{name}`: {error}")
    else:
        lines.append("- none")
    lines.extend(["", "## Exp9 Entry Mapping", ""])
    lines.extend(
        [
            "- First gate manifest: `manifests/selected_primary_comp.repaired.pai_paths.jsonl` if PAI rewrites were written, otherwise `manifests/selected_primary_comp.repaired.jsonl`.",
            "- `TRAIN_MASK_MODE=partial`",
            "- `MASK_FROM_MANIFEST=true`",
            "- `LOSS_REGION_MODE=full` for the first target-domain gate.",
            "- Stage1 DPO only; keep Stage2 frozen SFT/target-domain temporal weights.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    manifests_dir = output_root / "manifests"
    if not manifests_dir.exists():
        raise SystemExit(f"[error] manifests dir not found: {manifests_dir}")
    report_path = Path(args.report).expanduser() if args.report else output_root / "reports" / "d3_training_readiness_report.md"

    rows_by_name = {}
    parse_errors = {}
    manifest_counts = {}
    for name in d2_ready.REPAIRED_SELECTED_MANIFESTS:
        rows, errors = d2_ready.read_jsonl(manifests_dir / name)
        rows_by_name[name] = rows
        parse_errors[name] = errors
        manifest_counts[name] = len(rows) if not errors or rows else "MISSING"

    rewritten_manifests = []
    rewritten_by_name = {}
    for name, rows in rows_by_name.items():
        rewritten = d2_ready.maybe_write_rewritten_manifest(
            manifests_dir,
            name,
            rows,
            args.h20_prefix,
            args.pai_prefix,
        )
        if rewritten:
            rewritten_manifests.append(str(rewritten))
            rewritten_rows, rewritten_errors = d2_ready.read_jsonl(rewritten)
            rewritten_by_name[name] = rewritten_rows
            if rewritten_errors:
                parse_errors[f"{name}.pai_paths"] = rewritten_errors

    check_rows_by_name = {
        name: rewritten_by_name.get(name, rows)
        for name, rows in rows_by_name.items()
    }

    primary_comp = check_rows_by_name["selected_primary_comp.repaired.jsonl"]
    all_selected_rows = [row for rows in check_rows_by_name.values() for row in rows]

    path_checks = {
        name: d2_ready.path_audit(
            rows,
            d2_ready.TRAINING_PATH_FIELDS
            if "comp" in name and "nocomp" not in name
            else ["win_video_path", "raw_loser_video_path", "final_loser_video_path", "mask_path"],
        )
        for name, rows in check_rows_by_name.items()
    }
    consistency = {name: d2_ready.selected_consistency(rows, name) for name, rows in check_rows_by_name.items()}
    metadata = d2_ready.metadata_audit(all_selected_rows)
    sample = d2_ready.sample_audit(primary_comp, args)

    critical_issues = []
    for name, count in manifest_counts.items():
        if count != args.expected_selected_rows:
            critical_issues.append(f"{name} count {count} != {args.expected_selected_rows}")
    for name, errors in parse_errors.items():
        if errors:
            critical_issues.append(f"{name} parse errors: {len(errors)}")
    for name, audit in path_checks.items():
        if audit["missing"]:
            critical_issues.append(f"{name} missing paths: {audit['missing']}")
        if audit["h20_only"]:
            critical_issues.append(f"{name} still has H20-only paths after rewrite selection: {audit['h20_only']}")
    for name, item in consistency.items():
        if item.get("duplicate_sample_ids"):
            critical_issues.append(f"{name} duplicate_sample_ids={item['duplicate_sample_ids']}")
        if item.get("final_not_raw"):
            critical_issues.append(f"{name} final_not_raw={item['final_not_raw']}")
        if item.get("final_not_comp"):
            critical_issues.append(f"{name} final_not_comp={item['final_not_comp']}")
    if sample["issue_count"]:
        critical_issues.append(f"sample issues: {sample['issue_count']}")

    ready = not critical_issues
    payload = {
        "output_root": str(output_root),
        "manifest_counts": manifest_counts,
        "metadata": metadata,
        "path_audit": path_checks,
        "consistency": consistency,
        "sample_audit": sample,
        "parse_errors": parse_errors,
        "rewritten_manifests": rewritten_manifests,
        "critical_issues": critical_issues,
        "ready_for_exp9_stage1_gate": ready,
    }
    write_report(report_path, payload)

    print(f"[d3-ready] output_root={output_root}")
    print(f"[d3-ready] report={report_path}")
    for name, count in manifest_counts.items():
        print(f"[d3-ready] {name} {count}")
    print(f"[d3-ready] sampled={sample['sampled']} sample_issues={sample['issue_count']}")
    print(f"[d3-ready] rewritten_manifests={len(rewritten_manifests)}")
    print(f"[d3-ready] ready={ready}")
    if critical_issues:
        for issue in critical_issues[:20]:
            print(f"[d3-ready][issue] {issue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
