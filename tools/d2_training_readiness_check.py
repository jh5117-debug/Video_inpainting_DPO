#!/usr/bin/env python3
"""Final training-readiness checks for D2 repaired manifests.

This script is intentionally read-only for generated data. It does not
regenerate videos and does not overwrite original or repaired manifests. If an
H20-only path is found, it writes optional PAI-path rewritten copies next to the
repaired manifests.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


REPAIRED_SELECTED_MANIFESTS = [
    "selected_primary_comp.repaired.jsonl",
    "selected_primary_nocomp.repaired.jsonl",
    "selected_secondary_comp.repaired.jsonl",
    "selected_secondary_nocomp.repaired.jsonl",
]

TRAINING_PATH_FIELDS = [
    "win_video_path",
    "raw_loser_video_path",
    "comp_loser_video_path",
    "final_loser_video_path",
    "mask_path",
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.exists():
        return rows, [f"missing: {path}"]
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                errors.append(f"{path.name}:{line_no}: {exc}")
                continue
            if not isinstance(obj, dict):
                errors.append(f"{path.name}:{line_no}: expected object, got {type(obj).__name__}")
                continue
            rows.append(obj)
    return rows, errors


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def image_paths(path: Path) -> list[Path]:
    if not path.exists() or not path.is_dir():
        return []
    direct = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if direct:
        return direct
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def frame_dir_info(path: str) -> dict[str, Any]:
    p = Path(path)
    imgs = image_paths(p)
    if not imgs:
        return {
            "exists": p.exists(),
            "readable": False,
            "frames": 0,
            "size": "",
            "first": "",
            "error": "no image frames",
        }
    try:
        with Image.open(imgs[0]) as im:
            size = f"{im.width}x{im.height}"
            im.verify()
    except Exception as exc:
        return {
            "exists": True,
            "readable": False,
            "frames": len(imgs),
            "size": "",
            "first": str(imgs[0]),
            "error": str(exc),
        }
    return {
        "exists": True,
        "readable": True,
        "frames": len(imgs),
        "size": size,
        "first": str(imgs[0]),
        "error": "",
    }


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.int16)


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def comp_outside_diff(row: dict[str, Any], max_frames: int) -> dict[str, Any]:
    win = image_paths(Path(str(row.get("win_video_path", "") or "")))
    mask = image_paths(Path(str(row.get("mask_path", "") or "")))
    comp = image_paths(Path(str(row.get("comp_loser_video_path") or row.get("final_loser_video_path", "") or "")))
    n = min(len(win), len(mask), len(comp), max_frames)
    if n <= 0:
        return {"ok": False, "frames": n, "mean_abs": None, "max_abs": None, "error": "missing frames"}

    total_abs = 0.0
    total_count = 0
    max_abs = 0
    for i in range(n):
        w = load_rgb(win[i])
        c = load_rgb(comp[i])
        m = load_gray(mask[i])
        if w.shape[:2] != c.shape[:2] or w.shape[:2] != m.shape[:2]:
            return {
                "ok": False,
                "frames": i,
                "mean_abs": None,
                "max_abs": None,
                "error": f"shape mismatch win={w.shape[:2]} comp={c.shape[:2]} mask={m.shape[:2]}",
            }
        outside = m <= 127
        if not np.any(outside):
            continue
        diff = np.abs(c[outside] - w[outside])
        total_abs += float(diff.sum())
        total_count += int(diff.size)
        if diff.size:
            max_abs = max(max_abs, int(diff.max()))
    if total_count == 0:
        return {"ok": False, "frames": n, "mean_abs": None, "max_abs": None, "error": "no outside pixels"}
    mean_abs = total_abs / total_count
    return {"ok": True, "frames": n, "mean_abs": mean_abs, "max_abs": max_abs, "error": ""}


def path_audit(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    checked = Counter()
    empty = Counter()
    missing = Counter()
    h20_only = Counter()
    examples: dict[str, list[str]] = {"missing": [], "h20_only": []}
    for row in rows:
        for field in fields:
            value = str(row.get(field, "") or "")
            if not value:
                empty[field] += 1
                missing[field] += 1
                if len(examples["missing"]) < 20:
                    examples["missing"].append(f"{row.get('sample_id')} {field}=<empty>")
                continue
            checked[field] += 1
            if value.startswith("/home/nvme01/"):
                h20_only[field] += 1
                if len(examples["h20_only"]) < 20:
                    examples["h20_only"].append(f"{row.get('sample_id')} {field}={value}")
            if not Path(value).exists():
                missing[field] += 1
                if len(examples["missing"]) < 20:
                    examples["missing"].append(f"{row.get('sample_id')} {field}={value}")
    return {
        "checked": dict(checked),
        "empty": dict(empty),
        "missing": dict(missing),
        "h20_only": dict(h20_only),
        "examples": examples,
    }


def metadata_audit(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "source_dataset": dict(Counter(str(r.get("source_dataset")) for r in rows)),
        "generation_source": dict(Counter(str(r.get("generation_source")) for r in rows)),
        "generation_model": dict(Counter(str(r.get("generation_model")) for r in rows)),
        "mask_mode": dict(Counter(str(r.get("mask_mode")) for r in rows)),
        "num_masks_per_video": dict(Counter(str(r.get("num_masks_per_video")) for r in rows)),
        "data_asset": dict(Counter(str(r.get("data_asset")) for r in rows)),
        "canonical_num_frames": dict(Counter(str(r.get("canonical_num_frames")) for r in rows)),
        "canonical_height": dict(Counter(str(r.get("canonical_height")) for r in rows)),
        "canonical_width": dict(Counter(str(r.get("canonical_width")) for r in rows)),
        "diffueraser_inference_stack": dict(Counter(str(r.get("diffueraser_inference_stack")) for r in rows)),
        "diffueraser_prior_mode": dict(Counter(str(r.get("diffueraser_prior_mode")) for r in rows)),
    }


def selected_consistency(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    out = Counter()
    sample_ids = Counter(str(r.get("sample_id")) for r in rows)
    out["duplicate_sample_ids"] = sum(1 for count in sample_ids.values() if count > 1)
    out["unique_sample_ids"] = len(sample_ids)
    out["unique_pair_index"] = len({str(r.get("pair_index")) for r in rows})
    for row in rows:
        final_path = str(row.get("final_loser_video_path", "") or "")
        raw_path = str(row.get("raw_loser_video_path", "") or "")
        comp_path = str(row.get("comp_loser_video_path", "") or "")
        if "nocomp" in name and final_path != raw_path:
            out["final_not_raw"] += 1
        if "comp" in name and "nocomp" not in name and final_path != comp_path:
            out["final_not_comp"] += 1
    return dict(out)


def replace_prefix(value: Any, old_prefix: str, new_prefix: str) -> Any:
    if isinstance(value, str) and value.startswith(old_prefix):
        return new_prefix + value[len(old_prefix) :]
    return value


def maybe_write_rewritten_manifest(
    manifests_dir: Path,
    name: str,
    rows: list[dict[str, Any]],
    old_prefix: str,
    new_prefix: str,
) -> Path | None:
    has_h20_path = any(
        isinstance(value, str) and value.startswith(old_prefix)
        for row in rows
        for value in row.values()
    )
    if not has_h20_path:
        return None
    rewritten = [
        {key: replace_prefix(value, old_prefix, new_prefix) for key, value in row.items()}
        for row in rows
    ]
    out_name = name.replace(".jsonl", ".pai_paths.jsonl")
    out_path = manifests_dir / out_name
    write_jsonl(out_path, rewritten)
    return out_path


def sample_audit(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    sampled = rng.sample(rows, min(args.sample_size, len(rows))) if rows else []
    expected_sizes = {f"{args.canonical_width}x{args.canonical_height}", f"{args.canonical_height}x{args.canonical_width}"}

    frame_counter = Counter()
    size_counter = Counter()
    issues: list[str] = []
    outside_results: list[dict[str, Any]] = []
    for row in sampled:
        sid = row.get("sample_id", "<unknown>")
        mask_id = row.get("mask_id", "<unknown>")
        for field in TRAINING_PATH_FIELDS:
            info = frame_dir_info(str(row.get(field, "") or ""))
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
        diff = comp_outside_diff(row, args.canonical_num_frames)
        outside_results.append(diff)
        mean_abs = diff.get("mean_abs")
        max_abs = diff.get("max_abs")
        if (
            diff.get("error")
            or mean_abs is None
            or max_abs is None
            or mean_abs > args.outside_mean_abs_threshold
            or max_abs > args.outside_max_abs_threshold
        ):
            issues.append(f"{sid}/{mask_id} comp outside diff: {diff}")

    mean_values = [x["mean_abs"] for x in outside_results if x.get("mean_abs") is not None]
    max_values = [x["max_abs"] for x in outside_results if x.get("max_abs") is not None]
    outside_ok = sum(
        1
        for x in outside_results
        if not x.get("error")
        and x.get("mean_abs") is not None
        and x.get("max_abs") is not None
        and x["mean_abs"] <= args.outside_mean_abs_threshold
        and x["max_abs"] <= args.outside_max_abs_threshold
    )
    return {
        "sampled": len(sampled),
        "frame_counter": dict(frame_counter),
        "size_counter": dict(size_counter),
        "outside_ok": outside_ok,
        "outside_mean_abs_max": max(mean_values) if mean_values else None,
        "outside_max_abs_max": max(max_values) if max_values else None,
        "outside_mean_abs_threshold": args.outside_mean_abs_threshold,
        "outside_max_abs_threshold": args.outside_max_abs_threshold,
        "issues": issues[:200],
        "issue_count": len(issues),
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# D2 Training Readiness Report",
        "",
        f"- output_root: `{payload['output_root']}`",
        f"- ready_for_experiment_5_6_data_entry: `{payload['ready_for_experiment_5_6_data_entry']}`",
        f"- ready_for_experiment_7_8_data_entry: `{payload['ready_for_experiment_7_8_data_entry']}`",
        "- note: this report checks data and manifest readiness only; current training code still needs the generated-loser manifest dataset adapter before experiments 5/6/7/8 can start.",
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
    lines.extend(["", "## Training Entry Mapping", ""])
    lines.extend(
        [
            "- Experiment 5: `manifests/selected_primary_comp.repaired.jsonl`",
            "- Experiment 6: `manifests/selected_primary_nocomp.repaired.jsonl`",
            "- Experiment 7: `manifests/selected_primary_comp.repaired.jsonl` plus `mask_path`, `train_mask_mode=partial`, `loss_region_mode=full`",
            "- Experiment 8: `manifests/selected_primary_comp.repaired.jsonl` plus `mask_path`, `train_mask_mode=partial`, `loss_region_mode=region`",
            "",
            "## Current Training-Code Gap",
            "",
            "The repository currently does not yet expose `--preference_manifest`, `--mask_from_manifest`, `--train_mask_mode`, `--loss_region_mode`, or `--enable_dpo_diag`. Add the manifest dataset adapter before launching experiments 5/6/7/8.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_root",
        default="/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4",
    )
    parser.add_argument("--expected_selected_rows", type=int, default=10000)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--canonical_num_frames", type=int, default=16)
    parser.add_argument("--canonical_height", type=int, default=320)
    parser.add_argument("--canonical_width", type=int, default=512)
    parser.add_argument("--outside_mean_abs_threshold", type=float, default=0.5)
    parser.add_argument("--outside_max_abs_threshold", type=float, default=2.0)
    parser.add_argument("--h20_prefix", default="/home/nvme01/H20_Video_inpainting_DPO")
    parser.add_argument("--pai_prefix", default="/mnt/nas/hj/H20_Video_inpainting_DPO")
    parser.add_argument("--report", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    manifests_dir = output_root / "manifests"
    if not manifests_dir.exists():
        raise SystemExit(f"[error] manifests dir not found: {manifests_dir}")
    report_path = Path(args.report).expanduser() if args.report else output_root / "reports" / "d2_training_readiness_report.md"

    rows_by_name: dict[str, list[dict[str, Any]]] = {}
    parse_errors: dict[str, list[str]] = {}
    manifest_counts: dict[str, int | str] = {}
    for name in REPAIRED_SELECTED_MANIFESTS:
        rows, errors = read_jsonl(manifests_dir / name)
        rows_by_name[name] = rows
        parse_errors[name] = errors
        manifest_counts[name] = len(rows) if not errors or rows else "MISSING"

    primary_comp = rows_by_name["selected_primary_comp.repaired.jsonl"]
    primary_nocomp = rows_by_name["selected_primary_nocomp.repaired.jsonl"]
    all_selected_rows = [row for rows in rows_by_name.values() for row in rows]

    path_checks = {
        name: path_audit(rows, TRAINING_PATH_FIELDS if "comp" in name and "nocomp" not in name else [
            "win_video_path",
            "raw_loser_video_path",
            "final_loser_video_path",
            "mask_path",
        ])
        for name, rows in rows_by_name.items()
    }
    consistency = {name: selected_consistency(rows, name) for name, rows in rows_by_name.items()}
    metadata = metadata_audit(all_selected_rows)
    sample = sample_audit(primary_comp, args)

    rewritten_manifests: list[str] = []
    for name, rows in rows_by_name.items():
        rewritten = maybe_write_rewritten_manifest(
            manifests_dir,
            name,
            rows,
            args.h20_prefix,
            args.pai_prefix,
        )
        if rewritten:
            rewritten_manifests.append(str(rewritten))

    critical_issues: list[str] = []
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
            critical_issues.append(f"{name} has H20-only paths: {audit['h20_only']}")
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
        "ready_for_experiment_5_6_data_entry": ready,
        "ready_for_experiment_7_8_data_entry": ready,
    }
    write_report(report_path, payload)

    print(f"[d2-ready] output_root={output_root}")
    print(f"[d2-ready] report={report_path}")
    for name, count in manifest_counts.items():
        print(f"[d2-ready] {name} {count}")
    print(f"[d2-ready] sampled={sample['sampled']} sample_issues={sample['issue_count']}")
    print(f"[d2-ready] rewritten_manifests={len(rewritten_manifests)}")
    print(f"[d2-ready] ready={ready}")
    if critical_issues:
        for issue in critical_issues[:20]:
            print(f"[d2-ready][issue] {issue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
