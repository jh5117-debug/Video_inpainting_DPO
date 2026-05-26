#!/usr/bin/env python3
"""Audit and repair D2 VideoDPO partial-mask generated-loser manifests.

This script is intentionally post-generation only: it never regenerates video
frames and never overwrites the original manifests.  It adds explicit metadata
to repaired manifests and samples the generated frame directories for basic
readability, shape, and composition checks.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


MANIFEST_NAMES = [
    "candidates_all.jsonl",
    "candidates_all.scored.jsonl",
    "selected_primary_comp.jsonl",
    "selected_primary_nocomp.jsonl",
    "selected_secondary_comp.jsonl",
    "selected_secondary_nocomp.jsonl",
    "selection_events.jsonl",
]

REPAIR_MANIFEST_NAMES = [
    "candidates_all.jsonl",
    "candidates_all.scored.jsonl",
    "selected_primary_comp.jsonl",
    "selected_primary_nocomp.jsonl",
    "selected_secondary_comp.jsonl",
    "selected_secondary_nocomp.jsonl",
]

PATH_FIELDS = [
    "final_loser_video_path",
    "comp_loser_video_path",
    "raw_loser_video_path",
    "mask_path",
    "win_video_path",
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
        return {"exists": p.exists(), "frames": 0, "size": "", "first": "", "error": "no image frames"}
    try:
        with Image.open(imgs[0]) as im:
            size = f"{im.width}x{im.height}"
    except Exception as exc:
        return {"exists": True, "frames": len(imgs), "size": "", "first": str(imgs[0]), "error": str(exc)}
    return {"exists": True, "frames": len(imgs), "size": size, "first": str(imgs[0]), "error": ""}


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.int16)


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def comp_outside_diff(row: dict[str, Any], max_frames: int = 16) -> dict[str, Any]:
    win = image_paths(Path(str(row.get("win_video_path", ""))))
    mask = image_paths(Path(str(row.get("mask_path", ""))))
    comp = image_paths(Path(str(row.get("comp_loser_video_path") or row.get("final_loser_video_path", ""))))
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
        keep = m <= 127
        if not np.any(keep):
            continue
        diff = np.abs(c[keep] - w[keep])
        total_abs += float(diff.sum())
        total_count += int(diff.size)
        max_abs = max(max_abs, int(diff.max()) if diff.size else 0)
    if total_count == 0:
        return {"ok": False, "frames": n, "mean_abs": None, "max_abs": None, "error": "no keep-region pixels"}
    mean_abs = total_abs / total_count
    return {"ok": mean_abs <= 1e-6 and max_abs == 0, "frames": n, "mean_abs": mean_abs, "max_abs": max_abs, "error": ""}


def confirm_stack_prior(output_root: Path, user_stack: str, user_prior: str) -> tuple[str, str, list[str]]:
    if user_stack != "auto" and user_prior != "auto":
        return user_stack, user_prior, ["set from CLI"]

    evidence: list[str] = []
    stack = "unconfirmed"
    prior = "unconfirmed"
    shards_root = output_root / "_shards"
    checked = 0
    if shards_root.exists():
        for shard in sorted(p for p in shards_root.iterdir() if p.is_dir()):
            checked += 1
            if (shard / "work").exists() and any((shard / "work").rglob("run_or")):
                stack = "or"
                evidence.append(f"found run_or work dir under {shard.name}")
            log_root = shard / "logs"
            if log_root.exists():
                for log in log_root.rglob("*diffueraser.log"):
                    text = log.read_text(encoding="utf-8", errors="ignore")[-12000:]
                    if "run_OR.py" in text:
                        stack = "or"
                        evidence.append(f"found run_OR.py in {log}")
                    if "propainter_model_dir" in text or "--propainter_model_dir" in text:
                        prior = "propainter"
                        evidence.append(f"found propainter_model_dir in {log}")
                    if stack != "unconfirmed" and prior != "unconfirmed":
                        break
            if stack != "unconfirmed" and prior != "unconfirmed":
                break
            if checked >= 80:
                break

    if user_stack != "auto":
        stack = user_stack
    if user_prior != "auto":
        prior = user_prior
    if not evidence:
        evidence.append("no stack/prior evidence found in sampled shard work/log files")
    return stack, prior, evidence[:10]


def repair_row(
    row: dict[str, Any],
    manifest_name: str,
    stack: str,
    prior: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    repaired = dict(row)
    repaired["source_dataset"] = "videodpo"
    repaired["generation_source"] = "diffueraser_only"
    repaired["generation_model"] = "diffueraser"
    repaired["mask_mode"] = "partial"
    repaired["num_masks_per_video"] = 4
    repaired["data_asset"] = args.data_asset
    repaired["canonical_num_frames"] = args.canonical_num_frames
    repaired["canonical_height"] = args.canonical_height
    repaired["canonical_width"] = args.canonical_width
    repaired["diffueraser_inference_stack"] = stack
    repaired["diffueraser_prior_mode"] = prior

    canonical = repaired.get("canonical_setting")
    if isinstance(canonical, dict):
        canonical["canonical_num_frames"] = args.canonical_num_frames
        canonical["canonical_height"] = args.canonical_height
        canonical["canonical_width"] = args.canonical_width
        repaired["canonical_setting"] = canonical

    if "nocomp" in manifest_name:
        repaired["final_loser_video_path"] = repaired.get("raw_loser_video_path", "")
        repaired["final_loser_variant"] = "raw_loser"
        repaired["comp"] = False
    elif "comp" in manifest_name:
        repaired["final_loser_video_path"] = repaired.get("comp_loser_video_path", "")
        repaired["final_loser_variant"] = "comp_loser"
        repaired["comp"] = True

    return repaired


def path_audit(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    missing = Counter()
    h20_only = Counter()
    checked = Counter()
    for row in rows:
        for field in fields:
            value = str(row.get(field, "") or "")
            if not value:
                missing[field] += 1
                continue
            checked[field] += 1
            if value.startswith("/home/nvme01/"):
                h20_only[field] += 1
            if not Path(value).exists():
                missing[field] += 1
    return {"checked": dict(checked), "missing": dict(missing), "h20_only": dict(h20_only)}


def selected_consistency(rows: list[dict[str, Any]], manifest_name: str) -> dict[str, Any]:
    out = Counter()
    for row in rows:
        final_path = str(row.get("final_loser_video_path", "") or "")
        raw_path = str(row.get("raw_loser_video_path", "") or "")
        comp_path = str(row.get("comp_loser_video_path", "") or "")
        if "nocomp" in manifest_name:
            if final_path != raw_path:
                out["final_not_raw"] += 1
        elif "comp" in manifest_name:
            if final_path != comp_path:
                out["final_not_comp"] += 1
    return dict(out)


def sample_audit(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    if len(rows) > args.sample_size:
        sampled = rng.sample(rows, args.sample_size)
    else:
        sampled = list(rows)

    frame_issues: list[str] = []
    size_counter = Counter()
    frame_counter = Counter()
    outside = []
    for row in sampled:
        sid = row.get("sample_id", "<unknown>")
        mask_id = row.get("mask_id", "<unknown>")
        for field in PATH_FIELDS:
            info = frame_dir_info(str(row.get(field, "") or ""))
            if info["frames"]:
                frame_counter[(field, info["frames"])] += 1
            if info["size"]:
                size_counter[(field, info["size"])] += 1
            if info["error"] or info["frames"] != args.canonical_num_frames or info["size"] not in {"512x320", "320x512"}:
                frame_issues.append(f"{sid}/{mask_id} {field}: {info}")
        diff = comp_outside_diff(row, args.canonical_num_frames)
        outside.append(diff)
        if diff.get("error") or diff.get("max_abs") not in (0, None):
            frame_issues.append(f"{sid}/{mask_id} comp outside diff: {diff}")
    outside_ok = sum(1 for x in outside if x.get("ok"))
    max_abs_values = [x["max_abs"] for x in outside if x.get("max_abs") is not None]
    mean_abs_values = [x["mean_abs"] for x in outside if x.get("mean_abs") is not None]
    return {
        "sampled": len(sampled),
        "frame_counter": {f"{k[0]}:{k[1]}": v for k, v in frame_counter.items()},
        "size_counter": {f"{k[0]}:{k[1]}": v for k, v in size_counter.items()},
        "outside_ok": outside_ok,
        "outside_max_abs_max": max(max_abs_values) if max_abs_values else None,
        "outside_mean_abs_max": max(mean_abs_values) if mean_abs_values else None,
        "issues": frame_issues[:200],
    }


def write_report(
    path: Path,
    output_root: Path,
    manifest_counts: dict[str, int | str],
    errors: dict[str, list[str]],
    counters: dict[str, dict[str, int]],
    path_checks: dict[str, Any],
    consistency: dict[str, Any],
    sample: dict[str, Any],
    stack: str,
    prior: str,
    evidence: list[str],
    repaired_files: list[Path],
) -> None:
    lines = [
        "# D2 Post-Generation Audit",
        "",
        f"- output_root: `{output_root}`",
        f"- repaired_at: generated by `tools/d2_post_generation_audit_and_repair.py`",
        f"- diffueraser_inference_stack: `{stack}`",
        f"- diffueraser_prior_mode: `{prior}`",
        "",
        "## Manifest Counts",
        "",
    ]
    for name, count in manifest_counts.items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Metadata Counters", ""])
    for name, counter in counters.items():
        lines.append(f"- {name}: `{counter}`")
    lines.extend(["", "## Stack/Prior Evidence", ""])
    for item in evidence:
        lines.append(f"- {item}")
    lines.extend(["", "## Path Audit", ""])
    lines.append("```json")
    lines.append(json.dumps(path_checks, ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    lines.extend(["", "## Selected Manifest Consistency", ""])
    lines.append("```json")
    lines.append(json.dumps(consistency, ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    lines.extend(["", "## Random Sample Decode And Comp Audit", ""])
    lines.append("```json")
    lines.append(json.dumps(sample, ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    lines.extend(["", "## JSONL Parse Errors", ""])
    if any(errors.values()):
        for name, items in errors.items():
            for item in items[:50]:
                lines.append(f"- {name}: {item}")
    else:
        lines.append("- none")
    lines.extend(["", "## Repaired Manifests", ""])
    for repaired in repaired_files:
        lines.append(f"- `{repaired}`")
    lines.extend([
        "",
        "## Training Entry Mapping",
        "",
        "- Experiment 5: `manifests/selected_primary_comp.repaired.jsonl`",
        "- Experiment 6: `manifests/selected_primary_nocomp.repaired.jsonl`",
        "- Experiment 7: `manifests/selected_primary_comp.repaired.jsonl` plus `mask_path`, `M_train = M_gen`, full-video DPO loss",
        "- Experiment 8: `manifests/selected_primary_comp.repaired.jsonl` plus `mask_path`, region-weighted loss",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_root",
        default="/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4",
    )
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--canonical_num_frames", type=int, default=16)
    parser.add_argument("--canonical_height", type=int, default=320)
    parser.add_argument("--canonical_width", type=int, default=512)
    parser.add_argument("--data_asset", default="D2_videodpo_partialmask_k4_diffueraser_only")
    parser.add_argument("--diffueraser_inference_stack", default="auto", choices=["auto", "or", "br", "unconfirmed"])
    parser.add_argument("--diffueraser_prior_mode", default="auto", choices=["auto", "propainter", "noise", "none", "unconfirmed"])
    parser.add_argument("--report", default="")
    parser.add_argument("--no_repair", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    manifests_dir = output_root / "manifests"
    report_path = Path(args.report).expanduser() if args.report else output_root / "reports" / "d2_post_generation_audit.md"
    if not manifests_dir.exists():
        raise SystemExit(f"[error] manifests dir not found: {manifests_dir}")

    manifest_rows: dict[str, list[dict[str, Any]]] = {}
    errors: dict[str, list[str]] = {}
    manifest_counts: dict[str, int | str] = {}
    for name in MANIFEST_NAMES:
        rows, row_errors = read_jsonl(manifests_dir / name)
        manifest_rows[name] = rows
        errors[name] = row_errors
        manifest_counts[name] = len(rows) if not row_errors or rows else "MISSING"

    candidates = manifest_rows.get("candidates_all.jsonl", [])
    counters = {
        "status": dict(Counter(r.get("status") for r in candidates)),
        "generation_model": dict(Counter(r.get("generation_model") for r in candidates)),
        "generation_source_before": dict(Counter(r.get("generation_source") for r in candidates)),
        "diffueraser_stack_before": dict(Counter(r.get("diffueraser_inference_stack") for r in candidates)),
        "diffueraser_prior_before": dict(Counter(r.get("diffueraser_prior_mode") for r in candidates)),
    }
    stack, prior, evidence = confirm_stack_prior(
        output_root,
        args.diffueraser_inference_stack,
        args.diffueraser_prior_mode,
    )

    repaired_files: list[Path] = []
    if not args.no_repair:
        for name in REPAIR_MANIFEST_NAMES:
            repaired_rows = [repair_row(row, name, stack, prior, args) for row in manifest_rows.get(name, [])]
            repaired_name = name.replace(".jsonl", ".repaired.jsonl")
            repaired_path = manifests_dir / repaired_name
            write_jsonl(repaired_path, repaired_rows)
            repaired_files.append(repaired_path)
            manifest_rows[repaired_name] = repaired_rows

    selected_comp = manifest_rows.get("selected_primary_comp.repaired.jsonl") or manifest_rows.get("selected_primary_comp.jsonl", [])
    selected_nocomp = manifest_rows.get("selected_primary_nocomp.repaired.jsonl") or manifest_rows.get("selected_primary_nocomp.jsonl", [])
    path_checks = {
        "selected_primary_comp": path_audit(selected_comp, PATH_FIELDS),
        "selected_primary_nocomp": path_audit(selected_nocomp, ["final_loser_video_path", "raw_loser_video_path", "mask_path", "win_video_path"]),
    }
    consistency = {
        "selected_primary_comp": selected_consistency(selected_comp, "selected_primary_comp.jsonl"),
        "selected_primary_nocomp": selected_consistency(selected_nocomp, "selected_primary_nocomp.jsonl"),
        "sample_id_unique_selected_primary_comp": len({r.get("sample_id") for r in selected_comp}) == len(selected_comp),
        "pair_index_count_selected_primary_comp": len({r.get("pair_index") for r in selected_comp}),
    }
    sample = sample_audit(selected_comp, args) if selected_comp else {"sampled": 0, "issues": ["no selected_primary_comp rows"]}

    write_report(
        report_path,
        output_root,
        manifest_counts,
        errors,
        counters,
        path_checks,
        consistency,
        sample,
        stack,
        prior,
        evidence,
        repaired_files,
    )

    print(f"[d2-audit] output_root={output_root}")
    print(f"[d2-audit] report={report_path}")
    print(f"[d2-audit] stack={stack} prior={prior}")
    for name, count in manifest_counts.items():
        print(f"[d2-audit] {name} {count}")
    for repaired in repaired_files:
        print(f"[d2-audit] repaired={repaired}")
    if sample.get("issues"):
        print(f"[d2-audit] sample_issues={len(sample['issues'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
