#!/usr/bin/env python3
"""Exp26 VideoPainter shadow-dev confirmatory helpers.

This module is deliberately an orchestrator around existing project code. It
does not reimplement core image metrics or VideoPainter inference. It prepares
locked manifests, audits checkpoint/output identity, builds metric pair
manifests for all49 and frame1-48 protocols, and computes paired statistics
from the existing metric wrapper outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = PROJECT_ROOT / "exp26_videopainter_dpo_v2"
DEFAULT_RUN_ROOT = Path(
    "/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/"
    "exp26_videopainter_dpo_v2/shadowdev_confirmatory_20260625"
)
DEFAULT_TRAIN_ROOT = Path(
    "/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/"
    "exp26_videopainter_dpo_v2/vp_primary32_50step_20260625_171032"
)
DEFAULT_STEP0_CKPT = Path(
    "/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/"
    "third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch"
)

LOWER_IS_BETTER = {
    "whole_video_lpips",
    "ewarp_mask_region",
    "temporal_diff_delta_vs_gt",
    "outside_diff_mean",
    "outside_diff_max",
    "outside_region_diff_mean",
    "outside_region_diff_max",
}
PRIMARY_METRICS = [
    "whole_video_psnr",
    "whole_video_ssim",
    "whole_video_lpips",
    "strict_mask_pixel_psnr",
    "boundary_pixel_psnr",
    "boundary_psnr",
    "boundary_ssim",
    "outside_diff_mean",
    "outside_diff_max",
    "ewarp_mask_region",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def sha256_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    files = sorted(p for p in path.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    h = hashlib.sha256()
    samples: list[dict[str, Any]] = []
    for fp in files:
        rel = str(fp.relative_to(path))
        digest = sha256_file(fp)
        h.update(rel.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\0")
        if len(samples) < 12:
            samples.append({"relative_path": rel, "bytes": fp.stat().st_size, "sha256": digest})
    return {
        "exists": True,
        "path": str(path),
        "file_count": len(files),
        "bytes": total,
        "tree_sha256": h.hexdigest(),
        "sample_files": samples,
    }


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()
    except Exception as exc:  # noqa: BLE001 - audit should be best-effort.
        if args == ["branch", "--show-current"] and os.environ.get("EXP26_GIT_BRANCH"):
            return os.environ["EXP26_GIT_BRANCH"]
        if args == ["rev-parse", "HEAD"] and os.environ.get("EXP26_GIT_COMMIT"):
            return os.environ["EXP26_GIT_COMMIT"]
        if args == ["status", "--short"] and os.environ.get("EXP26_RUNTIME_SNAPSHOT"):
            return "runtime_snapshot_no_git"
        return f"ERROR:{exc!r}"


def numeric(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def frame_files(path: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts)


def read_rgb(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def read_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(path)
    if shape and arr.shape[:2] != shape:
        arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (arr > 127).astype(np.uint8)


def ensure_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def markdown_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        values = []
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                if math.isnan(val):
                    values.append("nan")
                else:
                    values.append(f"{val:.6f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def cmd_readback(args: argparse.Namespace) -> int:
    reports = [
        "reports/exp26_gate64_manifest_identity.json",
        "reports/exp26_vp_step0_baseline.md",
        "reports/exp26_vp_l0_l1.md",
        "reports/exp26_vp_10step.md",
        "reports/exp26_vp_50step_final.md",
        "reports/exp26_vp_50step_metrics.csv",
        "reports/exp26_vp_50step_statistics.json",
        "reports/exp26_vp_50step_diagnostics.csv",
        "reports/exp26_vp_50step_visual_review.csv",
    ]
    files_read = [
        "PRD/00_current_status.md",
        "PRD/01_experiment_matrix.md",
        "PRD/48_exp26_videopainter_dpo_v2.md",
        "experiment_registry/exp26_videopainter_dpo_v2/status.md",
        "experiment_registry/exp26_videopainter_dpo_v2/paths.yaml",
        "experiment_registry/exp26_videopainter_dpo_v2/config.yaml",
        "experiment_registry/exp26_videopainter_dpo_v2/metric_summary.md",
        "experiment_registry/exp26_videopainter_dpo_v2/qualitative_summary.md",
        *reports,
        "exp26_videopainter_dpo_v2/code/run_vp2_gate64_official_generation.py",
        "exp26_videopainter_dpo_v2/code/evaluate_vp2_step0_searchdev.py",
        "exp26_videopainter_dpo_v2/code/materialize_vp2_49f_sources.py",
        "exp26_videopainter_dpo_v2/code/generate_vp2_moving_br_masks.py",
        "exp26_videopainter_dpo_v2/code/review_gate64_official_outputs.py",
    ]
    train = EXP_ROOT / "manifests/vp2_gate64_primary32_final.jsonl"
    search = EXP_ROOT / "manifests/vp2_vor_bg_search_dev_32.jsonl"
    shadow = EXP_ROOT / "manifests/vp2_vor_bg_shadow_dev_32.jsonl"
    text = [
        "# Exp26 VideoPainter Shadow-Dev Confirmatory Readback",
        "",
        f"- branch: `{git_value(['branch', '--show-current'])}`",
        f"- HEAD: `{git_value(['rev-parse', 'HEAD'])}`",
        f"- status: `{git_value(['status', '--short']) or 'clean'}`",
        f"- training_manifest_sha256: `{sha256_text(train) if train.exists() else 'missing'}`",
        f"- search_dev_sha256: `{sha256_text(search) if search.exists() else 'missing'}`",
        f"- shadow_dev_sha256: `{sha256_text(shadow) if shadow.exists() else 'missing'}`",
        f"- run_root: `{args.run_root}`",
        f"- train_root: `{args.train_root}`",
        "- primary_comparison: `Step50 vs fixed Step0 official initialization`",
        "- secondary_trajectory: `Step10 and Step30 explanatory only`",
        "- banned_repeats: no training, no data reselection, no search-dev retuning, no left CLI modification",
        "- allowed_right_gpus: dynamic eligible subset of GPU0/GPU5/GPU6/GPU7 only",
        "",
        "## Files Read",
        "",
    ]
    for rel in files_read:
        path = PROJECT_ROOT / rel
        text.append(f"- `{rel}`: {'exists' if path.exists() else 'missing'}")
    text.extend(
        [
            "",
            "## Step Identity Targets",
            "",
            f"- Step0: `{args.step0_checkpoint}`",
            f"- Step10: `{args.train_root / 'checkpoint-10'}`",
            f"- Step30: `{args.train_root / 'checkpoint-30'}`",
            f"- Step50: `{args.train_root / 'checkpoint-50'}`",
            "",
            "Shadow-dev has not been used for checkpoint selection by this right-side workflow.",
        ]
    )
    out = PROJECT_ROOT / "reports/exp26_vp_shadowdev_readback.md"
    out.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(out)
    return 0


def cmd_integrity(args: argparse.Namespace) -> int:
    train = EXP_ROOT / "manifests/vp2_gate64_primary32_final.jsonl"
    search = EXP_ROOT / "manifests/vp2_vor_bg_search_dev_32.jsonl"
    shadow = EXP_ROOT / "manifests/vp2_vor_bg_shadow_dev_32.jsonl"
    train_rows = read_jsonl(train)
    search_rows = read_jsonl(search)
    shadow_rows = read_jsonl(shadow)
    train_groups = {str(r.get("scene_group") or r.get("source_sample_id") or r.get("sample_id")) for r in train_rows}
    search_groups = {str(r.get("scene_group") or r.get("source_sample_id") or r.get("sample_id")) for r in search_rows}
    shadow_groups = {str(r.get("scene_group") or r.get("source_sample_id") or r.get("sample_id")) for r in shadow_rows}
    rows: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    for row in shadow_rows:
        sid = str(row.get("sample_id"))
        group = str(row.get("scene_group") or row.get("source_sample_id") or sid)
        rows.append(
            {
                "sample_id": sid,
                "scene_group": group,
                "source_dataset": row.get("source_dataset", ""),
                "winner_member_path": row.get("winner_member_path", ""),
                "formal_49f": row.get("formal_49f", ""),
                "status": row.get("status", ""),
                "duplicate_sample_id": sid in seen_samples,
                "train_overlap": group in train_groups,
                "search_overlap": group in search_groups,
                "shadow_duplicate_group": list(shadow_groups).count(group) > 1,
            }
        )
        seen_samples.add(sid)
    summary = {
        "rows": len(shadow_rows),
        "scene_groups": len(shadow_groups),
        "shadow_sha256": sha256_text(shadow) if shadow.exists() else "",
        "train_overlap": len(shadow_groups & train_groups),
        "search_overlap": len(shadow_groups & search_groups),
        "duplicate_sample_ids": sum(1 for r in rows if r["duplicate_sample_id"]),
        "source_paths_exist_before_extraction": "member-path-only",
        "status": "SHADOW_DEV_INTEGRITY_PASS_PENDING_MATERIALIZATION"
        if len(shadow_rows) == 32 and len(shadow_groups) == 32 and not (shadow_groups & train_groups) and not (shadow_groups & search_groups)
        else "SHADOW_DEV_INTEGRITY_FAILED",
    }
    report_md = PROJECT_ROOT / "reports/exp26_vp_shadowdev_integrity_audit.md"
    report_csv = PROJECT_ROOT / "reports/exp26_vp_shadowdev_integrity_audit.csv"
    report_json = PROJECT_ROOT / "reports/exp26_vp_shadowdev_integrity.json"
    write_csv(report_csv, rows)
    report_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_md.write_text(
        "\n".join(
            [
                "# Exp26 Shadow-Dev Integrity Audit",
                "",
                f"- status: `{summary['status']}`",
                f"- rows: {summary['rows']}",
                f"- scene_groups: {summary['scene_groups']}",
                f"- train_overlap: {summary['train_overlap']}",
                f"- search_overlap: {summary['search_overlap']}",
                f"- manifest_sha256: `{summary['shadow_sha256']}`",
                "",
                "The locked shadow-dev manifest is source-only before this confirmatory run; 49F materialization and mask manifests must be created before inference.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if "PASS" in summary["status"] else 2


def cmd_preregister(args: argparse.Namespace) -> int:
    run_root = args.run_root
    mask_manifest = run_root / "gate64_mask_ready.jsonl"
    rows = read_jsonl(mask_manifest)
    if len(rows) != 32:
        raise ValueError(f"Expected 32 materialized shadow rows, got {len(rows)} from {mask_manifest}")
    confirm_manifest = EXP_ROOT / "manifests/vp2_shadowdev_confirmatory_32.jsonl"
    confirm_masks = EXP_ROOT / "manifests/vp2_shadowdev_confirmatory_masks.jsonl"
    out_rows = []
    status_rows = []
    for row in rows:
        sid = row["sample_id"]
        frame_dir = Path(row["frame_dir"])
        mask_dir = Path(row["mask_dir"])
        frame_count = len(frame_files(frame_dir))
        mask_count = len(frame_files(mask_dir))
        status = "OK" if frame_count == 49 and mask_count == 49 else "BAD_FRAME_COUNT"
        out = dict(row)
        out.update(
            {
                "shadow_confirmatory_protocol": "pre_registered_before_checkpoint_outputs",
                "inference_seed": args.seed,
                "first_frame_gt": True,
                "raw_comp_definition": "raw=model_output; comp=winner outside mask only",
                "num_inference_steps": 20,
                "guidance_scale": 6.0,
                "dtype": "bf16",
                "resolution": "720x480",
            }
        )
        out_rows.append(out)
        status_rows.append(
            {
                "sample_id": sid,
                "frame_dir": str(frame_dir),
                "mask_dir": str(mask_dir),
                "frame_count": frame_count,
                "mask_count": mask_count,
                "status": status,
                "scene_group": row.get("scene_group", ""),
            }
        )
    write_jsonl(confirm_manifest, out_rows)
    write_jsonl(confirm_masks, out_rows)
    summary = {
        "status": "PREREGISTERED" if all(r["status"] == "OK" for r in status_rows) else "FAILED",
        "rows": len(out_rows),
        "seed": args.seed,
        "confirm_manifest": str(confirm_manifest),
        "confirm_manifest_sha256": sha256_text(confirm_manifest),
        "mask_manifest": str(confirm_masks),
        "mask_manifest_sha256": sha256_text(confirm_masks),
        "runtime_mask_manifest": str(mask_manifest),
    }
    write_csv(PROJECT_ROOT / "reports/exp26_vp_shadowdev_preregistration_status.csv", status_rows)
    (PROJECT_ROOT / "reports/exp26_vp_shadowdev_preregistration.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (PROJECT_ROOT / "reports/exp26_vp_shadowdev_preregistration.md").write_text(
        "\n".join(
            [
                "# Exp26 Shadow-Dev Preregistration",
                "",
                f"- status: `{summary['status']}`",
                f"- rows: {summary['rows']}",
                f"- seed: `{summary['seed']}`",
                f"- confirm_manifest_sha256: `{summary['confirm_manifest_sha256']}`",
                f"- mask_manifest_sha256: `{summary['mask_manifest_sha256']}`",
                "- primary metrics: frame1-48 strict mask PSNR, boundary PSNR, mask LPIPS, Ewarp",
                "- Step10/Step30: trajectory only; no checkpoint reselection allowed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PREREGISTERED" else 2


def cmd_checkpoint_identity(args: argparse.Namespace) -> int:
    ckpts = {
        "step0": args.step0_checkpoint,
        "step10": args.train_root / "checkpoint-10",
        "step30": args.train_root / "checkpoint-30",
        "step50": args.train_root / "checkpoint-50",
    }
    rows: list[dict[str, Any]] = []
    base_tree = sha256_summary(args.step0_checkpoint).get("tree_sha256", "")
    for step, path in ckpts.items():
        summary = sha256_summary(path)
        model = path / "branch/diffusion_pytorch_model.safetensors"
        if not model.exists():
            model = path / "diffusion_pytorch_model.safetensors"
        trainer_state = path / "trainer_state.json"
        train_step = ""
        if trainer_state.exists():
            try:
                train_step = json.loads(trainer_state.read_text(encoding="utf-8")).get("global_step", "")
            except Exception:
                train_step = "unreadable"
        rows.append(
            {
                "step": step,
                "path": str(path),
                "exists": summary.get("exists"),
                "file_count": summary.get("file_count", 0),
                "bytes": summary.get("bytes", 0),
                "tree_sha256": summary.get("tree_sha256", ""),
                "model_file": str(model),
                "model_sha256": sha256_file(model) if model.exists() else "",
                "trainer_state": str(trainer_state),
                "trainer_global_step": train_step,
                "delta_vs_step0_tree": bool(step != "step0" and summary.get("tree_sha256") != base_tree),
                "fallback": False,
                "strict_load_result": "PENDING_RUNTIME_PREFLIGHT",
                "missing_keys": "",
                "unexpected_keys": "",
            }
        )
    if not any(bool(r["exists"]) for r in rows):
        status = "CHECKPOINT_IDENTITY_PENDING_PAI_PATHS"
    elif all(r["exists"] for r in rows) and rows[-1]["delta_vs_step0_tree"]:
        status = "CHECKPOINT_IDENTITY_PASS_PENDING_STRICT_LOAD"
    else:
        status = "CHECKPOINT_IDENTITY_FAILED"
    write_csv(PROJECT_ROOT / "reports/exp26_vp_shadowdev_checkpoint_identity.csv", rows)
    (PROJECT_ROOT / "reports/exp26_vp_shadowdev_checkpoint_identity.json").write_text(
        json.dumps({"status": status, "rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (PROJECT_ROOT / "reports/exp26_vp_shadowdev_checkpoint_identity.md").write_text(
        "\n".join(
            [
                "# Exp26 Shadow-Dev Checkpoint Identity",
                "",
                f"- status: `{status}`",
                f"- train_root: `{args.train_root}`",
                f"- step0_checkpoint: `{args.step0_checkpoint}`",
                "",
                markdown_table(rows, ["step", "exists", "file_count", "bytes", "trainer_global_step", "delta_vs_step0_tree", "strict_load_result"]),
            ]
        ),
        encoding="utf-8",
    )
    print(status)
    return 0 if "FAILED" not in status else 2


def make_no_first_dir(src: Path, dst: Path) -> Path:
    files = frame_files(src)
    if len(files) < 49:
        raise ValueError(f"{src} has {len(files)} frames, expected >=49")
    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.iterdir():
        if old.is_file() or old.is_symlink():
            old.unlink()
    for out_idx, fp in enumerate(files[1:49]):
        ensure_link_or_copy(fp, dst / f"{out_idx:05d}{fp.suffix.lower()}")
    return dst


def build_pair_manifests(args: argparse.Namespace, frame_range: str) -> dict[str, Path]:
    run_root = args.run_root
    mask_manifest = run_root / "gate64_mask_ready.jsonl"
    rows = read_jsonl(mask_manifest)
    if not rows:
        raise ValueError(f"Expected non-empty rows in {mask_manifest}, got 0")
    pair_root = run_root / "metric_pair_manifests" / frame_range
    derived_root = run_root / "derived_no_first_frame" if frame_range == "no_first_frame" else None
    outputs: dict[str, Path] = {}
    steps = [x.strip() for x in args.metric_steps.split(",") if x.strip()]
    variants = [x.strip() for x in args.metric_variants.split(",") if x.strip()]
    for step in steps:
        for variant in variants:
            pair_rows: list[dict[str, Any]] = []
            for row in rows:
                sid = row["sample_id"]
                gt = Path(row["frame_dir"])
                mask = Path(row["mask_dir"])
                pred = run_root / step / "official_generation" / f"{variant}_frames" / sid
                if frame_range == "no_first_frame":
                    assert derived_root is not None
                    gt = make_no_first_dir(Path(row["frame_dir"]), derived_root / "gt" / sid)
                    mask = make_no_first_dir(Path(row["mask_dir"]), derived_root / "mask" / sid)
                    pred = make_no_first_dir(pred, derived_root / step / variant / sid)
                pair_rows.append(
                    {
                        "sample_id": sid,
                        "model_label": f"{step}_{variant}_{frame_range}",
                        "gt_video_path": str(gt),
                        "prediction_video_path": str(pred),
                        "mask_path": str(mask),
                        "scene_group": row.get("scene_group", ""),
                        "source_dataset": row.get("source_dataset", ""),
                    }
                )
            out = pair_root / f"{step}_{variant}_pairs.jsonl"
            write_jsonl(out, pair_rows)
            outputs[f"{step}_{variant}"] = out
    return outputs


def run_metric_eval(args: argparse.Namespace, pair_manifest: Path, output_dir: Path, max_frames: int) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools/run_inpainting_metric_eval.py"),
        "--pair_manifest",
        str(pair_manifest),
        "--output_dir",
        str(output_dir),
        "--max_frames",
        str(max_frames),
        "--width",
        "720",
        "--height",
        "480",
        "--boundary_pixels",
        "4",
        "--device",
        args.device,
        "--compute_lpips",
        "--compute_ewarp",
        "--strict_missing",
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def cmd_metrics(args: argparse.Namespace) -> int:
    frame_ranges = [x.strip() for x in args.metric_ranges.split(",") if x.strip()]
    max_frames_by_range = {"all49": 49, "no_first_frame": 48}
    for frame_range in frame_ranges:
        max_frames = max_frames_by_range[frame_range]
        pairs = build_pair_manifests(args, frame_range)
        for key, pair_manifest in pairs.items():
            output_dir = args.run_root / "metrics" / frame_range / key
            summary = output_dir / "metrics/summary.csv"
            if args.skip_existing and summary.exists():
                continue
            run_metric_eval(args, pair_manifest, output_dir, max_frames=max_frames)
    print(args.run_root / "metrics")
    return 0


def per_sample_metrics(run_root: Path, frame_range: str, step: str, variant: str) -> list[dict[str, str]]:
    return read_csv(run_root / "metrics" / frame_range / f"{step}_{variant}" / "metrics/per_sample_metrics.csv")


def summary_metrics(run_root: Path, frame_range: str, step: str, variant: str) -> dict[str, str]:
    rows = read_csv(run_root / "metrics" / frame_range / f"{step}_{variant}" / "metrics/summary.csv")
    return rows[0] if rows else {}


def paired_stats(base_rows: list[dict[str, str]], cand_rows: list[dict[str, str]], metric: str, seed: int = 20260625) -> dict[str, Any]:
    base = {r["sample_id"]: numeric(r.get(metric)) for r in base_rows if r.get("status") == "ok"}
    cand = {r["sample_id"]: numeric(r.get(metric)) for r in cand_rows if r.get("status") == "ok"}
    ids = sorted(set(base) & set(cand))
    deltas = np.array([cand[i] - base[i] for i in ids], dtype=np.float64)
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return {"metric": metric, "n": 0}
    rng = np.random.default_rng(seed)
    boot = np.array([float(rng.choice(deltas, size=deltas.size, replace=True).mean()) for _ in range(10000)], dtype=np.float64)
    higher_good = metric not in LOWER_IS_BETTER
    wins = deltas > 0 if higher_good else deltas < 0
    prob = float((boot > 0).mean()) if higher_good else float((boot < 0).mean())
    loo = [float(np.delete(deltas, i).mean()) for i in range(deltas.size)] if deltas.size > 1 else []
    return {
        "metric": metric,
        "n": int(deltas.size),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "min_delta": float(deltas.min()),
        "max_delta": float(deltas.max()),
        "win_rate": float(wins.mean()),
        "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
        "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
        "probability_improved": prob,
        "leave_one_out_min": float(min(loo)) if loo else "",
        "leave_one_out_max": float(max(loo)) if loo else "",
    }


def first_existing(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        value = numeric(row.get(f"{key}_mean", row.get(key, "")))
        if math.isfinite(value):
            return value
    return float("nan")


def cmd_stats(args: argparse.Namespace) -> int:
    frame_ranges = [x.strip() for x in args.metric_ranges.split(",") if x.strip()]
    steps = [x.strip() for x in args.metric_steps.split(",") if x.strip()]
    variants = [x.strip() for x in args.metric_variants.split(",") if x.strip()]
    aggregate_rows: list[dict[str, Any]] = []
    for frame_range in frame_ranges:
        for step in steps:
            for variant in variants:
                row = summary_metrics(args.run_root, frame_range, step, variant)
                aggregate_rows.append({"frame_range": frame_range, "step": step, "variant": variant, **row})
    write_csv(PROJECT_ROOT / "reports/exp26_vp_shadowdev_aggregate_metrics.csv", aggregate_rows)
    paired_rows: list[dict[str, Any]] = []
    for frame_range in frame_ranges:
        for variant in variants:
            base_rows = per_sample_metrics(args.run_root, frame_range, "step0", variant)
            for step in steps:
                if step == "step0":
                    continue
                cand_rows = per_sample_metrics(args.run_root, frame_range, step, variant)
                for metric in PRIMARY_METRICS:
                    stat = paired_stats(base_rows, cand_rows, metric)
                    paired_rows.append({"frame_range": frame_range, "variant": variant, "comparison": f"{step}-step0", **stat})
    write_csv(PROJECT_ROOT / "reports/exp26_vp_shadowdev_paired_deltas.csv", paired_rows)
    primary = {
        row["metric"]: row
        for row in paired_rows
        if row["frame_range"] == "no_first_frame" and row["variant"] == "comp" and row["comparison"] == "step50-step0"
    }
    step0_comp = next((r for r in aggregate_rows if r["frame_range"] == "no_first_frame" and r["step"] == "step0" and r["variant"] == "comp"), {})
    step50_comp = next((r for r in aggregate_rows if r["frame_range"] == "no_first_frame" and r["step"] == "step50" and r["variant"] == "comp"), {})
    strict = primary.get("strict_mask_pixel_psnr", {})
    boundary = primary.get("boundary_pixel_psnr", {})
    lpips = primary.get("whole_video_lpips", {})
    ewarp = primary.get("ewarp_mask_region", {})
    whole_psnr_delta = first_existing(step50_comp, "whole_video_psnr") - first_existing(step0_comp, "whole_video_psnr")
    reasons: list[str] = []
    if not (strict.get("mean_delta", -999) > 0 and strict.get("probability_improved", 0) >= 0.90 and strict.get("win_rate", 0) >= 0.55):
        reasons.append("strict mask PSNR primary gate failed")
    if not (boundary.get("mean_delta", -999) > 0):
        reasons.append("boundary PSNR primary gate failed")
    if whole_psnr_delta < -0.02:
        reasons.append(f"whole comp PSNR dropped by {whole_psnr_delta:+.6f}")
    if lpips.get("mean_delta", 999) > 0.0005:
        reasons.append("LPIPS worsened beyond tolerance")
    if ewarp.get("mean_delta", 999) > 0.03:
        reasons.append("Ewarp worsened beyond tolerance")
    status = "VIDEOPAINTER_SHADOWDEV_METRIC_GATE_PASSED_PENDING_VISUAL_AND_SEED" if not reasons else "VIDEOPAINTER_SHADOWDEV_METRIC_GATE_FAILED"
    summary = {
        "status": status,
        "reasons": reasons,
        "primary_strict_mask_psnr": strict,
        "primary_boundary_psnr": boundary,
        "primary_lpips": lpips,
        "primary_ewarp": ewarp,
        "whole_comp_psnr_delta_no_first_frame": whole_psnr_delta,
    }
    (PROJECT_ROOT / "reports/exp26_vp_shadowdev_statistics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (PROJECT_ROOT / "reports/exp26_vp_shadowdev_metrics_and_statistics.md").write_text(
        "\n".join(
            [
                "# Exp26 VideoPainter Shadow-Dev Metrics And Statistics",
                "",
                f"- status: `{status}`",
                f"- whole_comp_psnr_delta_no_first_frame: {whole_psnr_delta:+.6f}",
                "",
                "## Primary Step50 - Step0 Comp Frame1-48",
                "",
                markdown_table(
                    [
                        {"metric": key, **val}
                        for key, val in primary.items()
                        if key in {"strict_mask_pixel_psnr", "boundary_pixel_psnr", "whole_video_lpips", "ewarp_mask_region"}
                    ],
                    ["metric", "n", "mean_delta", "median_delta", "win_rate", "bootstrap_ci_low", "bootstrap_ci_high", "probability_improved", "leave_one_out_min", "leave_one_out_max"],
                ),
                "",
                "Step10/Step30 are trajectory diagnostics only and are not used for checkpoint reselection.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def equality_fraction(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    eq = np.all(a == b, axis=2)
    if mask is not None:
        if mask.shape != eq.shape:
            mask = cv2.resize(mask, (eq.shape[1], eq.shape[0]), interpolation=cv2.INTER_NEAREST)
        sel = mask > 0
        if not np.any(sel):
            return float("nan")
        eq = eq[sel]
    return float(eq.mean())


def mae_region(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2)
    if mask is not None:
        if mask.shape != diff.shape:
            mask = cv2.resize(mask, (diff.shape[1], diff.shape[0]), interpolation=cv2.INTER_NEAREST)
        sel = mask > 0
        if not np.any(sel):
            return float("nan")
        diff = diff[sel]
    return float(np.mean(diff))


def cmd_leakage(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.run_root / "gate64_mask_ready.jsonl")
    out_rows: list[dict[str, Any]] = []
    for step in ("step0", "step10", "step30", "step50"):
        for row in rows:
            sid = row["sample_id"]
            gt_files = frame_files(Path(row["frame_dir"]))
            mask_files = frame_files(Path(row["mask_dir"]))
            raw_files = frame_files(args.run_root / step / "official_generation/raw_frames" / sid)
            comp_files = frame_files(args.run_root / step / "official_generation/comp_frames" / sid)
            n = min(len(gt_files), len(mask_files), len(raw_files), len(comp_files), 49)
            if n < 49:
                out_rows.append({"step": step, "sample_id": sid, "status": "MISSING_FRAMES", "frame_count": n})
                continue
            whole_eq_winner = []
            mask_eq_winner = []
            outside_eq_winner = []
            mask_mae_winner = []
            outside_mae_winner = []
            comp_mask_eq_winner = []
            raw_outside_eq_winner = []
            frame0_raw_eq = ""
            frames1_48_raw_eqs = []
            for i in range(49):
                gt = read_rgb(gt_files[i])
                mask = read_mask(mask_files[i], shape=gt.shape[:2])
                raw = read_rgb(raw_files[i])
                comp = read_rgb(comp_files[i])
                inv_mask = (mask <= 0).astype(np.uint8)
                whole_eq_winner.append(equality_fraction(raw, gt))
                mask_eq_winner.append(equality_fraction(raw, gt, mask))
                outside_eq_winner.append(equality_fraction(raw, gt, inv_mask))
                raw_outside_eq_winner.append(equality_fraction(raw, gt, inv_mask))
                comp_mask_eq_winner.append(equality_fraction(comp, gt, mask))
                mask_mae_winner.append(mae_region(raw, gt, mask))
                outside_mae_winner.append(mae_region(raw, gt, inv_mask))
                if i == 0:
                    frame0_raw_eq = equality_fraction(raw, gt)
                else:
                    frames1_48_raw_eqs.append(equality_fraction(raw, gt))
            out_rows.append(
                {
                    "step": step,
                    "sample_id": sid,
                    "status": "OK",
                    "frame_count": n,
                    "raw_whole_eq_winner_mean": float(np.nanmean(whole_eq_winner)),
                    "raw_mask_eq_winner_mean": float(np.nanmean(mask_eq_winner)),
                    "raw_outside_eq_winner_mean": float(np.nanmean(outside_eq_winner)),
                    "comp_mask_eq_winner_mean": float(np.nanmean(comp_mask_eq_winner)),
                    "raw_mask_mae_to_winner": float(np.nanmean(mask_mae_winner)),
                    "raw_outside_mae_to_winner": float(np.nanmean(outside_mae_winner)),
                    "frame0_raw_eq_winner": frame0_raw_eq,
                    "frames1_48_raw_eq_winner_mean": float(np.nanmean(frames1_48_raw_eqs)),
                    "expected_comp_outside_winner_copy": True,
                    "non_expected_gt_leakage_flag": bool(float(np.nanmean(frames1_48_raw_eqs)) > 0.999),
                }
            )
    write_csv(PROJECT_ROOT / "reports/exp26_vp_shadowdev_leakage_audit.csv", out_rows)
    flagged = [r for r in out_rows if r.get("non_expected_gt_leakage_flag")]
    status = "VALIDATION_BLOCKED_GT_LEAKAGE" if flagged else "NO_UNEXPECTED_WINNER_LEAKAGE_DETECTED"
    (PROJECT_ROOT / "reports/exp26_vp_shadowdev_leakage_audit.md").write_text(
        "\n".join(
            [
                "# Exp26 Shadow-Dev Leakage Audit",
                "",
                f"- status: `{status}`",
                f"- rows: {len(out_rows)}",
                f"- flagged: {len(flagged)}",
                "",
                "Comp is expected to copy winner outside mask. Raw frames and comp mask region are checked separately.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(status)
    return 0 if not flagged else 2


def cmd_dynamics(args: argparse.Namespace) -> int:
    diag = args.train_root / "dpo_diagnostics.csv"
    rows = read_csv(diag)
    if not rows:
        diag = args.train_root / "dpo_diagnostics" / "dpo_diagnostics.csv"
        rows = read_csv(diag)
    numeric_cols: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            val = numeric(value)
            if math.isfinite(val):
                numeric_cols.setdefault(key, []).append(val)
    summary: dict[str, Any] = {"diagnostics_path": str(diag), "rows": len(rows)}
    for key in ["grad_norm", "loss", "dpo_loss", "implicit_acc", "winner_improvement_mean", "loser_degradation_mean", "loser_dominant_ratio"]:
        vals = numeric_cols.get(key, [])
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            summary[f"{key}_max"] = float(np.nanmax(arr))
            summary[f"{key}_p95"] = float(np.nanquantile(arr, 0.95))
            summary[f"{key}_mean"] = float(np.nanmean(arr))
            summary[f"{key}_last"] = float(arr[-1])
    grad = numeric_cols.get("grad_norm", [])
    summary["grad_gt_10_count"] = int(sum(v > 10 for v in grad))
    summary["grad_gt_50_count"] = int(sum(v > 50 for v in grad))
    summary["grad_gt_100_count"] = int(sum(v > 100 for v in grad))
    summary["continue_100step_recommendation"] = "NO_100STEP_BY_PROTOCOL"
    (PROJECT_ROOT / "reports/exp26_vp_50step_dynamics_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(PROJECT_ROOT / "reports/exp26_vp_50step_dynamics_audit.csv", [summary])
    (PROJECT_ROOT / "reports/exp26_vp_50step_dynamics_audit.md").write_text(
        "\n".join(
            [
                "# Exp26 VideoPainter 50-Step Dynamics Audit",
                "",
                f"- diagnostics: `{diag}`",
                f"- rows: {summary['rows']}",
                f"- max_grad_norm: {summary.get('grad_norm_max', 'NA')}",
                f"- p95_grad_norm: {summary.get('grad_norm_p95', 'NA')}",
                "- conclusion: no 100-step continuation is authorized by this confirmatory protocol.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _candidate_path(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if str(path) and str(path) != "." and path.exists():
            return path
    return None


def _load_frames_for_tc_vfid(path: Path, frame_range: str) -> list[np.ndarray]:
    files = frame_files(path)
    if frame_range == "no_first_frame":
        files = files[1:49]
    else:
        files = files[:49]
    return [read_rgb(fp) for fp in files]


def _compute_tc_batched(tc_model: Any, frames_u8_rgb: list[np.ndarray], batch_size: int = 16) -> float:
    import torch
    import torch.nn.functional as F

    if len(frames_u8_rgb) < 2:
        return 1.0
    feats = []
    for idx in range(0, len(frames_u8_rgb), batch_size):
        batch = torch.stack(
            [tc_model.preprocess(Image.fromarray(frame)) for frame in frames_u8_rgb[idx : idx + batch_size]],
            dim=0,
        ).to(tc_model.device)
        with torch.no_grad():
            feat = tc_model.model.encode_image(batch)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        feats.append(feat.detach().cpu())
    features = torch.cat(feats, dim=0)
    sims = F.cosine_similarity(features[:-1], features[1:], dim=-1)
    return float(sims.mean().item())


def cmd_tc_vfid(args: argparse.Namespace) -> int:
    from inference import metrics as metric_backend

    device = args.device
    i3d_path = _candidate_path(
        [
            Path(os.environ["I3D_MODEL_PATH"]) if os.environ.get("I3D_MODEL_PATH") else Path(""),
            PROJECT_ROOT / "weights/i3d_rgb_imagenet.pt",
            Path("/home/hj/Video_inpainting_DPO/weights/i3d_rgb_imagenet.pt"),
            Path("/mnt/nas/hj/H20_Video_inpainting_DPO/weights/i3d_rgb_imagenet.pt"),
            Path("/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/weights/i3d_rgb_imagenet.pt"),
        ]
    )
    clip_dir = _candidate_path(
        [
            Path(os.environ["OPENCLIP_MODEL_DIR"]) if os.environ.get("OPENCLIP_MODEL_DIR") else Path(""),
            PROJECT_ROOT / "weights/open_clip_vit_h14",
            Path("/home/hj/.tmp/open_clip_vit_h14"),
            Path("/mnt/nas/hj/.tmp/open_clip_vit_h14"),
            Path("/mnt/workspace/hj/.tmp/open_clip_vit_h14"),
        ]
    )
    if i3d_path is None:
        raise FileNotFoundError("No i3d_rgb_imagenet.pt found for existing VFID backend")
    tc_model = metric_backend.TemporalConsistencyMetric(device=device, model_path=str(clip_dir) if clip_dir else None)
    i3d_model = metric_backend.init_i3d_model(str(i3d_path), device=device)
    rows = read_jsonl(args.run_root / "gate64_mask_ready.jsonl")
    per_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    ranges = tuple(x.strip() for x in args.tc_vfid_ranges.split(",") if x.strip())
    steps = tuple(x.strip() for x in args.tc_vfid_steps.split(",") if x.strip())
    variants = tuple(x.strip() for x in args.tc_vfid_variants.split(",") if x.strip())
    for frame_range in ranges:
        for step in steps:
            for variant in variants:
                print(f"[tc-vfid] {frame_range} {step} {variant}", flush=True)
                gt_acts: list[np.ndarray] = []
                pred_acts: list[np.ndarray] = []
                tc_vals: list[float] = []
                for row in rows:
                    sid = row["sample_id"]
                    gt_frames = _load_frames_for_tc_vfid(Path(row["frame_dir"]), frame_range)
                    pred_frames = _load_frames_for_tc_vfid(args.run_root / step / "official_generation" / f"{variant}_frames" / sid, frame_range)
                    if len(gt_frames) != len(pred_frames) or not gt_frames:
                        per_rows.append(
                            {
                                "frame_range": frame_range,
                                "step": step,
                                "variant": variant,
                                "sample_id": sid,
                                "status": "SKIPPED_FRAME_COUNT",
                                "tc": "",
                            }
                        )
                        continue
                    tc = _compute_tc_batched(tc_model, pred_frames)
                    gt_pil = [Image.fromarray(frame) for frame in gt_frames]
                    pred_pil = [Image.fromarray(frame) for frame in pred_frames]
                    gt_act, pred_act = metric_backend.calculate_i3d_activations(gt_pil, pred_pil, i3d_model, device)
                    gt_acts.append(gt_act)
                    pred_acts.append(pred_act)
                    tc_vals.append(tc)
                    per_rows.append(
                        {
                            "frame_range": frame_range,
                            "step": step,
                            "variant": variant,
                            "sample_id": sid,
                            "status": "ok",
                            "tc": tc,
                        }
                    )
                vfid = metric_backend.calculate_vfid(np.vstack(gt_acts), np.vstack(pred_acts)) if gt_acts and pred_acts else float("nan")
                summary_rows.append(
                    {
                        "frame_range": frame_range,
                        "step": step,
                        "variant": variant,
                        "rows": len(tc_vals),
                        "tc_mean": float(np.mean(tc_vals)) if tc_vals else float("nan"),
                        "tc_median": float(np.median(tc_vals)) if tc_vals else float("nan"),
                        "vfid": float(vfid),
                        "i3d_model_path": str(i3d_path),
                        "openclip_model_dir": str(clip_dir) if clip_dir else "auto_or_cache",
                    }
                )
    write_csv(PROJECT_ROOT / "reports/exp26_vp_shadowdev_tc_vfid_per_video.csv", per_rows)
    write_csv(PROJECT_ROOT / "reports/exp26_vp_shadowdev_tc_vfid_summary.csv", summary_rows)
    (PROJECT_ROOT / "reports/exp26_vp_shadowdev_tc_vfid.md").write_text(
        "\n".join(
            [
                "# Exp26 Shadow-Dev TC/VFID",
                "",
                "TC and VFID are computed through the existing `inference.metrics.py` backend.",
                f"- I3D: `{i3d_path}`",
                f"- OpenCLIP: `{clip_dir if clip_dir else 'auto_or_cache'}`",
                "",
                markdown_table(summary_rows, ["frame_range", "step", "variant", "rows", "tc_mean", "tc_median", "vfid"]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(PROJECT_ROOT / "reports/exp26_vp_shadowdev_tc_vfid_summary.csv")
    return 0


def _resize_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == width:
        return img
    scale = width / max(1, w)
    return cv2.resize(img, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def _label_band(width: int, text: str, *, height: int = 48) -> np.ndarray:
    band = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.putText(band, text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (240, 240, 240), 2, cv2.LINE_AA)
    return band


def _read_sheet(path: Path, width: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return np.full((260, width, 3), 32, dtype=np.uint8)
    return _resize_width(img, width)


def _stack_with_label(items: list[tuple[str, np.ndarray]], width: int) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for label, img in items:
        blocks.append(_label_band(width, label))
        blocks.append(img)
    return np.vstack(blocks)


def cmd_visual_review_pack(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.run_root / "gate64_mask_ready.jsonl")
    out_root = args.run_root / "visual_review_step0_vs_step50"
    anon_dir = out_root / "anonymous_pages"
    informed_dir = out_root / "informed_pages"
    sample_dir = out_root / "per_sample"
    for d in (anon_dir, informed_dir, sample_dir):
        d.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260626)
    width = 1280
    page_rows: list[dict[str, Any]] = []
    tiles: list[tuple[str, np.ndarray, np.ndarray]] = []
    for row in rows:
        sid = row["sample_id"]
        step0_ev = args.run_root / "step0/step0_review/evidence_sheets" / f"{sid}.jpg"
        step50_ev = args.run_root / "step50/step50_review/evidence_sheets" / f"{sid}.jpg"
        step0_cr = args.run_root / "step0/step0_review/crop_sheets" / f"{sid}.jpg"
        step50_cr = args.run_root / "step50/step50_review/crop_sheets" / f"{sid}.jpg"
        order = ["step0", "step50"]
        rng.shuffle(order)
        labels = {order[0]: "Method A", order[1]: "Method B"}
        sheets = {
            "step0": np.vstack([_read_sheet(step0_ev, width), _read_sheet(step0_cr, width)]),
            "step50": np.vstack([_read_sheet(step50_ev, width), _read_sheet(step50_cr, width)]),
        }
        anon = _stack_with_label(
            [
                (f"{sid} | {labels['step0'] if order[0] == 'step0' else labels['step50']} | anonymous", sheets[order[0]]),
                (f"{sid} | {labels['step0'] if order[1] == 'step0' else labels['step50']} | anonymous", sheets[order[1]]),
            ],
            width,
        )
        informed = _stack_with_label(
            [
                (f"{sid} | Step0 official baseline", sheets["step0"]),
                (f"{sid} | Step50 trained checkpoint", sheets["step50"]),
            ],
            width,
        )
        anon_path = sample_dir / f"{sid}_anonymous.jpg"
        informed_path = sample_dir / f"{sid}_informed.jpg"
        cv2.imwrite(str(anon_path), anon, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        cv2.imwrite(str(informed_path), informed, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        tiles.append((sid, _resize_width(anon, 920), _resize_width(informed, 920)))
        page_rows.append(
            {
                "sample_id": sid,
                "method_a": order[0],
                "method_b": order[1],
                "anonymous_sheet": str(anon_path),
                "informed_sheet": str(informed_path),
                "reviewer_pass": False,
                "final_class": "VISUAL_REVIEW_PENDING",
            }
        )
    for page_idx in range(0, len(tiles), 4):
        anon_blocks = []
        informed_blocks = []
        for sid, anon, informed in tiles[page_idx : page_idx + 4]:
            anon_blocks.append(_label_band(anon.shape[1], f"Anonymous review page {page_idx // 4 + 1} | {sid}", height=42))
            anon_blocks.append(anon)
            informed_blocks.append(_label_band(informed.shape[1], f"Informed review page {page_idx // 4 + 1} | {sid}", height=42))
            informed_blocks.append(informed)
        cv2.imwrite(str(anon_dir / f"page_{page_idx // 4:02d}.jpg"), np.vstack(anon_blocks), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        cv2.imwrite(str(informed_dir / f"page_{page_idx // 4:02d}.jpg"), np.vstack(informed_blocks), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    write_csv(PROJECT_ROOT / "reports/exp26_vp_shadowdev_visual_review_template.csv", page_rows)
    (PROJECT_ROOT / "reports/exp26_vp_shadowdev_visual_review_pack.md").write_text(
        "\n".join(
            [
                "# Exp26 Shadow-Dev Step0 vs Step50 Visual Review Pack",
                "",
                f"- samples: {len(rows)}",
                f"- anonymous_pages: `{anon_dir}`",
                f"- informed_pages: `{informed_dir}`",
                f"- per_sample: `{sample_dir}`",
                "",
                "This pack is evidence only; final reviewer classifications are recorded after opening the pages.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(out_root)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    p.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    p.add_argument("--step0-checkpoint", type=Path, default=DEFAULT_STEP0_CKPT)
    p.add_argument("--seed", type=int, default=20260619)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tc-vfid-ranges", default="all49,no_first_frame")
    p.add_argument("--tc-vfid-steps", default="step0,step10,step30,step50")
    p.add_argument("--tc-vfid-variants", default="raw,comp")
    p.add_argument("--metric-ranges", default="all49,no_first_frame")
    p.add_argument("--metric-steps", default="step0,step10,step30,step50")
    p.add_argument("--metric-variants", default="raw,comp")
    sub = p.add_subparsers(dest="command", required=True)
    for name in [
        "readback",
        "integrity",
        "preregister",
        "checkpoint-identity",
        "metrics",
        "stats",
        "leakage",
        "dynamics",
        "tc-vfid",
        "visual-review-pack",
    ]:
        sub.add_parser(name)
    args = p.parse_args()
    args.run_root = args.run_root.resolve()
    args.train_root = args.train_root.resolve()
    args.step0_checkpoint = args.step0_checkpoint.resolve()
    return globals()[f"cmd_{args.command.replace('-', '_')}"](args)


if __name__ == "__main__":
    raise SystemExit(main())
