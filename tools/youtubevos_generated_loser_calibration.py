#!/usr/bin/env python3
"""Run a YouTube-VOS partial-mask generated-loser calibration subset.

This mirrors the VideoDPO D2 generated-loser pipeline, but uses clean
YouTube-VOS frame directories as winners. It intentionally writes the same
JSONL manifest family as D2 so downstream data-only and task experiments can
reuse selection, path rewriting, and video inspection utilities.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.pai_videodpo_single_sample_generation_smoke import (
    CanonicalSetting,
    choose_frame_indices,
    diffueraser_inference_options,
    image_files,
    load_env_file,
    load_yaml,
    model_command,
    save_rgb_frames,
)
from tools.videodpo_generated_loser_calibration import (
    MODEL_NAMES,
    MODEL_PROMPT_USAGE,
    clean_dir,
    generation_source_label,
    jsonl_append,
    parse_csv,
    run_candidate,
)
from tools.videodpo_loser_candidate_scoring import compute_candidate_metrics
from tools.videodpo_loser_candidate_selection import select_manifests, write_calibration_report, write_jsonl
from tools.videodpo_mask_policy import generate_k_masks, load_policy


def safe_id(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text[:80] or "sample"


def resolve_youtubevos_root(path: str | Path) -> tuple[Path, Path, Path | None]:
    root = Path(path).expanduser().resolve()
    frames_root = root / "JPEGImages" if (root / "JPEGImages").is_dir() else root
    masks_root = root / "Annotations" if (root / "Annotations").is_dir() else None
    if not frames_root.is_dir():
        raise SystemExit(f"[error] YouTube-VOS frames root not found: {frames_root}")
    return root, frames_root, masks_root


def scan_frame_dirs(frames_root: Path, min_frames: int) -> list[Path]:
    out: list[Path] = []
    for path in sorted(p for p in frames_root.iterdir() if p.is_dir()):
        if len(image_files(path)) >= min_frames:
            out.append(path)
    return out


def load_captions(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"[error] caption_json must be a dict: {p}")
    return {str(k): str(v) for k, v in data.items()}


def prompt_for(video_dir: Path, captions: dict[str, str], prompt_mode: str) -> tuple[str, str]:
    if prompt_mode == "none":
        return "", "no_prompt"
    for key in (video_dir.name, f"ytbv_{video_dir.name}", str(video_dir)):
        value = captions.get(key, "").strip()
        if value:
            return value, "caption_json"
    words = video_dir.name.replace("_", " ").replace("-", " ")
    return f"A realistic video scene from YouTube-VOS showing {words}.", "fallback_video_id"


def read_canonical_frame_dir(
    frame_dir: Path,
    indices: list[int],
    width: int,
    height: int,
) -> list[np.ndarray]:
    files = image_files(frame_dir)
    frames: list[np.ndarray] = []
    for idx in indices:
        path = files[idx]
        frame_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"failed to decode frame: {path}")
        frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return frames


def image_dir_info(frame_dir: Path) -> tuple[int, int, float, int]:
    files = image_files(frame_dir)
    if not files:
        raise RuntimeError(f"no frames under {frame_dir}")
    first = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"failed to decode first frame: {files[0]}")
    h, w = first.shape[:2]
    return w, h, 24.0, len(files)


def build_setting(
    args: argparse.Namespace,
    root: Path,
    video_dir: Path,
    source_index: int,
    prompt: str,
    prompt_source: str,
    indices: list[int],
) -> tuple[CanonicalSetting, list[np.ndarray]]:
    raw_w, raw_h, raw_fps, raw_count = image_dir_info(video_dir)
    frames = read_canonical_frame_dir(video_dir, indices, args.width, args.height)
    setting = CanonicalSetting(
        official_config="youtubevos_partialmask_data",
        train_data_yaml="",
        video_root=str(root),
        pair_index=source_index,
        winner_video_index=source_index,
        loser_video_index=-1,
        winner_video_path=str(video_dir),
        loser_video_path="",
        prompt=prompt,
        raw_width=raw_w,
        raw_height=raw_h,
        raw_fps=raw_fps,
        raw_frame_count=raw_count,
        canonical_height=args.height,
        canonical_width=args.width,
        canonical_num_frames=args.num_frames,
        canonical_frame_stride=args.frame_stride,
        canonical_frame_indices=indices,
        canonical_frame_sampling=f"youtubevos {args.frame_selection}; prompt_source={prompt_source}",
        canonical_resize_policy="exact resize to canonical width/height matching VideoDPO generated-loser setting",
        canonical_crop_policy="none",
        canonical_normalization="saved as uint8 RGB PNG for generation; training normalizes separately",
        canonical_full_mask_value=0.0,
        generator_mask_png_value=255,
    )
    return setting, frames


def build_candidate_row(
    setting: CanonicalSetting,
    sample_id: str,
    source_video_id: str,
    annotation_dir: Path | None,
    prompt_source: str,
    prompt_model: str,
    mask_meta: dict[str, Any],
    model: str,
    generation_source: str,
    win_dir: Path,
    raw_dir: Path,
    comp_dir: Path,
    status: str,
    error_message: str,
) -> dict[str, Any]:
    prompt_input_mode, prompt_used_by_model = MODEL_PROMPT_USAGE[model]
    diffueraser_stack = ""
    diffueraser_prior_mode = ""
    if model == "diffueraser":
        diffueraser_stack, diffueraser_prior_mode = diffueraser_inference_options("partial")
    return {
        "sample_id": sample_id,
        "source_video_id": source_video_id,
        "pair_index": setting.pair_index,
        "prompt": setting.prompt,
        "prompt_source": prompt_source,
        "prompt_model": prompt_model,
        "prompt_input_mode": prompt_input_mode,
        "prompt_used_by_model": prompt_used_by_model,
        "win_video_path": str(win_dir),
        "youtubevos_frame_dir": setting.winner_video_path,
        "youtubevos_annotation_dir": str(annotation_dir) if annotation_dir and annotation_dir.exists() else "",
        "mask_id": mask_meta["mask_id"],
        "mask_path": mask_meta["mask_path"],
        "mask_mode": "partial",
        "mask_convention": "png_255_inpaint_region_0_keep_region",
        "comp": True,
        "generation_source": generation_source,
        "source_dataset": "youtubevos",
        "mask_policy": mask_meta["mask_policy"],
        "mask_area_ratio": mask_meta["area_ratio"],
        "mask_bbox": mask_meta["bbox"],
        "mask_motion_type": mask_meta["motion_type"],
        "mask_velocity": mask_meta["velocity"],
        "generation_model": model,
        "diffueraser_inference_stack": diffueraser_stack,
        "diffueraser_prior_mode": diffueraser_prior_mode,
        "raw_loser_video_path": str(raw_dir),
        "comp_loser_video_path": str(comp_dir),
        "final_loser_video_path": str(comp_dir),
        "raw_metrics": {},
        "comp_metrics": {},
        "quality_score": 0.0,
        "defect_bucket": "unscored",
        "status": status,
        "error_message": error_message if status != "OK" else "",
        "seed": mask_meta["seed"],
        "fps": setting.raw_fps,
        "num_frames": setting.canonical_num_frames,
        "height": setting.canonical_height,
        "width": setting.canonical_width,
        "mask_meta": mask_meta,
        "canonical_setting": asdict(setting),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YouTube-VOS generated-loser calibration subset.")
    parser.add_argument("--youtube_vos_root", default="data/external/ytbv_2019_full_resolution/train")
    parser.add_argument("--caption_json", default="")
    parser.add_argument("--prompt_model", default="")
    parser.add_argument("--prompt_mode", choices=["fallback", "none"], default="fallback")
    parser.add_argument("--output_root", default="data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4")
    parser.add_argument("--models", default="diffueraser")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--timeout_sec", type=int, default=3600)
    parser.add_argument("--frame_selection", choices=["seeded_random", "first"], default="seeded_random")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--mask_policy_config", default="configs/generation/videodpo_partialmask_policy_v1_medium_hard_k4.yaml")
    parser.add_argument("--selection_config", default="configs/generation/medium_hard_balanced_selection_v1.yaml")
    parser.add_argument("--calibration_report", default="PRD/youtubevos_generated_loser_calibration_report.md")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--count_only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load_env_file(Path("configs/paths/pai.detected.env"))
    models = parse_csv(args.models, MODEL_NAMES)
    generation_source = generation_source_label(models)
    root, frames_root, masks_root = resolve_youtubevos_root(args.youtube_vos_root)
    captions = load_captions(args.caption_json)
    policy = load_policy(args.mask_policy_config)
    selection_config = load_yaml(Path(args.selection_config))
    all_videos = scan_frame_dirs(frames_root, args.num_frames)
    if args.count_only:
        print(len(all_videos))
        return 0

    end_index = len(all_videos) if args.end_index < 0 else min(args.end_index, len(all_videos))
    videos = [(idx, all_videos[idx]) for idx in range(args.start_index, end_index)]
    if args.limit > 0:
        videos = videos[: args.limit]

    output_root = Path(args.output_root).resolve()
    manifests_dir = output_root / "manifests"
    reports_dir = output_root / "reports"
    candidates_manifest = manifests_dir / "candidates_all.jsonl"
    scored_manifest = manifests_dir / "candidates_all.scored.jsonl"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_existing and candidates_manifest.exists():
        candidates_manifest.unlink()
    if not args.skip_existing and scored_manifest.exists():
        scored_manifest.unlink()

    print(
        f"[youtubevos] videos={len(videos)} models={models} generation_source={generation_source} "
        f"frames_root={frames_root} output_root={output_root} prompt_mode={args.prompt_mode}"
    )
    print(f"[youtubevos] mask_policy={policy.policy_name} selection_policy={selection_config['policy_name']}")
    if args.dry_run:
        for idx, video_dir in videos[:20]:
            prompt, prompt_source = prompt_for(video_dir, captions, args.prompt_mode)
            print(f"[dry_run] {idx:06d} {video_dir.name} prompt_source={prompt_source} prompt={prompt}")
        return 0

    rows: list[dict[str, Any]] = []
    for source_index, video_dir in videos:
        source_video_id = video_dir.name
        sample_id = f"youtubevos_{source_index:06d}_{safe_id(source_video_id)}"
        prompt, prompt_source = prompt_for(video_dir, captions, args.prompt_mode)
        frame_count = len(image_files(video_dir))
        indices = choose_frame_indices(
            frame_count,
            args.num_frames,
            args.frame_stride,
            args.seed + source_index,
            args.frame_selection,
        )
        setting, frames = build_setting(args, root, video_dir, source_index, prompt, prompt_source, indices)
        sample_root = output_root / "candidates" / sample_id
        win_dir = sample_root / "win"
        if not args.skip_existing:
            clean_dir(sample_root)
        save_rgb_frames(frames, win_dir)
        mask_rows = generate_k_masks(policy, sample_id, output_root / "candidates", args.seed + source_index * 1009)
        annotation_dir = masks_root / source_video_id if masks_root else None
        print(f"[sample] {sample_id} masks={len(mask_rows)} prompt_source={prompt_source}")
        for mask_meta in mask_rows:
            mask_id = str(mask_meta["mask_id"])
            mask_dir = Path(mask_meta["mask_path"])
            for model in models:
                raw_dir = sample_root / mask_id / model / "raw"
                comp_dir = sample_root / mask_id / model / "comp"
                work_dir = output_root / "work" / sample_id / mask_id / model
                log_path = output_root / "logs" / sample_id / f"{mask_id}_{model}.log"
                if args.skip_existing and comp_dir.exists():
                    status, message = "OK", "skip_existing"
                else:
                    status, message = run_candidate(
                        model,
                        "partial",
                        setting,
                        win_dir,
                        mask_dir,
                        raw_dir,
                        comp_dir,
                        work_dir,
                        log_path,
                        args.timeout_sec,
                    )
                row = build_candidate_row(
                    setting,
                    sample_id,
                    source_video_id,
                    annotation_dir,
                    prompt_source,
                    args.prompt_model,
                    mask_meta,
                    model,
                    generation_source,
                    win_dir,
                    raw_dir,
                    comp_dir,
                    status,
                    message,
                )
                if status == "OK":
                    row["raw_metrics"] = compute_candidate_metrics(win_dir, raw_dir, mask_dir, selection_config)
                    row["comp_metrics"] = compute_candidate_metrics(win_dir, comp_dir, mask_dir, selection_config)
                    row["quality_score"] = row["comp_metrics"]["quality_score"]
                    row["defect_bucket"] = row["comp_metrics"]["defect_bucket"]
                rows.append(row)
                jsonl_append(candidates_manifest, row)
                print(f"[candidate] {sample_id} {mask_id} {model} {status} q={row['quality_score']:.4f}")

    write_jsonl(scored_manifest, rows)
    manifests, selection_events = select_manifests(rows, selection_config, "partial")
    for name, manifest_rows in manifests.items():
        if manifest_rows:
            write_jsonl(manifests_dir / f"{name}.jsonl", manifest_rows)
    write_jsonl(manifests_dir / "selection_events.jsonl", selection_events)
    write_calibration_report(Path(args.calibration_report), rows, manifests, selection_events, selection_config)
    reports_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.calibration_report, reports_dir / "generated_loser_calibration_report.md")
    print(f"[youtubevos] report={args.calibration_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
