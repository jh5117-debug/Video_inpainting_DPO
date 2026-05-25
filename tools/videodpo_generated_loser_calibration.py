#!/usr/bin/env python3
"""Run a VideoDPO generated-loser calibration subset.

This is a narrow PAI/H20 utility for the post-smoke, pre-full-generation stage.
It uses the canonical official DiffuEraser VideoDPO setting, generates either a
single full mask or K partial masks per winner, runs the selected generation
models, scores every candidate with cheap metrics, and writes selected
primary/secondary manifests.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.pai_videodpo_single_sample_generation_smoke import (
    CanonicalSetting,
    choose_frame_indices,
    compose_partial,
    load_env_file,
    load_yaml,
    make_full_masks,
    metadata_caption,
    image_files,
    model_command,
    raw_video_info,
    read_canonical_frames,
    read_json,
    resolve_videodpo_roots,
    save_mask_frames,
    save_rgb_frames,
    validate_frame_dir,
)
from tools.videodpo_loser_candidate_scoring import compute_candidate_metrics
from tools.videodpo_loser_candidate_selection import select_manifests, write_calibration_report, write_jsonl
from tools.videodpo_mask_policy import generate_k_masks, load_policy

MODEL_NAMES = ("diffueraser", "propainter", "cococo", "minimax_remover")
MODEL_GPU_ENV = {
    "diffueraser": "DIFFUERASER_GPU",
    "propainter": "PROPAINTER_GPU",
    "cococo": "COCOCO_GPU",
    "minimax_remover": "MINIMAX_REMOVER_GPU",
}
MODEL_PROMPT_USAGE = {
    "diffueraser": ("text_conditioned", True),
    "propainter": ("ignored_by_model", False),
    "cococo": ("text_conditioned", True),
    "minimax_remover": ("ignored_by_model", False),
}


def parse_csv(value: str, allowed: tuple[str, ...]) -> list[str]:
    if value.strip().lower() == "all":
        return list(allowed)
    items = [x.strip().lower().replace("-", "_") for x in value.split(",") if x.strip()]
    bad = [x for x in items if x not in allowed]
    if bad:
        raise SystemExit(f"[error] invalid models {bad}; allowed={list(allowed)}")
    return items


def jsonl_append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_frame_dir(src: Path, dst: Path) -> None:
    clean_dir(dst)
    for idx, path in enumerate(image_files(src)):
        shutil.copy2(path, dst / f"{idx:05d}{path.suffix.lower()}")


def generation_source_label(models: list[str]) -> str:
    if models == ["diffueraser"]:
        return "diffueraser_only"
    return "multi_source_" + "_".join(models)


def make_full_mask_meta(setting: CanonicalSetting, sample_id: str, sample_root: Path, seed: int) -> list[dict[str, Any]]:
    mask_id = "mask_000"
    mask_dir = sample_root / mask_id / "mask"
    save_mask_frames(
        make_full_masks(setting.canonical_num_frames, setting.canonical_height, setting.canonical_width),
        mask_dir,
    )
    width = setting.canonical_width
    height = setting.canonical_height
    frame_bbox = [[0, 0, width, height] for _ in range(setting.canonical_num_frames)]
    frame_area = [1.0 for _ in range(setting.canonical_num_frames)]
    return [
        {
            "mask_id": mask_id,
            "sample_id": sample_id,
            "mask_policy": "fullmask_v1",
            "seed": seed,
            "area_ratio": 1.0,
            "area_ratio_min": 1.0,
            "area_ratio_max": 1.0,
            "frame_area_ratio_min": 1.0,
            "frame_area_ratio_max": 1.0,
            "bbox": [0, 0, width, height],
            "bbox_ratio": [1.0, 1.0],
            "bbox_center_ratio": [0.5, 0.5],
            "bbox_margin_ratio": [0.0, 0.0, 0.0, 0.0],
            "motion_type": "full_static",
            "velocity": [0.0, 0.0],
            "motion_center_bounds": [0.0, 0.0, 1.0, 1.0],
            "motion_box_ratio": 1.0,
            "static_prob": 1.0,
            "mask_shape": "full_frame",
            "mask_location": "full_frame",
            "mask_motion": "static",
            "mask_dilation_iter": 0,
            "frame_level_bbox": frame_bbox,
            "frame_level_area_ratio": frame_area,
            "frames": [
                {
                    "frame": idx,
                    "bbox": [0, 0, width, height],
                    "area_ratio": 1.0,
                    "bbox_ratio": [1.0, 1.0],
                    "bbox_center_ratio": [0.5, 0.5],
                    "bbox_margin_ratio": [0.0, 0.0, 0.0, 0.0],
                }
                for idx in range(setting.canonical_num_frames)
            ],
            "mask_path": str(mask_dir),
        }
    ]


def derive_base(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any], int, int, int, int, float]:
    cfg_path = Path(args.official_config).resolve()
    cfg = load_yaml(cfg_path)
    train_params = cfg["data"]["params"]["train"]["params"]
    model_params = cfg.get("model", {}).get("params", {})
    os.environ.setdefault("BASE_MODEL_PATH", str(model_params.get("base_model_name_or_path", "")))
    os.environ.setdefault("VAE_PATH", str(model_params.get("vae_path", "")))

    train_yaml = Path(os.environ.get("VIDEO_DPO_TRAIN_DATA_YAML") or args.train_data_yaml).expanduser().resolve()
    if not train_yaml.exists():
        raise SystemExit(f"[error] train_data_yaml not found: {train_yaml}")
    roots = resolve_videodpo_roots(train_yaml)
    selected_root = roots[0]
    height = int(train_params.get("train_height") or train_params.get("resolution", [320, 512])[0])
    width = int(train_params.get("train_width") or train_params.get("resolution", [320, 512])[-1])
    nframes = int(train_params.get("video_length", 16))
    stride = int(train_params.get("frame_stride", 1))
    full_mask_value = float(train_params.get("full_mask_value", 0.0))
    return cfg_path, selected_root, cfg, height, width, nframes, stride, full_mask_value


def iter_valid_pairs(
    root: Path,
    start_index: int,
    end_index: int,
    limit: int,
    nframes: int,
    stride: int,
) -> list[tuple[int, dict[str, Any], dict[str, Any], dict[str, Any]]]:
    metadata = read_json(root / "metadata.json")
    pairs = read_json(root / "pair.json")
    if not isinstance(metadata, list) or not isinstance(pairs, list):
        raise RuntimeError(f"unexpected metadata/pair format under {root}")
    out = []
    for pair_index in range(start_index, len(pairs)):
        if end_index >= 0 and pair_index >= end_index:
            break
        pair = pairs[pair_index]
        winner = metadata[int(pair["video1"])]
        loser = metadata[int(pair["video2"])]
        winner_path = root / winner["basic"]["clip_path"]
        if not winner_path.exists():
            continue
        try:
            _, _, _, count = raw_video_info(winner_path)
            choose_frame_indices(count, nframes, stride, 0, "first")
        except Exception:
            continue
        out.append((pair_index, pair, winner, loser))
        if limit > 0 and len(out) >= limit:
            break
    return out


def make_setting(
    cfg_path: Path,
    train_yaml: Path,
    root: Path,
    pair_index: int,
    pair: dict[str, Any],
    winner: dict[str, Any],
    loser: dict[str, Any],
    height: int,
    width: int,
    nframes: int,
    stride: int,
    seed: int,
    frame_selection: str,
    full_mask_value: float,
) -> tuple[CanonicalSetting, list[Any]]:
    winner_idx = int(pair["video1"])
    loser_idx = int(pair["video2"])
    winner_path = (root / winner["basic"]["clip_path"]).resolve()
    loser_path = (root / loser["basic"]["clip_path"]).resolve()
    raw_w, raw_h, raw_fps, raw_count = raw_video_info(winner_path)
    frame_indices = choose_frame_indices(raw_count, nframes, stride, seed + pair_index, frame_selection)
    frames = read_canonical_frames(winner_path, frame_indices, width, height)
    caption = pair.get("frame_caption") or pair.get("caption")
    if isinstance(caption, list):
        caption = caption[0] if caption else ""
    prompt = str(caption or metadata_caption(winner) or metadata_caption(loser) or "")
    setting = CanonicalSetting(
        official_config=str(cfg_path),
        train_data_yaml=str(train_yaml),
        video_root=str(root),
        pair_index=pair_index,
        winner_video_index=winner_idx,
        loser_video_index=loser_idx,
        winner_video_path=str(winner_path),
        loser_video_path=str(loser_path),
        prompt=prompt,
        raw_width=raw_w,
        raw_height=raw_h,
        raw_fps=raw_fps,
        raw_frame_count=raw_count,
        canonical_height=height,
        canonical_width=width,
        canonical_num_frames=nframes,
        canonical_frame_stride=stride,
        canonical_frame_indices=frame_indices,
        canonical_frame_sampling=f"calibration uses {frame_selection} with seed={seed}+pair_index",
        canonical_resize_policy="exact resize to canonical width/height matching official dataset loader",
        canonical_crop_policy="none in current VideoDPOFullMaskDiffuEraserDataset",
        canonical_normalization="saved as uint8 RGB PNG for generation; training normalizes separately",
        canonical_full_mask_value=full_mask_value,
        generator_mask_png_value=255,
    )
    return setting, frames


def run_candidate(
    model: str,
    mask_mode: str,
    setting: CanonicalSetting,
    win_dir: Path,
    mask_dir: Path,
    raw_dir: Path,
    comp_dir: Path | None,
    work_dir: Path,
    log_path: Path,
    timeout_sec: int,
) -> tuple[str, str]:
    command_win_dir = win_dir
    command_mask_dir = mask_dir
    if model == "diffueraser":
        # DiffuEraser wrapper expects batch roots whose child sequence names
        # match. Other wrappers accept direct frame directories.
        batch_root = work_dir / "batch_inputs"
        sequence = "sample"
        command_win_dir = batch_root / "videos" / sequence
        command_mask_dir = batch_root / "masks" / sequence
        copy_frame_dir(win_dir, command_win_dir)
        copy_frame_dir(mask_dir, command_mask_dir)

    command = model_command(model, mask_mode, setting, command_win_dir, command_mask_dir, raw_dir, work_dir)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", "")
    gpu = os.environ.get(MODEL_GPU_ENV[model])
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[cmd] " + " ".join(shlex.quote(x) for x in command) + "\n")
        proc = subprocess.run(
            command,
            cwd=str(Path.cwd()),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec if timeout_sec > 0 else None,
            check=False,
        )
    if proc.returncode != 0:
        return "FAILED", f"returncode={proc.returncode}"
    decoded, h, w, validation = validate_frame_dir(
        raw_dir,
        setting.canonical_num_frames,
        setting.canonical_height,
        setting.canonical_width,
    )
    if validation != "OK":
        return "FAILED", validation
    if mask_mode == "partial":
        if comp_dir is None:
            return "FAILED", "missing comp_dir for partial mask candidate"
        outside = compose_partial(win_dir, raw_dir, mask_dir, comp_dir, setting.canonical_num_frames)
        if outside > 0:
            return "FAILED", f"comp outside-mask max diff is not zero: {outside}"
    return "OK", "returncode=0"


def build_candidate_row(
    setting: CanonicalSetting,
    sample_id: str,
    mask_meta: dict[str, Any],
    model: str,
    mask_mode: str,
    generation_source: str,
    win_dir: Path,
    raw_dir: Path,
    comp_dir: Path | None,
    status: str,
    error_message: str,
) -> dict[str, Any]:
    prompt_input_mode, prompt_used_by_model = MODEL_PROMPT_USAGE[model]
    return {
        "sample_id": sample_id,
        "source_video_id": Path(setting.winner_video_path).stem,
        "pair_index": setting.pair_index,
        "prompt": setting.prompt,
        "prompt_input_mode": prompt_input_mode,
        "prompt_used_by_model": prompt_used_by_model,
        "win_video_path": str(win_dir),
        "mask_id": mask_meta["mask_id"],
        "mask_path": mask_meta["mask_path"],
        "mask_mode": mask_mode,
        "mask_convention": "png_255_inpaint_region_0_keep_region",
        "comp": mask_mode == "partial",
        "generation_source": generation_source,
        "source_dataset": "videodpo",
        "mask_policy": mask_meta["mask_policy"],
        "mask_area_ratio": mask_meta["area_ratio"],
        "mask_bbox": mask_meta["bbox"],
        "mask_motion_type": mask_meta["motion_type"],
        "mask_velocity": mask_meta["velocity"],
        "generation_model": model,
        "raw_loser_video_path": str(raw_dir),
        "comp_loser_video_path": str(comp_dir) if comp_dir is not None else "",
        "final_loser_video_path": str(comp_dir if comp_dir is not None else raw_dir),
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
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VideoDPO generated-loser calibration subset.")
    parser.add_argument("--output_root", default="data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4")
    parser.add_argument("--models", default="all")
    parser.add_argument("--mask_mode", choices=["partial", "full"], default="partial")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--timeout_sec", type=int, default=3600)
    parser.add_argument("--frame_selection", choices=["seeded_random", "first"], default="seeded_random")
    parser.add_argument("--official_config", default="DPO_finetune/configs/official_diffueraser_stage1.yaml")
    parser.add_argument("--train_data_yaml", default="/mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.pai.yaml")
    parser.add_argument("--mask_policy_config", default="configs/generation/videodpo_partialmask_policy_v1_medium_hard_k4.yaml")
    parser.add_argument("--selection_config", default="configs/generation/medium_hard_balanced_selection_v1.yaml")
    parser.add_argument("--calibration_report", default="PRD/generated_loser_calibration_report.md")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load_env_file(Path("configs/paths/pai.detected.env"))
    models = parse_csv(args.models, MODEL_NAMES)
    generation_source = generation_source_label(models)
    output_root = Path(args.output_root).resolve()
    manifests_dir = output_root / "manifests"
    reports_dir = output_root / "reports"
    candidates_manifest = manifests_dir / "candidates_all.jsonl"
    scored_manifest = manifests_dir / "candidates_all.scored.jsonl"
    candidates_manifest.parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_existing and candidates_manifest.exists():
        candidates_manifest.unlink()
    if not args.skip_existing and scored_manifest.exists():
        scored_manifest.unlink()

    cfg_path, root, _, height, width, nframes, stride, full_mask_value = derive_base(args)
    train_yaml = Path(os.environ.get("VIDEO_DPO_TRAIN_DATA_YAML") or args.train_data_yaml).expanduser().resolve()
    policy = load_policy(args.mask_policy_config) if args.mask_mode == "partial" else None
    selection_config = load_yaml(Path(args.selection_config))
    pairs = iter_valid_pairs(root, args.start_index, args.end_index, args.limit, nframes, stride)

    print(f"[calibration] winners={len(pairs)} models={models} generation_source={generation_source} output_root={output_root}")
    print(f"[calibration] mask_mode={args.mask_mode} mask_policy={(policy.policy_name if policy else 'fullmask_v1')} selection_policy={selection_config['policy_name']}")
    if args.dry_run:
        print("[calibration] dry_run=true; no model inference")
        return 0

    rows: list[dict[str, Any]] = []
    for pair_index, pair, winner, loser in pairs:
        sample_id = f"videodpo_pair{pair_index:06d}"
        setting, frames = make_setting(
            cfg_path,
            train_yaml,
            root,
            pair_index,
            pair,
            winner,
            loser,
            height,
            width,
            nframes,
            stride,
            args.seed,
            args.frame_selection,
            full_mask_value,
        )
        sample_root = output_root / "candidates" / sample_id
        win_dir = sample_root / "win"
        if not args.skip_existing:
            clean_dir(sample_root)
        save_rgb_frames(frames, win_dir)
        if args.mask_mode == "partial":
            if policy is None:
                raise RuntimeError("partial mask mode requires a mask policy")
            mask_rows = generate_k_masks(policy, sample_id, output_root / "candidates", args.seed + pair_index * 1009)
        else:
            mask_rows = make_full_mask_meta(setting, sample_id, sample_root, args.seed + pair_index * 1009)
        print(f"[sample] {sample_id} masks={len(mask_rows)}")
        for mask_meta in mask_rows:
            mask_id = str(mask_meta["mask_id"])
            mask_dir = Path(mask_meta["mask_path"])
            for model in models:
                raw_dir = sample_root / mask_id / model / "raw"
                comp_dir = sample_root / mask_id / model / "comp" if args.mask_mode == "partial" else None
                work_dir = output_root / "work" / sample_id / mask_id / model
                log_path = output_root / "logs" / sample_id / f"{mask_id}_{model}.log"
                existing_dir = comp_dir if comp_dir is not None else raw_dir
                if args.skip_existing and existing_dir.exists():
                    status, message = "OK", "skip_existing"
                else:
                    status, message = run_candidate(
                        model,
                        args.mask_mode,
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
                    mask_meta,
                    model,
                    args.mask_mode,
                    generation_source,
                    win_dir,
                    raw_dir,
                    comp_dir,
                    status,
                    message,
                )
                if status == "OK":
                    row["raw_metrics"] = compute_candidate_metrics(win_dir, raw_dir, mask_dir, selection_config)
                    if args.mask_mode == "partial" and comp_dir is not None:
                        row["comp_metrics"] = compute_candidate_metrics(win_dir, comp_dir, mask_dir, selection_config)
                        row["quality_score"] = row["comp_metrics"]["quality_score"]
                        row["defect_bucket"] = row["comp_metrics"]["defect_bucket"]
                    else:
                        row["quality_score"] = row["raw_metrics"]["quality_score"]
                        row["defect_bucket"] = row["raw_metrics"]["defect_bucket"]
                rows.append(row)
                jsonl_append(candidates_manifest, row)
                print(f"[candidate] {sample_id} {mask_id} {model} {status} q={row['quality_score']:.4f}")

    write_jsonl(scored_manifest, rows)
    manifests, selection_events = select_manifests(rows, selection_config, args.mask_mode)
    for name, manifest_rows in manifests.items():
        if manifest_rows:
            write_jsonl(manifests_dir / f"{name}.jsonl", manifest_rows)
    write_jsonl(manifests_dir / "selection_events.jsonl", selection_events)
    write_calibration_report(Path(args.calibration_report), rows, manifests, selection_events, selection_config)
    reports_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.calibration_report, reports_dir / "generated_loser_calibration_report.md")
    print(f"[calibration] report={args.calibration_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
