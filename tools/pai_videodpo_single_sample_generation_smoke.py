#!/usr/bin/env python3
"""Canonical VideoDPO one-sample generation smoke for PAI.

This tool is intentionally narrow:

- it reads the same official DiffuEraser/VideoDPO config used by the completed
  PAI runs;
- it prepares one winner clip with the same canonical H/W/frame-count policy;
- it saves canonical full and partial masks;
- with ``--run_generation`` it calls the existing four model wrappers on that
  one sample only.

It never starts DPO training and never starts full data generation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_NAMES = ("diffueraser", "propainter", "cococo", "minimax_remover")


@dataclass
class CanonicalSetting:
    official_config: str
    train_data_yaml: str
    video_root: str
    pair_index: int
    winner_video_index: int
    loser_video_index: int
    winner_video_path: str
    loser_video_path: str
    prompt: str
    raw_width: int
    raw_height: int
    raw_fps: float
    raw_frame_count: int
    canonical_height: int
    canonical_width: int
    canonical_num_frames: int
    canonical_frame_stride: int
    canonical_frame_indices: list[int]
    canonical_frame_sampling: str
    canonical_resize_policy: str
    canonical_crop_policy: str
    canonical_normalization: str
    canonical_full_mask_value: float
    generator_mask_png_value: int


@dataclass
class SmokeResult:
    model: str
    mask_mode: str
    status: str
    raw_loser_frames_dir: str
    comp_loser_frames_dir: str | None
    final_loser_frames_dir: str | None
    log_path: str
    command: list[str]
    decoded_frames: int
    height: int | None
    width: int | None
    comp_outside_mask_max_abs_diff: float | None
    message: str


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - PAI dependency check
        raise SystemExit(f"[error] PyYAML is required to read {path}: {exc}") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(data, dict):
        raise SystemExit(f"[error] YAML did not parse as a dict: {path}")
    return data


def parse_csv(value: str, allowed: Iterable[str]) -> list[str]:
    allowed_set = set(allowed)
    if value.strip().lower() == "all":
        return list(allowed)
    items = [x.strip().lower().replace("-", "_") for x in value.split(",") if x.strip()]
    bad = [x for x in items if x not in allowed_set]
    if bad:
        raise SystemExit(f"[error] invalid values {bad}; allowed={sorted(allowed_set)}")
    return items


def first_existing(paths: Iterable[str | Path | None]) -> Path | None:
    for item in paths:
        if not item:
            continue
        path = Path(item).expanduser()
        if path.exists():
            return path.resolve()
    return None


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_videodpo_roots(train_yaml: Path) -> list[Path]:
    cfg = load_yaml(train_yaml)
    roots: list[Path] = []
    for item in cfg.get("META", []):
        path = Path(str(item)).expanduser()
        if not path.is_absolute():
            candidates = []
            if os.environ.get("VIDEODPO_DATA_BASE"):
                candidates.append(Path(os.environ["VIDEODPO_DATA_BASE"]).expanduser() / path)
            if os.environ.get("VIDEODPO_REPO"):
                candidates.append(Path(os.environ["VIDEODPO_REPO"]).expanduser() / path)
            candidates.extend([train_yaml.parent / path, Path.cwd() / path])
            path = next((p for p in candidates if p.exists()), candidates[0])
        roots.append(path.resolve())
    if not roots:
        raise SystemExit(f"[error] no META roots found in train data YAML: {train_yaml}")
    return roots


def metadata_caption(item: dict) -> str:
    captions = item.get("misc", {}).get("frame_caption", [])
    if isinstance(captions, list) and captions:
        return str(captions[0])
    if isinstance(captions, str):
        return captions
    return ""


def raw_video_info(path: Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return width, height, fps, count


def choose_frame_indices(total: int, nframes: int, stride: int, seed: int, policy: str) -> list[int]:
    all_frames = list(range(0, total, stride))
    if len(all_frames) < nframes:
        all_frames = list(range(0, total, 1))
    if len(all_frames) < nframes:
        raise ValueError(f"video has only {total} frames; need {nframes}")
    if policy == "first":
        start = 0
    else:
        start = random.Random(seed).randint(0, len(all_frames) - nframes)
    return all_frames[start : start + nframes]


def read_canonical_frames(path: Path, indices: list[int], width: int, height: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"failed to decode frame {idx} from {path}")
        frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def save_rgb_frames(frames: list[np.ndarray], out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        cv2.imwrite(str(out_dir / f"{idx:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def save_mask_frames(masks: list[np.ndarray], out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(masks):
        cv2.imwrite(str(out_dir / f"{idx:05d}.png"), mask.astype(np.uint8))


def make_full_masks(n: int, height: int, width: int) -> list[np.ndarray]:
    return [np.full((height, width), 255, dtype=np.uint8) for _ in range(n)]


def make_partial_masks(n: int, height: int, width: int, seed: int) -> tuple[list[np.ndarray], dict]:
    rng = random.Random(seed)
    masks: list[np.ndarray] = []
    boxes = []
    for _ in range(n):
        mask = np.zeros((height, width), dtype=np.uint8)
        box_w = rng.randint(max(16, width // 5), max(17, width // 2))
        box_h = rng.randint(max(16, height // 5), max(17, height // 2))
        x0 = rng.randint(0, max(0, width - box_w))
        y0 = rng.randint(0, max(0, height - box_h))
        mask[y0 : y0 + box_h, x0 : x0 + box_w] = 255
        masks.append(mask)
        boxes.append([x0, y0, x0 + box_w, y0 + box_h])
    area = float(np.mean([(m > 0).mean() for m in masks]))
    return masks, {"mask_area_ratio": area, "mask_bboxes": boxes}


def find_first_valid_pair(
    root: Path,
    start_index: int,
    nframes: int,
    stride: int,
) -> tuple[int, dict, dict, dict, list[dict]]:
    metadata = read_json(root / "metadata.json")
    pairs = read_json(root / "pair.json")
    if not isinstance(metadata, list) or not isinstance(pairs, list):
        raise RuntimeError(f"unexpected metadata/pair format under {root}")
    for pair_index in range(start_index, len(pairs)):
        pair = pairs[pair_index]
        winner_idx = int(pair["video1"])
        loser_idx = int(pair["video2"])
        winner = metadata[winner_idx]
        loser = metadata[loser_idx]
        winner_path = root / winner["basic"]["clip_path"]
        if not winner_path.exists():
            continue
        try:
            _, _, _, count = raw_video_info(winner_path)
            choose_frame_indices(count, nframes, stride, 0, "first")
        except Exception:
            continue
        return pair_index, pair, winner, loser, metadata
    raise RuntimeError(f"no valid pair found under {root} starting at index {start_index}")


def derive_canonical_setting(args: argparse.Namespace) -> tuple[CanonicalSetting, list[np.ndarray], Path, Path, Path, dict]:
    cfg_path = Path(args.official_config).resolve()
    cfg = load_yaml(cfg_path)
    train_params = cfg["data"]["params"]["train"]["params"]
    model_params = cfg.get("model", {}).get("params", {})

    train_yaml = Path(
        os.environ.get("VIDEO_DPO_TRAIN_DATA_YAML")
        or train_params.get("data_root")
        or args.train_data_yaml
    ).expanduser().resolve()
    if not train_yaml.exists():
        raise SystemExit(f"[error] train_data_yaml not found: {train_yaml}")

    height = int(train_params.get("train_height") or train_params.get("resolution", [320, 512])[0])
    width = int(train_params.get("train_width") or train_params.get("resolution", [320, 512])[-1])
    nframes = int(train_params.get("video_length", 16))
    stride = int(train_params.get("frame_stride", 1))
    full_mask_value = float(train_params.get("full_mask_value", 0.0))

    os.environ.setdefault("BASE_MODEL_PATH", str(model_params.get("base_model_name_or_path", "")))
    os.environ.setdefault("VAE_PATH", str(model_params.get("vae_path", "")))

    video_roots = resolve_videodpo_roots(train_yaml)
    selected_root = video_roots[0]
    pair_index, pair, winner, loser, _ = find_first_valid_pair(
        selected_root,
        args.pair_index,
        nframes,
        stride,
    )
    winner_idx = int(pair["video1"])
    loser_idx = int(pair["video2"])
    winner_path = (selected_root / winner["basic"]["clip_path"]).resolve()
    loser_path = (selected_root / loser["basic"]["clip_path"]).resolve()

    raw_w, raw_h, raw_fps, raw_count = raw_video_info(winner_path)
    frame_indices = choose_frame_indices(
        raw_count,
        nframes,
        stride,
        args.seed + pair_index,
        args.frame_selection,
    )
    frames = read_canonical_frames(winner_path, frame_indices, width, height)

    caption = pair.get("frame_caption") or pair.get("caption")
    if isinstance(caption, list):
        caption = caption[0] if caption else ""
    prompt = str(caption or metadata_caption(winner) or metadata_caption(loser) or "")

    setting = CanonicalSetting(
        official_config=str(cfg_path),
        train_data_yaml=str(train_yaml),
        video_root=str(selected_root),
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
        canonical_frame_sampling=(
            "training uses a random contiguous window over stride-filtered frames; "
            f"this smoke uses {args.frame_selection} with seed={args.seed}+pair_index"
        ),
        canonical_resize_policy=(
            "official dataset uses decord VideoReader(width=train_width, height=train_height); "
            "smoke mirrors this as exact resize to canonical width/height"
        ),
        canonical_crop_policy="none in current VideoDPOFullMaskDiffuEraserDataset",
        canonical_normalization="training tensor is uint8/255 -> [-1, 1]; smoke frames are saved as uint8 RGB PNG",
        canonical_full_mask_value=full_mask_value,
        generator_mask_png_value=255,
    )

    full_masks = make_full_masks(nframes, height, width)
    partial_masks, partial_meta = make_partial_masks(nframes, height, width, args.seed + pair_index + 1009)

    output_root = Path(args.output_root).resolve()
    video_dir = output_root / "inputs" / "videos" / "smoke_sample"
    full_mask_dir = output_root / "inputs" / "masks_full" / "smoke_sample"
    partial_mask_dir = output_root / "inputs" / "masks_partial" / "smoke_sample"
    save_rgb_frames(frames, video_dir)
    save_mask_frames(full_masks, full_mask_dir)
    save_mask_frames(partial_masks, partial_mask_dir)
    return setting, frames, video_dir, full_mask_dir, partial_mask_dir, partial_meta


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def validate_frame_dir(path: Path, expected_n: int, expected_h: int, expected_w: int) -> tuple[int, int | None, int | None, str]:
    if not path.exists():
        return 0, None, None, f"missing output dir: {path}"
    files = image_files(path)
    if not files:
        return 0, None, None, f"no frames under {path}"
    first = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
    if first is None:
        return len(files), None, None, f"failed to decode first output frame: {files[0]}"
    h, w = first.shape[:2]
    if len(files) != expected_n:
        return len(files), h, w, f"frame count mismatch expected={expected_n} got={len(files)}"
    if h != expected_h or w != expected_w:
        return len(files), h, w, f"shape mismatch expected={expected_h}x{expected_w} got={h}x{w}"
    return len(files), h, w, "OK"


def compose_partial(
    win_dir: Path,
    raw_dir: Path,
    mask_dir: Path,
    comp_dir: Path,
    expected_n: int,
) -> float:
    win_files = image_files(win_dir)
    raw_files = image_files(raw_dir)
    mask_files = image_files(mask_dir)
    if min(len(win_files), len(raw_files), len(mask_files)) < expected_n:
        raise RuntimeError("cannot compose: insufficient win/raw/mask frames")
    if comp_dir.exists():
        shutil.rmtree(comp_dir)
    comp_dir.mkdir(parents=True, exist_ok=True)
    outside_max = 0.0
    for idx in range(expected_n):
        win = cv2.imread(str(win_files[idx]), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(raw_files[idx]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_files[idx]), cv2.IMREAD_GRAYSCALE)
        if win is None or raw is None or mask is None:
            raise RuntimeError(f"cannot compose frame {idx}")
        raw = cv2.resize(raw, (win.shape[1], win.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask_bool = mask > 0
        out = win.copy()
        out[mask_bool] = raw[mask_bool]
        outside = ~mask_bool
        if outside.any():
            outside_max = max(outside_max, float(np.abs(out[outside].astype(np.int16) - win[outside].astype(np.int16)).max()))
        cv2.imwrite(str(comp_dir / f"{idx:05d}.png"), out)
    return outside_max


def python_for_model(model: str) -> str:
    env_name = {
        "diffueraser": "DIFFUERASER_PYTHON",
        "propainter": "PROPAINTER_PYTHON",
        "cococo": "COCOCO_PYTHON",
        "minimax_remover": "MINIMAX_REMOVER_PYTHON",
    }[model]
    return os.environ.get(env_name, sys.executable)


def third_party_root() -> Path | None:
    return first_existing([
        os.environ.get("THIRD_PARTY_VIDEO_INPAINTING_ROOT"),
        os.environ.get("THIRD_PARTY_ROOT"),
        Path.cwd() / "third_party_video_inpainting",
        "/mnt/nas/hj/H20_Video_inpainting_DPO/third_party_video_inpainting",
        "/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting",
    ])


def model_command(model: str, mask_mode: str, setting: CanonicalSetting, video_dir: Path, mask_dir: Path, out_dir: Path, work_dir: Path) -> list[str]:
    repo = Path.cwd().resolve()
    n = setting.canonical_num_frames
    w = setting.canonical_width
    h = setting.canonical_height
    prompt = setting.prompt
    py = python_for_model(model)
    third = third_party_root()

    if model == "diffueraser":
        base = first_existing([os.environ.get("BASE_MODEL_PATH"), "/mnt/nas/hj/weights/stable-diffusion-v1-5", repo / "weights" / "stable-diffusion-v1-5"])
        vae = first_existing([os.environ.get("VAE_PATH"), "/mnt/nas/hj/weights/sd-vae-ft-mse", repo / "weights" / "sd-vae-ft-mse"])
        diffueraser = first_existing([os.environ.get("DIFFUERASER_WEIGHT_ROOT"), repo / "weights" / "diffuEraser", repo / "weights" / "diffueraser"])
        propainter = first_existing([os.environ.get("PROPAINTER_WEIGHT_ROOT"), repo / "weights" / "propainter", third / "weights" / "propainter" if third else None])
        pcm = first_existing([os.environ.get("PCM_WEIGHTS_PATH"), "/mnt/nas/hj/weights/PCM_Weights", repo / "weights" / "PCM_Weights"])
        missing = [name for name, path in [("BASE_MODEL_PATH", base), ("VAE_PATH", vae), ("DIFFUERASER_WEIGHT_ROOT", diffueraser), ("PROPAINTER_WEIGHT_ROOT", propainter), ("PCM_WEIGHTS_PATH", pcm)] if path is None]
        if missing:
            raise RuntimeError(f"missing DiffuEraser assets: {missing}")
        return [
            py, "DPO_finetune/infer_diffueraser_candidate.py",
            "--video_root", str(video_dir.parent),
            "--mask_root", str(mask_dir.parent),
            "--output_dir", str(out_dir),
            "--work_dir", str(work_dir),
            "--project_root", str(repo),
            "--base_model_path", str(base),
            "--vae_path", str(vae),
            "--diffueraser_path", str(diffueraser),
            "--propainter_model_dir", str(propainter),
            "--pcm_weights_path", str(pcm),
            "--prompt", prompt,
            "--num_frames", str(n),
            "--width", str(w),
            "--height", str(h),
        ]

    if model == "propainter":
        propainter = first_existing([os.environ.get("PROPAINTER_WEIGHT_ROOT"), repo / "weights" / "propainter", third / "weights" / "propainter" if third else None])
        if propainter is None:
            raise RuntimeError("missing PROPAINTER_WEIGHT_ROOT")
        return [
            py, "DPO_finetune/infer_propainter_candidate.py",
            "--video_dir", str(video_dir),
            "--mask_dir", str(mask_dir),
            "--output_dir", str(out_dir),
            "--model_dir", str(propainter),
            "--num_frames", str(n),
            "--width", str(w),
            "--height", str(h),
        ]

    if model == "cococo":
        cococo_repo = first_existing([
            os.environ.get("COCOCO_REPO_ROOT"),
            third / "repos" / "COCOCO" if third else None,
            "/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/COCOCO",
        ])
        root = first_existing([os.environ.get("COCOCO_WEIGHT_ROOT"), third / "weights" / "COCOCO_weight" if third else None, repo / "weights" / "cococo"])
        if cococo_repo is None or root is None:
            raise RuntimeError("missing COCOCO_REPO_ROOT or COCOCO_WEIGHT_ROOT")
        model_path = first_existing([root / "cococo", root])
        sd_inpaint = first_existing([root / "stable-diffusion-v1-5-inpainting", os.environ.get("COCOCO_SD_INPAINT_ROOT")])
        if model_path is None or sd_inpaint is None:
            raise RuntimeError("missing COCOCO model_path or stable-diffusion-v1-5-inpainting")
        return [
            py, "DPO_finetune/infer_cococo_candidate.py",
            "--repo_dir", str(cococo_repo),
            "--video_dir", str(video_dir),
            "--mask_dir", str(mask_dir),
            "--output_dir", str(out_dir),
            "--work_dir", str(work_dir),
            "--model_path", str(model_path),
            "--pretrain_model_path", str(sd_inpaint),
            "--prompt", prompt or "a video",
            "--num_samples", "1",
            "--num_frames", str(n),
            "--width", str(w),
            "--height", str(h),
        ]

    if model == "minimax_remover":
        minimax_repo = first_existing([
            os.environ.get("MINIMAX_REMOVER_REPO_ROOT"),
            third / "repos" / "MiniMax-Remover" if third else None,
            "/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/repos/MiniMax-Remover",
        ])
        minimax = first_existing([os.environ.get("MINIMAX_REMOVER_WEIGHT_ROOT"), third / "weights" / "minimax" if third else None, repo / "weights" / "minimax_remover"])
        if minimax_repo is None or minimax is None:
            raise RuntimeError("missing MINIMAX_REMOVER_REPO_ROOT or MINIMAX_REMOVER_WEIGHT_ROOT")
        return [
            py, "DPO_finetune/infer_minimax_candidate.py",
            "--repo_dir", str(minimax_repo),
            "--video_dir", str(video_dir),
            "--mask_dir", str(mask_dir),
            "--output_dir", str(out_dir),
            "--model_dir", str(minimax),
            "--num_frames", str(n),
            "--width", str(w),
            "--height", str(h),
        ]

    raise RuntimeError(f"unknown model: {model}")


def run_model(
    model: str,
    mask_mode: str,
    setting: CanonicalSetting,
    video_dir: Path,
    mask_dir: Path,
    output_root: Path,
    run_generation: bool,
    timeout_sec: int,
) -> SmokeResult:
    raw_dir = output_root / "raw" / model / mask_mode
    work_dir = output_root / "work" / model / mask_mode
    log_path = output_root / "logs" / f"{model}_{mask_mode}.log"
    comp_dir = output_root / "comp" / model / mask_mode if mask_mode == "partial" else None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command: list[str] = []
    status = "NOT_RUN"
    message = "generation not requested"

    if run_generation:
        try:
            command = model_command(model, mask_mode, setting, video_dir, mask_dir, raw_dir, work_dir)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", "")
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
            status = "OK" if proc.returncode == 0 else "FAILED"
            message = f"returncode={proc.returncode}"
        except Exception as exc:
            status = "FAILED"
            message = str(exc)
            log_path.write_text(message + "\n", encoding="utf-8")

    decoded, h, w, validation = validate_frame_dir(
        raw_dir,
        setting.canonical_num_frames,
        setting.canonical_height,
        setting.canonical_width,
    )
    if status == "OK" and validation != "OK":
        status = "FAILED"
        message = validation

    outside_max = None
    final_dir = raw_dir
    if status == "OK" and mask_mode == "partial" and comp_dir is not None:
        try:
            outside_max = compose_partial(
                output_root / "inputs" / "videos" / "smoke_sample",
                raw_dir,
                mask_dir,
                comp_dir,
                setting.canonical_num_frames,
            )
            final_dir = comp_dir
            if outside_max > 0:
                status = "FAILED"
                message = f"comp outside-mask max diff is not zero: {outside_max}"
        except Exception as exc:
            status = "FAILED"
            message = f"comp failed: {exc}"

    return SmokeResult(
        model=model,
        mask_mode=mask_mode,
        status=status,
        raw_loser_frames_dir=str(raw_dir),
        comp_loser_frames_dir=str(comp_dir) if comp_dir else None,
        final_loser_frames_dir=str(final_dir) if final_dir else None,
        log_path=str(log_path),
        command=command,
        decoded_frames=decoded,
        height=h,
        width=w,
        comp_outside_mask_max_abs_diff=outside_max,
        message=message,
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tail_text(path: Path, lines: int) -> str:
    if lines <= 0 or not path.exists():
        return ""
    try:
        data = path.read_text(errors="replace").splitlines()
    except Exception as exc:
        return f"[could not read log: {exc}]"
    return "\n".join(data[-lines:])


def one_line_tail(path: Path, lines: int = 8) -> str:
    text = tail_text(path, lines)
    if not text:
        return ""
    return " | ".join(x.strip() for x in text.splitlines() if x.strip())[-1200:]


def write_report(path: Path, setting: CanonicalSetting, results: list[SmokeResult], run_generation: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PAI VideoDPO Single-Sample Generation Smoke",
        "",
        "## Canonical Setting",
        "",
        f"- train_data_yaml: `{setting.train_data_yaml}`",
        f"- video_root: `{setting.video_root}`",
        f"- pair_index: `{setting.pair_index}`",
        f"- winner_video_path: `{setting.winner_video_path}`",
        f"- prompt: `{setting.prompt}`",
        f"- raw: {setting.raw_width}x{setting.raw_height}, fps={setting.raw_fps:.4f}, frames={setting.raw_frame_count}",
        f"- canonical_height: {setting.canonical_height}",
        f"- canonical_width: {setting.canonical_width}",
        f"- canonical_num_frames: {setting.canonical_num_frames}",
        f"- canonical_frame_stride: {setting.canonical_frame_stride}",
        f"- canonical_frame_indices: `{setting.canonical_frame_indices}`",
        f"- canonical_frame_sampling: {setting.canonical_frame_sampling}",
        f"- canonical_resize_policy: {setting.canonical_resize_policy}",
        f"- canonical_crop_policy: {setting.canonical_crop_policy}",
        f"- canonical_full_mask_value: {setting.canonical_full_mask_value}",
        f"- generator_mask_png_value: {setting.generator_mask_png_value}",
        "",
        "## Results",
        "",
        f"- run_generation: {run_generation}",
        "",
        "| Model | Mask | Status | Decoded | H | W | Comp Outside Diff | Message |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in results:
        outside = "-" if item.comp_outside_mask_max_abs_diff is None else f"{item.comp_outside_mask_max_abs_diff:.6f}"
        lines.append(
            f"| {item.model} | {item.mask_mode} | {item.status} | {item.decoded_frames} | "
            f"{item.height or '-'} | {item.width or '-'} | {outside} | {item.message} |"
        )
    lines.extend([
        "",
        "## Gate",
        "",
        "Full data generation may start only for models whose full and partial smoke rows are `OK`.",
        "This smoke does not start DPO training.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def print_stdout_summary(
    setting: CanonicalSetting,
    results: list[SmokeResult],
    report_path: Path,
    manifest_path: Path,
    run_generation: bool,
    log_tail_lines: int,
) -> None:
    print("===== CANONICAL VIDEODPO SETTING =====")
    print(f"height={setting.canonical_height} width={setting.canonical_width} frames={setting.canonical_num_frames} stride={setting.canonical_frame_stride}")
    print(f"train_data_yaml={setting.train_data_yaml}")
    print(f"winner={setting.winner_video_path}")
    print(f"prompt={setting.prompt}")
    print(f"run_generation={run_generation}")
    print()
    print("===== SMOKE RESULT TABLE =====")
    print("| Model | Mask | Status | Decoded | H | W | Message | Log |")
    print("| --- | --- | --- | ---: | ---: | ---: | --- | --- |")
    for item in results:
        log_path = Path(item.log_path)
        msg = item.message.replace("\n", " ")[:500]
        print(
            f"| {item.model} | {item.mask_mode} | {item.status} | {item.decoded_frames} | "
            f"{item.height or '-'} | {item.width or '-'} | {msg} | {log_path} |"
        )
    print()
    print(f"REPORT={report_path}")
    print(f"MANIFEST={manifest_path}")

    failed = [r for r in results if run_generation and r.status != "OK"]
    if failed:
        print()
        print("===== FAILURE LOG TAILS =====")
        for item in failed:
            log_path = Path(item.log_path)
            print()
            print(f"### {item.model} {item.mask_mode} status={item.status}")
            print(f"message={item.message}")
            print(f"log={log_path}")
            tail = tail_text(log_path, log_tail_lines)
            print(tail if tail else "[no log tail available]")


def write_canonical_prd(path: Path, setting: CanonicalSetting) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "# VideoDPO Canonical Data Setting",
            "",
            "Verified from the completed official VideoDPO / official DiffuEraser PAI config path.",
            "",
            f"- canonical_height: {setting.canonical_height}",
            f"- canonical_width: {setting.canonical_width}",
            f"- canonical_num_frames: {setting.canonical_num_frames}",
            f"- canonical_frame_sampling: {setting.canonical_frame_sampling}",
            f"- canonical_frame_stride: {setting.canonical_frame_stride}",
            f"- canonical_resize_policy: {setting.canonical_resize_policy}",
            f"- canonical_crop_policy: {setting.canonical_crop_policy}",
            f"- canonical_normalization: {setting.canonical_normalization}",
            f"- canonical_full_mask_value: {setting.canonical_full_mask_value}",
            f"- generator_mask_png_value: {setting.generator_mask_png_value}",
            f"- train_data_yaml: `{setting.train_data_yaml}`",
            f"- source_config: `{setting.official_config}`",
            f"- source_video_root: `{setting.video_root}`",
            f"- verified_pair_index: {setting.pair_index}",
            f"- verified_winner_video_path: `{setting.winner_video_path}`",
            f"- verified_raw_video: {setting.raw_width}x{setting.raw_height}, fps={setting.raw_fps:.4f}, frames={setting.raw_frame_count}",
            f"- verified_prompt: `{setting.prompt}`",
            "",
            "Generation requirement: fullmask and partialmask loser generation must use frames saved after this canonical preprocessing, so `win`, `mask`, `raw_loser`, and `comp_loser` all share `[T,H,W] = "
            f"[{setting.canonical_num_frames},{setting.canonical_height},{setting.canonical_width}]`.",
            "",
            "Verification command:",
            "",
            "```bash",
            "python tools/pai_videodpo_single_sample_generation_smoke.py --models all --mask_modes full,partial --run_generation",
            "```",
            "",
        ]),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official_config", default="DPO_finetune/configs/official_diffueraser_stage1.yaml")
    parser.add_argument("--train_data_yaml", default="")
    parser.add_argument("--output_root", default="outputs/asset_smoke_tests/videodpo_single_sample")
    parser.add_argument("--report_path", default="")
    parser.add_argument("--manifest_path", default="")
    parser.add_argument("--canonical_prd_path", default="PRD/videodpo_canonical_data_setting.md")
    parser.add_argument("--models", default="all", help="comma-separated or all")
    parser.add_argument("--mask_modes", default="full,partial", help="comma-separated from full,partial")
    parser.add_argument("--pair_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame_selection", choices=["seeded_random", "first"], default="seeded_random")
    parser.add_argument("--run_generation", action="store_true")
    parser.add_argument("--timeout_sec", type=int, default=0)
    parser.add_argument("--print_log_tail_lines", type=int, default=80)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    load_env_file(Path("configs/paths/pai.detected.env"))
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    setting, _, video_dir, full_mask_dir, partial_mask_dir, partial_meta = derive_canonical_setting(args)
    (output_root / "canonical_setting.json").write_text(
        json.dumps({**asdict(setting), **partial_meta}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_canonical_prd(Path(args.canonical_prd_path), setting)

    models = parse_csv(args.models, MODEL_NAMES)
    mask_modes = parse_csv(args.mask_modes, ("full", "partial"))
    results: list[SmokeResult] = []
    manifest_rows: list[dict] = []
    for model in models:
        for mask_mode in mask_modes:
            mask_dir = full_mask_dir if mask_mode == "full" else partial_mask_dir
            result = run_model(
                model=model,
                mask_mode=mask_mode,
                setting=setting,
                video_dir=video_dir,
                mask_dir=mask_dir,
                output_root=output_root,
                run_generation=args.run_generation,
                timeout_sec=args.timeout_sec,
            )
            if args.run_generation and result.status != "OK" and Path(result.log_path).exists():
                tail = one_line_tail(Path(result.log_path), 8)
                if tail and tail not in result.message:
                    result.message = f"{result.message}; tail={tail}"
            results.append(result)
            row = {
                **asdict(setting),
                **asdict(result),
                "sample_id": f"videodpo_pair{setting.pair_index:06d}_{model}_{mask_mode}",
                "source_dataset": "videodpo",
                "canonical_win_frames_dir": str(video_dir),
                "mask_path": str(mask_dir),
                "mask_mode": mask_mode,
                "mask_convention": "png_255_inpaint_region_0_keep_region; training full_mask_value remains internal 0.0",
                "comp": mask_mode == "partial",
                "mask_area_ratio": 1.0 if mask_mode == "full" else partial_meta["mask_area_ratio"],
                "mask_bbox": "full" if mask_mode == "full" else partial_meta["mask_bboxes"],
            }
            manifest_rows.append(row)

    report_path = Path(args.report_path) if args.report_path else output_root / "report.md"
    manifest_path = Path(args.manifest_path) if args.manifest_path else output_root / "smoke_manifest.jsonl"
    write_report(report_path, setting, results, args.run_generation)
    write_jsonl(manifest_path, manifest_rows)

    print_stdout_summary(
        setting=setting,
        results=results,
        report_path=report_path,
        manifest_path=manifest_path,
        run_generation=args.run_generation,
        log_tail_lines=args.print_log_tail_lines,
    )
    failed = [r for r in results if args.run_generation and r.status != "OK"]
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
