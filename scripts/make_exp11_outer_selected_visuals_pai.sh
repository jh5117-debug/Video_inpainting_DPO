#!/usr/bin/env bash
set -euo pipefail

# Build selected qualitative evidence for the current best Exp11 boundary run.
# This script does not train and does not compute paper metrics from mp4. It
# reuses the fixed DAVIS50 raw6 hard-comp frame-wise wrapper only to save
# in-memory hard-composited frames/videos for visual inspection.

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp09_10_11_pai_sync}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/mnt/nas/hj/weights}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/mnt/nas/hj/conda_envs/diffueraser}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PREFIX}/bin/python}"

DAVIS_ROOT="${DAVIS_ROOT:-/mnt/workspace/hj/nas_hj/data/external/davis_432_240}"
DAVIS_VIDEO_ROOT="${DAVIS_VIDEO_ROOT:-${DAVIS_ROOT}/JPEGImages_432_240}"
DAVIS_MASK_ROOT="${DAVIS_MASK_ROOT:-${DAVIS_ROOT}/test_masks}"
DAVIS_GT_ROOT="${DAVIS_GT_ROOT:-${DAVIS_ROOT}/JPEGImages_432_240}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-${WEIGHTS_DIR}/stable-diffusion-v1-5}"
VAE_PATH="${VAE_PATH:-${WEIGHTS_DIR}/sd-vae-ft-mse}"
DIFFUERASER_WEIGHT_ROOT="${DIFFUERASER_WEIGHT_ROOT:-/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000}"
EXP11_OUTER_B075_S2_WEIGHTS="${EXP11_OUTER_B075_S2_WEIGHTS:-${OUTPUT_ROOT}/experiments/dpo/stage2/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s2_2000_davis_pai/last_weights}"
PROPAINTER_WEIGHT_ROOT="${PROPAINTER_WEIGHT_ROOT:-/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter}"
PCM_WEIGHTS_PATH="${PCM_WEIGHTS_PATH:-${WEIGHTS_DIR}/PCM_Weights}"
RAFT_MODEL_PATH="${RAFT_MODEL_PATH:-${PROPAINTER_WEIGHT_ROOT}/raft-things.pth}"

EVAL_GPU="${EVAL_GPU:-0}"
EVAL_WIDTH="${EVAL_WIDTH:-432}"
EVAL_HEIGHT="${EVAL_HEIGHT:-240}"
DAVIS_VIDEO_LENGTH="${DAVIS_VIDEO_LENGTH:-24}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-6}"
USE_PCM="${USE_PCM:-false}"
MASK_DILATION="${MASK_DILATION:-0}"

RUN_TAG="${RUN_TAG:-20260615_exp11_outer_b075_s2_selected_visuals}"
OUT_ROOT="${OUT_ROOT:-${OUTPUT_ROOT}/logs/target_eval/${RUN_TAG}}"
SELECTED_ROOT="${OUT_ROOT}/selected_davis"
VIS_ROOT="${OUT_ROOT}/visual_evidence"
LOG_DIR="${PROJECT_ROOT}/logs/pipelines"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${RUN_TAG}.log}"

CANDIDATES_DEFAULT="dog-agility,blackswan,lucia,soccerball,rhino,dance-jump,flamingo,boat"
CANDIDATES="${CANDIDATES:-${CANDIDATES_DEFAULT}}"

require_path() {
  local path="$1"
  local label="$2"
  [[ -e "${path}" ]] || { echo "missing ${label}: ${path}" >&2; exit 1; }
}

build_selected_roots() {
  rm -rf "${SELECTED_ROOT}"
  mkdir -p "${SELECTED_ROOT}/JPEGImages_432_240" "${SELECTED_ROOT}/test_masks"
  IFS=',' read -r -a names <<< "${CANDIDATES}"
  for name in "${names[@]}"; do
    name="${name// /}"
    require_path "${DAVIS_VIDEO_ROOT}/${name}" "DAVIS video ${name}"
    require_path "${DAVIS_MASK_ROOT}/${name}" "DAVIS mask ${name}"
    ln -s "${DAVIS_VIDEO_ROOT}/${name}" "${SELECTED_ROOT}/JPEGImages_432_240/${name}"
    ln -s "${DAVIS_MASK_ROOT}/${name}" "${SELECTED_ROOT}/test_masks/${name}"
  done
}

run_framewise_visual_eval() {
  local label="$1"
  local weights="$2"
  local save_path="$3"
  mkdir -p "${save_path}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON_BIN}" tools/run_davis50_framewise_protocol_eval.py \
    --video_root "${SELECTED_ROOT}/JPEGImages_432_240" \
    --mask_root "${SELECTED_ROOT}/test_masks" \
    --gt_root "${SELECTED_ROOT}/JPEGImages_432_240" \
    --base_model_path "${BASE_MODEL_PATH}" \
    --vae_path "${VAE_PATH}" \
    --propainter_model_dir "${PROPAINTER_WEIGHT_ROOT}" \
    --pcm_weights_path "${PCM_WEIGHTS_PATH}" \
    --input_size "${EVAL_WIDTH}x${EVAL_HEIGHT}" \
    --video_length "${DAVIS_VIDEO_LENGTH}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --use_pcm "${USE_PCM}" \
    --mask_dilation_iter "${MASK_DILATION}" \
    --raft_model_path "${RAFT_MODEL_PATH}" \
    --label "${label}" \
    --diffueraser_path "${weights}" \
    --save_path "${save_path}" \
    --save_videos \
    --save_comp_frames
}

compose_evidence() {
  "${PYTHON_BIN}" - "${SELECTED_ROOT}" "${OUT_ROOT}/framewise_metric/DiffuEraser-base" "${OUT_ROOT}/framewise_metric/Exp11_boundary_outer_b075_S2" "${VIS_ROOT}" "${CANDIDATES}" <<'PY'
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

selected_root = Path(sys.argv[1])
base_root = Path(sys.argv[2])
exp_root = Path(sys.argv[3])
out_root = Path(sys.argv[4])
candidates = [x.strip() for x in sys.argv[5].split(",") if x.strip()]

video_root = selected_root / "JPEGImages_432_240"
mask_root = selected_root / "test_masks"
side_dir = out_root / "side_by_side"
sheet_dir = out_root / "frame_contact_sheets"
frame_dir = out_root / "selected_frames"
for d in (side_dir, sheet_dir, frame_dir):
    d.mkdir(parents=True, exist_ok=True)


def read_rgb(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_sequence(root: Path, suffixes=(".png", ".jpg", ".jpeg")):
    files = sorted([p for p in root.iterdir() if p.suffix.lower() in suffixes])
    return [read_rgb(p) for p in files]


def resize_like(img, ref):
    h, w = ref.shape[:2]
    if img.shape[:2] == (h, w):
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def label(img, text):
    out = img.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def mask_overlay(gt, mask):
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = cv2.resize(mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    red = gt.copy()
    red[..., 0] = np.maximum(red[..., 0], 230)
    red[..., 1] = (red[..., 1] * 0.35).astype(np.uint8)
    red[..., 2] = (red[..., 2] * 0.35).astype(np.uint8)
    m = mask > 0
    out = gt.copy()
    out[m] = (0.55 * gt[m] + 0.45 * red[m]).astype(np.uint8)
    return out


def write_video(path, frames, fps=12):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


rows = []
for name in candidates:
    gt = load_sequence(video_root / name)[:24]
    masks = load_sequence(mask_root / name)[:24]
    base = load_sequence(base_root / name / "diffueraser_comp_frames")[:24]
    exp = load_sequence(exp_root / name / "diffueraser_comp_frames")[:24]
    n = min(len(gt), len(masks), len(base), len(exp))
    if n == 0:
        raise RuntimeError(f"no frames for {name}")
    gt, masks, base, exp = gt[:n], masks[:n], base[:n], exp[:n]
    frames = []
    chosen = sorted(set([0, min(6, n - 1), min(12, n - 1), min(18, n - 1), n - 1]))
    sheet_frames = []
    (frame_dir / name).mkdir(parents=True, exist_ok=True)
    for i, (g, m, b, e) in enumerate(zip(gt, masks, base, exp)):
        b = resize_like(b, g)
        e = resize_like(e, g)
        mo = mask_overlay(g, m)
        panel = np.concatenate([
            label(g, "GT"),
            label(mo, "mask overlay"),
            label(b, "SFT-48000"),
            label(e, "Exp11 outer b0.75 S2"),
        ], axis=1)
        frames.append(panel)
        if i in chosen:
            sheet_frames.append(panel)
            cv2.imwrite(str(frame_dir / name / f"{i:03d}.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    write_video(side_dir / f"{name}.mp4", frames)
    sheet = np.concatenate(sheet_frames, axis=0)
    cv2.imwrite(str(sheet_dir / f"{name}_frames.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    rows.append({"video": name, "side_by_side": str(side_dir / f"{name}.mp4"), "contact_sheet": str(sheet_dir / f"{name}_frames.jpg")})

with (out_root / "visual_evidence_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["video", "side_by_side", "contact_sheet"])
    writer.writeheader()
    writer.writerows(rows)

print(f"[visual-evidence] wrote {out_root}")
PY
}

main() {
  cd "${PROJECT_ROOT}"
  mkdir -p "${LOG_DIR}" "${OUT_ROOT}"
  require_path tools/run_davis50_framewise_protocol_eval.py "framewise eval wrapper"
  require_path "${DAVIS_VIDEO_ROOT}" "DAVIS video root"
  require_path "${DAVIS_MASK_ROOT}" "DAVIS mask root"
  require_path "${BASE_MODEL_PATH}" "base model"
  require_path "${VAE_PATH}" "vae"
  require_path "${DIFFUERASER_WEIGHT_ROOT}/unet_main/config.json" "SFT48000 unet"
  require_path "${EXP11_OUTER_B075_S2_WEIGHTS}/unet_main/config.json" "Exp11 outer S2 unet"
  require_path "${PROPAINTER_WEIGHT_ROOT}" "ProPainter weights"
  build_selected_roots
  run_framewise_visual_eval "DiffuEraser-base" "${DIFFUERASER_WEIGHT_ROOT}" "${OUT_ROOT}/framewise_metric/DiffuEraser-base"
  run_framewise_visual_eval "Exp11_boundary_outer_b075_S2" "${EXP11_OUTER_B075_S2_WEIGHTS}" "${OUT_ROOT}/framewise_metric/Exp11_boundary_outer_b075_S2"
  compose_evidence
  cat > "${OUT_ROOT}/README.md" <<EOF
# Exp11 outer b0.75 S2 selected visual evidence

Protocol: DAVIS selected videos, raw6, hard-comp, D+G off, no PCM, no mask dilation.

Candidates: ${CANDIDATES}

Outputs:
- metrics: \`${OUT_ROOT}/framewise_metric/*/metrics\`
- side-by-side videos: \`${VIS_ROOT}/side_by_side\`
- frame contact sheets: \`${VIS_ROOT}/frame_contact_sheets\`
- selected frame panels: \`${VIS_ROOT}/selected_frames\`
EOF
  echo "[done] ${OUT_ROOT}"
}

main "$@" 2>&1 | tee "${LOG_PATH}"
