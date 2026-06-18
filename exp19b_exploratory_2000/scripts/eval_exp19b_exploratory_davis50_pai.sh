#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
if [[ ! -d "${ROOT}" ]]; then
  ROOT="/mnt/nas/hj/H20_Video_inpainting_DPO"
fi
cd "${ROOT}"

export EXP19_CODE_ROOT="${EXP19_CODE_ROOT:-${ROOT}}"
export PYTHONPATH="${ROOT}:${EXP19_CODE_ROOT}:${PYTHONPATH:-}"

PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/mnt/nas/hj/weights}"
DAVIS_ROOT="${DAVIS_ROOT:-/mnt/workspace/hj/nas_hj/data/external/davis_432_240}"
VIDEO_ROOT="${VIDEO_ROOT:-${DAVIS_ROOT}/JPEGImages_432_240}"
MASK_ROOT="${MASK_ROOT:-${DAVIS_ROOT}/test_masks}"
GT_ROOT="${GT_ROOT:-${DAVIS_ROOT}/JPEGImages_432_240}"
BASE_MODEL="${BASE_MODEL:-${WEIGHTS_DIR}/stable-diffusion-v1-5}"
VAE="${VAE:-${WEIGHTS_DIR}/sd-vae-ft-mse}"
PROP="${PROP:-/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter}"
PCM="${PCM:-${WEIGHTS_DIR}/PCM_Weights}"
SFT48000="${SFT48000:-/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000}"
EXP11_STAGE2="${EXP11_STAGE2:-${OUTPUT_ROOT}/experiments/dpo/stage2/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s2_2000_davis_pai/last_weights}"
EXP19_ADAPTER="${EXP19_ADAPTER:-${OUTPUT_ROOT}/experiments/dpo/stage2/exp19b_boundary_flow_adapter_s2_2000_exploratory_from500_limit100/last_weights/flow_adapter.pt}"
FLOW_CACHE="${FLOW_CACHE:-${OUTPUT_ROOT}/data/cache/exp19_davis50_propainter_completed_flow}"
OUT="${OUT:-${OUTPUT_ROOT}/logs/target_eval/exp19b_exploratory_s2_2000_davis50}"

INPUT_SIZE="${INPUT_SIZE:-432x240}"
VIDEO_LENGTH="${VIDEO_LENGTH:-24}"
NFRAMES="${NFRAMES:-22}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-6}"
LIMIT_VIDEOS="${LIMIT_VIDEOS:-50}"
SEED="${SEED:-1234}"
DEVICE="${DEVICE:-cuda}"

args=(
  --video_root "${VIDEO_ROOT}"
  --mask_root "${MASK_ROOT}"
  --gt_root "${GT_ROOT}"
  --save_path "${OUT}"
  --flow_cache_root "${FLOW_CACHE}"
  --base_model_path "${BASE_MODEL}"
  --vae_path "${VAE}"
  --sft_weights "${SFT48000}"
  --exp11_weights "${EXP11_STAGE2}"
  --exp19_adapter "${EXP19_ADAPTER}"
  --propainter_model_dir "${PROP}"
  --pcm_weights_path "${PCM}"
  --input_size "${INPUT_SIZE}"
  --video_length "${VIDEO_LENGTH}"
  --nframes "${NFRAMES}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --limit_videos "${LIMIT_VIDEOS}"
  --seed "${SEED}"
  --device "${DEVICE}"
)

[[ "${COMPUTE_LPIPS:-1}" == "1" ]] && args+=(--compute_lpips)
[[ "${COMPUTE_EWARP:-1}" == "1" ]] && args+=(--compute_ewarp --raft_model_path "${RAFT_MODEL_PATH:-${PROP}/raft-things.pth}")
[[ "${COMPUTE_TC:-0}" == "1" ]] && args+=(--compute_tc --tc_model_path "${TC_MODEL_PATH:-${WEIGHTS_DIR}/open_clip_vit_h14}")
[[ "${COMPUTE_VFID:-0}" == "1" ]] && args+=(--compute_vfid --i3d_model_path "${I3D_MODEL_PATH:-${ROOT}/weights/i3d_rgb_imagenet.pt}")
[[ "${EXP19_SKIP_PREFLIGHT:-1}" == "1" ]] && args+=(--skip_preflight)

mkdir -p "$(dirname "${OUT}")" logs/pipelines reports
"${PY}" exp19_boundary_gated_flow_adapter_dpo/code/infer_exp19_flow_adapter_davis.py "${args[@]}"

cp "${OUT}/metrics/summary.csv" reports/exp19b_exploratory_2000_davis50_metric_summary.csv
cp "${OUT}/metrics/per_video_metrics.csv" reports/exp19b_exploratory_2000_davis50_per_video_metrics.csv
