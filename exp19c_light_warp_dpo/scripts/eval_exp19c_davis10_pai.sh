#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
if [[ ! -d "${ROOT}" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${ROOT}"

PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}"
DAVIS_ROOT="${DAVIS_ROOT:-/mnt/workspace/hj/nas_hj/data/external/davis_432_240}"
SAVE_PATH="${SAVE_PATH:-${OUTPUT_ROOT}/logs/target_eval/exp19c_light_warp_davis10}"
FLOW_CACHE="${FLOW_CACHE:-${OUTPUT_ROOT}/data/cache/exp19_davis10_propainter_completed_flow}"
BASE_MODEL="${BASE_MODEL:-/mnt/nas/hj/weights/stable-diffusion-v1-5}"
VAE="${VAE:-/mnt/nas/hj/weights/sd-vae-ft-mse}"
SFT48000="${SFT48000:-/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000}"
EXP11_STAGE2="${EXP11_STAGE2:-${OUTPUT_ROOT}/experiments/dpo/stage2/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s2_2000_davis_pai/last_weights}"
EXP19B="${EXP19B:-${OUTPUT_ROOT}/experiments/dpo/stage2/exp19b_boundary_flow_adapter_s2_500_limit100/last_weights/flow_adapter.pt}"
RUN_ROOT="${RUN_ROOT:-${OUTPUT_ROOT}/experiments/dpo/stage2/exp19c_light_warp_dpo}"
PROP="${PROP:-/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter}"
PCM="${PCM:-/mnt/nas/hj/weights/PCM_Weights}"
RAFT="${RAFT:-${PROP}/raft-things.pth}"
GPU="${EXP19C_EVAL_GPU:-0}"

mkdir -p "${SAVE_PATH}" logs/pipelines reports

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" exp19c_light_warp_dpo/code/eval_exp19c_davis10.py \
  --video_root "${DAVIS_ROOT}/JPEGImages_432_240" \
  --mask_root "${DAVIS_ROOT}/test_masks" \
  --gt_root "${DAVIS_ROOT}/JPEGImages_432_240" \
  --save_path "${SAVE_PATH}" \
  --flow_cache_root "${FLOW_CACHE}" \
  --base_model_path "${BASE_MODEL}" \
  --vae_path "${VAE}" \
  --sft_weights "${SFT48000}" \
  --exp11_weights "${EXP11_STAGE2}" \
  --exp19b_adapter "${EXP19B}" \
  --lambda000_adapter "${RUN_ROOT}/lambda000/last_weights/flow_adapter.pt" \
  --lambda005_adapter "${RUN_ROOT}/lambda005/last_weights/flow_adapter.pt" \
  --lambda010_adapter "${RUN_ROOT}/lambda010/last_weights/flow_adapter.pt" \
  --lambda020_adapter "${RUN_ROOT}/lambda020/last_weights/flow_adapter.pt" \
  --propainter_model_dir "${PROP}" \
  --pcm_weights_path "${PCM}" \
  --input_size 432x240 \
  --video_length 24 \
  --nframes 22 \
  --num_inference_steps 6 \
  --limit_videos 10 \
  --seed 1234 \
  --device cuda \
  --compute_lpips \
  --compute_ewarp \
  --raft_model_path "${RAFT}"
