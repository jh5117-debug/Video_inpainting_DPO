#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

EXP="exp10_region_local_dpo_s1s2_2000_davis_pai"
RUN_VERSION="${RUN_VERSION:-$(date +%Y%m%d_%H%M%S)_exp10_fresh_d3n16_val24}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG:-logs/pipelines/${EXP}_${RUN_VERSION}_fresh_gpus0_6_${TS}.log}"
PID_FILE="${PID_FILE:-logs/pipelines/${EXP}_fresh_gpus0_6.pid}"

mkdir -p logs/pipelines .tmp

echo "===== PAI EXP10 FRESH LAUNCH ====="
date
echo "RUN_VERSION=${RUN_VERSION}"
echo "LOG=${LOG}"

echo "===== GPU BEFORE ====="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits || true

RUN_EXPERIMENTS=exp10 \
RUN_VERSION="${RUN_VERSION}" \
PROJECT_ROOT="${PROJECT_ROOT}" \
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}" \
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-/mnt/nas/hj/H20_Video_inpainting_DPO/experiments}" \
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/mnt/nas/hj/conda_envs/diffueraser}" \
PROPAINTER_WEIGHT_ROOT="${PROPAINTER_WEIGHT_ROOT:-/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter}" \
RAFT_MODEL_PATH="${RAFT_MODEL_PATH:-/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter/raft-things.pth}" \
LINGBOT_PROCESS_NAME="${LINGBOT_PROCESS_NAME:-lingbot-worldmodel}" \
PROCESS_TITLE="${PROCESS_TITLE:-lingbot-worldmodel}" \
DPO_STAGE1_ENTRYPOINT="${DPO_STAGE1_ENTRYPOINT:-training/dpo/lingbot-worldmodel-stage1.py}" \
DPO_STAGE2_ENTRYPOINT="${DPO_STAGE2_ENTRYPOINT:-training/dpo/lingbot-worldmodel-stage2.py}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}" \
NUM_GPUS="${NUM_GPUS:-7}" \
EVAL_GPU="${EVAL_GPU:-0}" \
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29610}" \
MIXED_PRECISION="${MIXED_PRECISION:-bf16}" \
POLICY_DTYPE="${POLICY_DTYPE:-auto}" \
VAE_DTYPE="${VAE_DTYPE:-fp32}" \
REF_DTYPE="${REF_DTYPE:-bf16}" \
TEXT_DTYPE="${TEXT_DTYPE:-bf16}" \
SPLIT_POS_NEG_FORWARD="${SPLIT_POS_NEG_FORWARD:-1}" \
RESUME_FROM_CHECKPOINT=none \
POLICY_INIT_PATH= \
CKPT_STEPS="${CKPT_STEPS:-500}" \
CKPT_LIMIT="${CKPT_LIMIT:-3}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
setsid nohup bash scripts/launch_exp09_10_11_pai.sh > "${LOG}" 2>&1 < /dev/null &

echo $! > "${PID_FILE}"
sleep "${POST_LAUNCH_SLEEP:-35}"

echo "===== LAUNCHED ====="
echo "PID=$(cat "${PID_FILE}")"
echo "LOG=${LOG}"
ps -fp "$(cat "${PID_FILE}")" || true

echo "===== GPU AFTER ====="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits || true

echo "===== KEY LOG ====="
grep -a -nE 'Exp10|Stage1|Total optimization steps|Epoch|global_step=|dpo_diag|Traceback|FAILED|ERROR|OutOfMemory|SIGFPE|SIGTERM|Signal 15' "${LOG}" \
  | tail -80 || true
