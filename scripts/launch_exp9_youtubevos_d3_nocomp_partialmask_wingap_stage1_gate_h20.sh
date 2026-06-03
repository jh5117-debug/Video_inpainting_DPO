#!/bin/bash
set -euo pipefail

# H20 Exp9 target-domain D3 nocomp gate.
# Stage1 DPO only. Uses GPUs 0-5 by default. No DPO Stage2 and no VBench.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

export PROJECT_ROOT
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}}"
export EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${PROJECT_ROOT}/experiments}"
export WEIGHTS_DIR="${WEIGHTS_DIR:-${PROJECT_ROOT}/weights}"
if [[ ! -d "${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000" && -d "/home/nvme01/H20_Video_inpainting_DPO/weights" ]]; then
  export WEIGHTS_DIR="/home/nvme01/H20_Video_inpainting_DPO/weights"
fi
if [[ -z "${CONDA_ENV_PREFIX:-}" && -d "/home/nvme01/conda_envs/diffueraser" ]]; then
  export CONDA_ENV_PREFIX="/home/nvme01/conda_envs/diffueraser"
fi
export CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_PREFIX:-diffueraser}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export NUM_GPUS="${NUM_GPUS:-6}"

export EXP_NAME="${EXP_NAME:-exp9_youtubevos_d3_nocomp_partialmask_wingap_lose025_stage1_gate1500_h20}"
export RUN_VERSION="${RUN_VERSION:-$(date +%Y%m%d_%H%M%S)}"

# This launcher is a fixed H20 gate. Default behavior intentionally ignores
# stale shell variables from older runs; use EXP9_H20_ALLOW_CONFIG_OVERRIDE=true
# only for a deliberate custom rerun.
if [[ "${EXP9_H20_ALLOW_CONFIG_OVERRIDE:-false}" == "true" ]]; then
  export RUN_NAME="${RUN_NAME:-${EXP_NAME}_stage1}"
else
  export RUN_NAME="${EXP_NAME}_stage1"
  export STAGE1_MAX_STEPS="1500"
  export MAX_STEPS="1500"
  export CKPT_STEPS="500"
  export CKPT_LIMIT="5"
  export VAL_STEPS="999999"
  export LOGGING_STEPS="50"
fi

D3_ROOT="${D3_ROOT:-${PROJECT_ROOT}/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4}"
REPAIRED_MANIFEST="${D3_ROOT}/manifests/selected_primary_nocomp.repaired.jsonl"
DEFAULT_MANIFEST="${D3_ROOT}/manifests/selected_primary_nocomp.jsonl"
if [[ "${EXP9_H20_ALLOW_MANIFEST_OVERRIDE:-false}" == "true" && -n "${PREFERENCE_MANIFEST:-}" ]]; then
  echo "[exp9-h20-nocomp] using explicit PREFERENCE_MANIFEST override: ${PREFERENCE_MANIFEST}"
else
  if [[ -f "${REPAIRED_MANIFEST}" ]]; then
    export PREFERENCE_MANIFEST="${REPAIRED_MANIFEST}"
  else
    export PREFERENCE_MANIFEST="${DEFAULT_MANIFEST}"
  fi
fi

if [[ ! -f "${PREFERENCE_MANIFEST}" ]]; then
  echo "[exp9-h20-nocomp][ERROR] manifest missing: ${PREFERENCE_MANIFEST}" >&2
  exit 1
fi
if [[ "${PREFERENCE_MANIFEST}" != *"official_videodpo_diffueraser_youtubevos_partialmask_loser_k4"* ]]; then
  echo "[exp9-h20-nocomp][ERROR] Exp9 nocomp must use D3 YouTube-VOS manifest, got: ${PREFERENCE_MANIFEST}" >&2
  exit 1
fi
if grep -m1 -q '/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO' "${PREFERENCE_MANIFEST}"; then
  echo "[exp9-h20-nocomp][ERROR] manifest contains PAI paths; use H20-local paths." >&2
  exit 1
fi

export DPO_DATASET_TYPE="generated_loser_manifest"
export DPO_DATA_ROOT="${DPO_DATA_ROOT:-${PROJECT_ROOT}/data/external/DPO_Finetune_data}"
export TRAIN_MASK_MODE="partial"
export MASK_FROM_MANIFEST="true"
export LOSS_REGION_MODE="full"

export BETA_DPO="${BETA_DPO:-10}"
export WINNER_ABS_REG_WEIGHT="${WINNER_ABS_REG_WEIGHT:-0.05}"
export WINNER_GAP_REG_WEIGHT="${WINNER_GAP_REG_WEIGHT:-1.0}"
export WINNER_GAP_REG_MARGIN="${WINNER_GAP_REG_MARGIN:-0.0}"
export WINNER_GAP_REG_MODE="${WINNER_GAP_REG_MODE:-relu}"
export LOSE_GAP_WEIGHT="${LOSE_GAP_WEIGHT:-0.25}"
export DPO_LOSE_GAP_WEIGHT="${DPO_LOSE_GAP_WEIGHT:-${LOSE_GAP_WEIGHT}}"
export SFT_REG_WEIGHT="${SFT_REG_WEIGHT:-0.0}"

export STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-1500}"
export MAX_STEPS="${MAX_STEPS:-${STAGE1_MAX_STEPS}}"
export CKPT_STEPS="${CKPT_STEPS:-500}"
export CKPT_LIMIT="${CKPT_LIMIT:-5}"
export VAL_STEPS="${VAL_STEPS:-999999}"
export LOGGING_STEPS="${LOGGING_STEPS:-50}"

export REPORT_TO="${REPORT_TO:-none}"
export ENABLE_DPO_DIAG="${ENABLE_DPO_DIAG:-true}"
export DPO_DIAG_SAVE_CSV="${DPO_DIAG_SAVE_CSV:-true}"
export DPO_DIAG_SAVE_WANDB="${DPO_DIAG_SAVE_WANDB:-false}"
export DPO_DIAG_LOG_EVERY="${DPO_DIAG_LOG_EVERY:-10}"
export LINGBOT_PROCESS_NAME="${LINGBOT_PROCESS_NAME:-lingbot-worldphy}"
export TRAIN_HEIGHT="${TRAIN_HEIGHT:-320}"
export TRAIN_WIDTH="${TRAIN_WIDTH:-512}"
export RESOLUTION="${RESOLUTION:-512}"
export NFRAMES="${NFRAMES:-16}"
export NUM_WORKERS="${NUM_WORKERS:-0}"

# H20 has previously hit bf16/SIGFPE paths. Keep the gate conservative.
export MIXED_PRECISION="${MIXED_PRECISION:-no}"
export POLICY_DTYPE="${POLICY_DTYPE:-fp32}"
export VAE_DTYPE="${VAE_DTYPE:-fp32}"
export REF_DTYPE="${REF_DTYPE:-fp32}"
export TEXT_DTYPE="${TEXT_DTYPE:-fp32}"

mkdir -p "${OUTPUT_ROOT}/logs/train/${EXP_NAME}" "${OUTPUT_ROOT}/logs/pipelines" "${OUTPUT_ROOT}/reports"

{
  echo "[exp9-h20-nocomp] EXP_NAME=${EXP_NAME}"
  echo "[exp9-h20-nocomp] RUN_VERSION=${RUN_VERSION}"
  echo "[exp9-h20-nocomp] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NUM_GPUS=${NUM_GPUS}"
  echo "[exp9-h20-nocomp] manifest=${PREFERENCE_MANIFEST}"
  echo "[exp9-h20-nocomp] weights_dir=${WEIGHTS_DIR}"
  echo "[exp9-h20-nocomp] Stage1 only; DPO Stage2 and VBench are disabled."
  echo "[exp9-h20-nocomp] train_mask_mode=${TRAIN_MASK_MODE} mask_from_manifest=${MASK_FROM_MANIFEST}"
  echo "[exp9-h20-nocomp] beta=${BETA_DPO} winner_abs=${WINNER_ABS_REG_WEIGHT} winner_gap=${WINNER_GAP_REG_WEIGHT} lose_gap=${DPO_LOSE_GAP_WEIGHT}"
  echo "[exp9-h20-nocomp] max_steps=${MAX_STEPS} ckpt_steps=${CKPT_STEPS} ckpt_limit=${CKPT_LIMIT}"
  echo "[exp9-h20-nocomp] config_override=${EXP9_H20_ALLOW_CONFIG_OVERRIDE:-false} run_name=${RUN_NAME}"
} | tee "${OUTPUT_ROOT}/logs/pipelines/${EXP_NAME}.log"

exec bash "${PROJECT_ROOT}/training/dpo/scripts/03_dpo_stage1.sbatch" \
  > "${OUTPUT_ROOT}/logs/train/${EXP_NAME}/stage1.log" 2>&1
