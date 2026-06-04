#!/bin/bash
set -euo pipefail

# H20 Exp7-fix main gate.
# Requires small-mask 15-20% ProPainter-prior data. Refuses old D2 paths.

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
export CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/home/nvme01/conda_envs/diffueraser}"
export CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_PREFIX}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export NUM_GPUS="${NUM_GPUS:-6}"
export EXP_NAME="${EXP_NAME:-exp07_fix_smallmask_prior_wingap_lose025_stage1_gate1500_h20}"
export RUN_VERSION="${RUN_VERSION:-$(date +%Y%m%d_%H%M%S)}"
export RUN_NAME="${EXP_NAME}_stage1"

REG_DIR="${PROJECT_ROOT}/experiment_registry/exp07_fix_smallmask_prior"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/generated_losers/exp07_fix_videodpo_smallmask15_20_prior_k4}"
MANIFEST_REPAIRED="${DATA_ROOT}/manifests/selected_primary_comp.repaired.jsonl"
MANIFEST_DEFAULT="${DATA_ROOT}/manifests/selected_primary_comp.jsonl"
if [[ -z "${PREFERENCE_MANIFEST:-}" ]]; then
  if [[ -f "${MANIFEST_REPAIRED}" ]]; then
    export PREFERENCE_MANIFEST="${MANIFEST_REPAIRED}"
  else
    export PREFERENCE_MANIFEST="${MANIFEST_DEFAULT}"
  fi
fi

[[ -d "${REG_DIR}" ]] || { echo "[exp07-fix-h20][ERROR] registry missing: ${REG_DIR}" >&2; exit 1; }
[[ -f "${PREFERENCE_MANIFEST}" ]] || { echo "[exp07-fix-h20][ERROR] smallmask manifest missing: ${PREFERENCE_MANIFEST}" >&2; exit 1; }
if [[ "${PREFERENCE_MANIFEST}" != *"exp07_fix_videodpo_smallmask15_20_prior_k4"* ]]; then
  echo "[exp07-fix-h20][ERROR] refusing non-smallmask Exp7-fix manifest: ${PREFERENCE_MANIFEST}" >&2
  exit 1
fi
if grep -m1 -q '/mnt/workspace/' "${PREFERENCE_MANIFEST}"; then
  echo "[exp07-fix-h20][ERROR] manifest contains PAI paths; use H20-local repaired paths." >&2
  exit 1
fi
if [[ ! -d "${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000" ]]; then
  echo "[exp07-fix-h20][ERROR] missing SFT-48000 DiffuEraser weights under ${WEIGHTS_DIR}" >&2
  exit 1
fi

export DPO_DATASET_TYPE="generated_loser_manifest"
export TRAIN_MASK_MODE="partial"
export MASK_FROM_MANIFEST="true"
export LOSS_REGION_MODE="full"

export BETA_DPO="10"
export LOSE_GAP_WEIGHT="0.25"
export DPO_LOSE_GAP_WEIGHT="0.25"
export WINNER_ABS_REG_WEIGHT="0.05"
export WINNER_GAP_REG_WEIGHT="1.0"
export WINNER_GAP_REG_MARGIN="0.0"
export WINNER_GAP_REG_MODE="relu"
export SFT_REG_WEIGHT="0.0"

export STAGE1_MAX_STEPS="1500"
export MAX_STEPS="1500"
export CKPT_STEPS="500"
export CKPT_LIMIT="5"
export VAL_STEPS="999999"
export REPORT_TO="${REPORT_TO:-none}"
export ENABLE_DPO_DIAG="${ENABLE_DPO_DIAG:-true}"
export DPO_DIAG_SAVE_CSV="${DPO_DIAG_SAVE_CSV:-true}"
export DPO_DIAG_SAVE_WANDB="${DPO_DIAG_SAVE_WANDB:-false}"
export DPO_DIAG_LOG_EVERY="${DPO_DIAG_LOG_EVERY:-10}"
export TRAIN_HEIGHT="${TRAIN_HEIGHT:-320}"
export TRAIN_WIDTH="${TRAIN_WIDTH:-512}"
export RESOLUTION="${RESOLUTION:-512}"
export NFRAMES="${NFRAMES:-16}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export MIXED_PRECISION="${MIXED_PRECISION:-no}"
export POLICY_DTYPE="${POLICY_DTYPE:-fp32}"
export VAE_DTYPE="${VAE_DTYPE:-fp32}"
export REF_DTYPE="${REF_DTYPE:-fp32}"
export TEXT_DTYPE="${TEXT_DTYPE:-fp32}"

mkdir -p "${OUTPUT_ROOT}/logs/train/${EXP_NAME}" "${OUTPUT_ROOT}/logs/pipelines" "${OUTPUT_ROOT}/reports"
{
  echo "[exp07-fix-h20] EXP_NAME=${EXP_NAME}"
  echo "[exp07-fix-h20] RUN_VERSION=${RUN_VERSION}"
  echo "[exp07-fix-h20] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NUM_GPUS=${NUM_GPUS}"
  echo "[exp07-fix-h20] registry=${REG_DIR}"
  echo "[exp07-fix-h20] manifest=${PREFERENCE_MANIFEST}"
  echo "[exp07-fix-h20] weights=${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000"
  echo "[exp07-fix-h20] Stage1 only; no DPO Stage2; no VBench"
  echo "[exp07-fix-h20] train_mask_mode=${TRAIN_MASK_MODE} mask_from_manifest=${MASK_FROM_MANIFEST} loss_region_mode=${LOSS_REGION_MODE}"
  echo "[exp07-fix-h20] beta=${BETA_DPO} winner_abs=${WINNER_ABS_REG_WEIGHT} winner_gap=${WINNER_GAP_REG_WEIGHT} lose_gap=${DPO_LOSE_GAP_WEIGHT}"
  echo "[exp07-fix-h20] max_steps=${MAX_STEPS} ckpt_steps=${CKPT_STEPS} ckpt_limit=${CKPT_LIMIT}"
} | tee "${OUTPUT_ROOT}/logs/pipelines/${EXP_NAME}.log"

exec bash "${PROJECT_ROOT}/training/dpo/scripts/03_dpo_stage1.sbatch" \
  > "${OUTPUT_ROOT}/logs/train/${EXP_NAME}/stage1.log" 2>&1
