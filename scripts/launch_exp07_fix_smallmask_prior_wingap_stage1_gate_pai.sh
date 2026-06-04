#!/bin/bash
set -euo pipefail

# PAI Exp7-fix gate.
# Requires pre-generated small-mask 15-20% data with ProPainter-prior DiffuEraser
# generation. This launcher refuses to run on the old D2 manifest.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

export PROJECT_ROOT
export OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
export EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${OUTPUT_ROOT}/experiments}"
export WEIGHTS_DIR="${WEIGHTS_DIR:-/mnt/workspace/hj/nas_hj/weights}"
export CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/mnt/nas/hj/conda_envs/diffueraser}"
export CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_PREFIX}}"

export EXP_NAME="${EXP_NAME:-exp07_fix_smallmask_prior_wingap_lose025_stage1_gate1500}"
export RUN_VERSION="${RUN_VERSION:-$(date +%Y%m%d_%H%M%S)}"
export RUN_NAME="${EXP_NAME}_stage1"

REG_DIR="${PROJECT_ROOT}/experiment_registry/exp07_fix_smallmask_prior"
DATA_ROOT="${DATA_ROOT:-${OUTPUT_ROOT}/data/generated_losers/exp07_fix_videodpo_smallmask15_20_prior_k4}"
MANIFEST_DEFAULT="${DATA_ROOT}/manifests/selected_primary_comp.repaired.jsonl"
export PREFERENCE_MANIFEST="${PREFERENCE_MANIFEST:-${MANIFEST_DEFAULT}}"

[[ -d "${REG_DIR}" ]] || { echo "[exp07-fix-pai][ERROR] registry missing: ${REG_DIR}" >&2; exit 1; }
[[ -f "${PREFERENCE_MANIFEST}" ]] || { echo "[exp07-fix-pai][ERROR] smallmask manifest missing: ${PREFERENCE_MANIFEST}" >&2; exit 1; }

if [[ "${PREFERENCE_MANIFEST}" != *"exp07_fix_videodpo_smallmask15_20_prior_k4"* ]]; then
  echo "[exp07-fix-pai][ERROR] refusing non-smallmask Exp7-fix manifest: ${PREFERENCE_MANIFEST}" >&2
  exit 1
fi
if grep -m1 -q '/home/nvme01/' "${PREFERENCE_MANIFEST}"; then
  echo "[exp07-fix-pai][ERROR] manifest contains H20 paths; use PAI-local repaired paths." >&2
  exit 1
fi
if [[ ! -d "${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000" ]]; then
  echo "[exp07-fix-pai][ERROR] missing YouTube-VOS SFT-48000 DiffuEraser weights: ${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000" >&2
  exit 1
fi

export DPO_DATASET_TYPE="generated_loser_manifest"
export TRAIN_MASK_MODE="partial"
export MASK_FROM_MANIFEST="true"
export LOSS_REGION_MODE="full"

export BETA_DPO="10"
export LOSE_GAP_WEIGHT="0.25"
export DPO_LOSE_GAP_WEIGHT="${LOSE_GAP_WEIGHT}"
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
export NUM_GPUS="${NUM_GPUS:-8}"
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

mkdir -p "${OUTPUT_ROOT}/logs/train/${EXP_NAME}" "${OUTPUT_ROOT}/logs/pipelines" "${OUTPUT_ROOT}/reports"
{
  echo "[exp07-fix-pai] EXP_NAME=${EXP_NAME}"
  echo "[exp07-fix-pai] RUN_VERSION=${RUN_VERSION}"
  echo "[exp07-fix-pai] registry=${REG_DIR}"
  echo "[exp07-fix-pai] manifest=${PREFERENCE_MANIFEST}"
  echo "[exp07-fix-pai] weights=${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000"
  echo "[exp07-fix-pai] Stage1 only; no DPO Stage2; no VBench"
  echo "[exp07-fix-pai] train_mask_mode=${TRAIN_MASK_MODE} mask_from_manifest=${MASK_FROM_MANIFEST} loss_region_mode=${LOSS_REGION_MODE}"
  echo "[exp07-fix-pai] beta=${BETA_DPO} winner_abs=${WINNER_ABS_REG_WEIGHT} winner_gap=${WINNER_GAP_REG_WEIGHT} lose_gap=${DPO_LOSE_GAP_WEIGHT}"
  echo "[exp07-fix-pai] max_steps=${MAX_STEPS} ckpt_steps=${CKPT_STEPS} ckpt_limit=${CKPT_LIMIT}"
} | tee "${OUTPUT_ROOT}/logs/pipelines/${EXP_NAME}.log"

exec bash "${PROJECT_ROOT}/training/dpo/scripts/03_dpo_stage1.sbatch" \
  > "${OUTPUT_ROOT}/logs/train/${EXP_NAME}/stage1.log" 2>&1
