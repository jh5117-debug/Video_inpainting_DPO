#!/bin/bash
set -euo pipefail

# Prepared only. Do not run automatically.
# Purpose: train DPO Stage1 checkpoints for later DPO-S1 + frozen SFT-S2
# hybrid evaluation. This script does not launch Stage2 and does not run
# full VBench.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

export PROJECT_ROOT
export OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}"
export EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-/mnt/nas/hj/H20_Video_inpainting_DPO/experiments}"
export WEIGHTS_DIR="${WEIGHTS_DIR:-/mnt/nas/hj/weights}"
export CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/mnt/nas/hj/conda_envs/diffueraser}"
export CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_PREFIX}}"

export EXP_NAME="exp7_pm_stage1only_ckptsweep_wingap_lose025_beta10"
export RUN_NAME="${EXP_NAME}_stage1"
export RUN_VERSION="${RUN_VERSION:-$(date +%Y%m%d_%H%M%S)}"
export PREFERENCE_MANIFEST="/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4/manifests/selected_primary_comp.repaired.jsonl"

export DPO_DATASET_TYPE="generated_loser_manifest"
export DPO_DATA_ROOT="${DPO_DATA_ROOT:-${OUTPUT_ROOT}/DPO_Finetune_data}"
export TRAIN_MASK_MODE="partial"
export MASK_FROM_MANIFEST="true"
export LOSS_REGION_MODE="full"

export BETA_DPO="10"
export WINNER_ABS_REG_WEIGHT="0.05"
export WINNER_GAP_REG_WEIGHT="1.0"
export WINNER_GAP_REG_MARGIN="0.0"
export WINNER_GAP_REG_MODE="relu"
export LOSE_GAP_WEIGHT="0.25"
export DPO_LOSE_GAP_WEIGHT="${LOSE_GAP_WEIGHT}"
export SFT_REG_WEIGHT="0.0"

export STAGE1_MAX_STEPS="3000"
export MAX_STEPS="${STAGE1_MAX_STEPS}"
export CKPT_STEPS="500"
export CKPT_LIMIT="10"
export VAL_STEPS="999999"

export NUM_GPUS="${NUM_GPUS:-8}"
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

mkdir -p "${OUTPUT_ROOT}/logs/train/${EXP_NAME}" "${OUTPUT_ROOT}/logs/pipelines"

echo "[exp7-stage1only] EXP_NAME=${EXP_NAME}"
echo "[exp7-stage1only] RUN_VERSION=${RUN_VERSION}"
echo "[exp7-stage1only] Stage1 only; no DPO Stage2 and no full VBench will be launched."
echo "[exp7-stage1only] train_mask_mode=${TRAIN_MASK_MODE} mask_from_manifest=${MASK_FROM_MANIFEST}"
echo "[exp7-stage1only] max_steps=${MAX_STEPS} ckpt_steps=${CKPT_STEPS} ckpt_limit=${CKPT_LIMIT}"

exec bash "${PROJECT_ROOT}/training/dpo/scripts/03_dpo_stage1.sbatch" \
  > "${OUTPUT_ROOT}/logs/train/${EXP_NAME}/stage1.log" 2>&1
