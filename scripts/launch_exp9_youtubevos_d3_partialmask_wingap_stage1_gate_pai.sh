#!/bin/bash
set -euo pipefail

# Exp9 target-domain D3 gate.
# Stage1 DPO only. No DPO Stage2, no full VBench, no D2/D3 regeneration.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

export PROJECT_ROOT
export OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}"
export EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${OUTPUT_ROOT}/experiments}"
export WEIGHTS_DIR="${WEIGHTS_DIR:-/mnt/nas/hj/weights}"
export CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/mnt/nas/hj/conda_envs/diffueraser}"
export CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_PREFIX}}"

export EXP_NAME="${EXP_NAME:-exp9_youtubevos_d3_partialmask_wingap_lose025_stage1_gate1500}"
export RUN_VERSION="${RUN_VERSION:-$(date +%Y%m%d_%H%M%S)}"

# This launcher is a fixed gate. Default behavior intentionally ignores stale
# shell variables from older Exp5/Exp7 runs; use EXP9_ALLOW_CONFIG_OVERRIDE=true
# only for a deliberate custom rerun.
if [[ "${EXP9_ALLOW_CONFIG_OVERRIDE:-false}" == "true" ]]; then
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

D3_ROOT="${D3_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4}"
PAI_MANIFEST="${D3_ROOT}/manifests/selected_primary_comp.repaired.pai_paths.jsonl"
DEFAULT_MANIFEST="${D3_ROOT}/manifests/selected_primary_comp.repaired.jsonl"
if [[ "${EXP9_ALLOW_MANIFEST_OVERRIDE:-false}" == "true" && -n "${PREFERENCE_MANIFEST:-}" ]]; then
  echo "[exp9-stage1-gate] using explicit PREFERENCE_MANIFEST override: ${PREFERENCE_MANIFEST}"
else
  if [[ -f "${PAI_MANIFEST}" ]]; then
    export PREFERENCE_MANIFEST="${PAI_MANIFEST}"
  else
    export PREFERENCE_MANIFEST="${DEFAULT_MANIFEST}"
  fi
fi

if [[ ! -f "${PREFERENCE_MANIFEST}" ]]; then
  echo "[exp9-stage1-gate][ERROR] manifest missing: ${PREFERENCE_MANIFEST}" >&2
  exit 1
fi
if [[ "${PREFERENCE_MANIFEST}" != *"official_videodpo_diffueraser_youtubevos_partialmask_loser_k4"* ]]; then
  echo "[exp9-stage1-gate][ERROR] Exp9 must use D3 YouTube-VOS manifest, got: ${PREFERENCE_MANIFEST}" >&2
  echo "[exp9-stage1-gate][ERROR] Unset stale PREFERENCE_MANIFEST or set EXP9_ALLOW_MANIFEST_OVERRIDE=true only with a D3 manifest." >&2
  exit 1
fi
if grep -m1 -q '/home/nvme01/H20_Video_inpainting_DPO' "${PREFERENCE_MANIFEST}"; then
  echo "[exp9-stage1-gate][ERROR] manifest still contains H20 paths; use repaired PAI paths first." >&2
  exit 1
fi

export DPO_DATASET_TYPE="generated_loser_manifest"
export DPO_DATA_ROOT="${DPO_DATA_ROOT:-${OUTPUT_ROOT}/DPO_Finetune_data}"
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

mkdir -p "${OUTPUT_ROOT}/logs/train/${EXP_NAME}" "${OUTPUT_ROOT}/logs/pipelines" "${OUTPUT_ROOT}/reports"

{
  echo "[exp9-stage1-gate] EXP_NAME=${EXP_NAME}"
  echo "[exp9-stage1-gate] RUN_VERSION=${RUN_VERSION}"
  echo "[exp9-stage1-gate] manifest=${PREFERENCE_MANIFEST}"
  echo "[exp9-stage1-gate] Stage1 only; DPO Stage2 and VBench are disabled."
  echo "[exp9-stage1-gate] train_mask_mode=${TRAIN_MASK_MODE} mask_from_manifest=${MASK_FROM_MANIFEST}"
  echo "[exp9-stage1-gate] beta=${BETA_DPO} winner_abs=${WINNER_ABS_REG_WEIGHT} winner_gap=${WINNER_GAP_REG_WEIGHT} lose_gap=${DPO_LOSE_GAP_WEIGHT}"
  echo "[exp9-stage1-gate] max_steps=${MAX_STEPS} ckpt_steps=${CKPT_STEPS} ckpt_limit=${CKPT_LIMIT}"
  echo "[exp9-stage1-gate] config_override=${EXP9_ALLOW_CONFIG_OVERRIDE:-false} run_name=${RUN_NAME}"
} | tee "${OUTPUT_ROOT}/logs/pipelines/${EXP_NAME}.log"

exec bash "${PROJECT_ROOT}/training/dpo/scripts/03_dpo_stage1.sbatch" \
  > "${OUTPUT_ROOT}/logs/train/${EXP_NAME}/stage1.log" 2>&1
