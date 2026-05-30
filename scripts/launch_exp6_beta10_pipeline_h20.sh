#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

export PROJECT_ROOT
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}}"
export EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${PROJECT_ROOT}/experiments}"
export WEIGHTS_DIR="${WEIGHTS_DIR:-${PROJECT_ROOT}/weights}"
if [[ -z "${CONDA_ENV_PREFIX:-}" && -d "/home/nvme01/conda_envs/diffueraser" ]]; then
  export CONDA_ENV_PREFIX="/home/nvme01/conda_envs/diffueraser"
fi
export CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_PREFIX:-diffueraser}}"
export VBENCH_CONDA_ENV="${VBENCH_CONDA_ENV:-${CONDA_ENV}}"

export EXP_NAME="${EXP_NAME:-exp6_d2_nocomp_k4_beta10_s1s2_4000}"
DEFAULT_PAI_MANIFEST="/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4/manifests/selected_primary_nocomp.repaired.jsonl"
DEFAULT_H20_MANIFEST="${PROJECT_ROOT}/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4/manifests/selected_primary_nocomp.h20.jsonl"
DEFAULT_H20_REPAIRED="${PROJECT_ROOT}/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4/manifests/selected_primary_nocomp.repaired.jsonl"
if [[ -z "${PREFERENCE_MANIFEST:-}" ]]; then
  if [[ -f "${DEFAULT_H20_MANIFEST}" ]]; then
    export PREFERENCE_MANIFEST="${DEFAULT_H20_MANIFEST}"
  elif [[ -f "${DEFAULT_H20_REPAIRED}" ]]; then
    export PREFERENCE_MANIFEST="${DEFAULT_H20_REPAIRED}"
  else
    export PREFERENCE_MANIFEST="${DEFAULT_PAI_MANIFEST}"
  fi
fi
export TRAIN_MASK_MODE="${TRAIN_MASK_MODE:-full}"
export MASK_FROM_MANIFEST="${MASK_FROM_MANIFEST:-false}"
export LOSS_REGION_MODE="${LOSS_REGION_MODE:-full}"
export BETA_DPO="${BETA_DPO:-10}"
export STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-4000}"
export STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-4000}"
export NUM_GPUS="${NUM_GPUS:-8}"
export VAL_STEPS="${VAL_STEPS:-999999}"
export CKPT_STEPS="${CKPT_STEPS:-1000}"
export CKPT_LIMIT="${CKPT_LIMIT:-2}"
export REPORT_TO="${REPORT_TO:-none}"
export DPO_DIAG_SAVE_WANDB="${DPO_DIAG_SAVE_WANDB:-false}"
export ENABLE_DPO_DIAG="${ENABLE_DPO_DIAG:-true}"
export LINGBOT_PROCESS_NAME="${LINGBOT_PROCESS_NAME:-lingbot-worldphy}"
export TRAIN_HEIGHT="${TRAIN_HEIGHT:-320}"
export TRAIN_WIDTH="${TRAIN_WIDTH:-512}"
export RESOLUTION="${RESOLUTION:-512}"
export NFRAMES="${NFRAMES:-16}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export PROMPTS_FILE="${PROMPTS_FILE:-${PROJECT_ROOT}/external/VideoDPO/prompts/vbench_standard_prompts.txt}"
export QUAL30_SEED="${QUAL30_SEED:-42}"
export SKIP_QUAL30="${SKIP_QUAL30:-false}"
export SKIP_FULL_VBENCH="${SKIP_FULL_VBENCH:-false}"

# H20 can hit SIGFPE with bf16 policy forward/backward. Keep mixed bf16 for
# frozen/support modules, but run the trainable policy path in fp32.
export POLICY_DTYPE="${POLICY_DTYPE:-fp32}"
export VAE_DTYPE="${VAE_DTYPE:-auto}"
export REF_DTYPE="${REF_DTYPE:-auto}"
export TEXT_DTYPE="${TEXT_DTYPE:-auto}"

exec bash "${PROJECT_ROOT}/scripts/run_dpo_two_stage_vbench_pipeline.sh"
