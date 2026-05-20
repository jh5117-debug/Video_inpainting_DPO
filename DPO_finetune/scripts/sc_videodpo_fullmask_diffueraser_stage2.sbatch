#!/bin/bash
#SBATCH --job-name=VDPO_DE_S2
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=48:00:00
#SBATCH --output=logs/videodpo-fullmask-diffueraser-stage2-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="${PROJECT_NAME:-Video_inpainting_DPO}"
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
export PROJECT_ROOT

export WORLDMODELPHY_PROCESS_NAME="${WORLDMODELPHY_PROCESS_NAME:-lingbotworld-phy}"
export PROCESS_TITLE="${PROCESS_TITLE:-${WORLDMODELPHY_PROCESS_NAME}}"

export PROJECT_HOME="${PROJECT_HOME:-$(dirname "${PROJECT_ROOT}")}"
export PROJECT_DEV="${PROJECT_DEV:-${PROJECT_HOME}}"
export PROJECT_DATA="${PROJECT_DATA:-${PROJECT_ROOT}}"
export DATA_NAME="${DATA_NAME:-Video_inpainting_DPO}"
export DATA="${DATA:-${PROJECT_ROOT}}"

if [[ -z "${WEIGHTS_DIR:-}" ]]; then
  for candidate in \
    "/mnt/nas/hj/weights" \
    "/mnt/nas/hj/hf_fullmask_assets/fullmask_required_assets/weights" \
    "${PROJECT_ROOT}/weights" \
    "${DATA}/weights"; do
    if [[ -d "${candidate}/stable-diffusion-v1-5" && -d "${candidate}/sd-vae-ft-mse" ]]; then
      export WEIGHTS_DIR="${candidate}"
      break
    fi
  done
  export WEIGHTS_DIR="${WEIGHTS_DIR:-${DATA}/weights}"
fi

export VIDEODPO_REPO="${VIDEODPO_REPO:-${PROJECT_ROOT}/external/VideoDPO}"
export VIDEODPO_DATA_BASE="${VIDEODPO_DATA_BASE:-${PROJECT_ROOT}/data/VideoDPO}"
export DPO_DATA_ROOT="${DPO_DATA_ROOT:-${PROJECT_ROOT}/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml}"
export DPO_DATASET_TYPE="videodpo_fullmask"
export VAL_DATA_DIR="${VAL_DATA_DIR:-${PROJECT_ROOT}/data/external/davis_432_240}"
export EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${PROJECT_ROOT}/experiments}"

if [[ ! -e "${DPO_DATA_ROOT}" ]]; then
  for candidate in \
    "${PROJECT_ROOT}/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.yaml" \
    "${PROJECT_ROOT}/external/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml" \
    "${PROJECT_ROOT}/external/VideoDPO/configs/vc2_dpo/vidpro/train_data.yaml" \
    "${VIDEODPO_REPO}/configs/vc2_dpo/vidpro/train_data.absolute.yaml" \
    "${VIDEODPO_REPO}/configs/vc2_dpo/vidpro/train_data.yaml"; do
    if [[ -f "${candidate}" ]]; then
      export DPO_DATA_ROOT="${candidate}"
      break
    fi
  done
fi

if [[ -z "${BASELINE_UNET_PATH:-}" ]]; then
  for candidate in \
    "/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000" \
    "${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000" \
    "${WEIGHTS_DIR}/diffuEraser/converted_weights_step34000" \
    "${PROJECT_ROOT}/finetune-stage2/converted_weights_step48000" \
    "${PROJECT_ROOT}/finetune-stage2/converted_weights_step34000"; do
    if [[ -f "${candidate}/unet_main/config.json" ]]; then
      export BASELINE_UNET_PATH="${candidate}"
      break
    fi
  done
fi
export BASELINE_UNET_PATH="${BASELINE_UNET_PATH:-${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000}"
export REF_MODEL_PATH="${REF_MODEL_PATH:-${BASELINE_UNET_PATH}}"

if [[ -z "${PRETRAINED_DPO_S1:-}" ]]; then
  if [[ -n "${STAGE1_WEIGHTS_PATH:-}" ]]; then
    export PRETRAINED_DPO_S1="${STAGE1_WEIGHTS_PATH}"
  elif [[ -f "${EXPERIMENTS_DIR}/dpo/stage1/LATEST" ]]; then
    latest_run="$(head -n 1 "${EXPERIMENTS_DIR}/dpo/stage1/LATEST")"
    if [[ -d "${latest_run}/last_weights" ]]; then
      export PRETRAINED_DPO_S1="${latest_run}/last_weights"
    elif [[ -d "${latest_run}/best_weights" ]]; then
      export PRETRAINED_DPO_S1="${latest_run}/best_weights"
    fi
  fi
fi

if [[ -z "${PRETRAINED_DPO_S1:-}" && -d "${EXPERIMENTS_DIR}/dpo/stage1" ]]; then
  latest_weights="$(
    find "${EXPERIMENTS_DIR}/dpo/stage1" -maxdepth 2 -type d \
      \( -name last_weights -o -name best_weights \) -path '*fullmask*' \
      -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
  if [[ -n "${latest_weights}" ]]; then
    export PRETRAINED_DPO_S1="${latest_weights}"
  fi
fi

if [[ -z "${PRETRAINED_DPO_S1:-}" || ! -d "${PRETRAINED_DPO_S1}" ]]; then
  echo "[videodpo-fullmask-stage2][error] Stage1 DPO weights not found." >&2
  echo "Set PRETRAINED_DPO_S1 or STAGE1_WEIGHTS_PATH to a stage1 last_weights/best_weights directory." >&2
  exit 1
fi

export RUN_NAME="${RUN_NAME:-sc-videodpo-fullmask-diffueraser-stage2}"
export MAX_EPOCHS="${MAX_EPOCHS:-10}"
export MAX_STEPS="${MAX_STEPS-}"
export CKPT_STEPS="${CKPT_STEPS:-499}"
export VAL_STEPS="${VAL_STEPS:-999999}"
export BETA_DPO="${BETA_DPO:-5000}"
export NUM_GPUS="${NUM_GPUS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export LR="${LR:-6e-6}"
export RESOLUTION="${RESOLUTION:-512}"
export TRAIN_HEIGHT="${TRAIN_HEIGHT:-320}"
export TRAIN_WIDTH="${TRAIN_WIDTH:-512}"
export NFRAMES="${NFRAMES:-16}"
export MIXED_PRECISION="${MIXED_PRECISION:-no}"
export SPLIT_POS_NEG_FORWARD="${SPLIT_POS_NEG_FORWARD:-0}"
export USE_8BIT_ADAM="${USE_8BIT_ADAM:-0}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
export VIDEODPO_FRAME_STRIDE="${VIDEODPO_FRAME_STRIDE:-1}"
export VIDEODPO_FULL_MASK_VALUE="${VIDEODPO_FULL_MASK_VALUE:-0.0}"
export WANDB_PROJECT="${WANDB_PROJECT:-DPO_Diffueraser}"
export REPORT_TO="${REPORT_TO:-wandb}"

if [[ -z "${CONDA_ENV:-}" ]]; then
  if [[ -d "/mnt/nas/hj/conda_envs/diffueraser" ]]; then
    export CONDA_ENV="/mnt/nas/hj/conda_envs/diffueraser"
  elif [[ -d "$(dirname "${PROJECT_ROOT}")/conda_envs/diffueraser" ]]; then
    export CONDA_ENV="$(dirname "${PROJECT_ROOT}")/conda_envs/diffueraser"
  fi
fi

if [[ ! -d "${VIDEODPO_REPO}" ]]; then
  echo "[videodpo-fullmask-stage2][error] VIDEODPO_REPO not found: ${VIDEODPO_REPO}" >&2
  exit 1
fi
if [[ ! -e "${DPO_DATA_ROOT}" ]]; then
  echo "[videodpo-fullmask-stage2][error] DPO_DATA_ROOT not found: ${DPO_DATA_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${BASELINE_UNET_PATH}/unet_main/config.json" ]]; then
  echo "[videodpo-fullmask-stage2][error] BASELINE_UNET_PATH invalid: ${BASELINE_UNET_PATH}" >&2
  exit 1
fi

echo "[videodpo-fullmask-stage2] project_root=${PROJECT_ROOT}"
echo "[videodpo-fullmask-stage2] process_name=${WORLDMODELPHY_PROCESS_NAME}"
echo "[videodpo-fullmask-stage2] dpo_data_root=${DPO_DATA_ROOT}"
echo "[videodpo-fullmask-stage2] stage1_weights=${PRETRAINED_DPO_S1}"
echo "[videodpo-fullmask-stage2] baseline_unet=${BASELINE_UNET_PATH}"
echo "[videodpo-fullmask-stage2] ref_model=${REF_MODEL_PATH}"
echo "[videodpo-fullmask-stage2] weights_dir=${WEIGHTS_DIR}"
echo "[videodpo-fullmask-stage2] run_name=${RUN_NAME}"
echo "[videodpo-fullmask-stage2] max_epochs=${MAX_EPOCHS} max_steps=${MAX_STEPS:-auto} beta=${BETA_DPO} lr=${LR} grad_accum=${GRAD_ACCUM} train_size=${TRAIN_HEIGHT}x${TRAIN_WIDTH} split_pos_neg=${SPLIT_POS_NEG_FORWARD} report_to=${REPORT_TO}"

exec bash "${PROJECT_ROOT}/DPO_finetune/scripts/03_dpo_stage2.sbatch"
