#!/bin/bash
#SBATCH --job-name=VDPO_DE_S1
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=48:00:00
#SBATCH --output=logs/videodpo-fullmask-diffueraser-stage1-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="${PROJECT_NAME:-Video_inpainting_DPO}"
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${PROJECT_DEV:-}" ]]; then
    PROJECT_ROOT="${PROJECT_DEV}/${PROJECT_NAME}"
  else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  fi
fi
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

export PROJECT_HOME="${PROJECT_HOME:-/sc-projects/sc-proj-cc09-repair/hongyou}"
export PROJECT_DEV="${PROJECT_DEV:-${PROJECT_HOME}/dev}"
export PROJECT_DATA="${PROJECT_DATA:-${PROJECT_DEV}/data}"
export DATA_NAME="${DATA_NAME:-Video_inpainting_DPO}"
export DATA="${DATA:-${PROJECT_DATA}/${DATA_NAME}}"

export VIDEODPO_REPO="${VIDEODPO_REPO:-${PROJECT_ROOT}/external/VideoDPO}"
export VIDEODPO_DATA_BASE="${VIDEODPO_DATA_BASE:-${PROJECT_DATA}/VideoDPO}"
export DPO_DATA_ROOT="${DPO_DATA_ROOT:-${PROJECT_DATA}/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml}"
export DPO_DATASET_TYPE="videodpo_fullmask"

# Conceptually keep VideoDPO data/task and swap in the DiffuEraser model side.
# Engineering-wise this uses the opt-in full-mask adapter for VideoDPO pairs.
export RUN_NAME="${RUN_NAME:-sc-videodpo-fullmask-diffueraser-stage1-smoke}"
export MAX_STEPS="${MAX_STEPS:-1000}"
export CKPT_STEPS="${CKPT_STEPS:-500}"
export VAL_STEPS="${VAL_STEPS:-500}"
export LOGGING_STEPS="${LOGGING_STEPS:-1}"
export BETA_DPO="${BETA_DPO:-5000}"
export DPO_LOSE_GAP_WEIGHT="${DPO_LOSE_GAP_WEIGHT:-1.0}"
export NUM_GPUS="${NUM_GPUS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export LR="${LR:-6e-6}"
export RESOLUTION="${RESOLUTION:-512}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export VAE_DTYPE="${VAE_DTYPE:-auto}"
REQUESTED_SPLIT_POS_NEG_FORWARD="${SPLIT_POS_NEG_FORWARD:-0}"
ALLOW_UNSTABLE_FULLMASK_SPLIT="${ALLOW_UNSTABLE_FULLMASK_SPLIT:-0}"
case "${REQUESTED_SPLIT_POS_NEG_FORWARD,,}" in
  1|true|yes|on)
    case "${ALLOW_UNSTABLE_FULLMASK_SPLIT,,}" in
      1|true|yes|on)
        export SPLIT_POS_NEG_FORWARD="${REQUESTED_SPLIT_POS_NEG_FORWARD}"
        ;;
      *)
        echo "[videodpo-fullmask][warn] forcing SPLIT_POS_NEG_FORWARD=0; split forward is a known H20 step-0 SIGFPE path for this fullmask bridge." >&2
        echo "[videodpo-fullmask][warn] set ALLOW_UNSTABLE_FULLMASK_SPLIT=1 only for debugging that unstable path." >&2
        export SPLIT_POS_NEG_FORWARD=0
        ;;
    esac
    ;;
  *)
    export SPLIT_POS_NEG_FORWARD="${REQUESTED_SPLIT_POS_NEG_FORWARD}"
    ;;
esac
export USE_8BIT_ADAM="${USE_8BIT_ADAM:-0}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
export VIDEODPO_FRAME_STRIDE="${VIDEODPO_FRAME_STRIDE:-1}"
export VIDEODPO_FULL_MASK_VALUE="${VIDEODPO_FULL_MASK_VALUE:-0.0}"
export WANDB_PROJECT="${WANDB_PROJECT:-DPO_Diffueraser}"
export REPORT_TO="${REPORT_TO:-wandb}"

if [[ ! -d "${VIDEODPO_REPO}" ]]; then
  echo "[videodpo-fullmask][error] VIDEODPO_REPO not found: ${VIDEODPO_REPO}" >&2
  echo "[videodpo-fullmask][error] Run from ${PROJECT_ROOT}: git submodule update --init --recursive" >&2
  exit 1
fi
if [[ ! -e "${DPO_DATA_ROOT}" ]]; then
  echo "[videodpo-fullmask][error] DPO_DATA_ROOT not found: ${DPO_DATA_ROOT}" >&2
  echo "Set DPO_DATA_ROOT to the VideoDPO train_data.yaml or dataset root on SC." >&2
  exit 1
fi

echo "[videodpo-fullmask] project_root=${PROJECT_ROOT}"
echo "[videodpo-fullmask] videodpo_repo=${VIDEODPO_REPO}"
echo "[videodpo-fullmask] dpo_data_root=${DPO_DATA_ROOT}"
echo "[videodpo-fullmask] run_name=${RUN_NAME}"
echo "[videodpo-fullmask] max_steps=${MAX_STEPS} logging_steps=${LOGGING_STEPS} beta=${BETA_DPO} lr=${LR} grad_accum=${GRAD_ACCUM} split_pos_neg=${SPLIT_POS_NEG_FORWARD} report_to=${REPORT_TO}"

exec bash "${PROJECT_ROOT}/DPO_finetune/scripts/03_dpo_stage1.sbatch"
