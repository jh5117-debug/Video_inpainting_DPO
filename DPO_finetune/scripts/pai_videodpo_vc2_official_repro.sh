#!/usr/bin/env bash
# PAI launcher for a clean official VideoDPO VC2-DPO reproduction.
#
# Default mode uses the upstream VideoDPO code at the pinned commit and changes
# only the local data/checkpoint paths plus paper-scale training knobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

OFFICIAL_VIDEODPO_REPO="${OFFICIAL_VIDEODPO_REPO:-/mnt/nas/hj/official_repos/VideoDPO_official_1febdb4}"
VC2_DATA_YAML="${VC2_DATA_YAML:-/mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.pai.yaml}"
CONDA_ENV="${CONDA_ENV:-/mnt/nas/hj/conda_envs/videodpo}"

# Keep paper global batch = 8. With all 8 PAI GPUs, GRAD_ACCUM=1 gives 8.
NUM_GPUS="${NUM_GPUS:-8}"
DEVICE_LIST="${DEVICE_LIST:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_OPT_STEPS="${MAX_OPT_STEPS:-3000}"
CKPT_EVERY="${CKPT_EVERY:-500}"
NUM_WORKERS="${NUM_WORKERS:-16}"
BETA_DPO="${BETA_DPO:-5000.0}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"

RUN_NAME="${RUN_NAME:-pai-vc2-dpo-official-clean-step${MAX_OPT_STEPS}-gb8-gpu${DEVICE_LIST//,/}_${SECONDS}_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/videodpo_vc2_dpo_official_clean}"

# STRICT_OFFICIAL=1 keeps the upstream code unmodified. If the run reaches the
# post-train implicit test and that environment-specific test fails after saving
# checkpoints, rerun with STRICT_OFFICIAL=0. That mode still leaves the DPO loss,
# model, dataset and optimizer untouched, but applies small runtime compatibility
# patches for logging/OpenCLIP/post-train-test behavior.
STRICT_OFFICIAL="${STRICT_OFFICIAL:-1}"
if [[ "${STRICT_OFFICIAL}" == "1" ]]; then
  APPLY_DPO_DIAG_PATCH=0
  CLEAN_DEBUG_PRINT=0
  PATCH_OPENCLIP_BATCH_FIRST=0
  PATCH_LOGGER_COMPAT=0
  PATCH_SKIP_IMPLICIT_TEST=0
  DISABLE_IMAGE_LOGGER=0
else
  APPLY_DPO_DIAG_PATCH=0
  CLEAN_DEBUG_PRINT=1
  PATCH_OPENCLIP_BATCH_FIRST=1
  PATCH_LOGGER_COMPAT=1
  PATCH_SKIP_IMPLICIT_TEST=1
  DISABLE_IMAGE_LOGGER=1
fi

export PROJECT_ROOT OFFICIAL_VIDEODPO_REPO VC2_DATA_YAML CONDA_ENV
bash "${SCRIPT_DIR}/pai_prepare_official_videodpo_clone.sh"

code_status=""
if [[ "${STRICT_OFFICIAL}" == "1" ]]; then
  code_status="$(git -C "${OFFICIAL_VIDEODPO_REPO}" status --short --untracked-files=all | grep -vE '^[?][?] checkpoints/' || true)"
fi
if [[ "${STRICT_OFFICIAL}" == "1" && -n "${code_status}" ]]; then
  echo "[official-repro][error] strict mode requires clean official code before launch." >&2
  printf '%s\n' "${code_status}" >&2
  exit 1
fi

echo "[official-repro] run_name=${RUN_NAME}"
echo "[official-repro] strict_official=${STRICT_OFFICIAL}"
echo "[official-repro] official_repo=${OFFICIAL_VIDEODPO_REPO}"
echo "[official-repro] vc2_data_yaml=${VC2_DATA_YAML}"
echo "[official-repro] num_gpus=${NUM_GPUS} device_list=${DEVICE_LIST} batch_size=${BATCH_SIZE} grad_accum=${GRAD_ACCUM} max_opt_steps=${MAX_OPT_STEPS}"
echo "[official-repro] global_batch=$((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))"
echo "[official-repro] log_root=${LOG_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEVICE_LIST}}" \
PROJECT_ROOT="${PROJECT_ROOT}" \
VIDEODPO_REPO="${OFFICIAL_VIDEODPO_REPO}" \
VC2_DATA_YAML="${VC2_DATA_YAML}" \
LOG_ROOT="${LOG_ROOT}" \
RUN_NAME="${RUN_NAME}" \
CONDA_ENV="${CONDA_ENV}" \
NUM_GPUS="${NUM_GPUS}" \
DEVICE_LIST="${DEVICE_LIST}" \
BATCH_SIZE="${BATCH_SIZE}" \
GRAD_ACCUM="${GRAD_ACCUM}" \
MAX_OPT_STEPS="${MAX_OPT_STEPS}" \
CKPT_EVERY="${CKPT_EVERY}" \
NUM_WORKERS="${NUM_WORKERS}" \
BETA_DPO="${BETA_DPO}" \
ENABLE_WANDB="${ENABLE_WANDB}" \
EARLY_WANDB=0 \
WANDB_START_EVENT=0 \
APPLY_DPO_DIAG_PATCH="${APPLY_DPO_DIAG_PATCH}" \
CLEAN_DEBUG_PRINT="${CLEAN_DEBUG_PRINT}" \
PATCH_OPENCLIP_BATCH_FIRST="${PATCH_OPENCLIP_BATCH_FIRST}" \
PATCH_LOGGER_COMPAT="${PATCH_LOGGER_COMPAT}" \
PATCH_SKIP_IMPLICIT_TEST="${PATCH_SKIP_IMPLICIT_TEST}" \
DISABLE_IMAGE_LOGGER="${DISABLE_IMAGE_LOGGER}" \
GPU_PREFLIGHT="${GPU_PREFLIGHT:-1}" \
WORLDMODELPHY_PROCESS_NAME="${WORLDMODELPHY_PROCESS_NAME:-lingbotworld-phy}" \
PROCESS_TITLE="${PROCESS_TITLE:-lingbotworld-phy}" \
bash "${SCRIPT_DIR}/sc_videodpo_vc2_train.sbatch"
