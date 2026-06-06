#!/bin/bash
set -euo pipefail

# H20 Exp7-fix two-stage run.
# Uses small-mask 15-20% ProPainter-prior generated losers and the same
# regularized full-loss DPO setting used by Exp8a/Exp8c.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

export PROJECT_ROOT
if [[ -z "${OUTPUT_ROOT:-}" ]]; then
  if [[ -d "/home/nvme01/H20_Video_inpainting_DPO" ]]; then
    export OUTPUT_ROOT="/home/nvme01/H20_Video_inpainting_DPO"
  else
    export OUTPUT_ROOT="${PROJECT_ROOT}"
  fi
fi
export EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${OUTPUT_ROOT}/experiments}"
export WEIGHTS_DIR="${WEIGHTS_DIR:-${OUTPUT_ROOT}/weights}"
export CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/home/nvme01/conda_envs/diffueraser}"
export CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_PREFIX}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-7}"
export EXP_NAME="${EXP_NAME:-exp07_fix_smallmask_prior_wingap_lose025_s1s2_2000_h20}"
export RUN_VERSION="${RUN_VERSION:-$(date +%Y%m%d_%H%M%S)}"
export PIPELINE_TS="${PIPELINE_TS:-${RUN_VERSION}}"
export RUN_NAME="${EXP_NAME}"

REG_DIR="${PROJECT_ROOT}/experiment_registry/exp07_fix_smallmask_prior"
DATA_ROOT="${DATA_ROOT:-${OUTPUT_ROOT}/data/generated_losers/exp07_fix_videodpo_smallmask15_20_prior_k4}"
MANIFEST_REPAIRED="${DATA_ROOT}/manifests/selected_primary_comp.repaired.jsonl"
MANIFEST_DEFAULT="${DATA_ROOT}/manifests/selected_primary_comp.jsonl"
if [[ -z "${PREFERENCE_MANIFEST:-}" ]]; then
  if [[ -f "${MANIFEST_REPAIRED}" ]]; then
    export PREFERENCE_MANIFEST="${MANIFEST_REPAIRED}"
  else
    export PREFERENCE_MANIFEST="${MANIFEST_DEFAULT}"
  fi
fi

die() {
  echo "[exp07-fix-s1s2-h20][ERROR] $*" >&2
  exit 1
}

require_path() {
  local path="$1"
  local label="$2"
  [[ -e "$path" ]] || die "${label} not found: ${path}"
}

require_path "${REG_DIR}" "registry"
require_path "${PREFERENCE_MANIFEST}" "smallmask preference manifest"
require_path "${WEIGHTS_DIR}/stable-diffusion-v1-5" "stable-diffusion-v1-5"
require_path "${WEIGHTS_DIR}/sd-vae-ft-mse" "sd-vae-ft-mse"
require_path "${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000" "SFT-48000 DiffuEraser weights"
require_path "${PROJECT_ROOT}/scripts/run_dpo_two_stage_vbench_pipeline.sh" "two-stage pipeline"

if [[ "${PREFERENCE_MANIFEST}" != *"exp07_fix_videodpo_smallmask15_20_prior_k4"* ]]; then
  die "refusing non-smallmask Exp7-fix manifest: ${PREFERENCE_MANIFEST}"
fi
if grep -m1 -q '/mnt/workspace/' "${PREFERENCE_MANIFEST}"; then
  die "manifest contains PAI paths; use H20-local paths: ${PREFERENCE_MANIFEST}"
fi

export DPO_DATASET_TYPE="generated_loser_manifest"
export DPO_DATA_ROOT="${DATA_ROOT}"
export TRAIN_MASK_MODE="partial"
export MASK_FROM_MANIFEST="true"
export LOSS_REGION_MODE="full"

export BETA_DPO="${BETA_DPO:-10}"
export LOSE_GAP_WEIGHT="${LOSE_GAP_WEIGHT:-0.25}"
export DPO_LOSE_GAP_WEIGHT="${DPO_LOSE_GAP_WEIGHT:-${LOSE_GAP_WEIGHT}}"
export WINNER_ABS_REG_WEIGHT="${WINNER_ABS_REG_WEIGHT:-0.05}"
export WINNER_GAP_REG_WEIGHT="${WINNER_GAP_REG_WEIGHT:-1.0}"
export WINNER_GAP_REG_MARGIN="${WINNER_GAP_REG_MARGIN:-0.0}"
export WINNER_GAP_REG_MODE="${WINNER_GAP_REG_MODE:-relu}"
export SFT_REG_WEIGHT="${SFT_REG_WEIGHT:-0.0}"

export STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-2000}"
export STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-2000}"
export MAX_STEPS="${STAGE1_MAX_STEPS}"
export CKPT_STEPS="${CKPT_STEPS:-500}"
export CKPT_LIMIT="${CKPT_LIMIT:-5}"
export VAL_STEPS="${VAL_STEPS:-999999}"
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
export LOGGING_STEPS="${LOGGING_STEPS:-10}"

# H20 SIGFPE-safe profile from PRD: no mixed precision, no split forward, fp32 modules.
export MIXED_PRECISION="${MIXED_PRECISION:-no}"
export POLICY_DTYPE="${POLICY_DTYPE:-fp32}"
export VAE_DTYPE="${VAE_DTYPE:-fp32}"
export REF_DTYPE="${REF_DTYPE:-fp32}"
export TEXT_DTYPE="${TEXT_DTYPE:-fp32}"
export SPLIT_POS_NEG_FORWARD="${SPLIT_POS_NEG_FORWARD:-0}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29571}"

export REF_MODEL_PATH="${REF_MODEL_PATH:-${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000}"
export BASELINE_WEIGHTS_PATH="${BASELINE_WEIGHTS_PATH:-${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000}"
export BASELINE_UNET_PATH="${BASELINE_UNET_PATH:-${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000}"
export VAL_DATA_DIR="${VAL_DATA_DIR:-${OUTPUT_ROOT}/data/external/davis_432_240}"

# Keep the active training job focused on SIGFPE-safe H20 stability. The required
# Exp8-style DAVIS validation is handled by the posthoc watcher script after
# Stage2 finishes:
#   scripts/run_exp07_fix_smallmask_prior_posthoc_davis_val_h20.sh
export SKIP_QUAL30="${SKIP_QUAL30:-true}"
export SKIP_FULL_VBENCH="${SKIP_FULL_VBENCH:-true}"

mkdir -p "${OUTPUT_ROOT}/logs/pipelines" "${OUTPUT_ROOT}/logs/train/${EXP_NAME}" "${OUTPUT_ROOT}/reports"

python_bin="${PYTHON_BIN:-}"
if [[ -z "${python_bin}" ]]; then
  if [[ -x "${CONDA_ENV}/bin/python" ]]; then
    python_bin="${CONDA_ENV}/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    python_bin="$(command -v python3)"
  else
    python_bin="$(command -v python)"
  fi
fi

"${python_bin}" - <<'PY'
import json
import os
from pathlib import Path

manifest = Path(os.environ["PREFERENCE_MANIFEST"])
rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
if not rows:
    raise SystemExit(f"empty manifest: {manifest}")
required = ["win_video_path", "final_loser_video_path", "mask_path"]
for idx, row in enumerate(rows[:16]):
    for key in required:
        value = row.get(key)
        if not value or not Path(value).exists():
            raise SystemExit(f"row {idx} missing/unreadable {key}: {value}")
print(f"[exp07-fix-s1s2-h20] manifest_rows={len(rows)} sampled_paths_checked={min(16, len(rows))}")
PY

{
  echo "[exp07-fix-s1s2-h20] EXP_NAME=${EXP_NAME}"
  echo "[exp07-fix-s1s2-h20] RUN_VERSION=${RUN_VERSION}"
  echo "[exp07-fix-s1s2-h20] PROJECT_ROOT=${PROJECT_ROOT}"
  echo "[exp07-fix-s1s2-h20] OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "[exp07-fix-s1s2-h20] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NUM_GPUS=${NUM_GPUS}"
  echo "[exp07-fix-s1s2-h20] manifest=${PREFERENCE_MANIFEST}"
  echo "[exp07-fix-s1s2-h20] weights=${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000"
  echo "[exp07-fix-s1s2-h20] Stage1=${STAGE1_MAX_STEPS} Stage2=${STAGE2_MAX_STEPS}"
  echo "[exp07-fix-s1s2-h20] H20 precision: mixed=${MIXED_PRECISION} policy=${POLICY_DTYPE} vae=${VAE_DTYPE} ref=${REF_DTYPE} text=${TEXT_DTYPE} split=${SPLIT_POS_NEG_FORWARD}"
  echo "[exp07-fix-s1s2-h20] loss beta=${BETA_DPO} lose_gap=${DPO_LOSE_GAP_WEIGHT} winner_abs=${WINNER_ABS_REG_WEIGHT} winner_gap=${WINNER_GAP_REG_WEIGHT}"
} | tee "${OUTPUT_ROOT}/logs/pipelines/${EXP_NAME}_${RUN_VERSION}.precheck.log"

case "${EXP07_S1S2_PRECHECK_ONLY:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    echo "[exp07-fix-s1s2-h20] precheck-only complete"
    exit 0
    ;;
esac

exec bash "${PROJECT_ROOT}/scripts/run_dpo_two_stage_vbench_pipeline.sh"
