#!/bin/bash
set -euo pipefail

# Posthoc DAVIS validation watcher for the H20 Exp7-fix run.
# The original 20260606_142555 launch used the generic two-stage training
# pipeline and skipped eval. This script restores the intended Exp8-style
# validation contract without interrupting the active Stage2 process:
#   1. wait for Stage2 last_weights
#   2. validate DPO-S1 + SFT-S2 on DAVIS
#   3. validate DPO-S1 + DPO-S2 on DAVIS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

RUN_VERSION="${RUN_VERSION:-20260606_142555}"
EXP_NAME="${EXP_NAME:-exp07_fix_smallmask_prior_wingap_lose025_s1s2_2000_h20}"
STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-${EXP_NAME}_stage1}"
STAGE2_RUN_NAME="${STAGE2_RUN_NAME:-${EXP_NAME}_stage2}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/nvme01/H20_Video_inpainting_DPO}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${PROJECT_ROOT}}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${OUTPUT_ROOT}/experiments}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${OUTPUT_ROOT}/weights}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/home/nvme01/conda_envs/diffueraser}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PREFIX}/bin/python}"

DAVIS_ROOT="${DAVIS_ROOT:-${OUTPUT_ROOT}/data/external/davis_432_240}"
DAVIS_VIDEO_ROOT="${DAVIS_VIDEO_ROOT:-${DAVIS_ROOT}/JPEGImages_432_240}"
DAVIS_MASK_ROOT="${DAVIS_MASK_ROOT:-${DAVIS_ROOT}/test_masks}"
DAVIS_GT_ROOT="${DAVIS_GT_ROOT:-${DAVIS_ROOT}/JPEGImages_432_240}"
DAVIS_VIDEO_LENGTH="${DAVIS_VIDEO_LENGTH:-24}"
DAVIS_NUM_QUAL="${DAVIS_NUM_QUAL:-30}"
EVAL_WIDTH="${EVAL_WIDTH:-432}"
EVAL_HEIGHT="${EVAL_HEIGHT:-240}"
EVAL_GPU="${EVAL_GPU:-1}"
COMPUTE_LPIPS="${COMPUTE_LPIPS:-0}"
COMPUTE_EWARP="${COMPUTE_EWARP:-0}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-${WEIGHTS_DIR}/stable-diffusion-v1-5}"
VAE_PATH="${VAE_PATH:-${WEIGHTS_DIR}/sd-vae-ft-mse}"
DIFFUERASER_WEIGHT_ROOT="${DIFFUERASER_WEIGHT_ROOT:-${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000}"
SFT_STAGE2_WEIGHTS="${SFT_STAGE2_WEIGHTS:-${DIFFUERASER_WEIGHT_ROOT}}"
PCM_WEIGHTS_PATH="${PCM_WEIGHTS_PATH:-${WEIGHTS_DIR}/PCM_Weights}"
PROPAINTER_WEIGHT_ROOT="${PROPAINTER_WEIGHT_ROOT:-}"
RAFT_MODEL_PATH="${RAFT_MODEL_PATH:-}"

WAIT_FOR_STAGE2="${WAIT_FOR_STAGE2:-1}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-0}"

REPORT="${REPORT:-${OUTPUT_ROOT}/reports/exp07_fix_smallmask_prior_posthoc_davis_val_h20_${RUN_VERSION}.md}"
LOG_PREFIX="${LOG_PREFIX:-${OUTPUT_ROOT}/logs/target_eval/exp07_fix_smallmask_prior}"

die() {
  echo "[exp07-posthoc-davis][ERROR] $*" >&2
  exit 1
}

require_path() {
  local path="$1"
  local label="$2"
  [[ -e "${path}" ]] || die "${label} not found: ${path}"
}

bool_on() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    die "python not found; set PYTHON_BIN or CONDA_ENV_PREFIX"
  fi
fi

require_path "${PROJECT_ROOT}/scripts/launch_exp8c_youtubevos_gtwin_d3comp_fullloss_s1s2_2000_davis_pai.sh" "Exp8c DAVIS helper source"
require_path "${PROJECT_ROOT}/tools/build_diffueraser_dpoS1_sftS2_hybrid.py" "hybrid builder"
require_path "${PROJECT_ROOT}/tools/run_inpainting_metric_eval.py" "metric wrapper"
require_path "${PROJECT_ROOT}/inference/run_BR.py" "DiffuEraser inference entrypoint"
require_path "${DAVIS_VIDEO_ROOT}" "DAVIS images"
require_path "${DAVIS_MASK_ROOT}" "DAVIS masks"
require_path "${BASE_MODEL_PATH}" "stable-diffusion-v1-5"
require_path "${VAE_PATH}" "sd-vae-ft-mse"
require_path "${DIFFUERASER_WEIGHT_ROOT}/unet_main/config.json" "DiffuEraser base unet"
require_path "${DIFFUERASER_WEIGHT_ROOT}/brushnet/config.json" "DiffuEraser base brushnet"

STAGE1_RUN_DIR="${STAGE1_RUN_DIR:-${EXPERIMENTS_DIR}/dpo/stage1/${RUN_VERSION}_${STAGE1_RUN_NAME}}"
STAGE2_RUN_DIR="${STAGE2_RUN_DIR:-${EXPERIMENTS_DIR}/dpo/stage2/${RUN_VERSION}_${STAGE2_RUN_NAME}}"
STAGE1_LAST="${STAGE1_LAST:-${STAGE1_RUN_DIR}/last_weights}"
STAGE2_LAST="${STAGE2_LAST:-${STAGE2_RUN_DIR}/last_weights}"
HYBRID_DIR="${HYBRID_DIR:-${EXPERIMENTS_DIR}/hybrid/${RUN_VERSION}_${STAGE1_RUN_NAME}_dpoS1_sftS2}"
STAGE1_VAL_DIR="${STAGE1_VAL_DIR:-${LOG_PREFIX}_stage1_val_davis_${RUN_VERSION}}"
STAGE2_VAL_DIR="${STAGE2_VAL_DIR:-${LOG_PREFIX}_stage2_val_davis_${RUN_VERSION}}"

mkdir -p "$(dirname "${REPORT}")" "${OUTPUT_ROOT}/logs/target_eval"

{
  echo "# Exp7 Fix Small-Mask Prior Posthoc DAVIS Validation"
  echo
  echo "- run_version: \`${RUN_VERSION}\`"
  echo "- project_root: \`${PROJECT_ROOT}\`"
  echo "- output_root: \`${OUTPUT_ROOT}\`"
  echo "- stage1_run_dir: \`${STAGE1_RUN_DIR}\`"
  echo "- stage2_run_dir: \`${STAGE2_RUN_DIR}\`"
  echo "- eval_gpu: \`${EVAL_GPU}\`"
  echo "- davis_video_length: \`${DAVIS_VIDEO_LENGTH}\`"
  echo "- status: watcher_started"
} > "${REPORT}"

require_path "${STAGE1_LAST}/unet_main/config.json" "Stage1 last_weights unet"
require_path "${STAGE1_LAST}/brushnet/config.json" "Stage1 last_weights brushnet"

if bool_on "${WAIT_FOR_STAGE2}"; then
  start_ts="$(date +%s)"
  while [[ ! -f "${STAGE2_LAST}/unet_main/config.json" || ! -f "${STAGE2_LAST}/brushnet/config.json" ]]; do
    now_ts="$(date +%s)"
    waited=$((now_ts - start_ts))
    if [[ "${MAX_WAIT_SECONDS}" != "0" && "${waited}" -ge "${MAX_WAIT_SECONDS}" ]]; then
      die "timed out waiting for Stage2 last_weights after ${waited}s: ${STAGE2_LAST}"
    fi
    echo "[exp07-posthoc-davis] waiting for Stage2 last_weights (${waited}s): ${STAGE2_LAST}"
    sleep "${POLL_SECONDS}"
  done
else
  require_path "${STAGE2_LAST}/unet_main/config.json" "Stage2 last_weights unet"
  require_path "${STAGE2_LAST}/brushnet/config.json" "Stage2 last_weights brushnet"
fi

# Reuse the Exp8c DAVIS helper functions without executing its main().
helper_file="$(mktemp)"
awk '/^run_stage1\\(\\)/ {exit} {print}' \
  "${PROJECT_ROOT}/scripts/launch_exp8c_youtubevos_gtwin_d3comp_fullloss_s1s2_2000_davis_pai.sh" > "${helper_file}"
# shellcheck source=/dev/null
source "${helper_file}"
rm -f "${helper_file}"

echo "[exp07-posthoc-davis] build Stage1 hybrid: ${HYBRID_DIR}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/build_diffueraser_dpoS1_sftS2_hybrid.py" \
  --dpo_stage1_weights "${STAGE1_LAST}" \
  --sft_stage2_weights "${SFT_STAGE2_WEIGHTS}" \
  --output_dir "${HYBRID_DIR}" \
  --strict false \
  --report_path "${OUTPUT_ROOT}/reports/exp07_fix_smallmask_prior_stage1_hybrid_key_merge_report_h20.md"
require_path "${HYBRID_DIR}/last_weights/unet_main/config.json" "Stage1 hybrid full weights"

run_davis_validation "Exp7 Fix Stage1 DPO + SFT Stage2" \
  "DPO-S1_SFT-S2" \
  "${HYBRID_DIR}/last_weights" \
  "${STAGE1_VAL_DIR}"

run_davis_validation "Exp7 Fix Stage1 DPO + Stage2 DPO" \
  "DPO-S1_DPO-S2" \
  "${STAGE2_LAST}" \
  "${STAGE2_VAL_DIR}"

{
  echo
  echo "## Completed"
  echo
  echo "- status: complete"
  echo "- stage1_hybrid: \`${HYBRID_DIR}/last_weights\`"
  echo "- stage1_val: \`${STAGE1_VAL_DIR}\`"
  echo "- stage2_val: \`${STAGE2_VAL_DIR}\`"
  echo "- stage1_metrics: \`${STAGE1_VAL_DIR}/metrics/summary.csv\`"
  echo "- stage2_metrics: \`${STAGE2_VAL_DIR}/metrics/summary.csv\`"
} >> "${REPORT}"

echo "[exp07-posthoc-davis] complete"
echo "[exp07-posthoc-davis] report=${REPORT}"
