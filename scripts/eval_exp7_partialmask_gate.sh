#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/mnt/nas/hj/weights}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/mnt/nas/hj/conda_envs/diffueraser}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PREFIX}/bin/python}"

MANIFEST="${MANIFEST:-${OUTPUT_ROOT}/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4/manifests/selected_primary_comp.repaired.jsonl}"
EXP_NAME="${EXP_NAME:-exp7_gate1500}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EVAL_DIR="${EVAL_DIR:-${OUTPUT_ROOT}/logs/partialmask_eval/${EXP_NAME}_${TIMESTAMP}}"

STAGE1_RUN_DIR="${STAGE1_RUN_DIR:-${OUTPUT_ROOT}/experiments/dpo/stage1/20260601_065618_exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500_stage1}"
STAGE2_RUN_DIR="${STAGE2_RUN_DIR:-${OUTPUT_ROOT}/experiments/dpo/stage2/20260601_065618_exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500_stage2}"

BASE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH:-${WEIGHTS_DIR}/stable-diffusion-v1-5}"
VAE_PATH="${VAE_PATH:-${WEIGHTS_DIR}/sd-vae-ft-mse}"
BASE_WEIGHTS_DIR="${BASE_WEIGHTS_DIR:-${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000}"

NUM_SAMPLES="${NUM_SAMPLES:-30}"
NUM_SAMPLES_METRIC="${NUM_SAMPLES_METRIC:-100}"
SEED="${SEED:-42}"
HEIGHT="${HEIGHT:-320}"
WIDTH="${WIDTH:-512}"
FRAMES="${FRAMES:-16}"
FPS="${FPS:-10}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-20}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-12.0}"
TORCH_DTYPE="${TORCH_DTYPE:-bf16}"
VAE_DTYPE="${VAE_DTYPE:-fp32}"
DPO_SUMMARY_OUT="${DPO_SUMMARY_OUT:-${OUTPUT_ROOT}/reports/exp7_gate1500_dpo_diag_summary.md}"

die() {
  echo "[exp7-partialmask-eval][ERROR] $*" >&2
  exit 1
}

require_path() {
  local path="$1"
  local label="$2"
  [[ -e "$path" ]] || die "${label} not found: ${path}"
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

require_path "${MANIFEST}" "D2 manifest"
require_path "${BASE_MODEL_NAME_OR_PATH}" "stable-diffusion-v1-5"
require_path "${VAE_PATH}" "sd-vae-ft-mse"
require_path "${BASE_WEIGHTS_DIR}" "DiffuEraser-base weights"
require_path "${STAGE1_RUN_DIR}" "Exp7 Stage1 run dir"
require_path "${STAGE2_RUN_DIR}" "Exp7 Stage2 run dir"
require_path "${PROJECT_ROOT}/tools/eval_generated_loser_partialmask_model.py" "partial-mask eval tool"

mkdir -p "${EVAL_DIR}" "$(dirname "${DPO_SUMMARY_OUT}")"

echo "[exp7-partialmask-eval] output=${EVAL_DIR}"
echo "[exp7-partialmask-eval] manifest=${MANIFEST}"
echo "[exp7-partialmask-eval] stage1=${STAGE1_RUN_DIR}"
echo "[exp7-partialmask-eval] stage2=${STAGE2_RUN_DIR}"
echo "[exp7-partialmask-eval] num_samples=${NUM_SAMPLES} num_samples_metric=${NUM_SAMPLES_METRIC}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/eval_generated_loser_partialmask_model.py" \
  --manifest "${MANIFEST}" \
  --output_dir "${EVAL_DIR}" \
  --base_weights_dir "${BASE_WEIGHTS_DIR}" \
  --checkpoint "Stage1_ckpt500=${STAGE1_RUN_DIR}/checkpoint-500" \
  --checkpoint "Stage1_ckpt1000=${STAGE1_RUN_DIR}/checkpoint-1000" \
  --checkpoint "Stage1_last=${STAGE1_RUN_DIR}/last_weights" \
  --checkpoint "Stage2_last=${STAGE2_RUN_DIR}/last_weights" \
  --base_model_name_or_path "${BASE_MODEL_NAME_OR_PATH}" \
  --vae_path "${VAE_PATH}" \
  --num_samples "${NUM_SAMPLES}" \
  --num_samples_metric "${NUM_SAMPLES_METRIC}" \
  --seed "${SEED}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --frames "${FRAMES}" \
  --fps "${FPS}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --guidance_scale "${GUIDANCE_SCALE}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --vae_dtype "${VAE_DTYPE}" \
  --stage1_diag_csv "${STAGE1_RUN_DIR}/dpo_diagnostics.csv" \
  --stage2_diag_csv "${STAGE2_RUN_DIR}/dpo_diagnostics.csv" \
  --dpo_summary_out "${DPO_SUMMARY_OUT}" \
  --skip_existing

echo "[exp7-partialmask-eval] done"
echo "[exp7-partialmask-eval] side_by_side=${EVAL_DIR}/side_by_side"
echo "[exp7-partialmask-eval] metrics=${EVAL_DIR}/metrics/summary.csv"
echo "[exp7-partialmask-eval] report=${EVAL_DIR}/report.md"
echo "[exp7-partialmask-eval] dpo_summary=${DPO_SUMMARY_OUT}"
