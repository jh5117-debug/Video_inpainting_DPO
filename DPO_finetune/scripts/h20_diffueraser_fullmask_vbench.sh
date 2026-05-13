#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

WEIGHTS_DIR="${WEIGHTS_DIR:-${PROJECT_ROOT}/weights}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${PROJECT_ROOT}/experiments}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${WEIGHTS_DIR}/stable-diffusion-v1-5}"
VAE_PATH="${VAE_PATH:-${WEIGHTS_DIR}/sd-vae-ft-mse}"
VIDEODPO_REPO="${VIDEODPO_REPO:-${PROJECT_ROOT}/external/VideoDPO}"
VBENCH_ROOT="${VBENCH_ROOT:-${PROJECT_ROOT}/external/VBench}"
PROMPTS_FILE="${PROMPTS_FILE:-${VIDEODPO_REPO}/prompts/vbench_standard_prompts.txt}"
VBENCH_FULL_JSON="${VBENCH_FULL_JSON:-${VBENCH_ROOT}/vbench/VBench_full_info.json}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/logs/diffueraser_fullmask_vbench/$(date -u +%Y%m%d_%H%M%S)}"

CONDA_ENV="${CONDA_ENV:-/home/nvme01/conda_envs/diffueraser}"
VBENCH_CONDA_ENV="${VBENCH_CONDA_ENV:-${PROJECT_ROOT}/third_party_video_inpainting/envs/vbench}"
if [[ ! -d "${VBENCH_CONDA_ENV}" ]]; then
  VBENCH_CONDA_ENV="${VBENCH_CONDA_ENV_FALLBACK:-${CONDA_ENV}}"
fi

WEIGHTS_PATH="${WEIGHTS_PATH:-}"
if [[ -z "${WEIGHTS_PATH}" ]]; then
  latest_file="${EXPERIMENTS_DIR}/dpo/stage1/LATEST"
  if [[ -f "${latest_file}" ]]; then
    latest_run="$(head -n 1 "${latest_file}")"
    if [[ -d "${latest_run}/last_weights" ]]; then
      WEIGHTS_PATH="${latest_run}/last_weights"
    fi
  fi
fi

HEIGHT="${HEIGHT:-320}"
WIDTH="${WIDTH:-512}"
FRAMES="${FRAMES:-16}"
FPS="${FPS:-10}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-5}"
PROMPT_LIMIT="${PROMPT_LIMIT:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-12.0}"
SEED_BASE="${SEED_BASE:-20230211}"
FULL_MASK_VALUE="${FULL_MASK_VALUE:-0.0}"
TORCH_DTYPE="${TORCH_DTYPE:-bf16}"
VAE_DTYPE="${VAE_DTYPE:-fp32}"
GENERATE="${GENERATE:-1}"
RUN_VBENCH="${RUN_VBENCH:-1}"
DIMENSIONS="${DIMENSIONS:-subject_consistency background_consistency aesthetic_quality imaging_quality object_class multiple_objects color spatial_relationship scene temporal_style overall_consistency human_action temporal_flickering motion_smoothness dynamic_degree appearance_style}"

if [[ -z "${WEIGHTS_PATH}" || ! -d "${WEIGHTS_PATH}" ]]; then
  echo "[fullmask-vbench][error] WEIGHTS_PATH not found. Set WEIGHTS_PATH to a stage1 last_weights/best_weights directory." >&2
  exit 1
fi
if [[ ! -f "${PROMPTS_FILE}" ]]; then
  echo "[fullmask-vbench][error] PROMPTS_FILE not found: ${PROMPTS_FILE}" >&2
  exit 1
fi
if [[ "${RUN_VBENCH}" == "1" && "${PROMPT_LIMIT}" != "0" ]]; then
  echo "[fullmask-vbench][error] PROMPT_LIMIT is generation-smoke only. Set RUN_VBENCH=0 or use the full VBench prompt suite." >&2
  exit 1
fi
if [[ "${RUN_VBENCH}" == "1" && ! -f "${VBENCH_ROOT}/evaluate.py" ]]; then
  echo "[fullmask-vbench][error] VBench evaluate.py not found under ${VBENCH_ROOT}" >&2
  exit 1
fi

if [[ -n "${CONDA_BASE:-}" && -x "${CONDA_BASE}/bin/conda" ]]; then
  :
elif [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$("${CONDA_EXE}" info --base)"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  echo "[fullmask-vbench][error] conda not found; set CONDA_EXE or CONDA_BASE." >&2
  exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"

mkdir -p "${OUT_ROOT}" "${PROJECT_ROOT}/logs"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${PROJECT_ROOT}/.hf_cache}"
export TMPDIR="${TMPDIR:-${PROJECT_ROOT}/.tmp}"
mkdir -p "${HF_HOME}" "${TMPDIR}"

GEN_DIR="${OUT_ROOT}/vbench_standard_named"
EVAL_DIR="${OUT_ROOT}/vbench_eval"

echo "[fullmask-vbench] project_root=${PROJECT_ROOT}"
echo "[fullmask-vbench] weights_path=${WEIGHTS_PATH}"
echo "[fullmask-vbench] prompts=${PROMPTS_FILE}"
echo "[fullmask-vbench] out_root=${OUT_ROOT}"
echo "[fullmask-vbench] generate=${GENERATE} run_vbench=${RUN_VBENCH} samples_per_prompt=${SAMPLES_PER_PROMPT} prompt_limit=${PROMPT_LIMIT}"

if [[ "${GENERATE}" == "1" ]]; then
  conda activate "${CONDA_ENV}"
  python "${PROJECT_ROOT}/tools/generate_diffueraser_fullmask_vbench.py" \
    --base_model_name_or_path "${BASE_MODEL_PATH}" \
    --vae_path "${VAE_PATH}" \
    --weights_path "${WEIGHTS_PATH}" \
    --prompts_file "${PROMPTS_FILE}" \
    --output_dir "${GEN_DIR}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --frames "${FRAMES}" \
    --fps "${FPS}" \
    --samples_per_prompt "${SAMPLES_PER_PROMPT}" \
    --prompt_limit "${PROMPT_LIMIT}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --seed_base "${SEED_BASE}" \
    --full_mask_value "${FULL_MASK_VALUE}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --vae_dtype "${VAE_DTYPE}" \
    --skip_existing
fi

if [[ "${RUN_VBENCH}" == "1" ]]; then
  conda activate "${VBENCH_CONDA_ENV}"
  export PYTHONPATH="${VBENCH_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
  mkdir -p "${EVAL_DIR}"
  cd "${VBENCH_ROOT}"
  python evaluate.py \
    --videos_path "${GEN_DIR}" \
    --output_path "${EVAL_DIR}" \
    --full_json_dir "${VBENCH_FULL_JSON}" \
    --dimension ${DIMENSIONS} \
    --mode vbench_standard

  latest_json="$(ls -t "${EVAL_DIR}"/results_*_eval_results.json | head -n 1)"
  python "${PROJECT_ROOT}/tools/summarize_vbench_results.py" "${latest_json}" \
    --output_json "${EVAL_DIR}/summary.json" \
    --output_csv "${EVAL_DIR}/summary.csv"
fi

echo "[fullmask-vbench] done: ${OUT_ROOT}"
