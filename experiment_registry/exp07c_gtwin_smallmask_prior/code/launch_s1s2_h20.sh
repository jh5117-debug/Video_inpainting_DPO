#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
cd "${PROJECT_ROOT}"

EXP_ID="exp07c_gtwin_smallmask_prior"
EXP_NAME="${EXP_NAME:-exp07c_gtwin_smallmask_prior_wingap_lose025_s1s2_2000_h20}"
STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-exp07c_gtwin_smallmask_prior_wingap_lose025_s1_2000_h20}"
STAGE2_RUN_NAME="${STAGE2_RUN_NAME:-exp07c_gtwin_smallmask_prior_wingap_lose025_s2_2000_h20}"
RUN_VERSION="${RUN_VERSION:-$(date +%Y%m%d_%H%M%S)}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/nvme01/H20_Video_inpainting_DPO}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${OUTPUT_ROOT}/experiments}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${OUTPUT_ROOT}/weights}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/home/nvme01/conda_envs/diffueraser}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PREFIX}/bin/python}"

SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-${OUTPUT_ROOT}/data/generated_losers/exp07_fix_videodpo_smallmask15_20_prior_k4}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-${SOURCE_DATA_ROOT}/manifests/selected_primary_comp.jsonl}"
DATA_ROOT="${DATA_ROOT:-${OUTPUT_ROOT}/data/generated_losers/exp07c_gtwin_smallmask15_20_prior_k4}"
PREFERENCE_MANIFEST="${PREFERENCE_MANIFEST:-${DATA_ROOT}/manifests/selected_primary_comp.gtwin.jsonl}"
GTWIN_CACHE_ROOT="${GTWIN_CACHE_ROOT:-${DATA_ROOT}/gt_win_cache}"
VIDEO_DPO_TRAIN_DATA_YAML="${VIDEO_DPO_TRAIN_DATA_YAML:-${OUTPUT_ROOT}/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml}"
GTWIN_TOOL="${GTWIN_TOOL:-${PROJECT_ROOT}/experiment_registry/${EXP_ID}/code/prepare_gtwin_manifest.py}"

BASE_WEIGHTS="${BASE_WEIGHTS:-${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000}"
BASE_MODEL="${BASE_MODEL:-${WEIGHTS_DIR}/stable-diffusion-v1-5}"
VAE_PATH="${VAE_PATH:-${WEIGHTS_DIR}/sd-vae-ft-mse}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
EVAL_GPU_LIST="${EVAL_GPU_LIST:-${CUDA_VISIBLE_DEVICES}}"
FINAL_GPU="${FINAL_GPU:-0}"
MAIN_PROCESS_PORT_STAGE1="${MAIN_PROCESS_PORT_STAGE1:-29581}"
MAIN_PROCESS_PORT_STAGE2="${MAIN_PROCESS_PORT_STAGE2:-29582}"

STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-2000}"
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-2000}"
CKPT_STEPS="${CKPT_STEPS:-500}"
CKPT_LIMIT="${CKPT_LIMIT:-5}"

die() {
  echo "[${EXP_ID}][ERROR] $*" >&2
  exit 1
}

require_path() {
  local path="$1"
  local label="$2"
  [[ -e "${path}" ]] || die "${label} not found: ${path}"
}

mkdir -p "${OUTPUT_ROOT}/logs/pipelines" "${OUTPUT_ROOT}/reports" "${DATA_ROOT}/manifests"

require_path "${GTWIN_TOOL}" "Exp7c GT/winner manifest tool"
require_path "${SOURCE_MANIFEST}" "Exp7a source small-D2 manifest"
require_path "${VIDEO_DPO_TRAIN_DATA_YAML}" "VideoDPO train data YAML"
require_path "${BASE_WEIGHTS}/unet_main" "SFT DiffuEraser unet_main"
require_path "${BASE_WEIGHTS}/brushnet" "SFT DiffuEraser brushnet"
require_path "${BASE_MODEL}" "stable-diffusion-v1-5"
require_path "${VAE_PATH}" "sd-vae-ft-mse"
require_path "${PROJECT_ROOT}/training/dpo/scripts/03_dpo_stage1.sbatch" "Stage1 launcher"
require_path "${PROJECT_ROOT}/training/dpo/scripts/03_dpo_stage2.sbatch" "Stage2 launcher"
require_path "${PROJECT_ROOT}/tools/eval_generated_loser_partialmask_model.py" "small-D2 eval tool"
require_path "${PROJECT_ROOT}/tools/build_diffueraser_dpoS1_sftS2_hybrid.py" "hybrid builder"

if [[ ! -f "${PREFERENCE_MANIFEST}" ]]; then
  echo "[${EXP_ID}] preparing GT/winner manifest: ${PREFERENCE_MANIFEST}"
  "${PYTHON_BIN}" "${GTWIN_TOOL}" \
    --source_manifest "${SOURCE_MANIFEST}" \
    --train_data_yaml "${VIDEO_DPO_TRAIN_DATA_YAML}" \
    --output_root "${DATA_ROOT}" \
    --output_manifest "${PREFERENCE_MANIFEST}" \
    --cache_root "${GTWIN_CACHE_ROOT}" \
    --strict \
    --report_path "${OUTPUT_ROOT}/reports/exp7c_gtwin_manifest_prepare_${RUN_VERSION}.md"
else
  echo "[${EXP_ID}] GT/winner manifest exists: ${PREFERENCE_MANIFEST}"
fi

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

manifest = Path("${PREFERENCE_MANIFEST}")
rows = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
if not rows:
    raise SystemExit(f"empty manifest: {manifest}")
for idx, row in enumerate(rows[:16]):
    for key in ("win_video_path", "final_loser_video_path", "mask_path"):
        path = Path(str(row.get(key, "")))
        if not path.exists():
            raise SystemExit(f"row {idx} missing {key}: {path}")
print(f"[${EXP_ID}] manifest_rows={len(rows)} sampled_paths_checked={min(16, len(rows))}")
PY

{
  echo "[${EXP_ID}] EXP_NAME=${EXP_NAME}"
  echo "[${EXP_ID}] RUN_VERSION=${RUN_VERSION}"
  echo "[${EXP_ID}] PROJECT_ROOT=${PROJECT_ROOT}"
  echo "[${EXP_ID}] OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "[${EXP_ID}] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NUM_GPUS=${NUM_GPUS}"
  echo "[${EXP_ID}] manifest=${PREFERENCE_MANIFEST}"
  echo "[${EXP_ID}] source_manifest=${SOURCE_MANIFEST}"
  echo "[${EXP_ID}] train_data_yaml=${VIDEO_DPO_TRAIN_DATA_YAML}"
  echo "[${EXP_ID}] eval=VideoDPO small-D2 partial-mask"
} | tee "${OUTPUT_ROOT}/logs/pipelines/${EXP_NAME}_${RUN_VERSION}.precheck.log"

case "${EXP7C_PRECHECK_ONLY:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    echo "[${EXP_ID}] precheck-only complete"
    exit 0
    ;;
esac

common_training_env() {
  export PROJECT_ROOT OUTPUT_ROOT EXPERIMENTS_DIR WEIGHTS_DIR CONDA_ENV_PREFIX
  export DATA="${OUTPUT_ROOT}"
  export DPO_DATASET_TYPE="generated_loser_manifest"
  export DPO_DATA_ROOT="${DATA_ROOT}"
  export PREFERENCE_MANIFEST
  export TRAIN_MASK_MODE="partial"
  export MASK_FROM_MANIFEST="true"
  export LOSS_REGION_MODE="full"
  export ENABLE_DPO_DIAG="true"
  export DPO_DIAG_LOG_EVERY="${DPO_DIAG_LOG_EVERY:-10}"
  export DPO_DIAG_SAVE_CSV="true"
  export DPO_DIAG_SAVE_WANDB="false"
  export REPORT_TO="none"
  export BETA_DPO="${BETA_DPO:-10}"
  export SFT_REG_WEIGHT="${SFT_REG_WEIGHT:-0.0}"
  export LOSE_GAP_WEIGHT="${LOSE_GAP_WEIGHT:-0.25}"
  export DPO_LOSE_GAP_WEIGHT="${DPO_LOSE_GAP_WEIGHT:-${LOSE_GAP_WEIGHT}}"
  export WINNER_ABS_REG_WEIGHT="${WINNER_ABS_REG_WEIGHT:-0.05}"
  export WINNER_GAP_REG_WEIGHT="${WINNER_GAP_REG_WEIGHT:-1.0}"
  export WINNER_GAP_REG_MARGIN="${WINNER_GAP_REG_MARGIN:-0.0}"
  export WINNER_GAP_REG_MODE="${WINNER_GAP_REG_MODE:-relu}"
  export TRAIN_HEIGHT="${TRAIN_HEIGHT:-320}"
  export TRAIN_WIDTH="${TRAIN_WIDTH:-512}"
  export RESOLUTION="${RESOLUTION:-512}"
  export NFRAMES="${NFRAMES:-16}"
  export NUM_WORKERS="${NUM_WORKERS:-0}"
  export LOGGING_STEPS="${LOGGING_STEPS:-10}"
  export LR="${LR:-1e-6}"
  export LR_SCHEDULER="${LR_SCHEDULER:-constant}"
  export LR_WARMUP="${LR_WARMUP:-500}"
  export CKPT_STEPS CKPT_LIMIT
  export VAL_STEPS="${VAL_STEPS:-999999}"
  export REF_MODEL_PATH="${REF_MODEL_PATH:-${BASE_WEIGHTS}}"
  export BASELINE_UNET_PATH="${BASELINE_UNET_PATH:-${BASE_WEIGHTS}}"
  export VAL_DATA_DIR="${VAL_DATA_DIR:-${OUTPUT_ROOT}/data/external/davis_432_240}"
  export CUDA_VISIBLE_DEVICES NUM_GPUS
  export MIXED_PRECISION="${MIXED_PRECISION:-no}"
  export POLICY_DTYPE="${POLICY_DTYPE:-fp32}"
  export VAE_DTYPE="${VAE_DTYPE:-fp32}"
  export REF_DTYPE="${REF_DTYPE:-fp32}"
  export TEXT_DTYPE="${TEXT_DTYPE:-fp32}"
  export SPLIT_POS_NEG_FORWARD="${SPLIT_POS_NEG_FORWARD:-0}"
  export CHUNK_ALIGNED="${CHUNK_ALIGNED:-1}"
  export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
  export XFORMERS="${XFORMERS:-0}"
  export USE_8BIT_ADAM="${USE_8BIT_ADAM:-0}"
  export WANDB_PROJECT="${WANDB_PROJECT:-DPO_Diffueraser_Exp7c}"
  export WANDB_ENTITY="${WANDB_ENTITY:-jh5117-columbia-university}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
}

run_small_d2_eval() {
  local label="$1"
  local weights="$2"
  local eval_dir="$3"
  local log_dir="$4"
  local stage1_diag="$5"
  local stage2_diag="$6"

  require_path "${weights}/unet_main" "${label} unet_main"
  require_path "${weights}/brushnet" "${label} brushnet"
  mkdir -p "${eval_dir}" "${log_dir}"

  IFS=',' read -r -a gpus <<< "${EVAL_GPU_LIST}"
  local num_shards="${#gpus[@]}"
  if [[ "${num_shards}" -lt 1 ]]; then
    die "EVAL_GPU_LIST is empty"
  fi

  local common_args=(
    --manifest "${PREFERENCE_MANIFEST}"
    --output_dir "${eval_dir}"
    --base_weights_dir "${BASE_WEIGHTS}"
    --checkpoint "${label}=${weights}"
    --base_model_name_or_path "${BASE_MODEL}"
    --vae_path "${VAE_PATH}"
    --num_samples "${EVAL_NUM_SAMPLES:-30}"
    --num_samples_metric "${EVAL_NUM_SAMPLES_METRIC:-100}"
    --seed 42
    --height 320
    --width 512
    --frames 16
    --fps 10
    --num_inference_steps "${EVAL_NUM_INFERENCE_STEPS:-20}"
    --guidance_scale "${EVAL_GUIDANCE_SCALE:-12.0}"
    --torch_dtype fp32
    --vae_dtype fp32
    --no_d2_loser
    --skip_existing
    --stage1_diag_csv "${stage1_diag}"
    --dpo_summary_out "${eval_dir}/dpo_diag_summary.md"
  )
  if [[ -n "${stage2_diag}" && -f "${stage2_diag}" ]]; then
    common_args+=(--stage2_diag_csv "${stage2_diag}")
  fi

  echo "[${EXP_ID}] small-D2 eval ${label}: ${eval_dir}"
  local pids=()
  for shard_index in "${!gpus[@]}"; do
    local gpu="${gpus[${shard_index}]}"
    local shard_log="${log_dir}/shard_${shard_index}_gpu${gpu}.log"
    echo "[${EXP_ID}] eval shard=${shard_index}/${num_shards} gpu=${gpu} log=${shard_log}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" tools/eval_generated_loser_partialmask_model.py \
      "${common_args[@]}" \
      --num_shards "${num_shards}" \
      --shard_index "${shard_index}" \
      --generate_only \
      > "${shard_log}" 2>&1 &
    pids+=("$!")
  done

  local rc=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      rc=1
    fi
  done
  if [[ "${rc}" -ne 0 ]]; then
    die "small-D2 eval shard failed for ${label}; see ${log_dir}"
  fi

  CUDA_VISIBLE_DEVICES="${FINAL_GPU}" "${PYTHON_BIN}" tools/eval_generated_loser_partialmask_model.py \
    "${common_args[@]}" \
    > "${log_dir}/finalize_gpu${FINAL_GPU}.log" 2>&1

  echo "[${EXP_ID}] small-D2 eval complete label=${label}"
  echo "[${EXP_ID}] eval_dir=${eval_dir}"
  echo "[${EXP_ID}] side_by_side=${eval_dir}/side_by_side/${label}"
}

common_training_env

export RUN_VERSION
export RUN_NAME="${STAGE1_RUN_NAME}"
export MAX_STEPS="${STAGE1_MAX_STEPS}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT_STAGE1}"
echo "[${EXP_ID}] Stage1 start: ${RUN_VERSION}_${STAGE1_RUN_NAME}"
bash training/dpo/scripts/03_dpo_stage1.sbatch

STAGE1_DIR="${EXPERIMENTS_DIR}/dpo/stage1/${RUN_VERSION}_${STAGE1_RUN_NAME}"
STAGE1_LAST="${STAGE1_DIR}/last_weights"
STAGE1_DIAG="${STAGE1_DIR}/dpo_diagnostics.csv"
require_path "${STAGE1_LAST}/unet_main" "Stage1 last_weights unet_main"
require_path "${STAGE1_LAST}/brushnet" "Stage1 last_weights brushnet"

HYBRID_DIR="${EXPERIMENTS_DIR}/hybrid/${RUN_VERSION}_${STAGE1_RUN_NAME}_dpoS1_sftS2"
echo "[${EXP_ID}] build exp7c-1 hybrid: ${HYBRID_DIR}"
"${PYTHON_BIN}" tools/build_diffueraser_dpoS1_sftS2_hybrid.py \
  --dpo_stage1_weights "${STAGE1_LAST}" \
  --sft_stage2_weights "${BASE_WEIGHTS}" \
  --output_dir "${HYBRID_DIR}" \
  --report_path "${OUTPUT_ROOT}/reports/exp7c_stage1_hybrid_key_merge_${RUN_VERSION}.md"

run_small_d2_eval \
  "exp7c-1" \
  "${HYBRID_DIR}/last_weights" \
  "${OUTPUT_ROOT}/logs/partialmask_eval/exp7c_gtwin_smallD2_stage1_${RUN_VERSION}" \
  "${OUTPUT_ROOT}/logs/pipelines/exp7c_gtwin_smallD2_stage1_eval_${RUN_VERSION}" \
  "${STAGE1_DIAG}" \
  ""

common_training_env
export RUN_VERSION
export RUN_NAME="${STAGE2_RUN_NAME}"
export MAX_STEPS="${STAGE2_MAX_STEPS}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT_STAGE2}"
export PRETRAINED_DPO_S1="${STAGE1_LAST}"
echo "[${EXP_ID}] Stage2 start: ${RUN_VERSION}_${STAGE2_RUN_NAME}"
bash training/dpo/scripts/03_dpo_stage2.sbatch

STAGE2_DIR="${EXPERIMENTS_DIR}/dpo/stage2/${RUN_VERSION}_${STAGE2_RUN_NAME}"
STAGE2_LAST="${STAGE2_DIR}/last_weights"
STAGE2_DIAG="${STAGE2_DIR}/dpo_diagnostics.csv"
require_path "${STAGE2_LAST}/unet_main" "Stage2 last_weights unet_main"
require_path "${STAGE2_LAST}/brushnet" "Stage2 last_weights brushnet"

run_small_d2_eval \
  "exp7c-2" \
  "${STAGE2_LAST}" \
  "${OUTPUT_ROOT}/logs/partialmask_eval/exp7c_gtwin_smallD2_stage2_${RUN_VERSION}" \
  "${OUTPUT_ROOT}/logs/pipelines/exp7c_gtwin_smallD2_stage2_eval_${RUN_VERSION}" \
  "${STAGE1_DIAG}" \
  "${STAGE2_DIAG}"

REPORT="${OUTPUT_ROOT}/reports/exp7c_gtwin_smallmask_prior_s1s2_${RUN_VERSION}.md"
cat > "${REPORT}" <<EOF_REPORT
# Exp7c GT-Win Small-D2 S1/S2

- run_version: \`${RUN_VERSION}\`
- manifest: \`${PREFERENCE_MANIFEST}\`
- stage1: \`${STAGE1_DIR}\`
- stage1_val: \`${OUTPUT_ROOT}/logs/partialmask_eval/exp7c_gtwin_smallD2_stage1_${RUN_VERSION}\`
- stage2: \`${STAGE2_DIR}\`
- stage2_val: \`${OUTPUT_ROOT}/logs/partialmask_eval/exp7c_gtwin_smallD2_stage2_${RUN_VERSION}\`

EOF_REPORT

echo "[${EXP_ID}] complete"
echo "[${EXP_ID}] report=${REPORT}"

