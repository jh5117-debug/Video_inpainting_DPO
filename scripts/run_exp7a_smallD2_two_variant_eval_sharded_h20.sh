#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/nvme01/H20_Video_inpainting_DPO}"
PYTHON_BIN="${PYTHON_BIN:-/home/nvme01/conda_envs/diffueraser/bin/python}"
CUDA_VISIBLE_DEVICES_LIST="${CUDA_VISIBLE_DEVICES_LIST:-0,1,2,3,4,5,6,7}"
FINAL_GPU="${FINAL_GPU:-0}"

MANIFEST="${MANIFEST:-${OUTPUT_ROOT}/data/generated_losers/exp07_fix_videodpo_smallmask15_20_prior_k4/manifests/selected_primary_comp.jsonl}"
BASE_WEIGHTS="${BASE_WEIGHTS:-${OUTPUT_ROOT}/weights/diffuEraser/converted_weights_step48000}"
BASE_MODEL="${BASE_MODEL:-${OUTPUT_ROOT}/weights/stable-diffusion-v1-5}"
VAE_PATH="${VAE_PATH:-${OUTPUT_ROOT}/weights/sd-vae-ft-mse}"

S1_SFT_S2_WEIGHTS="${S1_SFT_S2_WEIGHTS:-${OUTPUT_ROOT}/logs/partialmask_eval/exp7a_fix_smallmask_prior_smallD2_20260607_223326/hybrids/Hybrid_DPO_S1_last__Official_DiffuEraser_base_Stage2/last_weights}"
S1_DPO_S2_WEIGHTS="${S1_DPO_S2_WEIGHTS:-${OUTPUT_ROOT}/experiments/dpo/stage2/20260606_142555_exp07_fix_smallmask_prior_wingap_lose025_s1s2_2000_h20_stage2/last_weights}"

RUN_VERSION="${RUN_VERSION:-$(date +%Y%m%d_%H%M%S)}"
EVAL_DIR="${EVAL_DIR:-${OUTPUT_ROOT}/logs/partialmask_eval/exp7a_fix_smallmask_prior_smallD2_two_variant_${RUN_VERSION}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs/pipelines/exp7a_smallD2_two_variant_eval_sharded_${RUN_VERSION}}"
mkdir -p "${EVAL_DIR}" "${LOG_DIR}"

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES_LIST}"
NUM_SHARDS="${#GPUS[@]}"
if [[ "${NUM_SHARDS}" -lt 1 ]]; then
  echo "[exp7a-sharded][ERROR] no GPUs configured" >&2
  exit 2
fi

COMMON_ARGS=(
  --manifest "${MANIFEST}"
  --output_dir "${EVAL_DIR}"
  --base_weights_dir "${BASE_WEIGHTS}"
  --checkpoint "DPO-S1_SFT-S2=${S1_SFT_S2_WEIGHTS}"
  --checkpoint "DPO-S1_DPO-S2=${S1_DPO_S2_WEIGHTS}"
  --base_model_name_or_path "${BASE_MODEL}"
  --vae_path "${VAE_PATH}"
  --num_samples 30
  --num_samples_metric 100
  --seed 42
  --height 320
  --width 512
  --frames 16
  --fps 10
  --num_inference_steps 20
  --guidance_scale 12.0
  --torch_dtype fp32
  --vae_dtype fp32
  --no_d2_loser
  --skip_existing
)

echo "[exp7a-sharded] project=${PROJECT_ROOT}"
echo "[exp7a-sharded] eval_dir=${EVAL_DIR}"
echo "[exp7a-sharded] log_dir=${LOG_DIR}"
echo "[exp7a-sharded] gpus=${CUDA_VISIBLE_DEVICES_LIST} num_shards=${NUM_SHARDS}"
echo "[exp7a-sharded] manifest=${MANIFEST}"

pids=()
for shard_index in "${!GPUS[@]}"; do
  gpu="${GPUS[${shard_index}]}"
  shard_log="${LOG_DIR}/shard_${shard_index}_gpu${gpu}.log"
  echo "[exp7a-sharded] launch shard=${shard_index} gpu=${gpu} log=${shard_log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" tools/eval_generated_loser_partialmask_model.py \
    "${COMMON_ARGS[@]}" \
    --num_shards "${NUM_SHARDS}" \
    --shard_index "${shard_index}" \
    --generate_only \
    > "${shard_log}" 2>&1 &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    rc=1
  fi
done

if [[ "${rc}" -ne 0 ]]; then
  echo "[exp7a-sharded][ERROR] at least one shard failed; see ${LOG_DIR}" >&2
  exit "${rc}"
fi

echo "[exp7a-sharded] all shards complete; finalizing metrics and side-by-side on gpu=${FINAL_GPU}"
CUDA_VISIBLE_DEVICES="${FINAL_GPU}" "${PYTHON_BIN}" tools/eval_generated_loser_partialmask_model.py \
  "${COMMON_ARGS[@]}" \
  > "${LOG_DIR}/finalize_gpu${FINAL_GPU}.log" 2>&1

echo "[exp7a-sharded] complete"
echo "[exp7a-sharded] eval_dir=${EVAL_DIR}"
echo "[exp7a-sharded] side_by_side=${EVAL_DIR}/side_by_side"
echo "[exp7a-sharded] metrics=${EVAL_DIR}/metrics/summary.csv"
