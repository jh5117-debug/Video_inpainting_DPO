#!/usr/bin/env bash
set -euo pipefail

WORKTREE="${WORKTREE:-/home/nvme01/H20_Video_inpainting_DPO_exp41_minimax_h20_parallel}"
PROJECT_ROOT="${PROJECT_ROOT:-${WORKTREE}}"
MINIMAX_REPO="${MINIMAX_REPO:-/home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/minimax/pai_abs/mnt/nas/hj/official_repos/MiniMax-Remover_28e12b4}"
MODEL_DIR="${MODEL_DIR:-/home/nvme01/H20_Video_inpainting_DPO/weights/minimax_remover/current}"
MANIFEST="${MANIFEST:-${WORKTREE}/exp41_h20_minimax_parallel_bf16/manifests/exp41_exp40_localdpo_v3_train_h20.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp41_h20_minimax_parallel_bf16/bf16_preflight_$(date +%Y%m%d_%H%M%S)}"
CONDA_BIN="${CONDA_BIN:-/home/nvme01/miniconda3/bin/conda}"
WAN_ENV="${WAN_ENV:-/home/nvme01/miniconda3/envs/wan}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_BIN} run --no-capture-output -p ${WAN_ENV} python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_BIN} run --no-capture-output -p ${WAN_ENV} torchrun}"

export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export XFORMERS_DISABLED=1
export DISABLE_XFORMERS=1
export FLASH_ATTENTION_FORCE_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

SCRIPT="${WORKTREE}/exp41_h20_minimax_parallel_bf16/bf16_minimax_preflight.py"
mkdir -p "${OUTPUT_ROOT}/logs"

common_args=(
  --repo-dir "${MINIMAX_REPO}"
  --project-root "${PROJECT_ROOT}"
  --model-dir "${MODEL_DIR}"
  --manifest "${MANIFEST}"
  --output-root "${OUTPUT_ROOT}"
  --save-checkpoint
)

run_single() {
  local case_name="$1"
  local visible="$2"
  shift 2
  echo "[exp41-bf16] ${case_name} CUDA_VISIBLE_DEVICES=${visible}"
  CUDA_VISIBLE_DEVICES="${visible}" ${PYTHON_BIN} "${SCRIPT}" \
    --case "${case_name}" \
    "${common_args[@]}" "$@" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/${case_name}.log"
}

run_ddp() {
  local case_name="$1"
  local nproc="$2"
  local visible="$3"
  shift 3
  echo "[exp41-bf16] ${case_name} nproc=${nproc} CUDA_VISIBLE_DEVICES=${visible}"
  CUDA_VISIBLE_DEVICES="${visible}" ${TORCHRUN_BIN} \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${nproc}" \
    "${SCRIPT}" \
    --case "${case_name}" \
    "${common_args[@]}" "$@" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/${case_name}.log"
}

echo "[exp41-bf16] output=${OUTPUT_ROOT}"
hostname | tee "${OUTPUT_ROOT}/hostname.txt"
date -Ins | tee "${OUTPUT_ROOT}/started_at.txt"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee "${OUTPUT_ROOT}/gpu_before.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "${OUTPUT_ROOT}/compute_before.csv" || true

run_single P0 0
run_single P1 0
run_single P2 0
run_single P3 0
run_single P4 0
run_single P5 0
run_ddp P6 2 0,1
run_ddp P7 8 0,1,2,3,4,5,6,7

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee "${OUTPUT_ROOT}/gpu_after.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "${OUTPUT_ROOT}/compute_after.csv" || true
echo "${OUTPUT_ROOT}" | tee "${OUTPUT_ROOT}/output_root.txt"
