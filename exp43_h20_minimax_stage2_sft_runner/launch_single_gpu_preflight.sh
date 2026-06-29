#!/usr/bin/env bash
set -euo pipefail

WORKTREE="${WORKTREE:-/home/nvme01/H20_Video_inpainting_DPO_exp43_minimax_stage2_sft}"
PROJECT_ROOT="${PROJECT_ROOT:-${WORKTREE}}"
MINIMAX_REPO="${MINIMAX_REPO:-/home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/minimax/pai_abs/mnt/nas/hj/official_repos/MiniMax-Remover_28e12b4}"
MODEL_DIR="${MODEL_DIR:-/home/nvme01/H20_Video_inpainting_DPO/weights/minimax_remover/current}"
MANIFEST="${MANIFEST:-${WORKTREE}/exp43_h20_minimax_stage2_sft_runner/manifests/exp43_preflight_train_h20.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp43_h20_minimax_stage2_sft_runner/bf16_preflight_$(date +%Y%m%d_%H%M%S)}"
CONDA_BIN="${CONDA_BIN:-/home/nvme01/miniconda3/bin/conda}"
WAN_ENV="${WAN_ENV:-/home/nvme01/miniconda3/envs/wan}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_BIN} run --no-capture-output -p ${WAN_ENV} python}"
GPU="${GPU:-0}"

export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export XFORMERS_DISABLED=1
export DISABLE_XFORMERS=1
export FLASH_ATTENTION_FORCE_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT="${WORKTREE}/exp43_h20_minimax_stage2_sft_runner/runner_stage2_sft_ladder.py"
mkdir -p "${OUTPUT_ROOT}/logs"

common_args=(
  --repo-dir "${MINIMAX_REPO}"
  --project-root "${PROJECT_ROOT}"
  --model-dir "${MODEL_DIR}"
  --manifest "${MANIFEST}"
  --output-root "${OUTPUT_ROOT}"
  --save-checkpoint
)

echo "[exp43-bf16] output=${OUTPUT_ROOT}"
hostname | tee "${OUTPUT_ROOT}/hostname.txt"
date -Ins | tee "${OUTPUT_ROOT}/started_at.txt"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee "${OUTPUT_ROOT}/gpu_before_single.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "${OUTPUT_ROOT}/compute_before_single.csv" || true

for case_name in P0 P1 P2 P3 P4 P5; do
  echo "[exp43-bf16] ${case_name} CUDA_VISIBLE_DEVICES=${GPU}"
  CUDA_VISIBLE_DEVICES="${GPU}" ${PYTHON_BIN} "${SCRIPT}" preflight \
    --case "${case_name}" \
    "${common_args[@]}" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/${case_name}.log"
done

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee "${OUTPUT_ROOT}/gpu_after_single.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "${OUTPUT_ROOT}/compute_after_single.csv" || true
echo "${OUTPUT_ROOT}" | tee "${OUTPUT_ROOT}/output_root.txt"
