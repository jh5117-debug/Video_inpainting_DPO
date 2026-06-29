#!/usr/bin/env bash
set -euo pipefail

WORKTREE="${WORKTREE:-/home/nvme01/H20_Video_inpainting_DPO_exp43_minimax_stage2_sft}"
PROJECT_ROOT="${PROJECT_ROOT:-${WORKTREE}}"
MINIMAX_REPO="${MINIMAX_REPO:-/home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/minimax/pai_abs/mnt/nas/hj/official_repos/MiniMax-Remover_28e12b4}"
MODEL_DIR="${MODEL_DIR:-/home/nvme01/H20_Video_inpainting_DPO/weights/minimax_remover/current}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp43_h20_minimax_stage2_sft_runner}"
LOG_ROOT="${LOG_ROOT:-/home/nvme01/H20_Video_inpainting_DPO/logs/autoresearch/exp43_h20_minimax_stage2_sft_runner}"
REPORTS_DIR="${REPORTS_DIR:-${WORKTREE}/reports}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${WORKTREE}/exp43_h20_minimax_stage2_sft_runner/manifests/exp43_stage2_sft_train.jsonl}"
SEARCH_MANIFEST="${SEARCH_MANIFEST:-${WORKTREE}/exp43_h20_minimax_stage2_sft_runner/manifests/exp43_stage2_sft_search.jsonl}"
SHADOW_MANIFEST="${SHADOW_MANIFEST:-${WORKTREE}/exp43_h20_minimax_stage2_sft_runner/manifests/exp43_stage2_sft_shadow.jsonl}"
CONDA_BIN="${CONDA_BIN:-/home/nvme01/miniconda3/bin/conda}"
WAN_ENV="${WAN_ENV:-/home/nvme01/miniconda3/envs/wan}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_BIN} run --no-capture-output -p ${WAN_ENV} torchrun}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_BIN} run --no-capture-output -p ${WAN_ENV} python}"

RECIPES="${RECIPES:-SFT-A SFT-B SFT-C}"
LRS="${LRS:-3e-5 1e-4 3e-4}"
TARGET_STEPS="${TARGET_STEPS:-30}"
LIMIT_SEARCH="${LIMIT_SEARCH:-24}"
LIMIT_SHADOW="${LIMIT_SHADOW:-24}"
CUDA_LIST="${CUDA_LIST:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-8}"
SEED="${SEED:-20260629}"

export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export XFORMERS_DISABLED=1
export DISABLE_XFORMERS=1
export FLASH_ATTENTION_FORCE_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT="${WORKTREE}/exp43_h20_minimax_stage2_sft_runner/runner_stage2_sft_ladder.py"
BRANCH="$(git -C "${WORKTREE}" branch --show-current)"
COMMIT="$(git -C "${WORKTREE}" rev-parse HEAD)"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${REPORTS_DIR}"

hostname | tee "${LOG_ROOT}/sft_ladder_hostname.txt"
date -Ins | tee "${LOG_ROOT}/sft_ladder_started_at.txt"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee "${LOG_ROOT}/gpu_before_sft_ladder.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "${LOG_ROOT}/compute_before_sft_ladder.csv" || true

for recipe in ${RECIPES}; do
  for lr in ${LRS}; do
    lr_slug="$(printf '%s' "${lr}" | tr '+-' 'pm')"
    run_id="${recipe}_lr${lr_slug}_step${TARGET_STEPS}"
    echo "[exp43-sft] train ${run_id}"
    CUDA_VISIBLE_DEVICES="${CUDA_LIST}" ${TORCHRUN_BIN} \
      --standalone \
      --nnodes=1 \
      --nproc_per_node="${NPROC}" \
      "${SCRIPT}" train-sft \
      --repo-dir "${MINIMAX_REPO}" \
      --project-root "${PROJECT_ROOT}" \
      --model-dir "${MODEL_DIR}" \
      --manifest "${TRAIN_MANIFEST}" \
      --output-root "${OUTPUT_ROOT}" \
      --log-root "${LOG_ROOT}" \
      --reports-dir "${REPORTS_DIR}" \
      --branch "${BRANCH}" \
      --commit "${COMMIT}" \
      --recipe "${recipe}" \
      --target-steps "${TARGET_STEPS}" \
      --lr "${lr}" \
      --dtype bf16 \
      --run-id "${run_id}" \
      --seed "${SEED}" \
      2>&1 | tee "${LOG_ROOT}/${run_id}_train.log"

    checkpoint="${OUTPUT_ROOT}/sft_ladder/${run_id}/checkpoints/checkpoint-${TARGET_STEPS}"
    echo "[exp43-sft] evaluate ${run_id} checkpoint=${checkpoint}"
    CUDA_VISIBLE_DEVICES="${CUDA_LIST%%,*}" ${PYTHON_BIN} "${SCRIPT}" evaluate-sft \
      --repo-dir "${MINIMAX_REPO}" \
      --project-root "${PROJECT_ROOT}" \
      --model-dir "${MODEL_DIR}" \
      --output-root "${OUTPUT_ROOT}" \
      --log-root "${LOG_ROOT}" \
      --reports-dir "${REPORTS_DIR}" \
      --branch "${BRANCH}" \
      --commit "${COMMIT}" \
      --run-id "${run_id}" \
      --recipe "${recipe}" \
      --lr "${lr}" \
      --target-steps "${TARGET_STEPS}" \
      --checkpoint "${checkpoint}" \
      --search-manifest "${SEARCH_MANIFEST}" \
      --shadow-manifest "${SHADOW_MANIFEST}" \
      --limit-search "${LIMIT_SEARCH}" \
      --limit-shadow "${LIMIT_SHADOW}" \
      --dtype bf16 \
      --seed "${SEED}" \
      2>&1 | tee "${LOG_ROOT}/${run_id}_eval.log"
  done
done

${PYTHON_BIN} "${SCRIPT}" summarize-sft --output-root "${OUTPUT_ROOT}" --reports-dir "${REPORTS_DIR}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee "${LOG_ROOT}/gpu_after_sft_ladder.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "${LOG_ROOT}/compute_after_sft_ladder.csv" || true
