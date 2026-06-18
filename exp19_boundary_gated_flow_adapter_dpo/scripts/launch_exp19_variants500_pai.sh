#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
export EXP19_CODE_ROOT="${EXP19_CODE_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp09_10_11_pai_sync}"
if [[ ! -d "${ROOT}" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${ROOT}"

PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}"
FLOW_CACHE="${FLOW_CACHE:-${OUTPUT_ROOT}/data/cache/exp19_propainter_completed_flow_limit100}"
FLOW_MANIFEST="${FLOW_MANIFEST:-${FLOW_CACHE}/manifests/exp19_train_with_completed_flow_limit100.jsonl}"
BASE_MODEL="${BASE_MODEL:-/mnt/nas/hj/weights/stable-diffusion-v1-5}"
VAE="${VAE:-/mnt/nas/hj/weights/sd-vae-ft-mse}"
EXP11_STAGE2="${EXP11_STAGE2:-${OUTPUT_ROOT}/experiments/dpo/stage2/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s2_2000_davis_pai/last_weights}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/experiments/dpo/stage2/exp19b_boundary_flow_adapter_s2_500_limit100}"
DIAG_CSV="${DIAG_CSV:-exp19_boundary_gated_flow_adapter_dpo/dpo_diag/exp19b_stage2_500_dpo_diagnostics.csv}"
GPU="${EXP19_GPU:-0}"

mkdir -p "$(dirname "${DIAG_CSV}")" "${RUN_DIR}" logs/pipelines
CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" exp19_boundary_gated_flow_adapter_dpo/code/train_exp19_stage2_adapter.py \
  --base_model_name_or_path "${BASE_MODEL}" \
  --vae_path "${VAE}" \
  --exp11_stage2_weights "${EXP11_STAGE2}" \
  --flow_manifest "${FLOW_MANIFEST}" \
  --output_dir "${RUN_DIR}" \
  --diag_csv "${DIAG_CSV}" \
  --report_path reports/exp19_isolated_wrapper_preflight.md \
  --max_train_steps 500 \
  --checkpointing_steps 250 \
  --mixed_precision bf16
