#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
if [[ ! -d "${ROOT}" ]]; then
  ROOT="/mnt/nas/hj/H20_Video_inpainting_DPO"
fi
cd "${ROOT}"

export EXP19_CODE_ROOT="${EXP19_CODE_ROOT:-${ROOT}}"
export PYTHONPATH="${ROOT}:${EXP19_CODE_ROOT}:${PYTHONPATH:-}"

PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/mnt/nas/hj/weights}"
BASE_MODEL="${BASE_MODEL:-${WEIGHTS_DIR}/stable-diffusion-v1-5}"
VAE="${VAE:-${WEIGHTS_DIR}/sd-vae-ft-mse}"
EXP11_STAGE2="${EXP11_STAGE2:-${OUTPUT_ROOT}/experiments/dpo/stage2/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s2_2000_davis_pai/last_weights}"
FLOW_CACHE="${FLOW_CACHE:-${OUTPUT_ROOT}/data/cache/exp19_propainter_completed_flow_limit100}"
FLOW_MANIFEST="${FLOW_MANIFEST:-${FLOW_CACHE}/manifests/exp19_train_with_completed_flow_limit100.jsonl}"
START_ADAPTER="${START_ADAPTER:-${OUTPUT_ROOT}/experiments/dpo/stage2/exp19b_boundary_flow_adapter_s2_500_limit100/last_weights/flow_adapter.pt}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/experiments/dpo/stage2/exp19b_boundary_flow_adapter_s2_2000_exploratory_from500_limit100}"
DIAG_CSV="${DIAG_CSV:-exp19b_exploratory_2000/dpo_diag/exp19b_exploratory_s2_2000_dpo_diagnostics.csv}"
REPORT_PATH="${REPORT_PATH:-reports/exp19b_exploratory_2000_preflight.md}"
GPU="${EXP19_GPU:-0}"

mkdir -p "$(dirname "${DIAG_CSV}")" "${RUN_DIR}" logs/pipelines reports

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" exp19c_light_warp_dpo/code/train_exp19c_stage2_adapter.py \
  --base_model_name_or_path "${BASE_MODEL}" \
  --vae_path "${VAE}" \
  --exp11_stage2_weights "${EXP11_STAGE2}" \
  --start_adapter "${START_ADAPTER}" \
  --flow_manifest "${FLOW_MANIFEST}" \
  --output_dir "${RUN_DIR}" \
  --diag_csv "${DIAG_CSV}" \
  --report_path "${REPORT_PATH}" \
  --variant_name exp19b_exploratory_s2_2000_from500_limit100 \
  --lambda_warp 0.0 \
  --max_train_steps 1500 \
  --checkpointing_steps 500 \
  --mixed_precision bf16 \
  --residual_scale 0.5 \
  --confidence_exponent 2.0
