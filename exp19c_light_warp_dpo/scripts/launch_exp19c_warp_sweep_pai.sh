#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
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
START_ADAPTER="${START_ADAPTER:-${OUTPUT_ROOT}/experiments/dpo/stage2/exp19b_boundary_flow_adapter_s2_500_limit100/last_weights/flow_adapter.pt}"
RUN_ROOT="${RUN_ROOT:-${OUTPUT_ROOT}/experiments/dpo/stage2/exp19c_light_warp_dpo}"
GPU="${EXP19C_GPU:-0}"
MAX_STEPS="${MAX_STEPS:-500}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-250}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.5}"
CONF_EXP="${CONF_EXP:-2.0}"

mkdir -p logs/pipelines exp19c_light_warp_dpo/dpo_diag exp19c_light_warp_dpo/reports "${RUN_ROOT}"

declare -A LAMBDAS=(
  [lambda000]="0.0"
  [lambda005]="0.005"
  [lambda010]="0.010"
  [lambda020]="0.020"
)

for name in lambda000 lambda005 lambda010 lambda020; do
  lambda="${LAMBDAS[$name]}"
  run_dir="${RUN_ROOT}/${name}"
  diag_csv="exp19c_light_warp_dpo/dpo_diag/${name}_dpo_diagnostics.csv"
  report_path="exp19c_light_warp_dpo/reports/${name}_preflight.md"
  echo "[Exp19c] start ${name} lambda=${lambda} $(date)"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" exp19c_light_warp_dpo/code/train_exp19c_stage2_adapter.py \
    --base_model_name_or_path "${BASE_MODEL}" \
    --vae_path "${VAE}" \
    --exp11_stage2_weights "${EXP11_STAGE2}" \
    --start_adapter "${START_ADAPTER}" \
    --flow_manifest "${FLOW_MANIFEST}" \
    --output_dir "${run_dir}" \
    --diag_csv "${diag_csv}" \
    --report_path "${report_path}" \
    --variant_name "exp19c_${name}" \
    --lambda_warp "${lambda}" \
    --max_train_steps "${MAX_STEPS}" \
    --checkpointing_steps "${CHECKPOINT_STEPS}" \
    --mixed_precision bf16 \
    --residual_scale "${RESIDUAL_SCALE}" \
    --confidence_exponent "${CONF_EXP}"
  echo "[Exp19c] done ${name} $(date)"
done
