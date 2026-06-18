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
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/experiments/dpo/stage2/exp19b_wrapper_preflight}"
mkdir -p reports
[[ -f "${FLOW_MANIFEST}" ]] || { echo "[exp19] missing flow manifest: ${FLOW_MANIFEST}" >&2; exit 2; }
"${PY}" exp19_boundary_gated_flow_adapter_dpo/code/train_exp19_stage2_adapter.py \
  --preflight_only \
  --base_model_name_or_path "${BASE_MODEL}" \
  --vae_path "${VAE}" \
  --exp11_stage2_weights "${EXP11_STAGE2}" \
  --flow_manifest "${FLOW_MANIFEST}" \
  --output_dir "${RUN_DIR}" \
  --report_path reports/exp19_isolated_wrapper_preflight.md
