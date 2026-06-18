#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
if [[ ! -d "${ROOT}" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${ROOT}"

mkdir -p logs/pipelines reports
LOG="logs/pipelines/exp19_wrapper_recovery.log"
{
  echo "[Exp19-wrapper] start $(date)"
  echo "[Exp19-wrapper] host $(hostname)"
  nvidia-smi || true
  python -m py_compile exp19_boundary_gated_flow_adapter_dpo/code/*.py
  bash -n exp19_boundary_gated_flow_adapter_dpo/scripts/*.sh
  bash exp19_boundary_gated_flow_adapter_dpo/scripts/prepare_exp19_flow_cache_limit100_pai.sh
  bash exp19_boundary_gated_flow_adapter_dpo/scripts/run_exp19_preflight_pai.sh
  bash exp19_boundary_gated_flow_adapter_dpo/scripts/launch_exp19_variants500_pai.sh
  set +e
  bash exp19_boundary_gated_flow_adapter_dpo/scripts/eval_exp19_davis10_pai.sh
  bash exp19_boundary_gated_flow_adapter_dpo/scripts/make_exp19_visuals_pai.sh
  set -e
  echo "[Exp19-wrapper] done $(date)"
} 2>&1 | tee "${LOG}"
