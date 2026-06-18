#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
if [[ ! -d "${ROOT}" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${ROOT}"

mkdir -p logs/pipelines reports exp19_boundary_gated_flow_adapter_dpo/reports
LOG="logs/pipelines/exp19_boundary_gated_flow_adapter_dpo_overnight.log"
PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"

{
  echo "[Exp19] start $(date)"
  echo "[Exp19] host $(hostname)"
  echo "[Exp19] root ${ROOT}"
  nvidia-smi || true
  "${PY}" -m py_compile exp19_boundary_gated_flow_adapter_dpo/code/*.py
  bash -n exp19_boundary_gated_flow_adapter_dpo/scripts/*.sh

  set +e
  bash exp19_boundary_gated_flow_adapter_dpo/scripts/run_exp19_preflight_pai.sh
  PREFLIGHT_STATUS=$?
  set -e
  if [[ "${PREFLIGHT_STATUS}" -ne 0 ]]; then
    echo "[Exp19] BLOCKED at architecture preflight; not exporting flow cache or training."
    cp reports/exp19_preflight_report.md exp19_boundary_gated_flow_adapter_dpo/reports/preflight_report.md || true
    cat > reports/exp19_final_report.md <<EOF
# Exp19 Final Report

Status:

\`\`\`text
BLOCKED_AT_ARCHITECTURE_PREFLIGHT
\`\`\`

The PAI launcher did not export full flow cache, train, or evaluate because the
requested multi-scale flow-adapter injection is unsafe through the shared
UNetMotionModel residual interface. See:

\`\`\`text
reports/exp19_preflight_report.md
\`\`\`
EOF
    cp reports/exp19_final_report.md exp19_boundary_gated_flow_adapter_dpo/reports/final_report.md || true
    exit "${PREFLIGHT_STATUS}"
  fi

  bash exp19_boundary_gated_flow_adapter_dpo/scripts/prepare_exp19_flow_cache_limit100_pai.sh
  bash exp19_boundary_gated_flow_adapter_dpo/scripts/launch_exp19_variants500_pai.sh
  bash exp19_boundary_gated_flow_adapter_dpo/scripts/eval_exp19_davis10_pai.sh
  bash exp19_boundary_gated_flow_adapter_dpo/scripts/make_exp19_visuals_pai.sh
  echo "[Exp19] done $(date)"
} 2>&1 | tee "${LOG}"
