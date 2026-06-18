#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
if [[ ! -d "${ROOT}" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${ROOT}"

PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
mkdir -p reports
"${PY}" exp19_boundary_gated_flow_adapter_dpo/code/train_exp19_stage2_adapter.py \
  --preflight_only \
  --report_path reports/exp19_preflight_report.md
