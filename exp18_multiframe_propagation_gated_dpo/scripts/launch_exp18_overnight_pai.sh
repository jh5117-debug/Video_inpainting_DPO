#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"

mkdir -p logs/pipelines
LOG="logs/pipelines/exp18_multiframe_propagation_gated_dpo_overnight.log"
{
  echo "[Exp18] start $(date)"
  bash exp18_multiframe_propagation_gated_dpo/scripts/prepare_exp18_cache_limit100_pai.sh
  bash exp18_multiframe_propagation_gated_dpo/scripts/launch_exp18_stage1_gates_pai.sh
  bash exp18_multiframe_propagation_gated_dpo/scripts/eval_exp18_davis10_pai.sh
  bash exp18_multiframe_propagation_gated_dpo/scripts/make_exp18_visuals_pai.sh
  echo "[Exp18] done $(date)"
} 2>&1 | tee "$LOG"
