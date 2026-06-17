#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"

EXP_DIR="exp18_multiframe_propagation_gated_dpo"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
EVAL_ROOT="${EXP18_EVAL_ROOT:-logs/target_eval/exp18_multiframe_propagation_gated_dpo_davis10}"
VIS_OUT="${EXP18_VIS_OUT:-${EXP_DIR}/visual_cases/davis10_index}"

"$PYTHON_BIN" "$EXP_DIR/code/make_exp18_visuals.py" \
  --eval_root "$EVAL_ROOT" \
  --output_dir "$VIS_OUT"

echo "[Exp18] visual index: $VIS_OUT"
