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
mkdir -p "$EXP_DIR/reports" reports "$EVAL_ROOT"

"$PYTHON_BIN" "$EXP_DIR/code/eval_exp18_variants.py" \
  --eval_root "$EVAL_ROOT" \
  --output_md reports/exp18_davis10_eval_status.md

echo "[Exp18] eval status written to reports/exp18_davis10_eval_status.md"
