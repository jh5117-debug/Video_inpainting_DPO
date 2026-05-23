#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: run_eval.sh"
  echo "Dry-run wrapper. Real eval uses DPO_finetune/scripts/h20_diffueraser_fullmask_vbench.sh."
  exit 0
fi

echo "[dry-run] Reuse DPO_finetune/scripts/h20_diffueraser_fullmask_vbench.sh and table scripts."
