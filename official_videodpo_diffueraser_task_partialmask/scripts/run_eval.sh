#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: run_eval.sh"
  echo "Dry-run placeholder. Evaluate task-partialmask outputs with PSNR/SSIM/VBench/SBS."
  exit 0
fi

echo "[dry-run] Evaluate partial-mask task outputs after explicit implementation."
