#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: run_eval.sh"
  echo "Dry-run placeholder. Evaluate with VBench, PSNR/SSIM, and qualitative SBS."
  exit 0
fi

echo "[dry-run] Evaluate generated/trained outputs with existing metric wrappers."
