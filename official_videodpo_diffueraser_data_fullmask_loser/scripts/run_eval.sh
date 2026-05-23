#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: run_eval.sh"
  echo "Dry-run placeholder. Reuse existing fullmask VBench/SBS evaluation wrappers."
  exit 0
fi

echo "[dry-run] Reuse tools/generate_diffueraser_fullmask_vbench.py and VBench table scripts."
