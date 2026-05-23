#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: run_train.sh"
  echo "Dry-run placeholder. Train with full-mask bridge using partialmask_comp generated data."
  exit 0
fi

echo "[dry-run] Reuse official_videodpo_diffueraser full-mask bridge; partial mask is not a training input."
