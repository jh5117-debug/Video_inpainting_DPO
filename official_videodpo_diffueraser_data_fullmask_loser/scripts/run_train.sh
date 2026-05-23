#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/lingbot_process.sh"

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: run_train.sh"
  echo "Dry-run placeholder. Reuse official_videodpo_diffueraser launchers with GENERATED_LOSER_ROOT manifest/data."
  exit 0
fi

echo "[dry-run] Train by reusing official_videodpo_diffueraser full-mask bridge."
echo "[dry-run] Set data root/manifest from: ${GENERATED_LOSER_ROOT:-$REPO_ROOT/data/generated_losers/fullmask}"
echo "[dry-run] Process name default: $LINGBOT_PROCESS_NAME"
