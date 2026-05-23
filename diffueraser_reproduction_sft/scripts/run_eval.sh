#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/lingbot_process.sh"

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: run_eval.sh"
  echo "Dry-run wrapper. Reuse existing DiffuEraser metric/VBench scripts."
  exit 0
fi

echo "[dry-run] Best setting: denoise_steps=6, use_pcm=false, gaussian_blur_after_composite=false."
echo "[dry-run] Process name default: $LINGBOT_PROCESS_NAME"
