#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/lingbot_process.sh"

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: run_train.sh"
  echo "Dry-run wrapper. Real PAI entrypoint: DPO_finetune/scripts/pai_official_diffueraser_stage.sh"
  exit 0
fi

echo "[dry-run] Reuse DPO_finetune/scripts/pai_official_diffueraser_stage.sh for stage1/stage2."
echo "[dry-run] Process name default: $LINGBOT_PROCESS_NAME"
