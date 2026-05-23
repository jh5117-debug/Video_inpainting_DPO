#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/lingbot_process.sh"

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: run_train.sh"
  echo "Dry-run wrapper. Real PAI training entrypoint: DPO_finetune/scripts/pai_videodpo_vc2_official_repro.sh"
  exit 0
fi

echo "[dry-run] Reuse DPO_finetune/scripts/pai_videodpo_vc2_official_repro.sh"
echo "[dry-run] Process name default: $LINGBOT_PROCESS_NAME"
