#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/lingbot_process.sh"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: run_train.sh

Dry-run placeholder for partial-mask task training.
This should not reuse full-mask bridge unchanged; it needs an explicit partial-mask task adapter.
EOF
  exit 0
fi

echo "[dry-run] Partial-mask task training is scaffolded only."
echo "[dry-run] First version should use same-mask data from partialmask_loser_comp."
echo "[dry-run] Process name default: $LINGBOT_PROCESS_NAME"
