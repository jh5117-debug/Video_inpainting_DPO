#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: run_eval.sh"
  echo "Dry-run wrapper. Reuse VC2 VBench and SBS scripts recorded in PRD/02_pai_runbook.md."
  exit 0
fi

echo "[dry-run] Reuse recorded VC2 full VBench and SBS evaluation wrappers."
