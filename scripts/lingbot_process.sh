#!/usr/bin/env bash
set -euo pipefail

export LINGBOT_PROCESS_NAME="${LINGBOT_PROCESS_NAME:-lingbot-phy}"

lingbot_exec_python() {
  local python_bin="${PYTHON:-python}"
  exec -a "$LINGBOT_PROCESS_NAME" "$python_bin" "$@"
}

lingbot_run_python() {
  local python_bin="${PYTHON:-python}"
  "$python_bin" "$@"
}
