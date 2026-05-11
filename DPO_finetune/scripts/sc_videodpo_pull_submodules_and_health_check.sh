#!/usr/bin/env bash
# One-command SC bootstrap after `git pull`: initialize repo-local submodules,
# then run the static VideoDPO/DiffuEraser health check. No GPU is requested.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "[bootstrap] project_root=${PROJECT_ROOT}"

if ! command -v git >/dev/null 2>&1; then
  echo "[bootstrap][error] git not found in PATH" >&2
  exit 1
fi

echo "[bootstrap] initializing/updating submodules"
git -C "${PROJECT_ROOT}" submodule update --init --recursive

echo "[bootstrap] running static health check"
exec bash "${SCRIPT_DIR}/sc_videodpo_health_check.sh"
