#!/usr/bin/env bash
# One-command SC helper: update repo-local dependencies, install the minimal
# VideoDPO extras into the selected conda env, then run the health check.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-${VIDEODPO_CONDA_ENV:-diffueraser}}"
INSTALL_MINIMAL="${INSTALL_MINIMAL:-1}"
EXPORT_ENV="${EXPORT_ENV:-1}"

echo "[fix-env] project_root=${PROJECT_ROOT}"
echo "[fix-env] conda_env=${CONDA_ENV}"
echo "[fix-env] install_minimal=${INSTALL_MINIMAL}"
echo "[fix-env] export_env=${EXPORT_ENV}"

if ! command -v git >/dev/null 2>&1; then
  echo "[fix-env][error] git not found in PATH" >&2
  exit 1
fi

echo "[fix-env] initializing/updating submodules"
git -C "${PROJECT_ROOT}" submodule update --init --recursive

echo "[fix-env] installing/smoking VideoDPO minimal env extras"
CONDA_ENV="${CONDA_ENV}" INSTALL_MINIMAL="${INSTALL_MINIMAL}" EXPORT_ENV="${EXPORT_ENV}" \
  bash "${SCRIPT_DIR}/videodpo_env_smoke_and_export.sh"

echo "[fix-env] running health check"
CONDA_ENV="${CONDA_ENV}" bash "${SCRIPT_DIR}/sc_videodpo_health_check.sh"

echo "[fix-env] done"
