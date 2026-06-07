#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
export PROJECT_ROOT

exec bash "${PROJECT_ROOT}/experiment_registry/exp07_fix_smallmask_prior/code/posthoc_davis_val_h20.sh" "$@"
