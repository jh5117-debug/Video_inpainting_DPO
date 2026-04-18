#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${PROJECT_ROOT}/inference/run_weight_sweep.sh" "$@"
