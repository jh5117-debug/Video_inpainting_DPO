#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
export PROJECT_ROOT

exec bash "${PROJECT_ROOT}/experiment_registry/exp08c_youtubevos_gtwin_d3comp_fullloss_davis_s1s2_2000/code/launch_s1s2_pai.sh" "$@"
