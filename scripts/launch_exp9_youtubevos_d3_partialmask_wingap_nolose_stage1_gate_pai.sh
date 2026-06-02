#!/bin/bash
set -euo pipefail

# Prepared next gate only. Do not run until Exp9 lose025 gate and target
# metric reports show loser-degradation artifacts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

export EXP_NAME="${EXP_NAME:-exp9_youtubevos_d3_partialmask_wingap_nolose_stage1_gate1000}"
export RUN_NAME="${RUN_NAME:-${EXP_NAME}_stage1}"
export STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:-1000}"
export MAX_STEPS="${MAX_STEPS:-${STAGE1_MAX_STEPS}}"
export CKPT_STEPS="${CKPT_STEPS:-500}"
export CKPT_LIMIT="${CKPT_LIMIT:-5}"
export LOSE_GAP_WEIGHT="0.0"
export DPO_LOSE_GAP_WEIGHT="0.0"

exec bash "${PROJECT_ROOT}/scripts/launch_exp9_youtubevos_d3_partialmask_wingap_stage1_gate_pai.sh"
