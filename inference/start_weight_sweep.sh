#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

EXP_ROOT="${EXP_ROOT:-${PROJECT_ROOT}/experiments/evaluation/weight_sweep}"
LOG_DIR="${EXP_ROOT}/logs_${TIMESTAMP}"
LOG_FILE="${LOG_DIR}/nohup_${TIMESTAMP}.log"
mkdir -p "${LOG_DIR}"

echo "Starting DiffuEraser weight sweep in background..."
echo "  Main log: ${LOG_FILE}"
echo "  Per-experiment logs: ${LOG_DIR}/"

LOG_DIR="${LOG_DIR}" nohup bash "${SCRIPT_DIR}/run_weight_sweep.sh" > "${LOG_FILE}" 2>&1 &
MAIN_PID=$!

echo "  PID: ${MAIN_PID}"
echo ""
echo "Monitor with:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "Stop with:"
echo "  kill ${MAIN_PID}"
