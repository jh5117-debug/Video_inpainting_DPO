#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp23_pool_sweep}"
cd "${PROJECT_ROOT}"

CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/mnt/nas/hj/conda_envs/diffueraser}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PREFIX}/bin/python}"
PHY_PYTHON="${PHY_PYTHON:-${CONDA_ENV_PREFIX}/bin/Phy}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[exp23][ERROR] python not found: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ ! -x "${PHY_PYTHON}" ]]; then
  cp --reflink=auto "${PYTHON_BIN}" "${PHY_PYTHON}" 2>/dev/null || cp "${PYTHON_BIN}" "${PHY_PYTHON}"
  chmod 755 "${PHY_PYTHON}"
fi

"${PHY_PYTHON}" - <<'PY'
import sys
import torch
print("[exp23] phy executable:", sys.executable)
print("[exp23] torch:", torch.__version__)
print("[exp23] cuda:", torch.version.cuda)
PY

export PHY_PYTHON
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PROCESS_TITLE=Phy
export SETPROCTITLE=Phy
export LINGBOT_PROCESS_NAME=Phy
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_SILENT=true
export WANDB_QUIET=true
export WANDB_CONSOLE=off

mkdir -p logs/pipelines exp23_two_stage_pool_morphology_sweep/runtime
LOG_PATH="${LOG_PATH:-logs/pipelines/exp23_phy_sweep_controller.log}"
PID_PATH="${PID_PATH:-exp23_two_stage_pool_morphology_sweep/runtime/exp23_phy_sweep_controller.pid}"

nohup "${PHY_PYTHON}" exp23_two_stage_pool_morphology_sweep/code/exp23_trial_runner.py \
  --pair-id "${PAIR_ID:-phaseA_scale1_pair001_outer2}" \
  --gpus "${CUDA_VISIBLE_DEVICES}" \
  --nproc-per-node 4 \
  --phy-python "${PHY_PYTHON}" \
  > "${LOG_PATH}" 2>&1 &

echo "$!" > "${PID_PATH}"
echo "[exp23] controller_pid=$(cat "${PID_PATH}")"
echo "[exp23] log_path=${PROJECT_ROOT}/${LOG_PATH}"
echo "[exp23] pid_path=${PROJECT_ROOT}/${PID_PATH}"
