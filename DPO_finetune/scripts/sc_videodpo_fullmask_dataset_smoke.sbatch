#!/bin/bash
#SBATCH --job-name=VDPO_FM_SMOKE
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/videodpo-fullmask-dataset-smoke-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="${PROJECT_NAME:-Video_inpainting_DPO}"
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${PROJECT_DEV:-}" ]]; then
    PROJECT_ROOT="${PROJECT_DEV}/${PROJECT_NAME}"
  else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  fi
fi
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

export PROJECT_HOME="${PROJECT_HOME:-/sc-projects/sc-proj-cc09-repair/hongyou}"
export PROJECT_DEV="${PROJECT_DEV:-${PROJECT_HOME}/dev}"
export PROJECT_DATA="${PROJECT_DATA:-${PROJECT_DEV}/data}"
export VIDEODPO_REPO="${VIDEODPO_REPO:-${PROJECT_ROOT}/external/VideoDPO}"
export VIDEODPO_DATA_BASE="${VIDEODPO_DATA_BASE:-${PROJECT_DATA}/VideoDPO}"

DPO_DATA_ROOT="${DPO_DATA_ROOT:-${PROJECT_DATA}/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${PROJECT_DATA}/Video_inpainting_DPO/weights}"
BASE_MODEL="${BASE_MODEL:-${WEIGHTS_DIR}/stable-diffusion-v1-5}"
CONDA_ENV="${CONDA_ENV:-${DIFFUERASER_CONDA_ENV:-diffueraser}}"

if [[ -n "${CONDA_BASE:-}" && -x "${CONDA_BASE}/bin/conda" ]]; then
  :
elif [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$("${CONDA_EXE}" info --base)"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  echo "[fullmask-smoke][error] conda not found; set CONDA_EXE or CONDA_BASE." >&2
  exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${PROJECT_ROOT}/logs"
export PYTHONPATH="${PROJECT_ROOT}:${VIDEODPO_REPO}:${PYTHONPATH:-}"
echo "[fullmask-smoke] project_root=${PROJECT_ROOT}"
echo "[fullmask-smoke] dpo_data_root=${DPO_DATA_ROOT}"
echo "[fullmask-smoke] base_model=${BASE_MODEL}"

python "${PROJECT_ROOT}/tools/smoke_videodpo_fullmask_dataset.py" \
  --dpo_data_root "${DPO_DATA_ROOT}" \
  --base_model_name_or_path "${BASE_MODEL}" \
  --resolution "${RESOLUTION:-512}" \
  --nframes "${NFRAMES:-16}" \
  --videodpo_frame_stride "${VIDEODPO_FRAME_STRIDE:-1}" \
  --videodpo_full_mask_value "${VIDEODPO_FULL_MASK_VALUE:-0.0}" \
  --index "${SMOKE_INDEX:-0}"
