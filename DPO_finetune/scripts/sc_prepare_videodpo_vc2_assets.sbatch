#!/bin/bash
#SBATCH --job-name=VDPO_ASSETS
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/videodpo-vc2-assets-%j.out

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

DEFAULT_VIDEODPO_REPO="${PROJECT_ROOT}/external/VideoDPO"
DEFAULT_VBENCH_ROOT="${PROJECT_ROOT}/external/VBench"
VIDEODPO_REPO="${VIDEODPO_REPO:-${DEFAULT_VIDEODPO_REPO}}"
VBENCH_ROOT="${VBENCH_ROOT:-${DEFAULT_VBENCH_ROOT}}"
HF_REPO_ID="${HF_REPO_ID:-JiaHuang01/vidpro10k-vc2-dataset}"
VC2_DATASET_ROOT="${VC2_DATASET_ROOT:-${PROJECT_DATA}/VideoDPO/data/vidpro-vc2-dpo-dataset}"
VC2_DATA_YAML="${VC2_DATA_YAML:-${PROJECT_DATA}/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml}"
HF_DOWNLOAD_DIR="${HF_DOWNLOAD_DIR:-${VC2_DATASET_ROOT}}"
DOWNLOAD_DATASET="${DOWNLOAD_DATASET:-1}"
CHECK_VBENCH="${CHECK_VBENCH:-1}"
INSTALL_VBENCH="${INSTALL_VBENCH:-0}"
CONDA_ENV="${CONDA_ENV:-${VIDEODPO_CONDA_ENV:-videodpo}}"

if [[ -n "${CONDA_BASE:-}" && -x "${CONDA_BASE}/bin/conda" ]]; then
  :
elif [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$("${CONDA_EXE}" info --base)"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  echo "[assets][error] conda not found; set CONDA_EXE or CONDA_BASE." >&2
  exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${PROJECT_ROOT}/logs" "$(dirname "${VC2_DATA_YAML}")" "$(dirname "${VC2_DATASET_ROOT}")"
echo "[assets] project_root=${PROJECT_ROOT}"
if command -v git >/dev/null 2>&1 && [[ -d "${PROJECT_ROOT}/.git" ]]; then
  echo "[assets] project_commit=$(git -C "${PROJECT_ROOT}" log -1 --oneline 2>/dev/null || true)"
fi
echo "[assets] project_dev=${PROJECT_DEV}"
echo "[assets] project_data=${PROJECT_DATA}"
echo "[assets] videodpo_repo=${VIDEODPO_REPO}"
echo "[assets] vbench_root=${VBENCH_ROOT}"
echo "[assets] vc2_dataset_root=${VC2_DATASET_ROOT}"
echo "[assets] vc2_data_yaml=${VC2_DATA_YAML}"

if [[ ! -f "${VIDEODPO_REPO}/scripts/train.py" ]]; then
  echo "[assets][error] VideoDPO submodule is missing or uninitialized: ${VIDEODPO_REPO}" >&2
  echo "[assets][error] Run: git submodule update --init --recursive" >&2
  exit 1
fi

if [[ "${DOWNLOAD_DATASET}" == "1" ]]; then
  python "${PROJECT_ROOT}/tools/prepare_videodpo_vc2_dataset.py" \
    --repo_id "${HF_REPO_ID}" \
    --target_root "${VC2_DATASET_ROOT}" \
    --download_dir "${HF_DOWNLOAD_DIR}" \
    --output_yaml "${VC2_DATA_YAML}"
else
  python "${PROJECT_ROOT}/tools/prepare_videodpo_vc2_dataset.py" \
    --target_root "${VC2_DATASET_ROOT}" \
    --output_yaml "${VC2_DATA_YAML}" \
    --skip_download
fi

if [[ "${CHECK_VBENCH}" == "1" ]]; then
  if [[ ! -f "${VBENCH_ROOT}/evaluate.py" ]]; then
    echo "[assets][error] VBench submodule is missing or uninitialized: ${VBENCH_ROOT}" >&2
    echo "[assets][error] Run: git submodule update --init --recursive" >&2
    exit 1
  fi
  if [[ "${INSTALL_VBENCH}" == "1" ]]; then
    echo "[assets][info] INSTALL_VBENCH=1 now means install/check VBench deps from the submodule; it will not git-clone another repo."
  fi
  if [[ -f "${VBENCH_ROOT}/requirements.txt" && "${INSTALL_VBENCH}" == "1" ]]; then
    python -m pip install -r "${VBENCH_ROOT}/requirements.txt"
  fi
  if [[ -f "${VBENCH_ROOT}/evaluate.py" ]]; then
    python "${VBENCH_ROOT}/evaluate.py" --help >/dev/null || true
    echo "[assets] VBench evaluate.py found."
  fi
fi

if [[ ! -f "${VIDEODPO_REPO}/prompts/vbench_standard_prompts.txt" ]]; then
  echo "[assets][warn] VideoDPO VBench prompt file missing: ${VIDEODPO_REPO}/prompts/vbench_standard_prompts.txt" >&2
fi

echo "[assets] done"
