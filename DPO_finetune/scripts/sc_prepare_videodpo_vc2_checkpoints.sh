#!/usr/bin/env bash
# Prepare official VideoDPO VC2 checkpoints inside the repo-local VideoDPO
# submodule. This follows external/VideoDPO/readme.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VIDEODPO_REPO="${VIDEODPO_REPO:-${PROJECT_ROOT}/external/VideoDPO}"
CONDA_ENV="${CONDA_ENV:-${VIDEODPO_CONDA_ENV:-videodpo}}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/VideoCrafter/VideoCrafter2/resolve/main/model.ckpt}"
CKPT_DIR="${CKPT_DIR:-${VIDEODPO_REPO}/checkpoints/vc2}"
MODEL_CKPT="${MODEL_CKPT:-${CKPT_DIR}/model.ckpt}"
REF_CKPT="${REF_CKPT:-${CKPT_DIR}/ref_model.ckpt}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-1}"
CREATE_REF="${CREATE_REF:-1}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
FORCE_REF="${FORCE_REF:-0}"

if [[ -n "${CONDA_BASE:-}" && -x "${CONDA_BASE}/bin/conda" ]]; then
  :
elif [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$("${CONDA_EXE}" info --base)"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  echo "[vc2-ckpt][error] conda not found; set CONDA_EXE or CONDA_BASE." >&2
  exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if [[ ! -f "${VIDEODPO_REPO}/utils/create_ref_model.py" ]]; then
  echo "[vc2-ckpt][error] VideoDPO submodule missing: ${VIDEODPO_REPO}" >&2
  echo "[vc2-ckpt][error] Run: git submodule update --init --recursive" >&2
  exit 1
fi

if ! conda env list 2>/dev/null | awk '{print $1}' | grep -Fxq "${CONDA_ENV}"; then
  echo "[vc2-ckpt][error] conda env not found: ${CONDA_ENV}" >&2
  echo "[vc2-ckpt][error] Run DPO_finetune/scripts/videodpo_env_smoke_and_export.sh first, or set CONDA_ENV." >&2
  exit 1
fi

mkdir -p "${CKPT_DIR}"
echo "[vc2-ckpt] videodpo_repo=${VIDEODPO_REPO}"
echo "[vc2-ckpt] model_ckpt=${MODEL_CKPT}"
echo "[vc2-ckpt] ref_ckpt=${REF_CKPT}"

if [[ "${DOWNLOAD_MODEL}" == "1" && ( ! -s "${MODEL_CKPT}" || "${FORCE_DOWNLOAD}" == "1" ) ]]; then
  tmp="${MODEL_CKPT}.tmp"
  rm -f "${tmp}"
  echo "[vc2-ckpt] downloading ${MODEL_URL}"
  if command -v wget >/dev/null 2>&1; then
    wget -O "${tmp}" "${MODEL_URL}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "${MODEL_URL}" -o "${tmp}"
  else
    echo "[vc2-ckpt][error] neither wget nor curl is available" >&2
    exit 1
  fi
  mv "${tmp}" "${MODEL_CKPT}"
else
  echo "[vc2-ckpt] model checkpoint already exists or DOWNLOAD_MODEL=0"
fi

if [[ ! -s "${MODEL_CKPT}" ]]; then
  echo "[vc2-ckpt][error] model checkpoint missing or empty: ${MODEL_CKPT}" >&2
  exit 1
fi

if [[ "${CREATE_REF}" == "1" && ( ! -s "${REF_CKPT}" || "${FORCE_REF}" == "1" ) ]]; then
  echo "[vc2-ckpt] creating ref_model.ckpt with official utility"
  (
    cd "${VIDEODPO_REPO}"
    conda run --no-capture-output -n "${CONDA_ENV}" python utils/create_ref_model.py
  )
else
  echo "[vc2-ckpt] ref checkpoint already exists or CREATE_REF=0"
fi

if [[ ! -s "${REF_CKPT}" ]]; then
  echo "[vc2-ckpt][error] ref checkpoint missing or empty: ${REF_CKPT}" >&2
  exit 1
fi

ls -lh "${MODEL_CKPT}" "${REF_CKPT}"
echo "[vc2-ckpt] done"
