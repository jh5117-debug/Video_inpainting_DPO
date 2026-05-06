#!/usr/bin/env bash
set -euo pipefail

H20_REPO="${H20_REPO:-/home/nvme01/H20_Video_inpainting_DPO}"
VIDEODPO_REPO="${VIDEODPO_REPO:-/home/nvme01/VideoDPO}"
PATCH_FILE="${PATCH_FILE:-${H20_REPO}/patches/videodpo/h20_videoinpaint_dpo_adapter.patch}"
LOG_ROOT="${LOG_ROOT:-${H20_REPO}/logs}"

mkdir -p "${LOG_ROOT}"

if [[ ! -d "${H20_REPO}" ]]; then
  echo "[apply] H20 repo not found: ${H20_REPO}" >&2
  exit 1
fi
if [[ ! -d "${VIDEODPO_REPO}" ]]; then
  echo "[apply] VideoDPO repo not found: ${VIDEODPO_REPO}" >&2
  exit 1
fi
if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "[apply] patch not found: ${PATCH_FILE}" >&2
  exit 1
fi

cd "${VIDEODPO_REPO}"

echo "[apply] repo=${VIDEODPO_REPO}"
echo "[apply] patch=${PATCH_FILE}"
echo "[apply] logs=${LOG_ROOT}"

if git status --short | grep -q .; then
  DIRTY_PATCH="${LOG_ROOT}/pre_videodpo_adapter_dirty_$(date +%Y%m%d_%H%M%S).patch"
  git diff > "${DIRTY_PATCH}"
  git stash push -u -m "pre videodpo h20 adapter"
  echo "[apply] saved dirty diff to ${DIRTY_PATCH}"
fi

git fetch origin
git checkout -B h20-videoinpaint-dpo-adapter origin/main

if [[ -f data/video_inpainting_dpo_data.py && -f scripts_sh/launch_vc2_dpo_videoinpainting_h20_gpu6_7.sh ]]; then
  echo "[apply] adapter already appears to be present; skipping git am"
else
  git am "${PATCH_FILE}"
fi

export LOG_ROOT
bash scripts_sh/launch_vc2_dpo_videoinpainting_h20_gpu6_7.sh

LOG="$(ls -t "${LOG_ROOT}"/vc2_dpo_videoinpainting_h20_gpu6-7_*.stdout.log | head -n 1)"
echo "[apply] tail log: ${LOG}"
tail -f "${LOG}"
