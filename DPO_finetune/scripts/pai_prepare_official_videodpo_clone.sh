#!/usr/bin/env bash
# Clone a clean upstream VideoDPO checkout on PAI and attach local VC2 assets.
#
# This script intentionally keeps the official checkout separate from
# external/VideoDPO so previous diagnostics or inpainting-adapter patches cannot
# affect paper-reproduction runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

OFFICIAL_REPO_URL="${OFFICIAL_REPO_URL:-https://github.com/CIntellifusion/VideoDPO.git}"
OFFICIAL_VIDEODPO_REPO="${OFFICIAL_VIDEODPO_REPO:-/mnt/nas/hj/official_repos/VideoDPO_official_1febdb4}"
VIDEODPO_REF="${VIDEODPO_REF:-1febdb4}"
FETCH_OFFICIAL="${FETCH_OFFICIAL:-1}"
RESET_OFFICIAL_REPO="${RESET_OFFICIAL_REPO:-0}"

SOURCE_VC2_CKPT_DIR="${SOURCE_VC2_CKPT_DIR:-${PROJECT_ROOT}/external/VideoDPO/checkpoints/vc2}"
VC2_DATA_YAML="${VC2_DATA_YAML:-/mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.pai.yaml}"

mkdir -p "$(dirname "${OFFICIAL_VIDEODPO_REPO}")" "${PROJECT_ROOT}/logs"

echo "[official-videodpo] project_root=${PROJECT_ROOT}"
echo "[official-videodpo] repo_url=${OFFICIAL_REPO_URL}"
echo "[official-videodpo] repo=${OFFICIAL_VIDEODPO_REPO}"
echo "[official-videodpo] ref=${VIDEODPO_REF}"
echo "[official-videodpo] source_vc2_ckpt_dir=${SOURCE_VC2_CKPT_DIR}"
echo "[official-videodpo] vc2_data_yaml=${VC2_DATA_YAML}"

if [[ -d "${OFFICIAL_VIDEODPO_REPO}/.git" ]]; then
  dirty_status="$(git -C "${OFFICIAL_VIDEODPO_REPO}" status --short --untracked-files=all | grep -vE '^[?][?] checkpoints/' || true)"
  if [[ -n "${dirty_status}" ]]; then
    if [[ "${RESET_OFFICIAL_REPO}" != "1" ]]; then
      echo "[official-videodpo][error] existing clean repo has local changes:" >&2
      printf '%s\n' "${dirty_status}" >&2
      echo "[official-videodpo][error] Set RESET_OFFICIAL_REPO=1 to discard changes in this isolated official clone." >&2
      exit 1
    fi
    git -C "${OFFICIAL_VIDEODPO_REPO}" reset --hard
    git -C "${OFFICIAL_VIDEODPO_REPO}" clean -fdx
  fi
  if [[ "${FETCH_OFFICIAL}" == "1" ]]; then
    git -C "${OFFICIAL_VIDEODPO_REPO}" fetch origin
  fi
else
  git clone "${OFFICIAL_REPO_URL}" "${OFFICIAL_VIDEODPO_REPO}"
fi

git -C "${OFFICIAL_VIDEODPO_REPO}" checkout --detach "${VIDEODPO_REF}"

mkdir -p "${OFFICIAL_VIDEODPO_REPO}/checkpoints/vc2"
for name in model.ckpt ref_model.ckpt; do
  src="${SOURCE_VC2_CKPT_DIR}/${name}"
  dst="${OFFICIAL_VIDEODPO_REPO}/checkpoints/vc2/${name}"
  if [[ ! -s "${src}" ]]; then
    echo "[official-videodpo][error] missing VC2 asset: ${src}" >&2
    exit 1
  fi
  ln -sfn "${src}" "${dst}"
done

if [[ ! -s "${VC2_DATA_YAML}" ]]; then
  echo "[official-videodpo][error] VC2_DATA_YAML missing: ${VC2_DATA_YAML}" >&2
  exit 1
fi

echo "[official-videodpo] commit=$(git -C "${OFFICIAL_VIDEODPO_REPO}" log -1 --oneline)"
echo "[official-videodpo] status:"
git -C "${OFFICIAL_VIDEODPO_REPO}" status --short
echo "[official-videodpo] dpo_loss snippet:"
grep -n 'def dpo_loss' -A12 "${OFFICIAL_VIDEODPO_REPO}/lvdm/models/ddpm3d.py" | head -30
echo "[official-videodpo] dataset target:"
grep -n 'target: data.video_data.TextVideoDPO' -A6 "${OFFICIAL_VIDEODPO_REPO}/configs/vc2_dpo/config.yaml"
echo "[official-videodpo] ready"
