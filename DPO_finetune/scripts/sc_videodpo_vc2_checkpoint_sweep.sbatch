#!/bin/bash
#SBATCH --job-name=VDPO_CKSW
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --output=logs/videodpo-vc2-ckpt-sweep-%j.out

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

VIDEODPO_REPO="${VIDEODPO_REPO:-${PROJECT_ROOT}/external/VideoDPO}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DATA}/experiments/videodpo_vc2_dpo}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-${RUN_NAME:-}}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-}"
if [[ -z "${TRAIN_RUN_DIR}" ]]; then
  if [[ -z "${TRAIN_RUN_NAME}" ]]; then
    echo "[ckpt-sweep][error] Set TRAIN_RUN_NAME or TRAIN_RUN_DIR." >&2
    exit 1
  fi
  TRAIN_RUN_DIR="${LOG_ROOT}/${TRAIN_RUN_NAME}"
fi

CKPT_DIR="${CKPT_DIR:-${TRAIN_RUN_DIR}/checkpoints/trainstep_checkpoints}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_DATA}/experiments/videodpo_vbench_sweeps/$(basename "${TRAIN_RUN_DIR}")}"
SELECTION_DIR="${SELECTION_DIR:-${TRAIN_RUN_DIR}/checkpoints/selected_vbench}"
SCORE_KEY="${SCORE_KEY:-total_score}"
LINK_MODE="${LINK_MODE:-hardlink}"
PRUNE_AFTER="${PRUNE_AFTER:-0}"
INCLUDE_LAST_IN_SWEEP="${INCLUDE_LAST_IN_SWEEP:-0}"
INCLUDE_BASE_IN_SWEEP="${INCLUDE_BASE_IN_SWEEP:-0}"
MAX_CKPTS="${MAX_CKPTS:-0}"
SELECT_BEST="${SELECT_BEST:-1}"

mkdir -p "${OUT_ROOT}" "${PROJECT_ROOT}/logs"

if [[ ! -d "${TRAIN_RUN_DIR}" ]]; then
  echo "[ckpt-sweep][error] TRAIN_RUN_DIR not found: ${TRAIN_RUN_DIR}" >&2
  exit 1
fi
if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "[ckpt-sweep][error] CKPT_DIR not found: ${CKPT_DIR}" >&2
  exit 1
fi

manifest="${OUT_ROOT}/checkpoint_manifest.tsv"
: > "${manifest}"

count=0
while IFS= read -r ckpt; do
  label="$(basename "${ckpt}")"
  label="${label%.ckpt}"
  label="${label//[^A-Za-z0-9_.=-]/_}"
  printf '%s\t%s\n' "${label}" "${ckpt}" >> "${manifest}"
  count=$((count + 1))
  if [[ "${MAX_CKPTS}" != "0" && "${count}" -ge "${MAX_CKPTS}" ]]; then
    break
  fi
done < <(find "${CKPT_DIR}" -maxdepth 1 -type f -name '*.ckpt' | sort -V)

if [[ "${INCLUDE_LAST_IN_SWEEP}" == "1" && -f "${TRAIN_RUN_DIR}/checkpoints/last.ckpt" ]]; then
  printf '%s\t%s\n' "last" "${TRAIN_RUN_DIR}/checkpoints/last.ckpt" >> "${manifest}"
fi

if [[ "${INCLUDE_BASE_IN_SWEEP}" == "1" && -f "${VIDEODPO_REPO}/checkpoints/vc2/model.ckpt" ]]; then
  printf '%s\t%s\n' "vc2_base" "${VIDEODPO_REPO}/checkpoints/vc2/model.ckpt" >> "${manifest}"
fi

if [[ ! -s "${manifest}" ]]; then
  echo "[ckpt-sweep][error] no checkpoints found in ${CKPT_DIR}" >&2
  exit 1
fi

echo "[ckpt-sweep] train_run_dir=${TRAIN_RUN_DIR}"
echo "[ckpt-sweep] ckpt_dir=${CKPT_DIR}"
echo "[ckpt-sweep] out_root=${OUT_ROOT}"
echo "[ckpt-sweep] selection_dir=${SELECTION_DIR}"
echo "[ckpt-sweep] manifest=${manifest}"
cat "${manifest}"

ckpt_specs=""
while IFS=$'\t' read -r label ckpt; do
  if [[ -z "${ckpt_specs}" ]]; then
    ckpt_specs="${label}:${ckpt}"
  else
    ckpt_specs="${ckpt_specs},${label}:${ckpt}"
  fi
done < "${manifest}"

CKPT_SPECS="${ckpt_specs}" \
OUT_ROOT="${OUT_ROOT}" \
bash "${PROJECT_ROOT}/DPO_finetune/scripts/sc_videodpo_vc2_vbench.sbatch"

if [[ "${SELECT_BEST}" != "1" ]]; then
  echo "[ckpt-sweep] SELECT_BEST=${SELECT_BEST}; skipping checkpoint selection"
  exit 0
fi

selector_args=(
  "${PROJECT_ROOT}/tools/select_best_vbench_checkpoint.py"
  --sweep_root "${OUT_ROOT}"
  --checkpoint_manifest "${manifest}"
  --train_run_dir "${TRAIN_RUN_DIR}"
  --selection_dir "${SELECTION_DIR}"
  --score_key "${SCORE_KEY}"
  --link_mode "${LINK_MODE}"
)
if [[ "${PRUNE_AFTER}" == "1" ]]; then
  selector_args+=(--prune_trainstep_checkpoints)
fi

python "${selector_args[@]}"

echo "[ckpt-sweep] done"
echo "[ckpt-sweep] selected checkpoints: ${SELECTION_DIR}"
