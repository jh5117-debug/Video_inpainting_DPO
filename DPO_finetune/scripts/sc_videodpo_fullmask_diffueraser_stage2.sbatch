#!/bin/bash
#SBATCH --job-name=VDPO_DE_S2
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=48:00:00
#SBATCH --output=logs/videodpo-fullmask-diffueraser-stage2-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="${PROJECT_NAME:-Video_inpainting_DPO}"
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
export PROJECT_ROOT

export WORLDMODELPHY_PROCESS_NAME="${WORLDMODELPHY_PROCESS_NAME:-lingbotworld-phy}"
export PROCESS_TITLE="${PROCESS_TITLE:-${WORLDMODELPHY_PROCESS_NAME}}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"

export PROJECT_HOME="${PROJECT_HOME:-$(dirname "${PROJECT_ROOT}")}"
export PROJECT_DEV="${PROJECT_DEV:-${PROJECT_HOME}}"
export PROJECT_DATA="${PROJECT_DATA:-${PROJECT_ROOT}}"
export DATA_NAME="${DATA_NAME:-Video_inpainting_DPO}"
export DATA="${DATA:-${PROJECT_ROOT}}"

if [[ -z "${WEIGHTS_DIR:-}" ]]; then
  for candidate in \
    "/mnt/nas/hj/weights" \
    "/mnt/nas/hj/hf_fullmask_assets/fullmask_required_assets/weights" \
    "${PROJECT_ROOT}/weights" \
    "${DATA}/weights"; do
    if [[ -d "${candidate}/stable-diffusion-v1-5" && -d "${candidate}/sd-vae-ft-mse" ]]; then
      export WEIGHTS_DIR="${candidate}"
      break
    fi
  done
  export WEIGHTS_DIR="${WEIGHTS_DIR:-${DATA}/weights}"
fi

export VIDEODPO_REPO="${VIDEODPO_REPO:-${PROJECT_ROOT}/external/VideoDPO}"
export VIDEODPO_DATA_BASE="${VIDEODPO_DATA_BASE:-${PROJECT_ROOT}/data/VideoDPO}"
export DPO_DATA_ROOT="${DPO_DATA_ROOT:-${PROJECT_ROOT}/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml}"
export DPO_DATASET_TYPE="videodpo_fullmask"
export VAL_DATA_DIR="${VAL_DATA_DIR:-${PROJECT_ROOT}/data/external/davis_432_240}"
export EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${PROJECT_ROOT}/experiments}"

if [[ ! -e "${DPO_DATA_ROOT}" ]]; then
  for candidate in \
    "${PROJECT_ROOT}/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.yaml" \
    "${PROJECT_ROOT}/external/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml" \
    "${PROJECT_ROOT}/external/VideoDPO/configs/vc2_dpo/vidpro/train_data.yaml" \
    "${VIDEODPO_REPO}/configs/vc2_dpo/vidpro/train_data.absolute.yaml" \
    "${VIDEODPO_REPO}/configs/vc2_dpo/vidpro/train_data.yaml"; do
    if [[ -f "${candidate}" ]]; then
      export DPO_DATA_ROOT="${candidate}"
      break
    fi
  done
fi

if [[ -z "${BASELINE_UNET_PATH:-}" ]]; then
  for candidate in \
    "/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000" \
    "${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000" \
    "${WEIGHTS_DIR}/diffuEraser/converted_weights_step34000" \
    "${PROJECT_ROOT}/finetune-stage2/converted_weights_step48000" \
    "${PROJECT_ROOT}/finetune-stage2/converted_weights_step34000"; do
    if [[ -f "${candidate}/unet_main/config.json" ]]; then
      export BASELINE_UNET_PATH="${candidate}"
      break
    fi
  done
fi
export BASELINE_UNET_PATH="${BASELINE_UNET_PATH:-${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000}"
export REF_MODEL_PATH="${REF_MODEL_PATH:-${BASELINE_UNET_PATH}}"

if [[ -z "${PRETRAINED_DPO_S1:-}" ]]; then
  if [[ -n "${STAGE1_WEIGHTS_PATH:-}" ]]; then
    export PRETRAINED_DPO_S1="${STAGE1_WEIGHTS_PATH}"
  elif [[ -f "${EXPERIMENTS_DIR}/dpo/stage1/LATEST" ]]; then
    latest_run="$(head -n 1 "${EXPERIMENTS_DIR}/dpo/stage1/LATEST")"
    if [[ -d "${latest_run}/last_weights" ]]; then
      export PRETRAINED_DPO_S1="${latest_run}/last_weights"
    elif [[ -d "${latest_run}/best_weights" ]]; then
      export PRETRAINED_DPO_S1="${latest_run}/best_weights"
    fi
  fi
fi

if [[ -z "${PRETRAINED_DPO_S1:-}" && -d "${EXPERIMENTS_DIR}/dpo/stage1" ]]; then
  latest_weights="$(
    find "${EXPERIMENTS_DIR}/dpo/stage1" -maxdepth 2 -type d \
      \( -name last_weights -o -name best_weights \) -path '*fullmask*' \
      -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
  if [[ -n "${latest_weights}" ]]; then
    export PRETRAINED_DPO_S1="${latest_weights}"
  fi
fi

if [[ -z "${PRETRAINED_DPO_S1:-}" || ! -d "${PRETRAINED_DPO_S1}" ]]; then
  echo "[videodpo-fullmask-stage2][error] Stage1 DPO weights not found." >&2
  echo "Set PRETRAINED_DPO_S1 or STAGE1_WEIGHTS_PATH to a stage1 last_weights/best_weights directory." >&2
  exit 1
fi

export RUN_NAME="${RUN_NAME:-sc-videodpo-fullmask-diffueraser-stage2}"
export MAX_EPOCHS="${MAX_EPOCHS:-10}"
export MAX_STEPS="${MAX_STEPS-}"
export CKPT_STEPS="${CKPT_STEPS:-499}"
export VAL_STEPS="${VAL_STEPS:-999999}"
export BETA_DPO="${BETA_DPO:-5000}"
export NUM_GPUS="${NUM_GPUS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export LR="${LR:-6e-6}"
export RESOLUTION="${RESOLUTION:-512}"
export TRAIN_HEIGHT="${TRAIN_HEIGHT:-320}"
export TRAIN_WIDTH="${TRAIN_WIDTH:-512}"
export NFRAMES="${NFRAMES:-16}"
export MIXED_PRECISION="${MIXED_PRECISION:-no}"
export SPLIT_POS_NEG_FORWARD="${SPLIT_POS_NEG_FORWARD:-0}"
export USE_8BIT_ADAM="${USE_8BIT_ADAM:-0}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
export VIDEODPO_FRAME_STRIDE="${VIDEODPO_FRAME_STRIDE:-1}"
export VIDEODPO_FULL_MASK_VALUE="${VIDEODPO_FULL_MASK_VALUE:-0.0}"
export WANDB_PROJECT="${WANDB_PROJECT:-DPO_Diffueraser}"
export REPORT_TO="${REPORT_TO:-wandb}"

if [[ -z "${CONDA_ENV:-}" ]]; then
  if [[ -d "/mnt/nas/hj/conda_envs/diffueraser" ]]; then
    export CONDA_ENV="/mnt/nas/hj/conda_envs/diffueraser"
  elif [[ -d "$(dirname "${PROJECT_ROOT}")/conda_envs/diffueraser" ]]; then
    export CONDA_ENV="$(dirname "${PROJECT_ROOT}")/conda_envs/diffueraser"
  fi
fi

if [[ ! -d "${VIDEODPO_REPO}" ]]; then
  echo "[videodpo-fullmask-stage2][error] VIDEODPO_REPO not found: ${VIDEODPO_REPO}" >&2
  exit 1
fi
if [[ ! -e "${DPO_DATA_ROOT}" ]]; then
  echo "[videodpo-fullmask-stage2][error] DPO_DATA_ROOT not found: ${DPO_DATA_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${BASELINE_UNET_PATH}/unet_main/config.json" ]]; then
  echo "[videodpo-fullmask-stage2][error] BASELINE_UNET_PATH invalid: ${BASELINE_UNET_PATH}" >&2
  exit 1
fi

if [[ "${DPO_DATASET_TYPE}" == "videodpo_fullmask" ]] && command -v python3 >/dev/null 2>&1; then
  DPO_DATA_ROOT="${DPO_DATA_ROOT}" \
  VIDEODPO_DATA_BASE="${VIDEODPO_DATA_BASE:-}" \
  VIDEODPO_REPO="${VIDEODPO_REPO:-}" \
  PROJECT_ROOT="${PROJECT_ROOT}" \
  python3 - <<'PY'
from pathlib import Path
import os
import re
import sys


def read_meta(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
        cfg = yaml.safe_load(text) or {}
        return [str(x) for x in cfg.get("META", [])]
    except Exception:
        metas = []
        in_meta = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("META:"):
                in_meta = True
                metas.extend(re.findall(r"['\"]([^'\"]+)['\"]", stripped.split(":", 1)[1]))
                continue
            if in_meta and stripped.startswith("-"):
                metas.append(stripped[1:].strip().strip("'\""))
            elif in_meta and stripped and not line.startswith((" ", "\t")):
                in_meta = False
        return metas


def valid_root(root: Path) -> bool:
    return (root / "metadata.json").is_file() and (root / "pair.json").is_file()


raw = Path(os.environ["DPO_DATA_ROOT"]).expanduser()
project_root = Path(os.environ["PROJECT_ROOT"]).expanduser()
cwd = Path.cwd()
checked = []

if raw.is_dir():
    roots = [raw]
elif raw.is_file() and raw.suffix.lower() in {".yaml", ".yml"}:
    roots = []
    for item in read_meta(raw):
        meta = Path(item).expanduser()
        if meta.is_absolute():
            candidates = [meta]
        else:
            candidates = []
            if os.environ.get("VIDEODPO_DATA_BASE"):
                candidates.append(Path(os.environ["VIDEODPO_DATA_BASE"]).expanduser() / meta)
            if os.environ.get("VIDEODPO_REPO"):
                candidates.append(Path(os.environ["VIDEODPO_REPO"]).expanduser() / meta)
            candidates.extend([raw.parent / meta, project_root / meta, cwd / meta])
        seen = set()
        unique = []
        for candidate in candidates:
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        checked.extend(unique)
        match = next((candidate for candidate in unique if valid_root(candidate)), None)
        if match is None:
            print("[videodpo-fullmask-stage2][error] VideoDPO META root is missing metadata.json/pair.json.", file=sys.stderr)
            print(f"  train_data={raw}", file=sys.stderr)
            print(f"  meta={item}", file=sys.stderr)
            print("  checked:", file=sys.stderr)
            for candidate in unique:
                print(f"    - {candidate}", file=sys.stderr)
            sys.exit(1)
        roots.append(match)
else:
    print(f"[videodpo-fullmask-stage2][error] DPO_DATA_ROOT is not a directory/yaml: {raw}", file=sys.stderr)
    sys.exit(1)

if not roots:
    print(f"[videodpo-fullmask-stage2][error] no META roots in {raw}", file=sys.stderr)
    sys.exit(1)

for root in roots:
    if not valid_root(root):
        print(f"[videodpo-fullmask-stage2][error] missing metadata.json/pair.json under {root}", file=sys.stderr)
        sys.exit(1)
    print(f"[videodpo-fullmask-stage2] data_root_ok={root}")
PY
fi

echo "[videodpo-fullmask-stage2] project_root=${PROJECT_ROOT}"
echo "[videodpo-fullmask-stage2] process_name=${WORLDMODELPHY_PROCESS_NAME}"
echo "[videodpo-fullmask-stage2] dpo_data_root=${DPO_DATA_ROOT}"
echo "[videodpo-fullmask-stage2] stage1_weights=${PRETRAINED_DPO_S1}"
echo "[videodpo-fullmask-stage2] baseline_unet=${BASELINE_UNET_PATH}"
echo "[videodpo-fullmask-stage2] ref_model=${REF_MODEL_PATH}"
echo "[videodpo-fullmask-stage2] weights_dir=${WEIGHTS_DIR}"
echo "[videodpo-fullmask-stage2] run_name=${RUN_NAME}"
echo "[videodpo-fullmask-stage2] max_epochs=${MAX_EPOCHS} max_steps=${MAX_STEPS:-auto} beta=${BETA_DPO} lr=${LR} grad_accum=${GRAD_ACCUM} train_size=${TRAIN_HEIGHT}x${TRAIN_WIDTH} split_pos_neg=${SPLIT_POS_NEG_FORWARD} report_to=${REPORT_TO}"

exec bash "${PROJECT_ROOT}/DPO_finetune/scripts/03_dpo_stage2.sbatch"
