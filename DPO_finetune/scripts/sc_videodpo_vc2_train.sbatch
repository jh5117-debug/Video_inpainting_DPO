#!/bin/bash
#SBATCH --job-name=VDPO_VC2
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/dpo-stage1-%j.out

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
VC2_DATA_YAML="${VC2_DATA_YAML:-${PROJECT_DATA}/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DATA}/experiments/videodpo_vc2_dpo}"
RUN_NAME="${RUN_NAME:-sc-vc2-dpo-$(date -u +%Y%m%d_%H%M%S)}"
CONFIG="${CONFIG:-configs/vc2_dpo/config.yaml}"
NUM_GPUS="${NUM_GPUS:-8}"
DEVICE_LIST="${DEVICE_LIST:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_OPT_STEPS="${MAX_OPT_STEPS:-}"
CKPT_EVERY="${CKPT_EVERY:-499}"
NUM_WORKERS="${NUM_WORKERS:-16}"
BETA_DPO="${BETA_DPO:-5000.0}"
MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-65000 -n 1)}"
CONDA_ENV="${CONDA_ENV:-${VIDEODPO_CONDA_ENV:-videodpo}}"
ENABLE_WANDB="${ENABLE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-DPO_Diffueraser}"
WANDB_ENTITY="${WANDB_ENTITY:-jh5117-columbia-university}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-VideoDPO_VC2}"
WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-False}"
EARLY_WANDB="${EARLY_WANDB:-1}"

if [[ -n "${CONDA_BASE:-}" && -x "${CONDA_BASE}/bin/conda" ]]; then
  :
elif [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$("${CONDA_EXE}" info --base)"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  echo "[vc2-train][error] conda not found; set CONDA_EXE or CONDA_BASE." >&2
  exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if [[ ! -d "${VIDEODPO_REPO}" ]]; then
  echo "[vc2-train][error] VIDEODPO_REPO not found: ${VIDEODPO_REPO}" >&2
  echo "[vc2-train][error] Run from ${PROJECT_ROOT}: git submodule update --init --recursive" >&2
  exit 1
fi
if [[ ! -f "${VC2_DATA_YAML}" ]]; then
  echo "[vc2-train][error] VC2_DATA_YAML not found: ${VC2_DATA_YAML}" >&2
  echo "Run DPO_finetune/scripts/sc_prepare_videodpo_vc2_assets.sbatch first, or set VC2_DATA_YAML." >&2
  exit 1
fi
if [[ ! -f "${VIDEODPO_REPO}/checkpoints/vc2/model.ckpt" ]]; then
  echo "[vc2-train][error] VC2 base checkpoint missing: ${VIDEODPO_REPO}/checkpoints/vc2/model.ckpt" >&2
  exit 1
fi
if [[ ! -f "${VIDEODPO_REPO}/checkpoints/vc2/ref_model.ckpt" ]]; then
  echo "[vc2-train][error] VC2 ref checkpoint missing: ${VIDEODPO_REPO}/checkpoints/vc2/ref_model.ckpt" >&2
  exit 1
fi

RUN_DIR="${LOG_ROOT}/${RUN_NAME}"
mkdir -p "${LOG_ROOT}" "${PROJECT_ROOT}/logs" "${RUN_DIR}"
LAUNCH_LOG="${LAUNCH_LOG:-${RUN_DIR}/slurm_launch_stdout.log}"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEVICE_LIST}}"
export PYTHONPATH="${VIDEODPO_REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_PROJECT WANDB_ENTITY
export WANDB_DIR="${PROJECT_ROOT}/.wandb_cache"
export WANDB_CACHE_DIR="${PROJECT_ROOT}/.wandb_cache"
export WANDB_DATA_DIR="${PROJECT_ROOT}/.wandb_cache"
export WANDB_CONFIG_DIR="${PROJECT_ROOT}/.wandb_cache/config"
WANDB_RUN_ID="${WANDB_RUN_ID:-$(printf '%s' "${RUN_NAME}-${SLURM_JOB_ID:-manual}" | tr -cs 'A-Za-z0-9_.-' '-')}"
WANDB_RUN_ID="$(printf '%s' "${WANDB_RUN_ID}" | sed 's/^-//; s/-$//')"
WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_RUN_ID WANDB_RESUME
if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.netrc" ]]; then
  WANDB_API_KEY_FROM_NETRC=$(grep -A2 'api.wandb.ai' "${HOME}/.netrc" | grep password | awk '{print $2}' | head -n 1 || true)
  if [[ -n "${WANDB_API_KEY_FROM_NETRC}" ]]; then
    export WANDB_API_KEY="${WANDB_API_KEY_FROM_NETRC}"
  fi
fi
export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export TRANSFORMERS_CACHE="${PROJECT_ROOT}/.hf_cache"
export TMPDIR="${TMPDIR:-${PROJECT_ROOT}/.tmp}"
mkdir -p "${WANDB_DIR}" "${WANDB_CONFIG_DIR}" "${HF_HOME}" "${TMPDIR}"

wandb_event() {
  local event="$1"
  shift || true
  case "${ENABLE_WANDB,,}:${EARLY_WANDB,,}" in
    1:1|1:true|1:yes|1:on|true:1|true:true|true:yes|true:on|yes:1|yes:true|yes:yes|yes:on|on:1|on:true|on:yes|on:on)
      python "${PROJECT_ROOT}/tools/wandb_job_event.py" \
        --event "${event}" \
        --project "${WANDB_PROJECT}" \
        --entity "${WANDB_ENTITY}" \
        --group "${WANDB_RUN_GROUP}" \
        --run-name "${RUN_NAME}" \
        --run-id "${WANDB_RUN_ID}" \
        --run-dir "${WANDB_DIR}" \
        --log-file "${LAUNCH_LOG}" \
        "$@" || true
      ;;
  esac
}

on_error() {
  local rc="$?"
  trap - ERR
  echo "[vc2-train][error] failed with exit_code=${rc}"
  wandb_event failed --exit-code "${rc}" --tail-lines 260
  exit "${rc}"
}
trap on_error ERR

echo "[vc2-train] videodpo_repo=${VIDEODPO_REPO}"
echo "[vc2-train] data_yaml=${VC2_DATA_YAML}"
echo "[vc2-train] log_root=${LOG_ROOT}"
echo "[vc2-train] launch_log=${LAUNCH_LOG}"
echo "[vc2-train] run_name=${RUN_NAME}"
echo "[vc2-train] num_gpus=${NUM_GPUS} device_list=${DEVICE_LIST}"
echo "[vc2-train] max_opt_steps=${MAX_OPT_STEPS:-official-config-max_epochs} grad_accum=${GRAD_ACCUM} beta=${BETA_DPO}"
echo "[vc2-train] enable_wandb=${ENABLE_WANDB} early_wandb=${EARLY_WANDB} wandb_project=${WANDB_PROJECT} wandb_entity=${WANDB_ENTITY} wandb_dir=${WANDB_DIR} wandb_run_id=${WANDB_RUN_ID}"

wandb_event start \
  --config "run_name=${RUN_NAME}" \
  --config "num_gpus=${NUM_GPUS}" \
  --config "device_list=${DEVICE_LIST}" \
  --config "beta_dpo=${BETA_DPO}" \
  --config "vc2_data_yaml=${VC2_DATA_YAML}" \
  --config "videodpo_repo=${VIDEODPO_REPO}"

echo "[vc2-train] dependency preflight"
python - <<'PY'
import importlib
import os
import sys

videodpo_repo = os.environ["VIDEODPO_REPO"]
sys.path.insert(0, videodpo_repo)
os.chdir(videodpo_repo)

required = [
    "torch",
    "pytorch_lightning",
    "omegaconf",
    "kornia",
    "wandb",
    "open_clip",
    "transformers",
    "fairscale",
    "timm",
    "peft",
    "decord",
    "av",
]
missing = []
for name in required:
    try:
        module = importlib.import_module(name)
        print(f"[vc2-train][preflight][OK] {name} {getattr(module, '__version__', '')}".rstrip())
    except Exception as exc:
        print(f"[vc2-train][preflight][FAIL] {name}: {type(exc).__name__}: {exc}")
        missing.append(name)

try:
    importlib.import_module("lvdm.modules.encoders.condition")
    print("[vc2-train][preflight][OK] lvdm.modules.encoders.condition")
except Exception as exc:
    print(f"[vc2-train][preflight][FAIL] lvdm.modules.encoders.condition: {type(exc).__name__}: {exc}")
    missing.append("lvdm.modules.encoders.condition")

if missing:
    env = os.environ.get("CONDA_ENV", "diffueraser")
    print("[vc2-train][preflight][ERROR] missing/import-failing modules:", ", ".join(missing))
    print("[vc2-train][preflight][FIX] CONDA_ENV=%s INSTALL_MINIMAL=1 bash DPO_finetune/scripts/videodpo_env_smoke_and_export.sh" % env)
    raise SystemExit(2)
PY

MAX_STEPS_ARG=()
if [[ -n "${MAX_OPT_STEPS}" ]]; then
  MAX_STEPS_ARG=(lightning.trainer.max_steps="${MAX_OPT_STEPS}")
fi

LOGGER_ARGS=()
case "${ENABLE_WANDB,,}" in
  1|true|yes|on)
    LOGGER_ARGS=(
      lightning.logger.target=pytorch_lightning.loggers.WandbLogger
      lightning.logger.params.project="${WANDB_PROJECT}"
      lightning.logger.params.entity="${WANDB_ENTITY}"
      lightning.logger.params.name="${RUN_NAME}"
      lightning.logger.params.id="${WANDB_RUN_ID}"
      lightning.logger.params.resume="${WANDB_RESUME}"
      lightning.logger.params.save_dir="${LOG_ROOT}"
      lightning.logger.params.group="${WANDB_RUN_GROUP}"
      lightning.logger.params.log_model="${WANDB_LOG_MODEL}"
    )
    ;;
  *)
    echo "[vc2-train] W&B disabled; using official VideoDPO TensorBoard logger."
    ;;
esac

cd "${VIDEODPO_REPO}"
python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port "${MASTER_PORT}" \
  scripts/train.py \
  -t \
  --devices "${DEVICE_LIST}" \
  --base "${CONFIG}" \
  --name "${RUN_NAME}" \
  --logdir "${LOG_ROOT}" \
  --auto_resume True \
  lightning.trainer.num_nodes=1 \
  lightning.trainer.devices="${NUM_GPUS}" \
  lightning.trainer.accelerator=gpu \
  lightning.trainer.accumulate_grad_batches="${GRAD_ACCUM}" \
  lightning.trainer.log_every_n_steps=1 \
  lightning.trainer.precision=32 \
  lightning.callbacks.metrics_over_trainsteps_checkpoint.params.every_n_train_steps="${CKPT_EVERY}" \
  model.params.beta_dpo="${BETA_DPO}" \
  data.params.batch_size="${BATCH_SIZE}" \
  data.params.num_workers="${NUM_WORKERS}" \
  data.params.train.params.data_root="${VC2_DATA_YAML}" \
  "data.params.train.params.resolution=[320,512]" \
  data.params.train.params.video_length=16 \
  data.params.train.params.frame_stride=1 \
  "${LOGGER_ARGS[@]}" \
  "${MAX_STEPS_ARG[@]}"

trap - ERR
wandb_event finished --exit-code 0 --tail-lines 120
echo "[vc2-train] done"
echo "[vc2-train] checkpoints should be under ${LOG_ROOT}/${RUN_NAME}/checkpoints"
