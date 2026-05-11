#!/bin/bash
#SBATCH --job-name=VDPO_VC2
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=72:00:00
#SBATCH --output=logs/videodpo-vc2-train-%j.out

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

VIDEODPO_REPO="${VIDEODPO_REPO:-${PROJECT_DEV}/VideoDPO}"
VC2_DATA_YAML="${VC2_DATA_YAML:-${PROJECT_DATA}/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DATA}/experiments/videodpo_vc2_dpo}"
RUN_NAME="${RUN_NAME:-sc-vc2-dpo-$(date -u +%Y%m%d_%H%M%S)}"
CONFIG="${CONFIG:-configs/vc2_dpo/config.yaml}"
NUM_GPUS="${NUM_GPUS:-4}"
DEVICE_LIST="${DEVICE_LIST:-0,1,2,3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_OPT_STEPS="${MAX_OPT_STEPS:-}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
NUM_WORKERS="${NUM_WORKERS:-10}"
BETA_DPO="${BETA_DPO:-5000.0}"
MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-65000 -n 1)}"
CONDA_ENV="${CONDA_ENV:-${VIDEODPO_CONDA_ENV:-videodpo}}"

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

mkdir -p "${LOG_ROOT}" "${PROJECT_ROOT}/logs"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEVICE_LIST}}"
export PYTHONPATH="${VIDEODPO_REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING=1

echo "[vc2-train] videodpo_repo=${VIDEODPO_REPO}"
echo "[vc2-train] data_yaml=${VC2_DATA_YAML}"
echo "[vc2-train] log_root=${LOG_ROOT}"
echo "[vc2-train] run_name=${RUN_NAME}"
echo "[vc2-train] num_gpus=${NUM_GPUS} device_list=${DEVICE_LIST}"
echo "[vc2-train] max_opt_steps=${MAX_OPT_STEPS:-official-config-max_epochs} grad_accum=${GRAD_ACCUM} beta=${BETA_DPO}"

MAX_STEPS_ARG=()
if [[ -n "${MAX_OPT_STEPS}" ]]; then
  MAX_STEPS_ARG=(lightning.trainer.max_steps="${MAX_OPT_STEPS}")
fi

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
  "${MAX_STEPS_ARG[@]}"

echo "[vc2-train] done"
echo "[vc2-train] checkpoints should be under ${LOG_ROOT}/${RUN_NAME}/checkpoints"
