#!/usr/bin/env bash
# Run DiffuEraser inside the official VideoDPO Lightning train/config skeleton.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

STAGE="${STAGE:-stage1}"
case "${STAGE}" in
  stage1|stage2) ;;
  *) echo "[official-diffueraser][error] STAGE must be stage1 or stage2, got: ${STAGE}" >&2; exit 1 ;;
esac

OFFICIAL_SOURCE_REPO="${OFFICIAL_SOURCE_REPO:-/mnt/nas/hj/official_repos/VideoDPO_official_1febdb4}"
OFFICIAL_VIDEODPO_REPO="${OFFICIAL_VIDEODPO_REPO:-/mnt/nas/hj/official_repos/VideoDPO_official_diffueraser_1febdb4}"
OFFICIAL_REPO_URL="${OFFICIAL_REPO_URL:-https://github.com/CIntellifusion/VideoDPO.git}"
VIDEODPO_REF="${VIDEODPO_REF:-1febdb4}"
RESET_OFFICIAL_REPO="${RESET_OFFICIAL_REPO:-1}"
FETCH_OFFICIAL="${FETCH_OFFICIAL:-0}"

CONDA_ENV="${CONDA_ENV:-/mnt/nas/hj/conda_envs/diffueraser}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/mnt/nas/hj/weights}"
VC2_DATA_YAML="${VC2_DATA_YAML:-/mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.pai.yaml}"
BASE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH:-${WEIGHTS_DIR}/stable-diffusion-v1-5}"
VAE_PATH="${VAE_PATH:-${WEIGHTS_DIR}/sd-vae-ft-mse}"
REF_MODEL_PATH="${REF_MODEL_PATH:-${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000}"
BASELINE_UNET_PATH="${BASELINE_UNET_PATH:-${REF_MODEL_PATH}}"
PRETRAINED_DPO_S1="${PRETRAINED_DPO_S1:-}"

NUM_GPUS="${NUM_GPUS:-4}"
DEVICE_LIST="${DEVICE_LIST:-4,5,6,7}"
CUDA_DEVICE_LIST="${CUDA_VISIBLE_DEVICES:-${DEVICE_LIST}}"
PL_DEVICE_LIST="${PL_DEVICE_LIST:-}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_OPT_STEPS="${MAX_OPT_STEPS:-1}"
CKPT_EVERY="${CKPT_EVERY:-1}"
NUM_WORKERS="${NUM_WORKERS:-16}"
LR="${LR:-6e-6}"
BETA_DPO="${BETA_DPO:-5000.0}"
LOSE_GAP_WEIGHT="${LOSE_GAP_WEIGHT:-1.0}"
PRECISION="${PRECISION:-32}"
FORWARD_DTYPE="${FORWARD_DTYPE:-fp32}"
VAE_DTYPE="${VAE_DTYPE:-fp32}"
REF_DTYPE="${REF_DTYPE:-fp32}"
TEXT_DTYPE="${TEXT_DTYPE:-fp32}"
SPLIT_POS_NEG_FORWARD="${SPLIT_POS_NEG_FORWARD:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
USE_8BIT_ADAM="${USE_8BIT_ADAM:-0}"
MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-65000 -n 1)}"
TMPDIR="${TMPDIR:-/tmp/hj_videodpo_diffueraser_tmp}"

RUN_NAME="${RUN_NAME:-pai-official-diffueraser-${STAGE}-smoke-gpu${DEVICE_LIST//,/}-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/official_diffueraser_${STAGE}}"
RUN_DIR="${LOG_ROOT}/${RUN_NAME}"
LAUNCH_LOG="${LAUNCH_LOG:-${RUN_DIR}/slurm_launch_stdout.log}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/DPO_finetune/configs/official_diffueraser_${STAGE}.yaml}"
SKIP_TEST_PATCH="${SKIP_TEST_PATCH:-${PROJECT_ROOT}/patches/videodpo/sc_vc2_skip_implicit_test.patch}"

mkdir -p "$(dirname "${OFFICIAL_VIDEODPO_REPO}")" "${RUN_DIR}" "${TMPDIR}"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

echo "[official-diffueraser] stage=${STAGE}"
echo "[official-diffueraser] project_root=${PROJECT_ROOT}"
echo "[official-diffueraser] official_repo=${OFFICIAL_VIDEODPO_REPO}"
echo "[official-diffueraser] source_repo=${OFFICIAL_SOURCE_REPO}"
echo "[official-diffueraser] config=${CONFIG}"
echo "[official-diffueraser] run_name=${RUN_NAME}"
echo "[official-diffueraser] device_list=${DEVICE_LIST} num_gpus=${NUM_GPUS} batch=${BATCH_SIZE} grad_accum=${GRAD_ACCUM} global_batch=$((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM)) max_steps=${MAX_OPT_STEPS}"
echo "[official-diffueraser] cuda_visible_devices=${CUDA_DEVICE_LIST}"
echo "[official-diffueraser] data=${VC2_DATA_YAML}"
echo "[official-diffueraser] ref_model=${REF_MODEL_PATH}"
echo "[official-diffueraser] baseline_unet=${BASELINE_UNET_PATH}"
echo "[official-diffueraser] pretrained_dpo_s1=${PRETRAINED_DPO_S1:-unset}"

if [[ ! -d "${OFFICIAL_VIDEODPO_REPO}/.git" ]]; then
  if [[ -d "${OFFICIAL_SOURCE_REPO}/.git" ]]; then
    echo "[official-diffueraser] cloning from local source clone"
    git clone --no-hardlinks "${OFFICIAL_SOURCE_REPO}" "${OFFICIAL_VIDEODPO_REPO}"
  else
    echo "[official-diffueraser] cloning from ${OFFICIAL_REPO_URL}"
    git clone "${OFFICIAL_REPO_URL}" "${OFFICIAL_VIDEODPO_REPO}"
  fi
fi

if [[ "${RESET_OFFICIAL_REPO}" == "1" ]]; then
  git -C "${OFFICIAL_VIDEODPO_REPO}" reset --hard
  git -C "${OFFICIAL_VIDEODPO_REPO}" clean -fdx
fi
if [[ "${FETCH_OFFICIAL}" == "1" ]]; then
  git -C "${OFFICIAL_VIDEODPO_REPO}" fetch origin
else
  echo "[official-diffueraser] fetch_skipped=1"
fi
git -C "${OFFICIAL_VIDEODPO_REPO}" checkout --detach "${VIDEODPO_REF}"

if ! grep -q "if args.test:" "${OFFICIAL_VIDEODPO_REPO}/scripts/train.py"; then
  echo "[official-diffueraser] applying skip-post-train-test patch"
  git -C "${OFFICIAL_VIDEODPO_REPO}" apply "${SKIP_TEST_PATCH}"
fi

if [[ "${STAGE}" == "stage2" ]]; then
  if [[ -z "${PRETRAINED_DPO_S1}" || ! -d "${PRETRAINED_DPO_S1}" ]]; then
    echo "[official-diffueraser][error] stage2 requires PRETRAINED_DPO_S1 pointing to Stage1 last_weights/best_weights." >&2
    exit 1
  fi
fi

if [[ -f "${VAE_PATH}/vae/config.json" ]]; then
  echo "[official-diffueraser] vae_layout=sd-subfolder"
elif [[ -f "${VAE_PATH}/config.json" ]]; then
  echo "[official-diffueraser] vae_layout=standalone"
else
  echo "[official-diffueraser][error] VAE path invalid: ${VAE_PATH}" >&2
  echo "[official-diffueraser][error] expected either ${VAE_PATH}/vae/config.json or ${VAE_PATH}/config.json" >&2
  exit 1
fi

for required in \
  "${CONFIG}" \
  "${VC2_DATA_YAML}" \
  "${BASE_MODEL_NAME_OR_PATH}/tokenizer" \
  "${BASE_MODEL_NAME_OR_PATH}/scheduler" \
  "${REF_MODEL_PATH}/unet_main/config.json" \
  "${REF_MODEL_PATH}/brushnet/config.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "[official-diffueraser][error] missing required path: ${required}" >&2
    exit 1
  fi
done

if [[ -n "${CONDA_BASE:-}" && -x "${CONDA_BASE}/bin/conda" ]]; then
  :
elif [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$("${CONDA_EXE}" info --base)"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
else
  echo "[official-diffueraser][error] conda not found; set CONDA_EXE or CONDA_BASE." >&2
  exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if [[ -z "${PL_DEVICE_LIST}" ]]; then
  PL_DEVICE_LIST="$(seq -s, 0 "$((NUM_GPUS - 1))")"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}"
export PYTHONPATH="${PROJECT_ROOT}:${OFFICIAL_VIDEODPO_REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export NCCL_ASYNC_ERROR_HANDLING=1
export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export TRANSFORMERS_CACHE="${PROJECT_ROOT}/.hf_cache"
export TMPDIR
export PROCESS_TITLE="${PROCESS_TITLE:-lingbotworld-phy}"
export WORLDMODELPHY_PROCESS_NAME="${WORLDMODELPHY_PROCESS_NAME:-lingbotworld-phy}"
mkdir -p "${HF_HOME}" "${TMPDIR}"
echo "[official-diffueraser] lightning_devices=${PL_DEVICE_LIST}"

python - <<'PY'
import importlib
mods = [
    "pytorch_lightning",
    "diffusers",
    "official_videodpo_diffueraser.models",
    "official_videodpo_diffueraser.data",
    "libs.brushnet_CA",
    "libs.unet_motion_model",
]
for name in mods:
    mod = importlib.import_module(name)
    print(f"[official-diffueraser][preflight][OK] {name} {getattr(mod, '__version__', '')}")
PY

GRAD_CKPT_VALUE=true
case "${GRADIENT_CHECKPOINTING,,}" in 0|false|no|off) GRAD_CKPT_VALUE=false ;; esac
SPLIT_VALUE=true
case "${SPLIT_POS_NEG_FORWARD,,}" in 0|false|no|off) SPLIT_VALUE=false ;; esac
USE_8BIT_VALUE=false
case "${USE_8BIT_ADAM,,}" in 1|true|yes|on) USE_8BIT_VALUE=true ;; esac

cd "${OFFICIAL_VIDEODPO_REPO}"
python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port "${MASTER_PORT}" \
  scripts/train.py \
  -t \
  --devices "${PL_DEVICE_LIST}" \
  --base "${CONFIG}" \
  --name "${RUN_NAME}" \
  --logdir "${LOG_ROOT}" \
  --auto_resume True \
  lightning.trainer.num_nodes=1 \
  lightning.trainer.devices="${NUM_GPUS}" \
  lightning.trainer.accelerator=gpu \
  lightning.trainer.accumulate_grad_batches="${GRAD_ACCUM}" \
  lightning.trainer.log_every_n_steps=1 \
  lightning.trainer.precision="${PRECISION}" \
  lightning.callbacks.metrics_over_trainsteps_checkpoint.params.every_n_train_steps="${CKPT_EVERY}" \
  data.params.batch_size="${BATCH_SIZE}" \
  data.params.num_workers="${NUM_WORKERS}" \
  data.params.train.params.data_root="${VC2_DATA_YAML}" \
  data.params.train.params.base_model_name_or_path="${BASE_MODEL_NAME_OR_PATH}" \
  data.params.train.params.video_length=16 \
  "data.params.train.params.resolution=[320,512]" \
  data.params.train.params.train_height=320 \
  data.params.train.params.train_width=512 \
  data.params.train.params.frame_stride=1 \
  data.params.train.params.full_mask_value=0.0 \
  model.params.base_model_name_or_path="${BASE_MODEL_NAME_OR_PATH}" \
  model.params.vae_path="${VAE_PATH}" \
  model.params.ref_model_path="${REF_MODEL_PATH}" \
  model.params.baseline_unet_path="${BASELINE_UNET_PATH}" \
  model.params.pretrained_dpo_stage1="${PRETRAINED_DPO_S1}" \
  model.params.learning_rate="${LR}" \
  model.params.beta_dpo="${BETA_DPO}" \
  model.params.lose_gap_weight="${LOSE_GAP_WEIGHT}" \
  model.params.forward_dtype="${FORWARD_DTYPE}" \
  model.params.vae_dtype="${VAE_DTYPE}" \
  model.params.ref_dtype="${REF_DTYPE}" \
  model.params.text_dtype="${TEXT_DTYPE}" \
  model.params.gradient_checkpointing="${GRAD_CKPT_VALUE}" \
  model.params.split_pos_neg_forward="${SPLIT_VALUE}" \
  model.params.use_8bit_adam="${USE_8BIT_VALUE}" \
  lightning.trainer.max_steps="${MAX_OPT_STEPS}"

echo "[official-diffueraser] done"
echo "[official-diffueraser] run_dir=${RUN_DIR}"
echo "[official-diffueraser] lightning_dir=${LOG_ROOT}/${RUN_NAME}"
