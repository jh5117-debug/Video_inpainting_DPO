#!/bin/bash
#SBATCH --job-name=DPO_S1
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
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_NAME="${PROJECT_NAME:-Video_inpainting_DPO}"
DATA_NAME="${DATA_NAME:-Video_inpainting_DPO}"
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${PROJECT_DEV:-}" ]]; then
    PROJECT_ROOT="${PROJECT_DEV}/${PROJECT_NAME}"
  elif [[ -n "${PROJECT_HOME:-}" ]]; then
    PROJECT_ROOT="${PROJECT_HOME}/dev/${PROJECT_NAME}"
  else
    PROJECT_ROOT="${DEFAULT_PROJECT_ROOT}"
  fi
fi
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
export WORLDMODELPHY_PROCESS_NAME="${WORLDMODELPHY_PROCESS_NAME:-worldmodelphy}"
export PROCESS_TITLE="${PROCESS_TITLE:-${WORLDMODELPHY_PROCESS_NAME}}"
export LINGBOT_PROCESS_NAME="${LINGBOT_PROCESS_NAME:-lingbot-phy}"
PROJECT_PARENT="$(dirname "${PROJECT_ROOT}")"
PROJECT_HOME="${PROJECT_HOME:-/sc-projects/sc-proj-cc09-repair/hongyou}"
PROJECT_DEV="${PROJECT_DEV:-${PROJECT_HOME}/dev}"
PROJECT_DATA="${PROJECT_DATA:-${PROJECT_DEV}/data}"
DATA="${DATA:-${PROJECT_DATA}/${DATA_NAME}}"
DEFAULT_CONDA_ENV_PREFIX="${PROJECT_DEV}/conda_envs/diffueraser"

NUM_GPUS=${NUM_GPUS:-8}
WEIGHTS_DIR=${WEIGHTS_DIR:-${DATA}/weights}
DPO_DATA_ROOT=${DPO_DATA_ROOT:-${DATA}/DPO_Finetune_data}
DPO_DATASET_TYPE=${DPO_DATASET_TYPE:-diffueraser_inpainting}
PREFERENCE_MANIFEST=${PREFERENCE_MANIFEST:-}
TRAIN_MASK_MODE=${TRAIN_MASK_MODE:-full}
MASK_FROM_MANIFEST=${MASK_FROM_MANIFEST:-false}
LOSS_REGION_MODE=${LOSS_REGION_MODE:-full}
GAP_NORMALIZATION=${GAP_NORMALIZATION:-raw}
GAP_EPS=${GAP_EPS:-1e-6}
LOSE_GAP_CLIP_TAU=${LOSE_GAP_CLIP_TAU:-}
MASK_REGION_WEIGHT=${MASK_REGION_WEIGHT:-1.0}
BOUNDARY_REGION_WEIGHT=${BOUNDARY_REGION_WEIGHT:-0.5}
OUTSIDE_REGION_WEIGHT=${OUTSIDE_REGION_WEIGHT:-0.05}
DPO_GAP_TRACE_CSV=${DPO_GAP_TRACE_CSV:-}
DPO_GAP_SAMPLES_JSONL_GZ=${DPO_GAP_SAMPLES_JSONL_GZ:-}
ENABLE_DPO_DIAG=${ENABLE_DPO_DIAG:-true}
DPO_DIAG_LOG_EVERY=${DPO_DIAG_LOG_EVERY:-10}
DPO_DIAG_SAVE_CSV=${DPO_DIAG_SAVE_CSV:-true}
DPO_DIAG_SAVE_WANDB=${DPO_DIAG_SAVE_WANDB:-true}
VAL_DATA_DIR=${VAL_DATA_DIR:-${DATA}/davis_432_240}
REF_MODEL_PATH=${REF_MODEL_PATH:-}
POLICY_INIT_PATH=${POLICY_INIT_PATH:-}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-latest}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
LR=${LR:-1e-6}
LR_SCHEDULER=${LR_SCHEDULER:-constant}
LR_WARMUP=${LR_WARMUP:-500}
MAX_EPOCHS=${MAX_EPOCHS:-}
if [[ -z "${MAX_STEPS+x}" ]]; then
  if [[ "${DPO_DATASET_TYPE}" == "videodpo_fullmask" ]]; then
    MAX_STEPS=""
    MAX_EPOCHS="${MAX_EPOCHS:-10}"
  else
    MAX_STEPS="20000"
  fi
else
  MAX_STEPS="${MAX_STEPS}"
fi
CKPT_STEPS=${CKPT_STEPS:-2000}
CKPT_LIMIT=${CKPT_LIMIT:-1}
VAL_STEPS=${VAL_STEPS:-2000}
LOGGING_STEPS=${LOGGING_STEPS:-300}
DPO_DIAGNOSTICS=${DPO_DIAGNOSTICS:-1}
RESOLUTION=${RESOLUTION:-512}
TRAIN_HEIGHT=${TRAIN_HEIGHT:-${RESOLUTION}}
TRAIN_WIDTH=${TRAIN_WIDTH:-${RESOLUTION}}
NFRAMES=${NFRAMES:-16}
SEED=${SEED:-42}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
VAE_DTYPE=${VAE_DTYPE:-auto}
POLICY_DTYPE=${POLICY_DTYPE:-auto}
REF_DTYPE=${REF_DTYPE:-auto}
TEXT_DTYPE=${TEXT_DTYPE:-auto}
WANDB_PROJECT=${WANDB_PROJECT:-DPO_Diffueraser}
REPORT_TO=${REPORT_TO:-wandb}
BETA_DPO=${BETA_DPO:-500.0}
DAVIS_OVERSAMPLE=${DAVIS_OVERSAMPLE:-10}
VIDEODPO_FRAME_STRIDE=${VIDEODPO_FRAME_STRIDE:-1}
VIDEODPO_CLIP_LENGTH=${VIDEODPO_CLIP_LENGTH:-1.0}
VIDEODPO_FULL_MASK_VALUE=${VIDEODPO_FULL_MASK_VALUE:-0.0}
EXPERIMENTS_DIR=${EXPERIMENTS_DIR:-${DATA}/experiments}
RUN_NAME=${RUN_NAME:-sc-dpo-stage1-newdata-bf16-raw6-nodil}
RUN_VERSION=${RUN_VERSION:-}
WANDB_ENTITY=${WANDB_ENTITY:-jh5117-columbia-university}
SFT_REG_WEIGHT=${SFT_REG_WEIGHT:-0.0}
LOSE_GAP_WEIGHT=${LOSE_GAP_WEIGHT:-${DPO_LOSE_GAP_WEIGHT:-1.0}}
DPO_LOSE_GAP_WEIGHT="${LOSE_GAP_WEIGHT}"
WINNER_ABS_REG_WEIGHT=${WINNER_ABS_REG_WEIGHT:-0.0}
WINNER_GAP_REG_WEIGHT=${WINNER_GAP_REG_WEIGHT:-0.0}
WINNER_GAP_REG_MARGIN=${WINNER_GAP_REG_MARGIN:-0.0}
WINNER_GAP_REG_MODE=${WINNER_GAP_REG_MODE:-relu}
CHUNK_ALIGNED=${CHUNK_ALIGNED:-1}
XFORMERS=${XFORMERS:-0}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}
SPLIT_POS_NEG_FORWARD=${SPLIT_POS_NEG_FORWARD:-1}
USE_8BIT_ADAM=${USE_8BIT_ADAM:-0}
DEBUG_FIRST_BATCH_STAGES=${DEBUG_FIRST_BATCH_STAGES:-0}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-0}
VAL_NUM_INFERENCE_STEPS=${VAL_NUM_INFERENCE_STEPS:-6}
VAL_MASK_DILATION_ITER=${VAL_MASK_DILATION_ITER:-0}

CONDA_PREFIX_READY=0
if [[ -n "${CONDA_ENV_PREFIX:-}" && -x "${CONDA_ENV_PREFIX}/bin/python" ]]; then
  CONDA_ENV="${CONDA_ENV_PREFIX}"
  CONDA_PREFIX_READY=1
elif [[ -z "${CONDA_ENV:-}" && -x "${DEFAULT_CONDA_ENV_PREFIX}/bin/python" ]]; then
  CONDA_ENV="${DEFAULT_CONDA_ENV_PREFIX}"
  CONDA_PREFIX_READY=1
elif [[ -n "${CONDA_ENV_PREFIX:-}" ]]; then
  CONDA_ENV="${CONDA_ENV_PREFIX}"
elif [[ -z "${CONDA_ENV:-}" && -d "${DEFAULT_CONDA_ENV_PREFIX}" ]]; then
  CONDA_ENV="${DEFAULT_CONDA_ENV_PREFIX}"
else
  CONDA_ENV="${CONDA_ENV:-diffueraser}"
fi

if [[ "${CONDA_PREFIX_READY}" == "1" ]]; then
  export CONDA_PREFIX="${CONDA_ENV}"
  export PATH="${CONDA_ENV}/bin:${PATH}"
else
  if [[ -n "${CONDA_BASE:-}" && -x "${CONDA_BASE}/bin/conda" ]]; then
    :
  elif [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$("${CONDA_EXE}" info --base)"
  elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
  elif [[ -x "${PROJECT_PARENT}/miniconda3/bin/conda" ]]; then
    CONDA_BASE="${PROJECT_PARENT}/miniconda3"
  else
    echo "conda not found; set CONDA_EXE, CONDA_ENV_PREFIX, or install Miniconda." >&2
    exit 1
  fi

  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "PROJECT_ROOT does not exist: $PROJECT_ROOT" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

echo "[dpo-stage1] dataset_type=${DPO_DATASET_TYPE}"
echo "[dpo-stage1] dpo_data_root=${DPO_DATA_ROOT}"
if [[ "${DPO_DATASET_TYPE}" == "videodpo_fullmask" ]]; then
  echo "[dpo-stage1] videodpo_frame_stride=${VIDEODPO_FRAME_STRIDE} full_mask_value=${VIDEODPO_FULL_MASK_VALUE}"
fi

REF_MODEL_ARG=()
if [[ -n "${REF_MODEL_PATH}" ]]; then
  REF_MODEL_ARG=(--ref_model_path "${REF_MODEL_PATH}")
fi

POLICY_INIT_ARG=()
if [[ -n "${POLICY_INIT_PATH}" ]]; then
  POLICY_INIT_ARG=(--policy_init_path "${POLICY_INIT_PATH}")
fi

RESUME_FROM_CHECKPOINT_ARG=()
case "${RESUME_FROM_CHECKPOINT,,}" in
  ""|0|false|no|off|none)
    ;;
  *)
    RESUME_FROM_CHECKPOINT_ARG=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
    ;;
esac

RUN_VERSION_ARG=()
if [[ -n "${RUN_VERSION}" ]]; then
  RUN_VERSION_ARG=(--run_version "${RUN_VERSION}")
fi

WANDB_ENTITY_ARG=()
if [[ -n "${WANDB_ENTITY}" ]]; then
  WANDB_ENTITY_ARG=(--wandb_entity "${WANDB_ENTITY}")
fi

MAIN_PROCESS_PORT_ARG=()
if [[ -n "${MAIN_PROCESS_PORT}" && "${MAIN_PROCESS_PORT}" != "0" && "${MAIN_PROCESS_PORT,,}" != "auto" ]]; then
  MAIN_PROCESS_PORT_ARG=(--main_process_port "${MAIN_PROCESS_PORT}")
fi

MAX_STEPS_ARG=()
if [[ -n "${MAX_STEPS}" ]]; then
  MAX_STEPS_ARG=(--max_train_steps "${MAX_STEPS}")
fi

MAX_EPOCHS_ARG=()
if [[ -n "${MAX_EPOCHS}" ]]; then
  MAX_EPOCHS_ARG=(--num_train_epochs "${MAX_EPOCHS}")
fi

CHUNK_ARG=()
case "${CHUNK_ALIGNED,,}" in
  1|true|yes|on)
    CHUNK_ARG=(--chunk_aligned)
    ;;
esac

XFORMERS_ARG=()
case "${XFORMERS,,}" in
  1|true|yes|on)
    XFORMERS_ARG=(--enable_xformers)
    ;;
esac

GRADIENT_CHECKPOINTING_ARG=()
case "${GRADIENT_CHECKPOINTING,,}" in
  0|false|no|off)
    GRADIENT_CHECKPOINTING_ARG=(--disable_gradient_checkpointing)
    ;;
esac

SPLIT_POS_NEG_FORWARD_ARG=()
case "${SPLIT_POS_NEG_FORWARD,,}" in
  1|true|yes|on)
    SPLIT_POS_NEG_FORWARD_ARG=(--split_pos_neg_forward)
    ;;
esac

USE_8BIT_ADAM_ARG=()
case "${USE_8BIT_ADAM,,}" in
  1|true|yes|on)
    USE_8BIT_ADAM_ARG=(--use_8bit_adam)
    ;;
esac

DEBUG_FIRST_BATCH_STAGES_ARG=()
case "${DEBUG_FIRST_BATCH_STAGES,,}" in
  1|true|yes|on)
    DEBUG_FIRST_BATCH_STAGES_ARG=(--debug_first_batch_stages)
    ;;
esac

DPO_DIAGNOSTICS_ARG=()
case "${DPO_DIAGNOSTICS,,}" in
  0|false|no|off)
    DPO_DIAGNOSTICS_ARG=(--disable_dpo_diagnostics)
    ;;
esac

# === 所有缓存/日志/artifacts 必须在项目目录内，禁止写入 home ===
if [[ "${REPORT_TO,,}" == "wandb" ]]; then
  export WANDB_DIR="${PROJECT_ROOT}/.wandb_cache"
  export WANDB_CACHE_DIR="${PROJECT_ROOT}/.wandb_cache"
  export WANDB_DATA_DIR="${PROJECT_ROOT}/.wandb_cache"
  export WANDB_CONFIG_DIR="${PROJECT_ROOT}/.wandb_cache/config"
  if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.netrc" ]]; then
    WANDB_API_KEY_FROM_NETRC=$(grep -A2 'api.wandb.ai' "${HOME}/.netrc" | grep password | awk '{print $2}' | head -n 1 || true)
    if [[ -n "${WANDB_API_KEY_FROM_NETRC}" ]]; then
      export WANDB_API_KEY="${WANDB_API_KEY_FROM_NETRC}"
    fi
  fi
  mkdir -p "${WANDB_DIR}" "${WANDB_CONFIG_DIR}"
fi
export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export TRANSFORMERS_CACHE="${PROJECT_ROOT}/.hf_cache"
export TMPDIR="${TMPDIR:-${PROJECT_ROOT}/.tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
mkdir -p "${HF_HOME}" "${TMPDIR}"

python training/dpo/scripts/run_stage1.py \
  --num_gpus ${NUM_GPUS} \
  --weights_dir ${WEIGHTS_DIR} \
  --dpo_data_root ${DPO_DATA_ROOT} \
  --dpo_dataset_type ${DPO_DATASET_TYPE} \
  --preference_manifest "${PREFERENCE_MANIFEST}" \
  --train_mask_mode "${TRAIN_MASK_MODE}" \
  --mask_from_manifest "${MASK_FROM_MANIFEST}" \
  --loss_region_mode "${LOSS_REGION_MODE}" \
  --gap_normalization "${GAP_NORMALIZATION}" \
  --gap_eps "${GAP_EPS}" \
  --lose_gap_clip_tau "${LOSE_GAP_CLIP_TAU}" \
  --mask_region_weight "${MASK_REGION_WEIGHT}" \
  --boundary_region_weight "${BOUNDARY_REGION_WEIGHT}" \
  --outside_region_weight "${OUTSIDE_REGION_WEIGHT}" \
  --dpo_gap_trace_csv "${DPO_GAP_TRACE_CSV}" \
  --dpo_gap_samples_jsonl_gz "${DPO_GAP_SAMPLES_JSONL_GZ}" \
  --enable_dpo_diag "${ENABLE_DPO_DIAG}" \
  --dpo_diag_log_every "${DPO_DIAG_LOG_EVERY}" \
  --dpo_diag_save_csv "${DPO_DIAG_SAVE_CSV}" \
  --dpo_diag_save_wandb "${DPO_DIAG_SAVE_WANDB}" \
  --val_data_dir ${VAL_DATA_DIR} \
  --experiments_dir ${EXPERIMENTS_DIR} \
  --run_name ${RUN_NAME} \
  --batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACCUM} \
  --num_workers ${NUM_WORKERS} \
  --learning_rate ${LR} \
  --lr_scheduler ${LR_SCHEDULER} \
  --lr_warmup_steps ${LR_WARMUP} \
  "${MAX_STEPS_ARG[@]}" \
  "${MAX_EPOCHS_ARG[@]}" \
  --checkpointing_steps ${CKPT_STEPS} \
  --checkpoints_total_limit ${CKPT_LIMIT} \
  --validation_steps ${VAL_STEPS} \
  --logging_steps ${LOGGING_STEPS} \
  --val_num_inference_steps ${VAL_NUM_INFERENCE_STEPS} \
  --val_mask_dilation_iter ${VAL_MASK_DILATION_ITER} \
  --resolution ${RESOLUTION} \
  --train_height ${TRAIN_HEIGHT} \
  --train_width ${TRAIN_WIDTH} \
  --nframes ${NFRAMES} \
  --seed ${SEED} \
  --mixed_precision ${MIXED_PRECISION} \
  --vae_dtype ${VAE_DTYPE} \
  --policy_dtype ${POLICY_DTYPE} \
  --ref_dtype ${REF_DTYPE} \
  --text_dtype ${TEXT_DTYPE} \
  --report_to ${REPORT_TO} \
  --wandb_project ${WANDB_PROJECT} \
  --beta_dpo ${BETA_DPO} \
  --sft_reg_weight ${SFT_REG_WEIGHT} \
  --lose_gap_weight ${DPO_LOSE_GAP_WEIGHT} \
  --winner_abs_reg_weight ${WINNER_ABS_REG_WEIGHT} \
  --winner_gap_reg_weight ${WINNER_GAP_REG_WEIGHT} \
  --winner_gap_reg_margin ${WINNER_GAP_REG_MARGIN} \
  --winner_gap_reg_mode ${WINNER_GAP_REG_MODE} \
  --davis_oversample ${DAVIS_OVERSAMPLE} \
  --videodpo_frame_stride ${VIDEODPO_FRAME_STRIDE} \
  --videodpo_clip_length ${VIDEODPO_CLIP_LENGTH} \
  --videodpo_full_mask_value ${VIDEODPO_FULL_MASK_VALUE} \
  "${MAIN_PROCESS_PORT_ARG[@]}" \
  "${WANDB_ENTITY_ARG[@]}" \
  "${CHUNK_ARG[@]}" \
  "${XFORMERS_ARG[@]}" \
  "${GRADIENT_CHECKPOINTING_ARG[@]}" \
  "${SPLIT_POS_NEG_FORWARD_ARG[@]}" \
  "${USE_8BIT_ADAM_ARG[@]}" \
  "${DEBUG_FIRST_BATCH_STAGES_ARG[@]}" \
  "${DPO_DIAGNOSTICS_ARG[@]}" \
  "${REF_MODEL_ARG[@]}" \
  "${POLICY_INIT_ARG[@]}" \
  "${RESUME_FROM_CHECKPOINT_ARG[@]}" \
  "${RUN_VERSION_ARG[@]}"
