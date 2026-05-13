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
WANDB_START_EVENT="${WANDB_START_EVENT:-0}"
APPLY_DPO_DIAG_PATCH="${APPLY_DPO_DIAG_PATCH:-1}"
DPO_DIAG_EVERY="${DPO_DIAG_EVERY:-300}"
DPO_DIAG_PATCH="${DPO_DIAG_PATCH:-${PROJECT_ROOT}/patches/videodpo/sc_vc2_dpo_diagnostics.patch}"
CLEAN_DEBUG_PRINT="${CLEAN_DEBUG_PRINT:-1}"
DEBUG_PRINT_PATCH="${DEBUG_PRINT_PATCH:-${PROJECT_ROOT}/patches/videodpo/sc_vc2_remove_debug_training_step_print.patch}"
PATCH_OPENCLIP_BATCH_FIRST="${PATCH_OPENCLIP_BATCH_FIRST:-1}"
OPENCLIP_BATCH_FIRST_PATCH="${OPENCLIP_BATCH_FIRST_PATCH:-${PROJECT_ROOT}/patches/videodpo/sc_vc2_open_clip_batch_first.patch}"
PATCH_LOGGER_COMPAT="${PATCH_LOGGER_COMPAT:-1}"
LOGGER_COMPAT_PATCH="${LOGGER_COMPAT_PATCH:-${PROJECT_ROOT}/patches/videodpo/sc_vc2_wandb_logger_compat.patch}"
PATCH_SKIP_IMPLICIT_TEST="${PATCH_SKIP_IMPLICIT_TEST:-1}"
SKIP_IMPLICIT_TEST_PATCH="${SKIP_IMPLICIT_TEST_PATCH:-${PROJECT_ROOT}/patches/videodpo/sc_vc2_skip_implicit_test.patch}"
DISABLE_IMAGE_LOGGER="${DISABLE_IMAGE_LOGGER:-1}"
GPU_PREFLIGHT="${GPU_PREFLIGHT:-1}"

export PROJECT_ROOT PROJECT_HOME PROJECT_DEV PROJECT_DATA
export VIDEODPO_REPO VC2_DATA_YAML LOG_ROOT RUN_NAME CONFIG
export NUM_GPUS DEVICE_LIST BATCH_SIZE GRAD_ACCUM MAX_OPT_STEPS CKPT_EVERY NUM_WORKERS BETA_DPO CONDA_ENV
export APPLY_DPO_DIAG_PATCH DPO_DIAG_EVERY DPO_DIAG_PATCH CLEAN_DEBUG_PRINT DEBUG_PRINT_PATCH PATCH_OPENCLIP_BATCH_FIRST OPENCLIP_BATCH_FIRST_PATCH PATCH_LOGGER_COMPAT LOGGER_COMPAT_PATCH PATCH_SKIP_IMPLICIT_TEST SKIP_IMPLICIT_TEST_PATCH DISABLE_IMAGE_LOGGER GPU_PREFLIGHT WANDB_START_EVENT

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
if [[ ! -f "${VIDEODPO_REPO}/scripts/train.py" || ! -f "${VIDEODPO_REPO}/lvdm/models/ddpm3d.py" || ! -f "${VIDEODPO_REPO}/configs/vc2_dpo/config.yaml" ]]; then
  echo "[vc2-train][error] VIDEODPO_REPO is incomplete: ${VIDEODPO_REPO}" >&2
  echo "[vc2-train][error] Missing one of scripts/train.py, lvdm/models/ddpm3d.py, configs/vc2_dpo/config.yaml." >&2
  echo "[vc2-train][error] Fix the submodule first: git submodule update --init --recursive external/VideoDPO" >&2
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
export WANDB_CONSOLE="${WANDB_CONSOLE:-wrap}"
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

apply_dpo_diag_patch() {
  case "${APPLY_DPO_DIAG_PATCH,,}" in
    1|true|yes|on)
      if [[ ! -f "${DPO_DIAG_PATCH}" ]]; then
        echo "[vc2-train][error] DPO diagnostics patch missing: ${DPO_DIAG_PATCH}" >&2
        exit 1
      fi
      if grep -q "_log_dpo_diag_table" "${VIDEODPO_REPO}/lvdm/models/ddpm3d.py"; then
        echo "[vc2-train] DPO diagnostics patch already applied in ${VIDEODPO_REPO}"
      else
        echo "[vc2-train] applying DPO diagnostics patch: ${DPO_DIAG_PATCH}"
        git -C "${VIDEODPO_REPO}" apply "${DPO_DIAG_PATCH}"
      fi
      ;;
    *)
      echo "[vc2-train] DPO diagnostics patch disabled by APPLY_DPO_DIAG_PATCH=${APPLY_DPO_DIAG_PATCH}"
      ;;
  esac
}

apply_debug_print_patch() {
  case "${CLEAN_DEBUG_PRINT,,}" in
    1|true|yes|on)
      if [[ ! -f "${DEBUG_PRINT_PATCH}" ]]; then
        echo "[vc2-train][error] debug-print cleanup patch missing: ${DEBUG_PRINT_PATCH}" >&2
        exit 1
      fi
      if grep -q '^[[:space:]]*print("test")' "${VIDEODPO_REPO}/lvdm/models/ddpm3d.py"; then
        echo "[vc2-train] removing VideoDPO training_step debug print: ${DEBUG_PRINT_PATCH}"
        git -C "${VIDEODPO_REPO}" apply "${DEBUG_PRINT_PATCH}"
      else
        echo "[vc2-train] VideoDPO training_step debug print already absent"
      fi
      ;;
    *)
      echo "[vc2-train] debug-print cleanup disabled by CLEAN_DEBUG_PRINT=${CLEAN_DEBUG_PRINT}"
      ;;
  esac
}

apply_openclip_batch_first_patch() {
  case "${PATCH_OPENCLIP_BATCH_FIRST,,}" in
    1|true|yes|on)
      if [[ ! -f "${OPENCLIP_BATCH_FIRST_PATCH}" ]]; then
        echo "[vc2-train][error] OpenCLIP batch-first patch missing: ${OPENCLIP_BATCH_FIRST_PATCH}" >&2
        exit 1
      fi
      if grep -q 'batch_first = getattr(self.model.transformer, "batch_first", False)' "${VIDEODPO_REPO}/lvdm/modules/encoders/condition.py"; then
        echo "[vc2-train] OpenCLIP batch-first patch already applied"
      else
        echo "[vc2-train] applying OpenCLIP batch-first patch: ${OPENCLIP_BATCH_FIRST_PATCH}"
        git -C "${VIDEODPO_REPO}" apply "${OPENCLIP_BATCH_FIRST_PATCH}"
      fi
      ;;
    *)
      echo "[vc2-train] OpenCLIP batch-first patch disabled by PATCH_OPENCLIP_BATCH_FIRST=${PATCH_OPENCLIP_BATCH_FIRST}"
      ;;
  esac
}

apply_logger_compat_patch() {
  case "${PATCH_LOGGER_COMPAT,,}" in
    1|true|yes|on)
      if [[ ! -f "${LOGGER_COMPAT_PATCH}" ]]; then
        echo "[vc2-train][error] logger compatibility patch missing: ${LOGGER_COMPAT_PATCH}" >&2
        exit 1
      fi
      if grep -Eq 'Skip W&B video log|Skip tensorboard video log' "${VIDEODPO_REPO}/utils/callbacks.py"; then
        echo "[vc2-train] W&B logger compatibility patch already applied"
      else
        echo "[vc2-train] applying W&B logger compatibility patch: ${LOGGER_COMPAT_PATCH}"
        git -C "${VIDEODPO_REPO}" apply "${LOGGER_COMPAT_PATCH}"
      fi
      ;;
    *)
      echo "[vc2-train] W&B logger compatibility patch disabled by PATCH_LOGGER_COMPAT=${PATCH_LOGGER_COMPAT}"
      ;;
  esac
}

apply_skip_implicit_test_patch() {
  case "${PATCH_SKIP_IMPLICIT_TEST,,}" in
    1|true|yes|on)
      if [[ ! -f "${SKIP_IMPLICIT_TEST_PATCH}" ]]; then
        echo "[vc2-train][error] skip implicit test patch missing: ${SKIP_IMPLICIT_TEST_PATCH}" >&2
        exit 1
      fi
      if grep -q 'if args.test:' "${VIDEODPO_REPO}/scripts/train.py" && ! grep -q 'if args.test or not trainer.interrupted' "${VIDEODPO_REPO}/scripts/train.py"; then
        echo "[vc2-train] implicit post-train test already disabled"
      else
        echo "[vc2-train] applying skip implicit post-train test patch: ${SKIP_IMPLICIT_TEST_PATCH}"
        git -C "${VIDEODPO_REPO}" apply "${SKIP_IMPLICIT_TEST_PATCH}"
      fi
      ;;
    *)
      echo "[vc2-train] skip implicit post-train test patch disabled by PATCH_SKIP_IMPLICIT_TEST=${PATCH_SKIP_IMPLICIT_TEST}"
      ;;
  esac
}

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
  wandb_event failed --exit-code "${rc}" --tail-lines 400
  exit "${rc}"
}
on_signal() {
  local sig="$1"
  local rc=128
  trap - ERR TERM INT HUP
  echo "[vc2-train][error] received signal ${sig}; logging Slurm tail to W&B before exit"
  wandb_event failed --exit-code "${rc}" --tail-lines 400 --config "signal=${sig}"
  exit "${rc}"
}
trap on_error ERR
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap 'on_signal HUP' HUP

echo "[vc2-train] videodpo_repo=${VIDEODPO_REPO}"
echo "[vc2-train] data_yaml=${VC2_DATA_YAML}"
echo "[vc2-train] log_root=${LOG_ROOT}"
echo "[vc2-train] launch_log=${LAUNCH_LOG}"
echo "[vc2-train] run_name=${RUN_NAME}"
echo "[vc2-train] num_gpus=${NUM_GPUS} device_list=${DEVICE_LIST}"
echo "[vc2-train] max_opt_steps=${MAX_OPT_STEPS:-official-config-max_epochs} grad_accum=${GRAD_ACCUM} beta=${BETA_DPO}"
echo "[vc2-train] dpo_diag_every=${DPO_DIAG_EVERY} apply_dpo_diag_patch=${APPLY_DPO_DIAG_PATCH} patch=${DPO_DIAG_PATCH}"
echo "[vc2-train] clean_debug_print=${CLEAN_DEBUG_PRINT} debug_print_patch=${DEBUG_PRINT_PATCH}"
echo "[vc2-train] patch_openclip_batch_first=${PATCH_OPENCLIP_BATCH_FIRST} openclip_batch_first_patch=${OPENCLIP_BATCH_FIRST_PATCH}"
echo "[vc2-train] patch_logger_compat=${PATCH_LOGGER_COMPAT} logger_compat_patch=${LOGGER_COMPAT_PATCH}"
echo "[vc2-train] patch_skip_implicit_test=${PATCH_SKIP_IMPLICIT_TEST} skip_implicit_test_patch=${SKIP_IMPLICIT_TEST_PATCH}"
echo "[vc2-train] disable_image_logger=${DISABLE_IMAGE_LOGGER}"
echo "[vc2-train] gpu_preflight=${GPU_PREFLIGHT} slurm_job_gpus=${SLURM_JOB_GPUS:-unset} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[vc2-train] enable_wandb=${ENABLE_WANDB} early_wandb=${EARLY_WANDB} wandb_start_event=${WANDB_START_EVENT} wandb_project=${WANDB_PROJECT} wandb_entity=${WANDB_ENTITY} wandb_dir=${WANDB_DIR} wandb_run_id=${WANDB_RUN_ID}"

case "${WANDB_START_EVENT,,}" in
  1|true|yes|on)
    wandb_event start \
      --config "run_name=${RUN_NAME}" \
      --config "num_gpus=${NUM_GPUS}" \
      --config "device_list=${DEVICE_LIST}" \
      --config "beta_dpo=${BETA_DPO}" \
      --config "dpo_diag_every=${DPO_DIAG_EVERY}" \
      --config "apply_dpo_diag_patch=${APPLY_DPO_DIAG_PATCH}" \
      --config "clean_debug_print=${CLEAN_DEBUG_PRINT}" \
      --config "patch_openclip_batch_first=${PATCH_OPENCLIP_BATCH_FIRST}" \
      --config "patch_logger_compat=${PATCH_LOGGER_COMPAT}" \
      --config "patch_skip_implicit_test=${PATCH_SKIP_IMPLICIT_TEST}" \
      --config "disable_image_logger=${DISABLE_IMAGE_LOGGER}" \
      --config "gpu_preflight=${GPU_PREFLIGHT}" \
      --config "vc2_data_yaml=${VC2_DATA_YAML}" \
      --config "videodpo_repo=${VIDEODPO_REPO}"
    ;;
  *)
    echo "[vc2-train] W&B start marker disabled; training or failure hook will create/update the run."
    ;;
esac

apply_dpo_diag_patch
apply_debug_print_patch
apply_openclip_batch_first_patch
apply_logger_compat_patch
apply_skip_implicit_test_patch

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
    "open_clip",
    "transformers",
    "huggingface_hub",
    "diffusers",
    "fairscale",
    "timm",
    "peft",
    "decord",
    "av",
]
if os.environ.get("ENABLE_WANDB", "1").lower() in {"1", "true", "yes", "on"}:
    required.append("wandb")
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

try:
    importlib.import_module("lvdm.models.ddpm3d")
    print("[vc2-train][preflight][OK] lvdm.models.ddpm3d")
except Exception as exc:
    print(f"[vc2-train][preflight][FAIL] lvdm.models.ddpm3d: {type(exc).__name__}: {exc}")
    missing.append("lvdm.models.ddpm3d")

if missing:
    env = os.environ.get("CONDA_ENV", "diffueraser")
    print("[vc2-train][preflight][ERROR] missing/import-failing modules:", ", ".join(missing))
    print("[vc2-train][preflight][FIX] CONDA_ENV=%s INSTALL_MINIMAL=1 bash DPO_finetune/scripts/videodpo_env_smoke_and_export.sh" % env)
    raise SystemExit(2)
PY

case "${GPU_PREFLIGHT,,}" in
  1|true|yes|on)
    echo "[vc2-train] gpu preflight"
    echo "[vc2-train][gpu-preflight] host=$(hostname)"
    echo "[vc2-train][gpu-preflight] SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
    echo "[vc2-train][gpu-preflight] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
    echo "[vc2-train][gpu-preflight] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    nvidia-smi -L || true
    python - <<'PY'
import os
import sys

import torch

requested = int(os.environ.get("NUM_GPUS", "1"))
print(f"[vc2-train][gpu-preflight] torch={torch.__version__} torch.version.cuda={torch.version.cuda}")
print(f"[vc2-train][gpu-preflight] torch.cuda.is_available()={torch.cuda.is_available()}")
try:
    count = torch.cuda.device_count()
except Exception as exc:
    print(f"[vc2-train][gpu-preflight][ERROR] torch.cuda.device_count failed: {type(exc).__name__}: {exc}")
    raise SystemExit(3)
print(f"[vc2-train][gpu-preflight] torch.cuda.device_count()={count} requested={requested}")
if not torch.cuda.is_available() or count < requested:
    print(
        "[vc2-train][gpu-preflight][ERROR] CUDA is not visible inside this Slurm allocation; "
        "likely node/GPU allocation/driver issue. Requeue on another node or contact SC admin."
    )
    raise SystemExit(3)
for idx in range(count):
    try:
        props = torch.cuda.get_device_properties(idx)
        print(f"[vc2-train][gpu-preflight][OK] cuda:{idx} name={props.name} total_memory_gb={props.total_memory / (1024 ** 3):.1f}")
    except Exception as exc:
        print(f"[vc2-train][gpu-preflight][ERROR] cuda:{idx} properties failed: {type(exc).__name__}: {exc}")
        raise SystemExit(3)
try:
    torch.cuda.set_device(0)
    probe = torch.ones(1, device="cuda")
    torch.cuda.synchronize()
    print(f"[vc2-train][gpu-preflight][OK] cuda tensor probe={float(probe.item())}")
except Exception as exc:
    print(f"[vc2-train][gpu-preflight][ERROR] cuda tensor probe failed: {type(exc).__name__}: {exc}")
    raise SystemExit(3)
PY
    ;;
  *)
    echo "[vc2-train] GPU preflight disabled by GPU_PREFLIGHT=${GPU_PREFLIGHT}"
    ;;
esac

MAX_STEPS_ARG=()
if [[ -n "${MAX_OPT_STEPS}" ]]; then
  MAX_STEPS_ARG=(lightning.trainer.max_steps="${MAX_OPT_STEPS}")
fi

IMAGE_LOGGER_ARGS=()
case "${DISABLE_IMAGE_LOGGER,,}" in
  1|true|yes|on)
    IMAGE_LOGGER_ARGS=(
      lightning.callbacks.batch_logger.params.batch_frequency=-1
      lightning.callbacks.image_logger.params.batch_frequency=-1
    )
    ;;
esac

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
  "${IMAGE_LOGGER_ARGS[@]}" \
  "${LOGGER_ARGS[@]}" \
  "${MAX_STEPS_ARG[@]}"

trap - ERR
wandb_event finished --exit-code 0 --tail-lines 120
echo "[vc2-train] done"
echo "[vc2-train] checkpoints should be under ${LOG_ROOT}/${RUN_NAME}/checkpoints"
