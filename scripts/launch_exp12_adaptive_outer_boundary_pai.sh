#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
cd "${PROJECT_ROOT}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${OUTPUT_ROOT}/experiments}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/mnt/nas/hj/conda_envs/diffueraser}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PREFIX}/bin/python}"

SOURCE_D3_ROOT="${SOURCE_D3_ROOT:-${WORKSPACE_ROOT}/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4}"
SOURCE_D3_MANIFEST="${SOURCE_D3_MANIFEST:-${SOURCE_D3_ROOT}/manifests/selected_primary_comp.repaired.pai_paths.jsonl}"
D3_ROOT="${D3_ROOT:-${WORKSPACE_ROOT}/data/generated_losers/exp09_10_11_youtubevos_gtwin_d3comp_pai}"
PREFERENCE_MANIFEST="${PREFERENCE_MANIFEST:-${D3_ROOT}/manifests/selected_primary_comp.gtwin.pai_paths.jsonl}"
DAVIS_ROOT="${DAVIS_ROOT:-/mnt/workspace/hj/nas_hj/data/external/davis_432_240}"
DIFFUERASER_WEIGHT_ROOT="${DIFFUERASER_WEIGHT_ROOT:-/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000}"
SFT_STAGE2_WEIGHTS="${SFT_STAGE2_WEIGHTS:-${DIFFUERASER_WEIGHT_ROOT}}"
PROPAINTER_WEIGHT_ROOT="${PROPAINTER_WEIGHT_ROOT:-/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter}"
RAFT_MODEL_PATH="${RAFT_MODEL_PATH:-${PROPAINTER_WEIGHT_ROOT}/raft-things.pth}"
I3D_MODEL_PATH="${I3D_MODEL_PATH:-${PROJECT_ROOT}/weights/i3d_rgb_imagenet.pt}"
TC_MODEL_PATH="${TC_MODEL_PATH:-${PROJECT_ROOT}/weights/open_clip_vit_h14}"

RUN_VERSION_ROOT="${RUN_VERSION_ROOT:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p logs/pipelines reports

source_helpers() {
  # Reuse train/validation helpers from the fixed Exp9/10/11 launcher without
  # executing its main function.
  # shellcheck disable=SC1090
  source <(sed '$d' scripts/launch_exp09_10_11_pai.sh)
}

set_fixed_training_env() {
  export PROJECT_ROOT OUTPUT_ROOT WORKSPACE_ROOT EXPERIMENTS_DIR CONDA_ENV_PREFIX PYTHON_BIN
  export SOURCE_D3_ROOT SOURCE_D3_MANIFEST D3_ROOT PREFERENCE_MANIFEST
  export DAVIS_ROOT DIFFUERASER_WEIGHT_ROOT SFT_STAGE2_WEIGHTS PROPAINTER_WEIGHT_ROOT RAFT_MODEL_PATH
  export I3D_MODEL_PATH TC_MODEL_PATH
  export AUTO_PREPARE_GTWIN="${AUTO_PREPARE_GTWIN:-1}"
  export ALLOW_HOME_NVME01_PATHS="${ALLOW_HOME_NVME01_PATHS:-0}"
  export CHECK_RAFT_LOAD="${CHECK_RAFT_LOAD:-1}"
  export TRAIN_HEIGHT="${TRAIN_HEIGHT:-320}" TRAIN_WIDTH="${TRAIN_WIDTH:-512}" RESOLUTION="${RESOLUTION:-512}"
  export NFRAMES="${NFRAMES:-16}" DAVIS_VIDEO_LENGTH="${DAVIS_VIDEO_LENGTH:-24}" DAVIS_NUM_QUAL="${DAVIS_NUM_QUAL:-30}"
  export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-6}" USE_PCM="${USE_PCM:-false}"
  export MASK_DILATION="${MASK_DILATION:-0}" APPLY_GAUSSIAN_BLUR="${APPLY_GAUSSIAN_BLUR:-false}" HARD_COMP="${HARD_COMP:-true}"
  export DAVIS_METRIC_PROTOCOL="${DAVIS_METRIC_PROTOCOL:-framewise_hard_comp}"
  export COMPUTE_LPIPS="${COMPUTE_LPIPS:-1}" COMPUTE_VFID="${COMPUTE_VFID:-1}" COMPUTE_TC="${COMPUTE_TC:-1}" COMPUTE_EWARP="${COMPUTE_EWARP:-0}"
  export GAP_NORMALIZATION="log_ratio" GAP_EPS="${GAP_EPS:-1e-6}" LOSE_GAP_CLIP_TAU="${LOSE_GAP_CLIP_TAU:-1.0}"
  export BETA_DPO="${BETA_DPO:-10}" SFT_REG_WEIGHT="${SFT_REG_WEIGHT:-0.0}" LOSE_GAP_WEIGHT="${LOSE_GAP_WEIGHT:-0.25}"
  export WINNER_ABS_REG_WEIGHT="${WINNER_ABS_REG_WEIGHT:-0.05}" WINNER_GAP_REG_WEIGHT="${WINNER_GAP_REG_WEIGHT:-1.0}" WINNER_GAP_REG_MARGIN="${WINNER_GAP_REG_MARGIN:-0.0}"
  export LOSS_REGION_MODE="region" MASK_REGION_WEIGHT="${MASK_REGION_WEIGHT:-1.0}" OUTSIDE_REGION_WEIGHT="${OUTSIDE_REGION_WEIGHT:-0.05}"
  export BOUNDARY_MODE="outer" BOUNDARY_REGION_WEIGHT="0.75"
  export ADAPTIVE_NORM_MODE="batch_zscore" ADAPTIVE_NORM_EPS="${ADAPTIVE_NORM_EPS:-1e-6}"
  export MAX_STEPS="${MAX_STEPS:-2000}" CKPT_STEPS="${CKPT_STEPS:-500}" CKPT_LIMIT="${CKPT_LIMIT:-5}" VAL_STEPS="${VAL_STEPS:-999999}" LOGGING_STEPS="${LOGGING_STEPS:-10}"
  export MIXED_PRECISION="${MIXED_PRECISION:-bf16}" POLICY_DTYPE="${POLICY_DTYPE:-auto}" VAE_DTYPE="${VAE_DTYPE:-fp32}" REF_DTYPE="${REF_DTYPE:-bf16}" TEXT_DTYPE="${TEXT_DTYPE:-bf16}"
  export SPLIT_POS_NEG_FORWARD="${SPLIT_POS_NEG_FORWARD:-1}" REPORT_TO="${REPORT_TO:-none}" DPO_DIAG_LOG_EVERY="${DPO_DIAG_LOG_EVERY:-10}" DPO_DIAG_SAVE_WANDB="${DPO_DIAG_SAVE_WANDB:-false}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-none}"
  export LINGBOT_PROCESS_NAME="${LINGBOT_PROCESS_NAME:-lingbot-worldphy}"
  export PROCESS_TITLE="${PROCESS_TITLE:-${LINGBOT_PROCESS_NAME}}"
}

check_source_manifest_contract() {
  [[ -f "${SOURCE_D3_MANIFEST}" ]] || { echo "[precheck][ERROR] missing source manifest: ${SOURCE_D3_MANIFEST}" >&2; return 2; }
  if grep -q "/home/nvme01" "${SOURCE_D3_MANIFEST}"; then
    echo "[precheck][ERROR] source manifest contains /home/nvme01: ${SOURCE_D3_MANIFEST}" >&2
    return 2
  fi
}

run_custom_s1s2() {
  local variant="$1"
  local current_stage1_label="$2"
  local current_stage2_label="$3"

  echo "[${EXP_TAG}] variant=${variant} stage1=${STAGE1_RUN_NAME} stage2=${STAGE2_RUN_NAME}"
  run_stage1
  STAGE1_RUN_DIR="$(latest_run_dir stage1 "${STAGE1_RUN_NAME}")"
  require_path "${STAGE1_RUN_DIR}" "${EXP_TAG} Stage1 run dir"
  STAGE1_LAST="${STAGE1_RUN_DIR}/last_weights"
  require_path "${STAGE1_LAST}/unet_main/config.json" "${EXP_TAG} Stage1 last_weights unet_main"
  require_path "${STAGE1_LAST}/brushnet/config.json" "${EXP_TAG} Stage1 last_weights brushnet"
  require_path "${STAGE1_RUN_DIR}/dpo_diagnostics.csv" "${EXP_TAG} Stage1 dpo_diagnostics.csv"
  summarize_diag_csv "${STAGE1_RUN_DIR}/dpo_diagnostics.csv" "${EXP_DIR}/runs/${variant}/dpo_diag_stage1_summary.md" "${EXP_TAG} ${variant} Stage1"

  HYBRID_DIR="${OUTPUT_ROOT}/experiments/hybrid/${RUN_VERSION}_${STAGE1_RUN_NAME}_dpoS1_sftS2"
  echo "[${EXP_TAG}] build Stage1 hybrid: ${HYBRID_DIR}"
  "${PYTHON_BIN}" tools/build_diffueraser_dpoS1_sftS2_hybrid.py \
    --dpo_stage1_weights "${STAGE1_LAST}" \
    --sft_stage2_weights "${SFT_STAGE2_WEIGHTS}" \
    --output_dir "${HYBRID_DIR}" \
    --strict false \
    --report_path "reports/${variant}_stage1_hybrid_key_merge_report.md"

  STAGE1_VAL_DIR="${OUTPUT_ROOT}/logs/target_eval/${variant}_stage1_val_davis_${RUN_VERSION}"
  run_davis_validation "${EXP_TAG}" "${EXP_TAG} ${variant} Stage1 DPO + SFT Stage2" "${current_stage1_label}" "${HYBRID_DIR}/last_weights" "${STAGE1_VAL_DIR}"

  run_stage2 "${STAGE1_LAST}"
  STAGE2_RUN_DIR="$(latest_run_dir stage2 "${STAGE2_RUN_NAME}")"
  require_path "${STAGE2_RUN_DIR}" "${EXP_TAG} Stage2 run dir"
  STAGE2_LAST="${STAGE2_RUN_DIR}/last_weights"
  require_path "${STAGE2_LAST}/unet_main/config.json" "${EXP_TAG} Stage2 last_weights unet_main"
  require_path "${STAGE2_LAST}/brushnet/config.json" "${EXP_TAG} Stage2 last_weights brushnet"
  require_path "${STAGE2_RUN_DIR}/dpo_diagnostics.csv" "${EXP_TAG} Stage2 dpo_diagnostics.csv"
  summarize_diag_csv "${STAGE2_RUN_DIR}/dpo_diagnostics.csv" "${EXP_DIR}/runs/${variant}/dpo_diag_stage2_summary.md" "${EXP_TAG} ${variant} Stage2"

  STAGE2_VAL_DIR="${OUTPUT_ROOT}/logs/target_eval/${variant}_stage2_val_davis_${RUN_VERSION}"
  run_davis_validation "${EXP_TAG}" "${EXP_TAG} ${variant} Stage1 DPO + Stage2 DPO" "${current_stage2_label}" "${STAGE2_LAST}" "${STAGE2_VAL_DIR}"

  cat > "${EXP_DIR}/runs/${variant}/status.md" <<EOF
# ${variant} Status

status: complete
stage1_run_dir: ${STAGE1_RUN_DIR}
stage2_run_dir: ${STAGE2_RUN_DIR}
stage1_val_dir: ${STAGE1_VAL_DIR}
stage2_val_dir: ${STAGE2_VAL_DIR}
boundary_mode: outer
boundary_weight: 0.75
adaptive_norm_mode: batch_zscore
metric_protocol: DAVIS50 raw6 hard-comp / D+G off / no PCM / frame-wise in-memory metric
EOF
}

run_exp12_outer_worker() {
  set_fixed_training_env
  source_helpers
  set_fixed_training_env
  export RUN_VERSION="${EXP12_OUTER_RUN_VERSION:-${RUN_VERSION_ROOT}_exp12_adaptive_outer}"
  export EXP_TAG="Exp12AdaptiveOuter"
  export EXP_DIR="exp12_adaptive_outer_boundary"
  export REG_DIR="experiment_registry/exp12_adaptive_outer_boundary"
  export CUDA_VISIBLE_DEVICES="${EXP12_OUTER_CUDA_VISIBLE_DEVICES:-4,5,6,7}" NUM_GPUS="${EXP12_OUTER_NUM_GPUS:-4}" EVAL_GPU="${EXP12_OUTER_EVAL_GPU:-4}" MAIN_PROCESS_PORT="${EXP12_OUTER_MAIN_PROCESS_PORT:-29622}"
  export DPO_STAGE1_ENTRYPOINT="exp12_adaptive_outer_boundary/code/train_stage1.py"
  export DPO_STAGE2_ENTRYPOINT="exp12_adaptive_outer_boundary/code/train_stage2.py"
  export WANDB_PROJECT="${WANDB_PROJECT:-DPO_Diffueraser_Exp12_AdaptiveOuter}"
  local variant="exp12_batch_adaptive_outer_b075_s1s2_2000"
  mkdir -p "${EXP_DIR}/runs/${variant}"
  export EXP_NAME="${variant}"
  export STAGE1_RUN_NAME="${variant}_s1_2000_davis_pai"
  export STAGE2_RUN_NAME="${variant}_s2_2000_davis_pai"
  export CURRENT_STAGE1_LABEL="${variant}_DPO-S1_SFT-S2"
  export CURRENT_STAGE2_LABEL="${variant}_DPO-S1_DPO-S2"
  run_custom_s1s2 "${variant}" "${CURRENT_STAGE1_LABEL}" "${CURRENT_STAGE2_LABEL}"
  cat > "${EXP_DIR}/status.md" <<EOF
# Exp12 Adaptive Outer Boundary Status

status: worker_complete
run_version: ${RUN_VERSION}
gpu: ${CUDA_VISIBLE_DEVICES}
launched_variant: ${variant}
boundary_mode: outer
boundary_weight: 0.75
adaptive_norm_mode: batch_zscore
metric_protocol: DAVIS50 raw6 hard-comp / D+G off / no PCM / frame-wise in-memory metric
EOF
}

main() {
  local mode="${1:-}"
  case "${mode}" in
    __worker)
      run_exp12_outer_worker
      return
      ;;
  esac

  set_fixed_training_env
  check_source_manifest_contract
  source_helpers
  set_fixed_training_env
  precheck_common
  "${PYTHON_BIN}" -m py_compile \
    exp12_adaptive_outer_boundary/code/train_stage1.py \
    exp12_adaptive_outer_boundary/code/train_stage2.py
  bash -n scripts/launch_exp12_adaptive_outer_boundary_pai.sh

  local log="logs/pipelines/exp12_adaptive_outer_boundary_gpu4_7.log"
  nohup bash "$0" __worker > "${log}" 2>&1 < /dev/null &
  echo $! > logs/pipelines/exp12_adaptive_outer_boundary_gpu4_7.pid

  echo "===== EXP12 ADAPTIVE OUTER LAUNCH STARTED ====="
  echo "pid=$(cat logs/pipelines/exp12_adaptive_outer_boundary_gpu4_7.pid) log=${log} gpu=${EXP12_OUTER_CUDA_VISIBLE_DEVICES:-4,5,6,7}"
  echo "variant=exp12_batch_adaptive_outer_b075_s1s2_2000"
  echo "metric_protocol=DAVIS50 raw6 hard-comp / D+G off / no PCM / frame-wise in-memory metric"
  echo "No process was killed by this launcher."
}

main "$@"
