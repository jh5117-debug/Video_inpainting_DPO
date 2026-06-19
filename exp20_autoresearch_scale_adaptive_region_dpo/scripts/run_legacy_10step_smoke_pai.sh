#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/mnt/nas/hj/conda_envs/diffueraser/bin/accelerate}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/mnt/nas/hj/weights}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${WEIGHTS_DIR}/stable-diffusion-v1-5}"
VAE_PATH="${VAE_PATH:-${WEIGHTS_DIR}/sd-vae-ft-mse}"
SFT48000_WEIGHTS="${SFT48000_WEIGHTS:-/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000}"
DPO_DATA_ROOT="${DPO_DATA_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/exp09_10_11_youtubevos_gtwin_d3comp_pai}"
PREFERENCE_MANIFEST="${PREFERENCE_MANIFEST:-${DPO_DATA_ROOT}/manifests/selected_primary_comp.gtwin.pai_paths.jsonl}"
VAL_DATA_DIR="${VAL_DATA_DIR:-/mnt/workspace/hj/nas_hj/data/external/davis_432_240}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp20_autoresearch_scale_adaptive_region_dpo}"
RUN_ID="${RUN_ID:-legacy_exact_10step_smoke_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${OUTPUT_ROOT}/smoke/${RUN_ID}"
LOG_DIR="${LOG_DIR:-/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp20}"
GPU_ID="${GPU_ID:-0}"
LOCK_PATH="/tmp/exp20_gpu_${GPU_ID}.lock"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29620}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}" reports

exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  echo "[exp20-smoke][ERROR] GPU lock busy: ${LOCK_PATH}" >&2
  exit 75
fi

export PROJECT_ROOT
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export BOUNDARY_MODE="outer"
export LINGBOT_PROCESS_NAME="exp20-smoke"
export PROCESS_TITLE="exp20-smoke"

LOG_PATH="${LOG_DIR}/${RUN_ID}.log"
echo "[exp20-smoke] gpu=${GPU_ID} run_dir=${RUN_DIR} log=${LOG_PATH}"

"${ACCELERATE_BIN}" launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  exp20_autoresearch_scale_adaptive_region_dpo/code/train_exp20_stage1.py \
  --base_model_name_or_path "${BASE_MODEL_PATH}" \
  --vae_path "${VAE_PATH}" \
  --ref_model_path "${SFT48000_WEIGHTS}" \
  --policy_init_path "${SFT48000_WEIGHTS}" \
  --dpo_data_root "${DPO_DATA_ROOT}" \
  --dpo_dataset_type generated_loser_manifest \
  --preference_manifest "${PREFERENCE_MANIFEST}" \
  --train_mask_mode partial \
  --mask_from_manifest true \
  --loss_region_mode region \
  --gap_normalization log_ratio \
  --gap_eps 1e-6 \
  --lose_gap_clip_tau 1.0 \
  --mask_region_weight 1.0 \
  --boundary_region_weight 0.75 \
  --outside_region_weight 0.05 \
  --radius_mode legacy_latent_exact \
  --legacy_exact true \
  --output_dir "${RUN_DIR}" \
  --logging_dir logs-dpo-stage1 \
  --val_data_dir "${VAL_DATA_DIR}" \
  --resolution 512 \
  --train_height 320 \
  --train_width 512 \
  --nframes 16 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --dataloader_num_workers 0 \
  --learning_rate 1e-6 \
  --lr_scheduler constant \
  --lr_warmup_steps 500 \
  --max_train_steps 10 \
  --checkpointing_steps 10 \
  --checkpoints_total_limit 3 \
  --validation_steps 999999 \
  --logging_steps 1 \
  --val_num_inference_steps 6 \
  --val_mask_dilation_iter 0 \
  --mixed_precision bf16 \
  --vae_dtype fp32 \
  --policy_dtype auto \
  --ref_dtype bf16 \
  --text_dtype bf16 \
  --beta_dpo 10 \
  --sft_reg_weight 0.0 \
  --lose_gap_weight 0.25 \
  --winner_abs_reg_weight 0.05 \
  --winner_gap_reg_weight 1.0 \
  --winner_gap_reg_margin 0.0 \
  --winner_gap_reg_mode relu \
  --dpo_diag_log_every 1 \
  --dpo_diag_save_csv true \
  --dpo_diag_save_wandb false \
  --report_to none \
  --seed 20260619 \
  --split_pos_neg_forward \
  --set_grads_to_none \
  2>&1 | tee "${LOG_PATH}"

test -f "${RUN_DIR}/dpo_diagnostics.csv"
test -f "${RUN_DIR}/train_timing.json"
test -f "${RUN_DIR}/last_weights/unet_main/config.json"
test -f "${RUN_DIR}/last_weights/brushnet/config.json"

"${PYTHON_BIN}" exp20_autoresearch_scale_adaptive_region_dpo/code/check_smoke_checkpoint.py \
  --last-weights "${RUN_DIR}/last_weights" \
  --sft-weights "${SFT48000_WEIGHTS}" \
  --report reports/exp20_checkpoint_reload_audit.md

cat > reports/exp20_legacy_10step_smoke.md <<EOF
# Exp20 Legacy 10-Step Smoke

- status: REAL_10STEP_SMOKE_PASSED
- checkpoint_reload: CHECKPOINT_RELOAD_PASSED
- run_dir: \`${RUN_DIR}\`
- log: \`${LOG_PATH}\`
- gpu: \`${GPU_ID}\`
- manifest: \`${PREFERENCE_MANIFEST}\`
- effective_global_batch: 4
- world_size: 1
- per_device_batch: 1
- gradient_accumulation_steps: 4
- max_train_steps: 10
- checkpoint: \`${RUN_DIR}/checkpoint-10\`
- last_weights: \`${RUN_DIR}/last_weights\`
- dpo_diag: \`${RUN_DIR}/dpo_diagnostics.csv\`

The run used real DiffuEraser Stage1 models, SFT-48000 initialization/reference, shared winner/loser noise and timesteps, frozen reference, and Exp20 legacy_exact region maps.
EOF

echo "[exp20-smoke] passed run_dir=${RUN_DIR}"
