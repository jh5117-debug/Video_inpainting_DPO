#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"

EXP_DIR="exp18_multiframe_propagation_gated_dpo"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

RUN_VERSION="${RUN_VERSION:-$(date +%Y%m%d_%H%M%S)}"
CACHE_ROOT="${EXP18_CACHE_ROOT:-${OUTPUT_ROOT}/data/cache/exp18_multiframe_propagation_cache_limit100}"
PROP_MANIFEST="${EXP18_PROP_MANIFEST:-${CACHE_ROOT}/manifests/exp18_train_with_multiframe_prop_limit100.jsonl}"
SFT48000="${SFT48000:-/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/mnt/nas/hj/weights}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${WEIGHTS_DIR}/stable-diffusion-v1-5}"
VAE_PATH="${VAE_PATH:-${WEIGHTS_DIR}/sd-vae-ft-mse}"
DAVIS="${DAVIS:-/mnt/workspace/hj/nas_hj/data/external/davis_432_240}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

mkdir -p "$EXP_DIR/reports" "$EXP_DIR/dpo_diag" "$EXP_DIR/runs" logs/pipelines reports

fail() { echo "[Exp18][BLOCKED] $*" >&2; exit 2; }
[[ -f "$PROP_MANIFEST" ]] || fail "missing propagation manifest: $PROP_MANIFEST"
[[ -f "$SFT48000/unet_main/config.json" ]] || fail "missing SFT-48000 unet: $SFT48000"
[[ -f "$SFT48000/brushnet/config.json" ]] || fail "missing SFT-48000 brushnet: $SFT48000"
[[ -d "$BASE_MODEL_PATH" ]] || fail "missing base model: $BASE_MODEL_PATH"
[[ -d "$VAE_PATH" ]] || fail "missing VAE: $VAE_PATH"
[[ -d "$DAVIS" ]] || fail "missing DAVIS: $DAVIS"

"$PYTHON_BIN" -m py_compile "$EXP_DIR"/code/*.py
bash -n "$EXP_DIR"/scripts/*.sh

run_variant() {
  local name="$1"
  local confidence_mode="$2"
  local lambda_prop="$3"
  local lambda_gen="$4"
  local lambda_boundary="$5"
  local out="${OUTPUT_ROOT}/experiments/dpo/stage1/${RUN_VERSION}_${name}_s1_500_pai"
  mkdir -p "$out"
  echo "[Exp18] running ${name}: ${out}"
  "$PYTHON_BIN" "$EXP_DIR/code/train_exp18_stage1.py" \
    --base_model_name_or_path "$BASE_MODEL_PATH" \
    --vae_path "$VAE_PATH" \
    --ref_model_path "$SFT48000" \
    --policy_init_path "$SFT48000" \
    --dpo_data_root "$(dirname "$(dirname "$PROP_MANIFEST")")" \
    --dpo_dataset_type generated_loser_manifest \
    --preference_manifest "$PROP_MANIFEST" \
    --train_mask_mode partial \
    --mask_from_manifest true \
    --loss_region_mode region \
    --gap_normalization log_ratio \
    --gap_eps 1e-6 \
    --lose_gap_clip_tau 1.0 \
    --mask_region_weight 1.0 \
    --boundary_region_weight 0.75 \
    --outside_region_weight 0.05 \
    --output_dir "$out" \
    --logging_dir "logs-${name}" \
    --val_data_dir "$DAVIS" \
    --resolution 512 \
    --train_height 320 \
    --train_width 512 \
    --nframes 16 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 0 \
    --learning_rate 1e-6 \
    --lr_scheduler constant \
    --lr_warmup_steps 0 \
    --max_train_steps "${EXP18_MAX_STEPS:-500}" \
    --checkpointing_steps 250 \
    --checkpoints_total_limit 3 \
    --validation_steps 999999 \
    --logging_steps 10 \
    --vae_dtype fp32 \
    --policy_dtype auto \
    --ref_dtype bf16 \
    --text_dtype bf16 \
    --mixed_precision bf16 \
    --beta_dpo 10 \
    --sft_reg_weight 0.0 \
    --lose_gap_weight 0.25 \
    --winner_abs_reg_weight 0.05 \
    --winner_gap_reg_weight 1.0 \
    --winner_gap_reg_margin 0.0 \
    --report_to none \
    --dpo_diag_log_every 10 \
    --dpo_diag_save_csv true \
    --dpo_diag_save_wandb false \
    --split_pos_neg_forward \
    --set_grads_to_none \
    --confidence_mode "$confidence_mode" \
    --tau_conf "${TAU_CONF:-0.5}" \
    --lambda_prop "$lambda_prop" \
    --lambda_gen "$lambda_gen" \
    --lambda_boundary_extra "$lambda_boundary"
  cp "$out/dpo_diagnostics.csv" "$EXP_DIR/dpo_diag/${name}_stage1_500_dpo_diagnostics.csv"
}

run_variant exp18a_prop_only flow_agreement 0.1 0.0 0.1
run_variant exp18b_prop_gen flow_agreement 0.1 0.05 0.1
run_variant exp18c_oracle oracle 0.1 0.05 0.1

echo "[Exp18] stage1 gates completed"
