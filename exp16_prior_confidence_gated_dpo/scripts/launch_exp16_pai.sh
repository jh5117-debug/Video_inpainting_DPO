#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"

EXP_DIR="exp16_prior_confidence_gated_dpo"
RUN_VERSION="${RUN_VERSION:-$(date +%Y%m%d_%H%M%S)}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/mnt/nas/hj/conda_envs/diffueraser}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PREFIX}/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

SOURCE_MANIFEST="${SOURCE_MANIFEST:-${WORKSPACE_ROOT}/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4/manifests/selected_primary_comp.repaired.pai_paths.jsonl}"
GTWIN_MANIFEST="${GTWIN_MANIFEST:-${WORKSPACE_ROOT}/data/generated_losers/exp09_10_11_youtubevos_gtwin_d3comp_pai/manifests/selected_primary_comp.gtwin.pai_paths.jsonl}"
MANIFEST="${EXP16_MANIFEST:-$GTWIN_MANIFEST}"

CACHE_ROOT="${EXP16_CACHE_ROOT:-${OUTPUT_ROOT}/data/cache/exp16_propainter_prior_cache_limit100}"
PRIOR_MANIFEST="${EXP16_PRIOR_MANIFEST:-${CACHE_ROOT}/manifests/exp16_train_with_prior_limit100.jsonl}"
SFT48000="${SFT48000:-/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000}"
DAVIS="${DAVIS:-/mnt/workspace/hj/nas_hj/data/external/davis_432_240}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/mnt/nas/hj/weights}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${WEIGHTS_DIR}/stable-diffusion-v1-5}"
VAE_PATH="${VAE_PATH:-${WEIGHTS_DIR}/sd-vae-ft-mse}"
PROPAINTER="${PROPAINTER_WEIGHT_ROOT:-/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

mkdir -p "$EXP_DIR/reports" "$EXP_DIR/cache" "$EXP_DIR/dpo_diag" "$EXP_DIR/runs" \
  "$EXP_DIR/manifests" logs/pipelines reports "$CACHE_ROOT"

fail() {
  echo "[Exp16][ERROR] $*" >&2
  exit 2
}

require_path() {
  local path="$1"
  local label="$2"
  [[ -e "$path" ]] || fail "missing ${label}: ${path}"
}

{
  echo "# Exp16 PAI Precheck"
  echo
  echo "- generated_at: $(date)"
  echo "- repo: \`$ROOT\`"
  echo "- run_version: \`$RUN_VERSION\`"
  echo "- source_manifest: \`$SOURCE_MANIFEST\`"
  echo "- training_manifest: \`$MANIFEST\`"
  echo "- prior_manifest: \`$PRIOR_MANIFEST\`"
  echo "- cache_root: \`$CACHE_ROOT\`"
  echo "- SFT-48000: \`$SFT48000\`"
  echo "- base_model_path: \`$BASE_MODEL_PATH\`"
  echo "- vae_path: \`$VAE_PATH\`"
  echo "- DAVIS: \`$DAVIS\`"
  echo "- ProPainter weights: \`$PROPAINTER\`"
  echo "- python: \`$PYTHON_BIN\`"
  echo "- CUDA_VISIBLE_DEVICES: \`${CUDA_VISIBLE_DEVICES:-unset}\`"
  echo
  for path in "$SOURCE_MANIFEST" "$MANIFEST" "$SFT48000" "$BASE_MODEL_PATH" "$VAE_PATH" "$DAVIS" "$PROPAINTER"; do
    if [[ -e "$path" ]]; then
      echo "- exists: \`$path\`"
    else
      echo "- missing: \`$path\`"
    fi
  done
} > reports/exp16_prior_confidence_pai_precheck.md

require_path "$MANIFEST" "GT-win generated-loser manifest"
require_path "$SFT48000/unet_main/config.json" "SFT-48000 unet_main"
require_path "$SFT48000/brushnet/config.json" "SFT-48000 brushnet"
require_path "$BASE_MODEL_PATH" "stable-diffusion-v1-5"
require_path "$VAE_PATH" "sd-vae-ft-mse"
require_path "$DAVIS" "DAVIS eval root"
require_path "$PROPAINTER/ProPainter.pth" "ProPainter checkpoint"
require_path "$PROPAINTER/recurrent_flow_completion.pth" "ProPainter flow completion checkpoint"
require_path "$PROPAINTER/raft-things.pth" "ProPainter RAFT checkpoint"

"$PYTHON_BIN" -m py_compile "$EXP_DIR"/code/*.py
bash -n "$EXP_DIR"/scripts/*.sh

if [[ ! -f "$PRIOR_MANIFEST" ]]; then
  echo "[Exp16] missing prior manifest; generating limit=100 ProPainter prior cache"
  "$PYTHON_BIN" "$EXP_DIR/code/precompute_propainter_prior_cache.py" \
    --input_manifest "$MANIFEST" \
    --output_root "$CACHE_ROOT" \
    --propainter_model_dir "$PROPAINTER" \
    --limit 100 \
    --resume \
    --nframes 16 \
    --width 432 \
    --height 240 \
    --ref_stride 3 \
    --neighbor_length 25 \
    --subvideo_length 80 \
    --mask_dilation 0
else
  echo "[Exp16] reusing existing prior manifest: $PRIOR_MANIFEST"
fi

require_path "$PRIOR_MANIFEST" "Exp16 real ProPainter prior manifest"
cp "${CACHE_ROOT}/reports/prior_cache_report.md" reports/exp16_propainter_prior_cache_limit100_report.md 2>/dev/null || true

"$PYTHON_BIN" "$EXP_DIR/code/audit_prior_confidence_cache.py" \
  --manifest "$PRIOR_MANIFEST" \
  --output_md reports/exp16_prior_confidence_limit100_audit.md \
  --output_csv "$EXP_DIR/manifests/prior_confidence_limit100.csv" \
  --limit 100 \
  --nframes 16 \
  --width 432 \
  --height 240 \
  --alpha 5.0

PREFLIGHT_DIR="${EXP_DIR}/runs/preflight_limit100"
mkdir -p "$PREFLIGHT_DIR"
echo "[Exp16] running real trainer preflight"
"$PYTHON_BIN" "$EXP_DIR/code/train_exp16_stage1.py" \
  --base_model_name_or_path "$BASE_MODEL_PATH" \
  --vae_path "$VAE_PATH" \
  --ref_model_path "$SFT48000" \
  --policy_init_path "$SFT48000" \
  --dpo_data_root "$(dirname "$(dirname "$MANIFEST")")" \
  --dpo_dataset_type generated_loser_manifest \
  --preference_manifest "$PRIOR_MANIFEST" \
  --train_mask_mode partial \
  --mask_from_manifest true \
  --loss_region_mode region \
  --gap_normalization log_ratio \
  --gap_eps 1e-6 \
  --lose_gap_clip_tau 1.0 \
  --mask_region_weight 1.0 \
  --boundary_region_weight 0.75 \
  --outside_region_weight 0.05 \
  --output_dir "$PREFLIGHT_DIR" \
  --logging_dir logs-exp16-preflight \
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
  --max_train_steps 1 \
  --checkpointing_steps 999999 \
  --checkpoints_total_limit 1 \
  --validation_steps 999999 \
  --logging_steps 1 \
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
  --dpo_diag_log_every 1 \
  --dpo_diag_save_csv true \
  --dpo_diag_save_wandb false \
  --split_pos_neg_forward \
  --set_grads_to_none \
  --confidence_mode gt_error \
  --confidence_alpha 5.0 \
  --lambda_prior 0.1 \
  --lambda_gen 0.05 \
  --lambda_boundary_extra 0.1 \
  --preflight_only

cp "$PREFLIGHT_DIR/dpo_diagnostics.csv" "$EXP_DIR/dpo_diag/preflight_dpo_diagnostics.csv"
cat > reports/exp16_preflight_limit100_report.md <<EOF
# Exp16 Preflight Limit100 Report

- generated_at: $(date)
- status: passed
- prior_manifest: \`$PRIOR_MANIFEST\`
- output_dir: \`$PREFLIGHT_DIR\`
- dpo_diag: \`$EXP_DIR/dpo_diag/preflight_dpo_diagnostics.csv\`

The preflight uses the isolated Exp16 trainer, real ProPainter prior frames,
VAE-encoded latent targets, and reconstructed predicted x0 latent. It is not a
frozen-reference epsilon proxy.
EOF

if [[ "${EXP16_RUN_STAGE1_500:-1}" != "1" ]]; then
  echo "[Exp16] preflight passed; Stage1 500 skipped by EXP16_RUN_STAGE1_500=${EXP16_RUN_STAGE1_500:-unset}"
  exit 0
fi

STAGE1_DIR="${OUTPUT_ROOT}/experiments/dpo/stage1/${RUN_VERSION}_exp16_prior_confidence_s1_500_limit100_pai"
mkdir -p "$STAGE1_DIR"
echo "[Exp16] starting Stage1 500 limit100: $STAGE1_DIR"
"$PYTHON_BIN" "$EXP_DIR/code/train_exp16_stage1.py" \
  --base_model_name_or_path "$BASE_MODEL_PATH" \
  --vae_path "$VAE_PATH" \
  --ref_model_path "$SFT48000" \
  --policy_init_path "$SFT48000" \
  --dpo_data_root "$(dirname "$(dirname "$MANIFEST")")" \
  --dpo_dataset_type generated_loser_manifest \
  --preference_manifest "$PRIOR_MANIFEST" \
  --train_mask_mode partial \
  --mask_from_manifest true \
  --loss_region_mode region \
  --gap_normalization log_ratio \
  --gap_eps 1e-6 \
  --lose_gap_clip_tau 1.0 \
  --mask_region_weight 1.0 \
  --boundary_region_weight 0.75 \
  --outside_region_weight 0.05 \
  --output_dir "$STAGE1_DIR" \
  --logging_dir logs-exp16-stage1-500 \
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
  --max_train_steps 500 \
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
  --confidence_mode gt_error \
  --confidence_alpha 5.0 \
  --lambda_prior 0.1 \
  --lambda_gen 0.05 \
  --lambda_boundary_extra 0.1

cp "$STAGE1_DIR/dpo_diagnostics.csv" "$EXP_DIR/dpo_diag/stage1_500_dpo_diagnostics.csv"
cat > reports/exp16_stage1_500_limit100_report.md <<EOF
# Exp16 Stage1 500 Limit100 Report

- generated_at: $(date)
- status: completed
- run_dir: \`$STAGE1_DIR\`
- prior_manifest: \`$PRIOR_MANIFEST\`
- dpo_diag: \`$EXP_DIR/dpo_diag/stage1_500_dpo_diagnostics.csv\`

Stage2 and full 2000+2000 training were not launched.
EOF

echo "[Exp16] Stage1 500 limit100 completed"
