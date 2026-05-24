#!/usr/bin/env bash
set -uo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root" || exit 1

export LINGBOT_PROCESS_NAME="${LINGBOT_PROCESS_NAME:-lingbot-phy}"

if [ -f configs/paths/pai.detected.env ]; then
  # shellcheck disable=SC1091
  source configs/paths/pai.detected.env
fi

source_dataset="${SOURCE_DATASET:-${VIDEO_DPO_TRAIN_DATA_YAML:-${VIDEO_DPO_DATA_ROOT:-}}}"
output_root="${OUTPUT_ROOT:-$repo_root/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4}"
model_name="${MODEL_NAME:-all}"
num_masks="${NUM_MASKS_PER_VIDEO:-4}"
seed="${SEED:-20260523}"
limit="${LIMIT:-}"
mask_policy_config="${MASK_POLICY_CONFIG:-configs/generation/videodpo_partialmask_policy_v1_medium_hard_k4.yaml}"
selection_policy_config="${SELECTION_POLICY_CONFIG:-configs/generation/medium_hard_balanced_selection_v1.yaml}"

mkdir -p \
  "$output_root/manifests" \
  "$output_root/candidates" \
  "$output_root/masks" \
  "$output_root/reports" \
  "$output_root/logs"

models=(diffueraser propainter cococo minimax_remover)
if [ "$model_name" != "all" ]; then
  models=("$model_name")
fi

for model in "${models[@]}"; do
  for comp in true false; do
    prefix="nocomp"
    [ "$comp" = "true" ] && prefix="comp"
    manifest="$output_root/manifests/${prefix}_${model}.schema.json"
    echo "[partialmask-k4-plan] model=$model comp=$comp manifest=$manifest"
    args=(
      -m tools.offline_loser_generation
      --source_dataset "${source_dataset:-UNCONFIRMED}" \
      --output_root "$output_root" \
      --model_name "$model" \
      --mask_mode partial \
      --mask_convention "canonical_320x512_16f_png_255_inpaint_0_keep_comp_normalized_before_manifest" \
      --comp "$comp" \
      --offline true \
      --num_masks_per_video "$num_masks" \
      --seed "$seed" \
      --save_manifest "$manifest" \
      --mask_policy_config "$mask_policy_config" \
      --selection_policy_config "$selection_policy_config" \
      --dry_run \
      --allow_missing_assets
    )
    if [ -n "$limit" ]; then
      args+=(--limit "$limit")
    fi
    python "${args[@]}"
  done
done

echo "[partialmask-k4-plan] output_root=$output_root"
echo "[partialmask-k4-plan] mask_policy=$mask_policy_config selection_policy=$selection_policy_config"
echo "[partialmask-k4-plan] dry-run only; candidates_all and selected manifests must be produced by calibration/full generation."
