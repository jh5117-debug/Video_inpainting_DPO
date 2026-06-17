#!/usr/bin/env bash
set -euo pipefail

# Exp17 overnight gate launcher.
# Runs only Stage1 gates and DAVIS10 sanity evals. It never launches Stage2.

PROJECT_ROOT="${EXP17_PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
cd "${PROJECT_ROOT}"

RUN_VERSION="${RUN_VERSION:-$(date +%Y%m%d_%H%M%S)_exp17_saturation_positive}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${OUTPUT_ROOT}/experiments}"
OUT_ROOT="${OUT_ROOT:-${OUTPUT_ROOT}/logs/target_eval/exp17_saturation_positive_dpo_${RUN_VERSION}_davis10}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs/pipelines/exp17_saturation_positive_dpo_${RUN_VERSION}}"
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" exp17_saturation_positive_dpo/runs exp17_saturation_positive_dpo/reports

PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/mnt/nas/hj/conda_envs/diffueraser}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
MAIN_PROCESS_PORT_BASE="${MAIN_PROCESS_PORT_BASE:-29717}"
EVAL_GPU="${EVAL_GPU:-0}"

MANIFEST="${MANIFEST:-${PROJECT_ROOT}/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4/manifests/selected_primary_comp.repaired.pai_paths.jsonl}"
SFT48000="${SFT48000:-/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000}"
EXP11_OUTER_B075_S2="${EXP11_OUTER_B075_S2:-${OUTPUT_ROOT}/experiments/dpo/stage2/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s2_2000_davis_pai/last_weights}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/mnt/nas/hj/weights}"
DAVIS_ROOT="${DAVIS_ROOT:-/mnt/workspace/hj/nas_hj/data/external/davis_432_240}"
VIDEO_ROOT="${VIDEO_ROOT:-${DAVIS_ROOT}/JPEGImages_432_240}"
MASK_ROOT="${MASK_ROOT:-${DAVIS_ROOT}/test_masks}"
PROP="${PROP:-/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter}"
BASE_MODEL="${BASE_MODEL:-${WEIGHTS_DIR}/stable-diffusion-v1-5}"
VAE="${VAE:-${WEIGHTS_DIR}/sd-vae-ft-mse}"
PCM="${PCM:-${WEIGHTS_DIR}/PCM_Weights}"
RAFT_MODEL_PATH="${RAFT_MODEL_PATH:-${PROP}/raft-things.pth}"

VIDEOS="${VIDEOS:-boat,rhino,dog-agility,blackswan,lucia,bear,dance-jump,soccerball,kite-surf,breakdance}"
VIDEO_LENGTH="${VIDEO_LENGTH:-24}"
SUBSET_ROOT="${OUT_ROOT}/davis10_subset"
SUBSET_VIDEO_ROOT="${SUBSET_ROOT}/JPEGImages_432_240"
SUBSET_MASK_ROOT="${SUBSET_ROOT}/test_masks"

require_path() {
  local path="$1"
  local label="$2"
  [[ -e "${path}" ]] || { echo "[missing] ${label}: ${path}" >&2; exit 2; }
}

echo "[exp17] hostname=$(hostname)"
echo "[exp17] project=${PROJECT_ROOT}"
echo "[exp17] run_version=${RUN_VERSION}"
echo "[exp17] cuda=${CUDA_VISIBLE_DEVICES}"
nvidia-smi || true

require_path "${PY}" "python"
require_path "${MANIFEST}" "generated loser manifest"
require_path "${SFT48000}" "SFT-48000"
require_path "${EXP11_OUTER_B075_S2}" "Exp11 outer b0.75 S2"
require_path "${VIDEO_ROOT}" "DAVIS video root"
require_path "${MASK_ROOT}" "DAVIS mask root"
require_path "${BASE_MODEL}" "base model"
require_path "${VAE}" "VAE"
require_path "${PROP}" "ProPainter weights"
require_path "exp17_saturation_positive_dpo/code/train_exp17_stage1.py" "Exp17 trainer"
require_path "tools/run_davis50_framewise_protocol_eval.py" "fixed DAVIS eval wrapper"
require_path "tools/build_diffueraser_dpoS1_sftS2_hybrid.py" "hybrid checkpoint builder"

"${PY}" -m py_compile exp17_saturation_positive_dpo/code/*.py
bash -n exp17_saturation_positive_dpo/scripts/*.sh

rm -rf "${SUBSET_ROOT}"
mkdir -p "${SUBSET_VIDEO_ROOT}" "${SUBSET_MASK_ROOT}"
IFS=',' read -r -a VIDEO_ARRAY <<< "${VIDEOS}"
for video in "${VIDEO_ARRAY[@]}"; do
  video="$(echo "${video}" | xargs)"
  require_path "${VIDEO_ROOT}/${video}" "DAVIS video ${video}"
  require_path "${MASK_ROOT}/${video}" "DAVIS mask ${video}"
  ln -s "${VIDEO_ROOT}/${video}" "${SUBSET_VIDEO_ROOT}/${video}"
  ln -s "${MASK_ROOT}/${video}" "${SUBSET_MASK_ROOT}/${video}"
done
printf "%s\n" "${VIDEO_ARRAY[@]}" > "${OUT_ROOT}/selected_videos.txt"

run_eval() {
  local label="$1"
  local weights="$2"
  local out="${OUT_ROOT}/${label}"
  local log="${LOG_ROOT}/${label}_eval.log"
  echo "[exp17] eval label=${label} weights=${weights}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PY}" tools/run_davis50_framewise_protocol_eval.py \
    --video_root "${SUBSET_VIDEO_ROOT}" \
    --mask_root "${SUBSET_MASK_ROOT}" \
    --gt_root "${SUBSET_VIDEO_ROOT}" \
    --base_model_path "${BASE_MODEL}" \
    --vae_path "${VAE}" \
    --propainter_model_dir "${PROP}" \
    --pcm_weights_path "${PCM}" \
    --diffueraser_path "${weights}" \
    --save_path "${out}" \
    --label "${label}" \
    --input_size 432x240 \
    --video_length "${VIDEO_LENGTH}" \
    --num_inference_steps 6 \
    --use_pcm false \
    --mask_dilation_iter 0 \
    --raft_model_path "${RAFT_MODEL_PATH}" \
    --save_videos \
    --save_comp_frames > "${log}" 2>&1
}

if [[ ! -f "${OUT_ROOT}/SFT48000_baseline/metrics/summary.csv" ]]; then
  run_eval "SFT48000_baseline" "${SFT48000}"
fi
if [[ ! -f "${OUT_ROOT}/Exp11_boundary_outer_b075_S2/metrics/summary.csv" ]]; then
  run_eval "Exp11_boundary_outer_b075_S2" "${EXP11_OUTER_B075_S2}"
fi

run_stage1_variant() {
  local variant="$1"
  local exp17_variant="$2"
  local lambda_abs="$3"
  local lambda_pos="$4"
  local target_margin="$5"
  local port="$6"
  local stage_name="${variant}_s1_1000_pai"
  local stage_dir="${EXPERIMENTS_DIR}/dpo/stage1/${RUN_VERSION}_${stage_name}"
  local train_log="${LOG_ROOT}/${variant}_train.log"

  echo "[exp17] train ${variant} variant=${exp17_variant}"
  export PROJECT_ROOT OUTPUT_ROOT EXPERIMENTS_DIR DATA="${OUTPUT_ROOT}"
  export CONDA_ENV_PREFIX PYTHON_BIN="${PY}" DPO_ACCELERATE_PYTHON_BIN="${PY}"
  export DPO_STAGE1_ENTRYPOINT="exp17_saturation_positive_dpo/code/train_exp17_stage1.py"
  export CUDA_VISIBLE_DEVICES NUM_GPUS MAIN_PROCESS_PORT="${port}"
  export WEIGHTS_DIR DPO_DATA_ROOT="${PROJECT_ROOT}/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4"
  export DPO_DATASET_TYPE="generated_loser_manifest" PREFERENCE_MANIFEST="${MANIFEST}"
  export TRAIN_MASK_MODE="partial" MASK_FROM_MANIFEST="true"
  export LOSS_REGION_MODE="region" GAP_NORMALIZATION="log_ratio" GAP_EPS="1e-6" LOSE_GAP_CLIP_TAU="1.0"
  export MASK_REGION_WEIGHT="1.0" BOUNDARY_REGION_WEIGHT="0.75" OUTSIDE_REGION_WEIGHT="0.05" BOUNDARY_MODE="outer"
  export ENABLE_DPO_DIAG="true" DPO_DIAG_LOG_EVERY="10" DPO_DIAG_SAVE_CSV="true" DPO_DIAG_SAVE_WANDB="false"
  export VAL_DATA_DIR="${DAVIS_ROOT}" REF_MODEL_PATH="${SFT48000}"
  export RUN_NAME="${stage_name}" RUN_VERSION
  export MAX_STEPS="1000" CKPT_STEPS="500" CKPT_LIMIT="3" VAL_STEPS="999999" LOGGING_STEPS="10"
  export TRAIN_HEIGHT="240" TRAIN_WIDTH="432" RESOLUTION="240" NFRAMES="16"
  export BETA_DPO="10" SFT_REG_WEIGHT="0.0" LOSE_GAP_WEIGHT="0.25" DPO_LOSE_GAP_WEIGHT="0.25"
  export WINNER_ABS_REG_WEIGHT="${lambda_abs}" WINNER_GAP_REG_WEIGHT="${lambda_pos}" WINNER_GAP_REG_MARGIN="0.0"
  export REPORT_TO="none" WANDB_PROJECT="DPO_Diffueraser_Exp17" MIXED_PRECISION="bf16"
  export POLICY_DTYPE="auto" VAE_DTYPE="fp32" REF_DTYPE="bf16" TEXT_DTYPE="bf16"
  export SPLIT_POS_NEG_FORWARD="1" RESUME_FROM_CHECKPOINT="none"
  export EXP17_VARIANT="${exp17_variant}" EXP17_LAMBDA_ABS="${lambda_abs}" EXP17_LAMBDA_POS="${lambda_pos}" EXP17_MARGIN_POS="0.0"
  export EXP17_TARGET_MARGIN="${target_margin}" EXP17_SATURATION_KAPPA="5.0"
  export DPO_GAP_TRACE_CSV="${stage_dir}/dpo_gap_trace.csv"
  export DPO_GAP_SAMPLES_JSONL_GZ="${stage_dir}/dpo_gap_samples.jsonl.gz"

  "${PY}" -m accelerate.commands.launch \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision "${MIXED_PRECISION}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    exp17_saturation_positive_dpo/code/train_exp17_stage1.py \
    --base_model_name_or_path "${BASE_MODEL}" \
    --vae_path "${VAE}" \
    --ref_model_path "${REF_MODEL_PATH}" \
    --dpo_data_root "${DPO_DATA_ROOT}" \
    --dpo_dataset_type "${DPO_DATASET_TYPE}" \
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
    --val_data_dir "${VAL_DATA_DIR}" \
    --output_dir "${stage_dir}" \
    --train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 0 \
    --learning_rate 1e-6 \
    --lr_scheduler constant \
    --lr_warmup_steps 500 \
    --max_train_steps "${MAX_STEPS}" \
    --checkpointing_steps "${CKPT_STEPS}" \
    --checkpoints_total_limit "${CKPT_LIMIT}" \
    --validation_steps "${VAL_STEPS}" \
    --logging_steps "${LOGGING_STEPS}" \
    --val_num_inference_steps 6 \
    --val_mask_dilation_iter 0 \
    --resolution "${RESOLUTION}" \
    --train_height "${TRAIN_HEIGHT}" \
    --train_width "${TRAIN_WIDTH}" \
    --nframes "${NFRAMES}" \
    --seed 42 \
    --mixed_precision "${MIXED_PRECISION}" \
    --vae_dtype "${VAE_DTYPE}" \
    --policy_dtype "${POLICY_DTYPE}" \
    --ref_dtype "${REF_DTYPE}" \
    --text_dtype "${TEXT_DTYPE}" \
    --report_to "${REPORT_TO}" \
    --tracker_project_name "${WANDB_PROJECT}" \
    --beta_dpo "${BETA_DPO}" \
    --sft_reg_weight "${SFT_REG_WEIGHT}" \
    --lose_gap_weight "${DPO_LOSE_GAP_WEIGHT}" \
    --winner_abs_reg_weight "${WINNER_ABS_REG_WEIGHT}" \
    --winner_gap_reg_weight "${WINNER_GAP_REG_WEIGHT}" \
    --winner_gap_reg_margin "${WINNER_GAP_REG_MARGIN}" \
    --winner_gap_reg_mode relu \
    --davis_oversample 10 \
    --gradient_checkpointing \
    --chunk_aligned \
    --split_pos_neg_forward > "${train_log}" 2>&1
  require_path "${stage_dir}/last_weights" "${variant} last_weights"
  cp "${stage_dir}/dpo_diagnostics.csv" "exp17_saturation_positive_dpo/dpo_diag/${variant}_stage1_1000_dpo_diagnostics.csv" || true

  local hybrid="${EXPERIMENTS_DIR}/hybrid/${RUN_VERSION}_${variant}_dpoS1_sftS2"
  "${PY}" tools/build_diffueraser_dpoS1_sftS2_hybrid.py \
    --dpo_stage1_weights "${stage_dir}/last_weights" \
    --sft_stage2_weights "${SFT48000}" \
    --output_dir "${hybrid}" \
    --strict false \
    --report_path "reports/${variant}_hybrid_key_merge_report.md" > "${LOG_ROOT}/${variant}_hybrid.log" 2>&1
  require_path "${hybrid}/last_weights" "${variant} hybrid last_weights"
  run_eval "${variant}" "${hybrid}/last_weights"

  "${PY}" exp17_saturation_positive_dpo/code/make_exp17_visual_cases.py \
    --video_root "${SUBSET_VIDEO_ROOT}" \
    --mask_root "${SUBSET_MASK_ROOT}" \
    --sft_eval_root "${OUT_ROOT}/SFT48000_baseline" \
    --exp11_eval_root "${OUT_ROOT}/Exp11_boundary_outer_b075_S2" \
    --exp17_eval_root "${OUT_ROOT}/${variant}" \
    --exp17_label "${variant}" \
    --output_root "${OUT_ROOT}/visual_cases/${variant}" \
    --videos "${VIDEOS}" \
    --video_length "${VIDEO_LENGTH}" > "${LOG_ROOT}/${variant}_visual.log" 2>&1
}

run_stage1_variant "exp17a_positive_s1_1000" "positive" "0.05" "2.0" "1.0" "${MAIN_PROCESS_PORT_BASE}"
run_stage1_variant "exp17b_saturation_s1_1000" "saturation" "0.05" "1.0" "1.0" "$((MAIN_PROCESS_PORT_BASE + 1))"
run_stage1_variant "exp17c_combined_s1_1000" "combined" "0.05" "2.0" "1.0" "$((MAIN_PROCESS_PORT_BASE + 2))"

"${PY}" - "${OUT_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/metrics/summary.csv")):
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    row["result_dir"] = str(path.parents[1])
    rows.append(row)

out = root / "metrics"
out.mkdir(parents=True, exist_ok=True)
keys = []
for row in rows:
    for key in row:
        if key not in keys:
            keys.append(key)
with (out / "summary_all.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)

cols = [
    ("whole_video_psnr_mean", "PSNR"),
    ("whole_video_ssim_mean", "SSIM"),
    ("strict_mask_pixel_psnr_mean", "strict mask PSNR"),
    ("boundary_pixel_psnr_mean", "boundary PSNR"),
    ("mask_bbox_psnr_mean", "bbox PSNR"),
    ("mask_bbox_ssim_mean", "bbox SSIM"),
]
lines = ["# Exp17 DAVIS10 Gate Summary", "", "| Method | " + " | ".join(name for _, name in cols) + " | rows |", "|---|" + "|".join(["---:"] * (len(cols) + 1)) + "|"]
for row in rows:
    vals = []
    for col, _ in cols:
        val = row.get(col, "")
        vals.append("" if val == "" else f"{float(val):.4f}")
    lines.append(f"| {row.get('model_label','')} | " + " | ".join(vals) + f" | {row.get('rows','')} |")
(out / "summary_all.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

by_label = {r.get("model_label", ""): r for r in rows}
exp11 = by_label.get("Exp11_boundary_outer_b075_S2")
candidates = [r for r in rows if r.get("model_label", "").startswith("exp17")]
decision = {"best_variant": None, "metric_positive": False, "reason": "missing Exp11 or candidates"}
if exp11 and candidates:
    def f(row, key):
        try:
            return float(row.get(key, "nan"))
        except Exception:
            return float("nan")
    best = max(candidates, key=lambda r: (f(r, "whole_video_psnr_mean"), f(r, "strict_mask_pixel_psnr_mean")))
    decision["best_variant"] = best.get("model_label")
    decision["metric_positive"] = (
        f(best, "whole_video_psnr_mean") >= f(exp11, "whole_video_psnr_mean")
        and f(best, "whole_video_ssim_mean") >= f(exp11, "whole_video_ssim_mean")
        and f(best, "strict_mask_pixel_psnr_mean") >= f(exp11, "strict_mask_pixel_psnr_mean")
        and f(best, "boundary_pixel_psnr_mean") >= f(exp11, "boundary_pixel_psnr_mean")
    )
    decision["reason"] = "metric positive on DAVIS10" if decision["metric_positive"] else "no variant beats Exp11 on DAVIS10 primary metrics"
(out / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
print("\n".join(lines))
print(json.dumps(decision, indent=2))
PY

cat > "${OUT_ROOT}/report.md" <<EOF
# Exp17 Saturation-Positive DPO Overnight Gate

- run_version: ${RUN_VERSION}
- output: ${OUT_ROOT}
- logs: ${LOG_ROOT}
- protocol: DAVIS10 raw6, no PCM, no mask dilation, no Gaussian blur, hard comp, frame-wise metrics, no VBench
- variants: Exp17a positive, Exp17b saturation, Exp17c combined
- no Stage2 launched

Summary:

\`\`\`text
${OUT_ROOT}/metrics/summary_all.md
\`\`\`
EOF

cp "${OUT_ROOT}/metrics/summary_all.csv" "reports/exp17_davis10_gate_metric_summary.csv"
cp "${OUT_ROOT}/metrics/summary_all.md" "reports/exp17_davis10_gate_metric_summary.md"
cp "${OUT_ROOT}/metrics/decision.json" "reports/exp17_davis10_gate_decision.json"

echo "[exp17] completed gates"
echo "[exp17] OUT_ROOT=${OUT_ROOT}"
echo "[exp17] LOG_ROOT=${LOG_ROOT}"
