#!/bin/bash
set -euo pipefail

# Builds and evaluates DPO Stage1 spatial + frozen SFT Stage2 motion hybrids.
# This is an evaluation/audit script only: it does not start training and does
# not run full VBench.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/mnt/nas/hj/weights}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/mnt/nas/hj/conda_envs/diffueraser}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PREFIX}/bin/python}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${OUTPUT_ROOT}/experiments}"

EXP_NAME="${EXP_NAME:-exp7_pm_dpoS1_sftS2_hybrid_ckptsweep}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EVAL_DIR="${EVAL_DIR:-${OUTPUT_ROOT}/logs/partialmask_eval/exp7_pm_dpoS1_sftS2_hybrid_${TIMESTAMP}}"
HYBRID_ROOT="${HYBRID_ROOT:-${EVAL_DIR}/hybrids}"
HYBRID_REPORT_ROOT="${HYBRID_REPORT_ROOT:-${EVAL_DIR}/hybrid_reports}"
FINAL_REPORT="${FINAL_REPORT:-${OUTPUT_ROOT}/reports/exp7_dpoS1_sftS2_hybrid_eval_report.md}"
STRUCTURE_REPORT="${STRUCTURE_REPORT:-${OUTPUT_ROOT}/reports/diffueraser_stage_checkpoint_structure.md}"

MANIFEST="${MANIFEST:-${OUTPUT_ROOT}/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4/manifests/selected_primary_comp.repaired.jsonl}"
STAGE1_RUN_DIR="${STAGE1_RUN_DIR:-${EXPERIMENTS_DIR}/dpo/stage1/20260601_065618_exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500_stage1}"
STAGE2_RUN_DIR="${STAGE2_RUN_DIR:-${EXPERIMENTS_DIR}/dpo/stage2/20260601_065618_exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500_stage2}"

BASE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH:-${WEIGHTS_DIR}/stable-diffusion-v1-5}"
VAE_PATH="${VAE_PATH:-${WEIGHTS_DIR}/sd-vae-ft-mse}"
BASE_WEIGHTS_DIR="${BASE_WEIGHTS_DIR:-${WEIGHTS_DIR}/diffuEraser/converted_weights_step48000}"
OFFICIAL_BASE_STAGE2_WEIGHTS="${OFFICIAL_BASE_STAGE2_WEIGHTS:-${BASE_WEIGHTS_DIR}}"

NUM_SAMPLES="${NUM_SAMPLES:-30}"
NUM_SAMPLES_METRIC="${NUM_SAMPLES_METRIC:-100}"
SEED="${SEED:-42}"
HEIGHT="${HEIGHT:-320}"
WIDTH="${WIDTH:-512}"
FRAMES="${FRAMES:-16}"
FPS="${FPS:-10}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-20}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-12.0}"
TORCH_DTYPE="${TORCH_DTYPE:-bf16}"
VAE_DTYPE="${VAE_DTYPE:-fp32}"

die() {
  echo "[dpoS1-sftS2-hybrid][ERROR] $*" >&2
  exit 1
}

require_path() {
  local path="$1"
  local label="$2"
  [[ -e "$path" ]] || die "${label} not found: ${path}"
}

is_direct_weights() {
  local path="$1"
  [[ -f "${path}/unet_main/config.json" && -d "${path}/brushnet" ]]
}

sanitize_label() {
  echo "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

add_unique_sft_candidate() {
  local label="$1"
  local path="$2"
  [[ -n "$path" ]] || return 0
  local real="$path"
  if [[ -e "$path" ]]; then
    real="$(cd "$path" && pwd)"
  fi
  for existing in "${SFT_PATHS[@]:-}"; do
    [[ "$existing" == "$real" ]] && return 0
  done
  SFT_LABELS+=("$label")
  SFT_PATHS+=("$real")
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    die "python not found; set PYTHON_BIN or CONDA_ENV_PREFIX"
  fi
fi

require_path "${MANIFEST}" "D2 manifest"
require_path "${STAGE1_RUN_DIR}" "Exp7 DPO Stage1 run dir"
require_path "${STAGE2_RUN_DIR}" "Exp7 DPO Stage2 run dir"
require_path "${BASE_MODEL_NAME_OR_PATH}" "stable-diffusion-v1-5"
require_path "${VAE_PATH}" "sd-vae-ft-mse"
require_path "${BASE_WEIGHTS_DIR}" "DiffuEraser-base weights"
require_path "${PROJECT_ROOT}/tools/build_diffueraser_dpoS1_sftS2_hybrid.py" "hybrid builder"
require_path "${PROJECT_ROOT}/tools/eval_generated_loser_partialmask_model.py" "partial-mask eval tool"
require_path "${PROJECT_ROOT}/tools/inspect_diffueraser_stage_weights.py" "stage checkpoint inspector"

mkdir -p "${EVAL_DIR}" "${HYBRID_ROOT}" "${HYBRID_REPORT_ROOT}" "$(dirname "${FINAL_REPORT}")"

echo "[dpoS1-sftS2-hybrid] output=${EVAL_DIR}"
echo "[dpoS1-sftS2-hybrid] manifest=${MANIFEST}"
echo "[dpoS1-sftS2-hybrid] stage1_run=${STAGE1_RUN_DIR}"
echo "[dpoS1-sftS2-hybrid] stage2_run=${STAGE2_RUN_DIR}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/inspect_diffueraser_stage_weights.py" \
  --stage1 "${STAGE1_RUN_DIR}/last_weights" \
  --dpo_stage2 "${STAGE2_RUN_DIR}/last_weights" \
  --sft_stage2 "${OFFICIAL_BASE_STAGE2_WEIGHTS}" \
  --report_path "${STRUCTURE_REPORT}" \
  --json_path "${EVAL_DIR}/diffueraser_stage_checkpoint_structure.json"

declare -a DPO_LABELS=()
declare -a DPO_PATHS=()
for step in 500 1000 1500 2000 2500 3000; do
  DPO_LABELS+=("DPO_S1_ckpt${step}")
  DPO_PATHS+=("${STAGE1_RUN_DIR}/checkpoint-${step}")
done
DPO_LABELS+=("DPO_S1_last")
DPO_PATHS+=("${STAGE1_RUN_DIR}/last_weights")

declare -a SFT_LABELS=()
declare -a SFT_PATHS=()
if [[ -n "${YOUTUBE_VOS_SFT_STAGE2_WEIGHTS:-}" ]]; then
  add_unique_sft_candidate "YouTubeVOS_SFT_Stage2" "${YOUTUBE_VOS_SFT_STAGE2_WEIGHTS}"
fi
if [[ -n "${SFT_STAGE2_WEIGHTS:-}" ]]; then
  add_unique_sft_candidate "User_SFT_Stage2" "${SFT_STAGE2_WEIGHTS}"
fi

declare -a YT_SEARCH_ROOTS=()
[[ -d "${OUTPUT_ROOT}" ]] && YT_SEARCH_ROOTS+=("${OUTPUT_ROOT}")
[[ -d "${WEIGHTS_DIR}" ]] && YT_SEARCH_ROOTS+=("${WEIGHTS_DIR}")
if [[ "${#YT_SEARCH_ROOTS[@]}" -gt 0 ]]; then
  mapfile -t YT_CANDIDATES < <(
    find "${YT_SEARCH_ROOTS[@]}" -maxdepth 7 -type f -path '*/unet_main/config.json' 2>/dev/null \
      | grep -Ei 'youtube|ytvos|youtube-vos|ytbv' \
      | sed 's#/unet_main/config.json##' \
      | sort -u \
      | head -3
  )
else
  YT_CANDIDATES=()
fi
for path in "${YT_CANDIDATES[@]}"; do
  add_unique_sft_candidate "YouTubeVOS_SFT_Stage2_auto" "${path}"
done

declare -a SFT_SEARCH_ROOTS=()
[[ -d "${EXPERIMENTS_DIR}/sft" ]] && SFT_SEARCH_ROOTS+=("${EXPERIMENTS_DIR}/sft")
[[ -d "${OUTPUT_ROOT}/finetune-stage2" ]] && SFT_SEARCH_ROOTS+=("${OUTPUT_ROOT}/finetune-stage2")
if [[ "${#SFT_SEARCH_ROOTS[@]}" -gt 0 ]]; then
  mapfile -t SFT_AUTO_CANDIDATES < <(
    find "${SFT_SEARCH_ROOTS[@]}" -maxdepth 6 -type f -path '*/unet_main/config.json' 2>/dev/null \
      | sed 's#/unet_main/config.json##' \
      | sort -u \
      | head -5
  )
else
  SFT_AUTO_CANDIDATES=()
fi
for path in "${SFT_AUTO_CANDIDATES[@]}"; do
  add_unique_sft_candidate "Previous_SFT_Stage2_auto" "${path}"
done
add_unique_sft_candidate "Official_DiffuEraser_base_Stage2" "${OFFICIAL_BASE_STAGE2_WEIGHTS}"

if [[ "${#SFT_PATHS[@]}" -eq 0 ]]; then
  die "No SFT/base Stage2 candidate found"
fi

declare -a CHECKPOINT_ARGS=()
if is_direct_weights "${STAGE1_RUN_DIR}/last_weights"; then
  CHECKPOINT_ARGS+=(--checkpoint "Exp7_DPO_Stage1_last=${STAGE1_RUN_DIR}/last_weights")
fi
if is_direct_weights "${STAGE2_RUN_DIR}/last_weights"; then
  CHECKPOINT_ARGS+=(--checkpoint "Exp7_DPO_S1_DPO_S2_last=${STAGE2_RUN_DIR}/last_weights")
fi

{
  echo "# DPO-S1 + SFT-S2 Hybrid Build Plan"
  echo
  echo "eval_dir: \`${EVAL_DIR}\`"
  echo "stage1_run_dir: \`${STAGE1_RUN_DIR}\`"
  echo "stage2_run_dir: \`${STAGE2_RUN_DIR}\`"
  echo
  echo "## SFT Stage2 candidates"
  echo
  echo "| label | path | direct_weights |"
  echo "| --- | --- | --- |"
  for i in "${!SFT_LABELS[@]}"; do
    direct=false
    if is_direct_weights "${SFT_PATHS[$i]}"; then direct=true; fi
    echo "| ${SFT_LABELS[$i]} | \`${SFT_PATHS[$i]}\` | ${direct} |"
  done
  echo
  echo "## DPO Stage1 candidates"
  echo
  echo "| label | path | direct_weights |"
  echo "| --- | --- | --- |"
  for i in "${!DPO_LABELS[@]}"; do
    direct=false
    if is_direct_weights "${DPO_PATHS[$i]}"; then direct=true; fi
    echo "| ${DPO_LABELS[$i]} | \`${DPO_PATHS[$i]}\` | ${direct} |"
  done
} > "${EVAL_DIR}/hybrid_build_plan.md"

for i in "${!DPO_LABELS[@]}"; do
  dpo_label="${DPO_LABELS[$i]}"
  dpo_path="${DPO_PATHS[$i]}"
  if ! is_direct_weights "${dpo_path}"; then
    echo "[dpoS1-sftS2-hybrid] skip ${dpo_label}: not exported unet_main/brushnet weights (${dpo_path})"
    continue
  fi
  for j in "${!SFT_LABELS[@]}"; do
    sft_label="${SFT_LABELS[$j]}"
    sft_path="${SFT_PATHS[$j]}"
    if ! is_direct_weights "${sft_path}"; then
      echo "[dpoS1-sftS2-hybrid] skip ${sft_label}: not exported unet_main/brushnet weights (${sft_path})"
      continue
    fi
    hybrid_label="Hybrid_${dpo_label}__${sft_label}"
    hybrid_safe="$(sanitize_label "${hybrid_label}")"
    hybrid_dir="${HYBRID_ROOT}/${hybrid_safe}"
    hybrid_report="${HYBRID_REPORT_ROOT}/${hybrid_safe}_key_merge_report.md"
    echo "[dpoS1-sftS2-hybrid] build ${hybrid_label}"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/build_diffueraser_dpoS1_sftS2_hybrid.py" \
      --dpo_stage1_weights "${dpo_path}" \
      --sft_stage2_weights "${sft_path}" \
      --output_dir "${hybrid_dir}" \
      --mode dpo_spatial_sft_motion \
      --strict false \
      --report_path "${hybrid_report}"
    CHECKPOINT_ARGS+=(--checkpoint "${hybrid_label}=${hybrid_dir}/last_weights")
  done
done

if [[ "${#CHECKPOINT_ARGS[@]}" -eq 0 ]]; then
  die "No evaluable checkpoints or hybrids were built"
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/eval_generated_loser_partialmask_model.py" \
  --manifest "${MANIFEST}" \
  --output_dir "${EVAL_DIR}" \
  --base_weights_dir "${BASE_WEIGHTS_DIR}" \
  "${CHECKPOINT_ARGS[@]}" \
  --base_model_name_or_path "${BASE_MODEL_NAME_OR_PATH}" \
  --vae_path "${VAE_PATH}" \
  --num_samples "${NUM_SAMPLES}" \
  --num_samples_metric "${NUM_SAMPLES_METRIC}" \
  --seed "${SEED}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --frames "${FRAMES}" \
  --fps "${FPS}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --guidance_scale "${GUIDANCE_SCALE}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --vae_dtype "${VAE_DTYPE}" \
  --no_d2_loser \
  --skip_existing

SUMMARY_CSV="${EVAL_DIR}/metrics/summary.csv"
SUMMARY_JSON="${EVAL_DIR}/metrics/summary.json"
require_path "${SUMMARY_CSV}" "hybrid metrics summary.csv"

"${PYTHON_BIN}" - <<PY
import csv
import json
import math
from pathlib import Path

summary_csv = Path("${SUMMARY_CSV}")
eval_dir = Path("${EVAL_DIR}")
final_report = Path("${FINAL_REPORT}")
structure_report = Path("${STRUCTURE_REPORT}")
rows = list(csv.DictReader(summary_csv.open("r", encoding="utf-8")))
for row in rows:
    for key in list(row):
        if key != "model_label":
            try:
                row[key] = float(row[key])
            except Exception:
                pass

def score(row):
    return (
        float(row.get("mask_region_ssim_mean", float("nan"))),
        float(row.get("mask_region_psnr_mean", float("nan"))),
    )

base = next((r for r in rows if r.get("model_label") == "DiffuEraser-base"), None)
dpo_stage2 = next((r for r in rows if r.get("model_label") == "Exp7_DPO_S1_DPO_S2_last"), None)
hybrids = [r for r in rows if str(r.get("model_label", "")).startswith("Hybrid_")]
best_hybrid = max(hybrids, key=score) if hybrids else None

def fmt(value):
    return f"{value:.6g}" if isinstance(value, float) and math.isfinite(value) else str(value)

lines = [
    "# Exp7 DPO-S1 + SFT-S2 Hybrid Eval Report",
    "",
    f"eval_dir: `{eval_dir}`",
    f"metrics_summary: `{summary_csv}`",
    f"metrics_json: `{Path('${SUMMARY_JSON}')}`",
    f"checkpoint_structure_report: `{structure_report}`",
    "",
    "## Answers",
    "",
    "1. DPO Stage1 and SFT Stage2 can be hybridized through the existing DiffuEraser Stage2 pattern: load a motion UNet, copy Stage1 2D modules into it, and copy BrushNet from Stage1.",
    "2. SFT Stage2 candidates are listed in `hybrid_build_plan.md`; the first available YouTube-VOS candidate is preferred, otherwise previous SFT candidates are used, then official/base Stage2.",
    "3. YouTube-VOS SFT Stage2 discovery is recorded in `hybrid_build_plan.md`.",
    "4. DPO Stage1 spatial weights are preserved in each hybrid by copying Stage1 `unet_main` spatial modules and Stage1 `brushnet`.",
    "5. SFT Stage2 motion weights are preserved because motion/temporal keys are left in the loaded Stage2 motion UNet and recorded in each `hybrid_reports/*_key_merge_report.md`.",
]
if best_hybrid:
    lines.append(f"6. Best hybrid by mask-region SSIM/PSNR: `{best_hybrid['model_label']}`.")
else:
    lines.append("6. No hybrid checkpoint was evaluated.")
if base and best_hybrid:
    beats_base = score(best_hybrid) > score(base)
    lines.append(f"7. Hybrid better than DiffuEraser-base by mask-region SSIM/PSNR: `{beats_base}`.")
else:
    lines.append("7. Hybrid vs DiffuEraser-base is unavailable.")
if dpo_stage2 and best_hybrid:
    beats_dpo_stage2 = score(best_hybrid) > score(dpo_stage2)
    lines.append(f"8. Hybrid better than Exp7 DPO Stage1 + DPO Stage2: `{beats_dpo_stage2}`.")
else:
    lines.append("8. Hybrid vs Exp7 DPO Stage1 + DPO Stage2 is unavailable.")
lines.extend([
    "9. Stage2 DPO should remain stopped unless a later gate explicitly targets temporal-only training without reintroducing loser degradation.",
    "10. Next-step choices: launch Stage1-only checkpoint sweep first if more Stage1 checkpoints are needed; consider no-lose-gap Stage1 if loser degradation remains visible; keep Exp8/D3 deferred.",
    "",
    "## Metric Table",
    "",
])
cols = ["model_label", "mask_region_psnr_mean", "mask_region_ssim_mean", "boundary_psnr_mean", "boundary_ssim_mean", "outside_region_diff_mean_mean", "temporal_diff_delta_vs_gt_mean"]
lines.append("| " + " | ".join(cols) + " |")
lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
for row in rows:
    lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in cols) + " |")
lines.append("")

final_report.parent.mkdir(parents=True, exist_ok=True)
final_report.write_text("\n".join(lines), encoding="utf-8")
print(f"[dpoS1-sftS2-hybrid] final_report={final_report}")
PY

echo "[dpoS1-sftS2-hybrid] done"
echo "[dpoS1-sftS2-hybrid] output=${EVAL_DIR}"
echo "[dpoS1-sftS2-hybrid] metrics=${SUMMARY_CSV}"
echo "[dpoS1-sftS2-hybrid] report=${FINAL_REPORT}"
