#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp20_autoresearch}"
cd "${PROJECT_ROOT}"

PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
GPU_ID="${GPU_ID:-0}"
EXP_ROOT="${EXP_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}"
TRIAL_ROOT="${TRIAL_ROOT:-${EXP_ROOT}/experiments/dpo/exp20_autoresearch_scale_adaptive_region_dpo/trials}"
SHADOW_ROOT="${SHADOW_ROOT:-${EXP_ROOT}/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/dev_boundary_shadow_v1_roots}"
RESULTS_TSV="${RESULTS_TSV:-exp20_autoresearch_scale_adaptive_region_dpo/multiseed_equal_step_confirmation/shadow_results_raw.tsv}"

require_dir() {
  local path="$1"
  local label="$2"
  [[ -d "${path}" ]] || { echo "[missing] ${label}: ${path}" >&2; exit 2; }
}

require_dir "${SHADOW_ROOT}/JPEGImages_432_240" "shadow video root"
require_dir "${SHADOW_ROOT}/test_masks" "shadow mask root"

run_eval() {
  local label="$1"
  local config="$2"
  local trial_dir="$3"
  require_dir "${trial_dir}/last_weights" "${label} last_weights"
  if [[ -f "${trial_dir}/eval_shadow/${label}_shadow/metrics/summary.csv" ]]; then
    echo "[skip-existing-shadow] ${label}"
    return 0
  fi
  echo "[shadow-eval] gpu=${GPU_ID} label=${label}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PY}" exp20_autoresearch_scale_adaptive_region_dpo/code/evaluate_trial.py \
    --trial-config "${config}" \
    --trial-dir "${trial_dir}" \
    --results-tsv "${RESULTS_TSV}" \
    --candidate-weights "${trial_dir}/last_weights" \
    --dev-root "${SHADOW_ROOT}" \
    --gpu-id "${GPU_ID}" \
    --video-length 24 \
    --eval-subdir eval_shadow \
    --label-suffix _shadow \
    --compute-lpips \
    --compute-ewarp
}

run_eval "EQ_P0" \
  "exp20_autoresearch_scale_adaptive_region_dpo/equal_step_confirmation/configs/EQ_P0_1d8cd54758b73251.json" \
  "${TRIAL_ROOT}/EQ_P0_1d8cd54758b73251"
run_eval "EQ_P4" \
  "exp20_autoresearch_scale_adaptive_region_dpo/equal_step_confirmation/configs/EQ_P4_edbea07bb785e769.json" \
  "${TRIAL_ROOT}/EQ_P4_edbea07bb785e769"
run_eval "EQ_BF07" \
  "exp20_autoresearch_scale_adaptive_region_dpo/equal_step_confirmation/configs/EQ_BF07_2bc98e58514fb1da.json" \
  "${TRIAL_ROOT}/EQ_BF07_2bc98e58514fb1da"
run_eval "EQ_AD04" \
  "exp20_autoresearch_scale_adaptive_region_dpo/equal_step_confirmation/configs/EQ_AD04_77a0ed002ad3955d.json" \
  "${TRIAL_ROOT}/EQ_AD04_77a0ed002ad3955d"

for config in exp20_autoresearch_scale_adaptive_region_dpo/multiseed_equal_step_confirmation/configs/MSEQ_*.json; do
  trial_id=$("${PY}" - "${config}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["trial_id"])
PY
)
  hash_id=$("${PY}" - "${config}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["config_hash"])
PY
)
  run_eval "${trial_id}" "${config}" "${TRIAL_ROOT}/${trial_id}_${hash_id}"
done

echo "[done] shadow evals: ${RESULTS_TSV}"
