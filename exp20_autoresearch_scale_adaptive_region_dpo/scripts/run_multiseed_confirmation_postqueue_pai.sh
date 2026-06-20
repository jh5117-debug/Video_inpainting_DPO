#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp20_autoresearch}"
cd "${PROJECT_ROOT}"

BASE="exp20_autoresearch_scale_adaptive_region_dpo/multiseed_equal_step_confirmation"
PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
GPU_ID="${GPU_ID:-0}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
LOG_ROOT="${LOG_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp20/multiseed_confirmation_postqueue}"
mkdir -p "${LOG_ROOT}" reports

echo "[postqueue] waiting for MSEQ queue to finish"
while true; do
  queue=$(find "${BASE}/queue" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)
  running=$(find "${BASE}/running" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)
  done_count=$(find "${BASE}/done" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)
  crash=$(find "${BASE}/crash" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)
  echo "[postqueue] $(date +%F_%T) queue=${queue} running=${running} done=${done_count} crash=${crash}"
  if [[ "${crash}" -gt 0 ]]; then
    echo "[postqueue] crash files present; refusing to continue" >&2
    exit 11
  fi
  if [[ "${queue}" -eq 0 && "${running}" -eq 0 ]]; then
    break
  fi
  sleep "${SLEEP_SECONDS}"
done

echo "[postqueue] running shadow-dev baselines"
GPU_LIST="${GPU_ID}" MAX_PARALLEL=1 bash exp20_autoresearch_scale_adaptive_region_dpo/scripts/run_shadow_dev_baselines_pai.sh \
  > "${LOG_ROOT}/shadow_baselines.log" 2>&1

echo "[postqueue] running shadow-dev candidate evals"
GPU_ID="${GPU_ID}" bash exp20_autoresearch_scale_adaptive_region_dpo/scripts/evaluate_multiseed_shadow_pai.sh \
  > "${LOG_ROOT}/shadow_candidates.log" 2>&1

echo "[postqueue] backfilling full candidate metrics"
CUDA_VISIBLE_DEVICES="${GPU_ID}" DEVICE="cuda" bash exp20_autoresearch_scale_adaptive_region_dpo/scripts/backfill_multiseed_candidate_metrics_pai.sh \
  > "${LOG_ROOT}/backfill_candidate_metrics.log" 2>&1

echo "[postqueue] analyzing BF07/P4 confirmation"
"${PY}" exp20_autoresearch_scale_adaptive_region_dpo/code/analyze_bf07_p4_confirmation.py \
  > "${LOG_ROOT}/analyze_confirmation.log" 2>&1

echo "[postqueue] building visual pack"
"${PY}" exp20_autoresearch_scale_adaptive_region_dpo/code/make_bf07_p4_visual_pack.py \
  > "${LOG_ROOT}/make_visual_pack.log" 2>&1

echo "[postqueue] done"
