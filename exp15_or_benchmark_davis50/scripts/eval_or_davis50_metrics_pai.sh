#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
cd "${PROJECT_ROOT}"

PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
OUT_ROOT="${OUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp15_or_benchmark_davis50}"
MANIFEST="${MANIFEST:-exp15_or_benchmark_davis50/manifests/davis50_or_manifest.csv}"
METHODS="${METHODS:-propainter,videocomposer,cococo,floed,diffueraser_sft48000,videopainter,vace,ours_exp11_outer_b075_s2,minimax_remover}"
METRICS_DIR="${OUT_ROOT}/metrics"

"${PY}" exp15_or_benchmark_davis50/code/or_bg_metric_eval.py \
  --manifest "${MANIFEST}" \
  --output_root "${OUT_ROOT}" \
  --methods "${METHODS}" \
  --metrics_dir "${METRICS_DIR}" \
  --max_frames "${MAX_FRAMES:-0}" \
  --workers "${WORKERS:-8}"

cp "${METRICS_DIR}/summary.csv" reports/exp15_or_davis50_quantitative_summary.csv
cp "${METRICS_DIR}/summary.md" reports/exp15_or_davis50_quantitative_summary.md
echo "[exp15-or] metrics: ${METRICS_DIR}/summary.csv"
