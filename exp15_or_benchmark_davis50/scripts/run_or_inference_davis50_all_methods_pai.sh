#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
cd "${PROJECT_ROOT}"

PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
OUT_ROOT="${OUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp15_or_benchmark_davis50}"
MANIFEST="${MANIFEST:-exp15_or_benchmark_davis50/manifests/davis50_or_manifest.csv}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
METHODS="${METHODS:-propainter,videocomposer,cococo,floed,diffueraser_sft48000,videopainter,vace,ours_exp11_outer_b075_s2,minimax_remover}"

mkdir -p "${OUT_ROOT}" logs/pipelines reports

bash exp15_or_benchmark_davis50/scripts/prepare_davis50_or_dataset_pai.sh

echo "[exp15-or] starting DAVIS50 OR inference"
echo "[exp15-or] project=${PROJECT_ROOT}"
echo "[exp15-or] output=${OUT_ROOT}"
echo "[exp15-or] methods=${METHODS}"
echo "[exp15-or] gpu=${GPU}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" exp15_or_benchmark_davis50/code/run_or_davis50_inference.py \
  --manifest "${MANIFEST}" \
  --output_root "${OUT_ROOT}" \
  --project_root "${PROJECT_ROOT}" \
  --python "${PY}" \
  --methods "${METHODS}" \
  --gpu "${GPU}" \
  --max_frames "${MAX_FRAMES:-0}" \
  --resume

echo "[exp15-or] inference completed; running metrics"
bash exp15_or_benchmark_davis50/scripts/eval_or_davis50_metrics_pai.sh

echo "[exp15-or] making visual grids"
bash exp15_or_benchmark_davis50/scripts/make_or_davis50_visual_grids_pai.sh

echo "[exp15-or] done"
