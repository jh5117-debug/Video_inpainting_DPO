#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
cd "${PROJECT_ROOT}"

PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
OUT_ROOT="${OUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp15_or_benchmark_davis50}"
mkdir -p "${OUT_ROOT}" reports

"${PY}" exp15_or_benchmark_davis50/code/run_or_davis50_inference.py \
  --manifest exp15_or_benchmark_davis50/manifests/davis50_or_manifest.csv \
  --output_root "${OUT_ROOT}" \
  --project_root "${PROJECT_ROOT}" \
  --python "${PY}" \
  --methods "propainter,videocomposer,cococo,floed,diffueraser_sft48000,videopainter,vace,ours_exp11_outer_b075_s2,minimax_remover" \
  --max_frames 0 \
  --resume \
  --gpu "${CUDA_VISIBLE_DEVICES:-0}" \
  --check_only

echo "[exp15-or] Method runtime status: reports/exp15_or_method_runtime_status.md"
