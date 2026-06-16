#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
cd "${PROJECT_ROOT}"

PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
DAVIS_ROOT="${DAVIS_ROOT:-/mnt/nas/hj/data/external/davis_2017_full_resolution_or_eval50/DAVIS}"

"${PY}" exp15_or_benchmark_davis50/code/build_davis50_or_manifest.py \
  --pai_davis_root "${DAVIS_ROOT}" \
  --out_csv exp15_or_benchmark_davis50/manifests/davis50_or_manifest.csv \
  --report_md reports/exp15_davis50_or_dataset_audit.md

echo "[exp15-or] DAVIS50 manifest ready: exp15_or_benchmark_davis50/manifests/davis50_or_manifest.csv"
