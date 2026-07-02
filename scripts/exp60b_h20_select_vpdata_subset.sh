#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/nvme01/H20_Video_inpainting_DPO_exp60b_vp_vpdata_transfer}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/nvme01/H20_Video_inpainting_DPO/data/external/vpdata_exp60b_h20_staging/raw_subset}"
REPORTS_DIR="${REPORTS_DIR:-/home/nvme01/H20_Video_inpainting_DPO/reports/exp60b_vpdata_h20_download}"
MANIFEST_DIR="${MANIFEST_DIR:-${REPO_ROOT}/manifests}"

cd "${REPO_ROOT}"

python -m exp60b_videopainter_vpdata_d3mask.vpdata_subset \
  --output_root "${OUTPUT_ROOT}" \
  --manifest_dir "${MANIFEST_DIR}" \
  --reports_dir "${REPORTS_DIR}" \
  --seed 20260702 \
  --max_train 1000 \
  --max_test 100 \
  --source_filter pexels_only \
  "$@"

