#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
cd "${PROJECT_ROOT}"

PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
OUT_ROOT="${OUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp15_or_benchmark_davis50}"
MANIFEST="${MANIFEST:-exp15_or_benchmark_davis50/manifests/davis50_or_manifest.csv}"
METHODS="${METHODS:-propainter,videocomposer,cococo,floed,diffueraser_sft48000,videopainter,vace,ours_exp11_outer_b075_s2}"
VIS_DIR="${OUT_ROOT}/visual_grids"

"${PY}" exp15_or_benchmark_davis50/code/make_or_visual_grid.py \
  --manifest "${MANIFEST}" \
  --output_root "${OUT_ROOT}" \
  --methods "${METHODS}" \
  --visual_dir "${VIS_DIR}" \
  --max_frames "${VIS_MAX_FRAMES:-24}"

cat > reports/exp15_or_davis50_visual_case_report.md <<EOF
# Exp15 DAVIS50 OR Visual Case Report

- Visual grid root: \`${VIS_DIR}\`
- Manifest: \`${VIS_DIR}/visual_manifest.csv\`
- Rows: two-row grid with input/mask plus available methods; blocked methods are explicit placeholders.
- This report is generated after DAVIS50 inference and does not use composited frames.
EOF

echo "[exp15-or] visual grids: ${VIS_DIR}"
