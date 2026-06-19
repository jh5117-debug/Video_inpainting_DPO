#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
PREFERENCE_MANIFEST="${PREFERENCE_MANIFEST:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/exp09_10_11_youtubevos_gtwin_d3comp_pai/manifests/selected_primary_comp.gtwin.pai_paths.jsonl}"

"${PYTHON_BIN}" exp20_autoresearch_scale_adaptive_region_dpo/code/run_legacy_full_parity.py \
  --manifest "${PREFERENCE_MANIFEST}" \
  --output-md reports/exp20_legacy_full_parity.md \
  --output-csv reports/exp20_legacy_full_parity.csv \
  --registry-json experiment_registry/exp20_autoresearch_scale_adaptive_region_dpo/parity.json
