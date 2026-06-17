#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
if [[ ! -d "$ROOT" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"

EXP_DIR="exp18_multiframe_propagation_gated_dpo"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

MANIFEST="${EXP18_SOURCE_MANIFEST:-${WORKSPACE_ROOT}/data/generated_losers/exp09_10_11_youtubevos_gtwin_d3comp_pai/manifests/selected_primary_comp.gtwin.pai_paths.jsonl}"
CACHE_ROOT="${EXP18_CACHE_ROOT:-${OUTPUT_ROOT}/data/cache/exp18_multiframe_propagation_cache_limit100}"
LIMIT="${EXP18_LIMIT:-100}"

mkdir -p "$EXP_DIR/reports" reports "$CACHE_ROOT"
[[ -f "$MANIFEST" ]] || { echo "[Exp18][BLOCKED] missing manifest: $MANIFEST" >&2; exit 2; }
if grep -q "/home/nvme01" "$MANIFEST"; then
  echo "[Exp18][BLOCKED] manifest still contains /home/nvme01 paths: $MANIFEST" >&2
  exit 2
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
"$PYTHON_BIN" "$EXP_DIR/code/precompute_multiframe_propagation_cache.py" \
  --input_manifest "$MANIFEST" \
  --output_root "$CACHE_ROOT" \
  --limit "$LIMIT" \
  --nframes "${NFRAMES:-16}" \
  --width "${WIDTH:-432}" \
  --height "${HEIGHT:-240}" \
  --source_window "${SOURCE_WINDOW:-3}" \
  --tau_conf "${TAU_CONF:-0.5}" \
  --write_oracle \
  --resume

cp "$CACHE_ROOT/reports/propagation_cache_report.md" reports/exp18_propagation_cache_quality_limit100.md
cp "$CACHE_ROOT/reports/propagation_cache_quality.csv" reports/exp18_propagation_cache_quality_limit100.csv
echo "[Exp18] cache ready: $CACHE_ROOT"
