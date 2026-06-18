#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
if [[ ! -d "${ROOT}" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${ROOT}"

PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
MANIFEST="${MANIFEST:-${ROOT}/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4/manifests/selected_primary_comp.repaired.pai_paths.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO/data/cache/exp19_propainter_completed_flow_limit100}"
PROP="${PROP:-/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter}"

mkdir -p "${OUTPUT_ROOT}"
"${PY}" exp19_boundary_gated_flow_adapter_dpo/code/export_propainter_completed_flow.py \
  --input_manifest "${MANIFEST}" \
  --output_root "${OUTPUT_ROOT}" \
  --propainter_model_dir "${PROP}" \
  --limit 100 \
  --nframes 16 \
  --width 432 \
  --height 240 \
  --raft_iter 20 \
  --fp16 \
  --resume \
  --save_visuals

mkdir -p reports
cp "${OUTPUT_ROOT}/reports/flow_cache_quality_limit100.md" reports/exp19_flow_cache_quality_limit100.md
cp "${OUTPUT_ROOT}/reports/flow_cache_quality_limit100.csv" reports/exp19_flow_cache_quality_limit100.csv
