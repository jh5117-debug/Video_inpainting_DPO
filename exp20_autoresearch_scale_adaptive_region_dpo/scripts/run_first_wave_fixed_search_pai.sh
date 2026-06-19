#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp20_autoresearch}"
cd "${PROJECT_ROOT}"

PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
EXP_DIR="exp20_autoresearch_scale_adaptive_region_dpo"
DEV_ROOT="${DEV_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/dev_boundary_search_v1_e385cc27_roots}"
LOG_ROOT="${LOG_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp20/first_wave_fixed_$(date +%Y%m%d_%H%M%S)}"
MAX_TRIALS="${MAX_TRIALS:-6}"
GPUS_PER_TRIAL="${GPUS_PER_TRIAL:-1}"
MAX_MEMORY_MIB="${MAX_MEMORY_MIB:-1024}"

mkdir -p "${LOG_ROOT}" reports

test -f "${EXP_DIR}/manifests/dev_boundary_search_v1.jsonl"
test -d "${DEV_ROOT}/JPEGImages_432_240"
test -d "${DEV_ROOT}/test_masks"
test -f reports/exp20_dev_baselines.csv

"${PY}" "${EXP_DIR}/code/prepare_first_wave_configs.py" \
  --search-space "${EXP_DIR}/search_space.yaml" \
  --output-root "${EXP_DIR}/first_wave" \
  --dev-manifest "${EXP_DIR}/manifests/dev_boundary_search_v1.jsonl" \
  > "${LOG_ROOT}/prepare_first_wave_configs.log" 2>&1

"${PY}" "${EXP_DIR}/code/search_worker.py" \
  --exp-dir "${EXP_DIR}" \
  --queue-dir "${EXP_DIR}/first_wave/queue" \
  --max-trials "${MAX_TRIALS}" \
  --gpus-per-trial "${GPUS_PER_TRIAL}" \
  --max-memory-mib "${MAX_MEMORY_MIB}" \
  --dev-root "${DEV_ROOT}" \
  > "${LOG_ROOT}/search_worker.log" 2>&1

echo "LOG_ROOT=${LOG_ROOT}"
echo "RESULTS=${EXP_DIR}/results.tsv"
