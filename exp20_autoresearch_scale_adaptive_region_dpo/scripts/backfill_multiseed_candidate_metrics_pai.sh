#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp20_autoresearch}"
cd "${PROJECT_ROOT}"

PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
EXP_ROOT="${EXP_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}"
TRIAL_ROOT="${TRIAL_ROOT:-${EXP_ROOT}/experiments/dpo/exp20_autoresearch_scale_adaptive_region_dpo/trials}"
SEARCH_ROOT="${SEARCH_ROOT:-${EXP_ROOT}/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/dev_boundary_search_v1_e385cc27_roots}"
SHADOW_ROOT="${SHADOW_ROOT:-${EXP_ROOT}/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/dev_boundary_shadow_v1_roots}"
DEVICE="${DEVICE:-cuda}"
I3D_MODEL_PATH="${I3D_MODEL_PATH:-${PROJECT_ROOT}/weights/i3d_rgb_imagenet.pt}"
TC_MODEL_PATH="${TC_MODEL_PATH:-${PROJECT_ROOT}/weights/open_clip_vit_h14}"
RAFT_MODEL_PATH="${RAFT_MODEL_PATH:-/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter/raft-things.pth}"

require_file() {
  local path="$1"
  local label="$2"
  [[ -f "${path}" ]] || { echo "[missing] ${label}: ${path}" >&2; exit 2; }
}

require_dir() {
  local path="$1"
  local label="$2"
  [[ -d "${path}" ]] || { echo "[missing] ${label}: ${path}" >&2; exit 2; }
}

require_file "${I3D_MODEL_PATH}" "I3D model"
require_file "${TC_MODEL_PATH}/open_clip_pytorch_model.bin" "TC model"
require_file "${RAFT_MODEL_PATH}" "RAFT model"

search_label_args=(
  --label-dir "P0_s20260619=${TRIAL_ROOT}/EQ_P0_1d8cd54758b73251/eval_dev/EQ_P0"
  --label-dir "P4_s20260619=${TRIAL_ROOT}/EQ_P4_edbea07bb785e769/eval_dev/EQ_P4"
  --label-dir "BF07_s20260619=${TRIAL_ROOT}/EQ_BF07_2bc98e58514fb1da/eval_dev/EQ_BF07"
)
shadow_label_args=(
  --label-dir "P0_s20260619=${TRIAL_ROOT}/EQ_P0_1d8cd54758b73251/eval_shadow/EQ_P0_shadow"
  --label-dir "P4_s20260619=${TRIAL_ROOT}/EQ_P4_edbea07bb785e769/eval_shadow/EQ_P4_shadow"
  --label-dir "BF07_s20260619=${TRIAL_ROOT}/EQ_BF07_2bc98e58514fb1da/eval_shadow/EQ_BF07_shadow"
  --label-dir "AD04_s20260619=${TRIAL_ROOT}/EQ_AD04_77a0ed002ad3955d/eval_shadow/EQ_AD04_shadow"
)

for config in exp20_autoresearch_scale_adaptive_region_dpo/multiseed_equal_step_confirmation/configs/MSEQ_*.json; do
  trial_id=$("${PY}" - "${config}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["trial_id"])
PY
)
  hash_id=$("${PY}" - "${config}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["config_hash"])
PY
)
  method=$("${PY}" - "${config}" <<'PY'
import json, re, sys
trial_id = json.load(open(sys.argv[1]))["trial_id"]
m = re.match(r"MSEQ_([^_]+)_s(\d+)", trial_id)
print(f"{m.group(1)}_s{m.group(2)}" if m else trial_id)
PY
)
  search_label_args+=(--label-dir "${method}=${TRIAL_ROOT}/${trial_id}_${hash_id}/eval_dev/${trial_id}")
  shadow_label_args+=(--label-dir "${method}=${TRIAL_ROOT}/${trial_id}_${hash_id}/eval_shadow/${trial_id}_shadow")
done

for arg in "${search_label_args[@]}" "${shadow_label_args[@]}"; do
  if [[ "${arg}" == *=* ]]; then
    path="${arg#*=}"
    require_dir "${path}" "eval label dir"
  fi
done

"${PY}" exp20_autoresearch_scale_adaptive_region_dpo/code/backfill_existing_eval_metrics.py \
  --video-root "${SEARCH_ROOT}/JPEGImages_432_240" \
  --mask-root "${SEARCH_ROOT}/test_masks" \
  --gt-root "${SEARCH_ROOT}/JPEGImages_432_240" \
  "${search_label_args[@]}" \
  --output-csv reports/exp20_bf07_p4_multiseed_search_full_metrics.csv \
  --output-md reports/exp20_bf07_p4_multiseed_search_full_metrics.md \
  --video-length 24 \
  --device "${DEVICE}" \
  --compute-lpips \
  --compute-vfid \
  --compute-tc \
  --compute-ewarp \
  --i3d-model-path "${I3D_MODEL_PATH}" \
  --tc-model-path "${TC_MODEL_PATH}" \
  --raft-model-path "${RAFT_MODEL_PATH}"

"${PY}" exp20_autoresearch_scale_adaptive_region_dpo/code/backfill_existing_eval_metrics.py \
  --video-root "${SHADOW_ROOT}/JPEGImages_432_240" \
  --mask-root "${SHADOW_ROOT}/test_masks" \
  --gt-root "${SHADOW_ROOT}/JPEGImages_432_240" \
  "${shadow_label_args[@]}" \
  --output-csv reports/exp20_bf07_p4_multiseed_shadow_full_metrics.csv \
  --output-md reports/exp20_bf07_p4_multiseed_shadow_full_metrics.md \
  --video-length 24 \
  --device "${DEVICE}" \
  --compute-lpips \
  --compute-vfid \
  --compute-tc \
  --compute-ewarp \
  --i3d-model-path "${I3D_MODEL_PATH}" \
  --tc-model-path "${TC_MODEL_PATH}" \
  --raft-model-path "${RAFT_MODEL_PATH}"

echo "[done] full candidate metrics backfilled"
