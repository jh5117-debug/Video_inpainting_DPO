#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp20_autoresearch}"
cd "${PROJECT_ROOT}"

PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_dev_boundary_shadow_v1_baselines}"
EXP_ROOT="${EXP_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}"
OUT_ROOT="${OUT_ROOT:-${EXP_ROOT}/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-${EXP_ROOT}/logs/autoresearch/exp20/${RUN_TAG}}"
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" reports

DEV_ROOT="${DEV_ROOT:-${EXP_ROOT}/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/dev_boundary_shadow_v1_roots}"
VIDEO_ROOT="${VIDEO_ROOT:-${DEV_ROOT}/JPEGImages_432_240}"
MASK_ROOT="${MASK_ROOT:-${DEV_ROOT}/test_masks}"
GT_ROOT="${GT_ROOT:-${VIDEO_ROOT}}"
EXPECTED_VIDEOS="${EXPECTED_VIDEOS:-20}"

BASE_MODEL="${BASE_MODEL:-/mnt/nas/hj/weights/stable-diffusion-v1-5}"
VAE="${VAE:-/mnt/nas/hj/weights/sd-vae-ft-mse}"
PROP="${PROP:-/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter}"
PCM="${PCM:-/mnt/nas/hj/weights/PCM_Weights}"
SFT48000="${SFT48000:-/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000}"
EXP11_S1="${EXP11_S1:-/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s1_2000_davis_pai/last_weights}"
EXP11_S2="${EXP11_S2:-/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s2_2000_davis_pai/last_weights}"
EXP11_S1_HYBRID_DIR="${EXP11_S1_HYBRID_DIR:-${EXP_ROOT}/experiments/dpo/exp20_autoresearch_scale_adaptive_region_dpo/hybrids/exp11_s1_sft_s2}"
EXP11_S1_HYBRID="${EXP11_S1_HYBRID:-${EXP11_S1_HYBRID_DIR}/last_weights}"

VIDEO_LENGTH="${VIDEO_LENGTH:-24}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-6}"
USE_PCM="${USE_PCM:-false}"
MASK_DILATION_ITER="${MASK_DILATION_ITER:-0}"
INPUT_SIZE="${INPUT_SIZE:-432x240}"
GPU_LIST_CSV="${GPU_LIST:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"

COMPUTE_LPIPS="${COMPUTE_LPIPS:-1}"
COMPUTE_VFID="${COMPUTE_VFID:-1}"
COMPUTE_TC="${COMPUTE_TC:-1}"
COMPUTE_EWARP="${COMPUTE_EWARP:-1}"
SAVE_VIDEOS="${SAVE_VIDEOS:-1}"
SAVE_COMP_FRAMES="${SAVE_COMP_FRAMES:-1}"
I3D_MODEL_PATH="${I3D_MODEL_PATH:-${PROJECT_ROOT}/weights/i3d_rgb_imagenet.pt}"
TC_MODEL_PATH="${TC_MODEL_PATH:-${PROJECT_ROOT}/weights/open_clip_vit_h14}"
RAFT_MODEL_PATH="${RAFT_MODEL_PATH:-${PROP}/raft-things.pth}"

require_path() {
  local path="$1"
  local label="$2"
  [[ -e "${path}" ]] || { echo "[missing] ${label}: ${path}" >&2; exit 2; }
}

for path in "${VIDEO_ROOT}" "${MASK_ROOT}" "${GT_ROOT}" "${SFT48000}" "${EXP11_S1}" "${EXP11_S2}" "${BASE_MODEL}" "${VAE}" "${PROP}"; do
  require_path "${path}" "${path}"
done
if [[ "${COMPUTE_VFID}" == "1" ]]; then
  require_path "${I3D_MODEL_PATH}" "I3D model"
fi
if [[ "${COMPUTE_TC}" == "1" ]]; then
  require_path "${TC_MODEL_PATH}/open_clip_pytorch_model.bin" "TC model"
fi
if [[ "${COMPUTE_EWARP}" == "1" ]]; then
  require_path "${RAFT_MODEL_PATH}" "RAFT model"
fi

video_count=$(find "${VIDEO_ROOT}" -mindepth 1 -maxdepth 1 \( -type d -o -type l \) | wc -l)
mask_count=$(find "${MASK_ROOT}" -mindepth 1 -maxdepth 1 \( -type d -o -type l \) | wc -l)
if [[ "${video_count}" -ne "${EXPECTED_VIDEOS}" || "${mask_count}" -ne "${EXPECTED_VIDEOS}" ]]; then
  echo "[bad-shadow-root] expected ${EXPECTED_VIDEOS} video dirs and masks, got video=${video_count} mask=${mask_count}" >&2
  exit 3
fi

if [[ ! -d "${EXP11_S1_HYBRID}" ]]; then
  mkdir -p "${EXP11_S1_HYBRID_DIR}"
  "${PY}" tools/build_diffueraser_dpoS1_sftS2_hybrid.py \
    --dpo_stage1_weights "${EXP11_S1}" \
    --sft_stage2_weights "${SFT48000}" \
    --output_dir "${EXP11_S1_HYBRID_DIR}" \
    --report_path "reports/exp20_exp11_s1_sft_s2_hybrid_audit.md" \
    --strict false \
    > "${LOG_ROOT}/build_exp11_s1_sft_s2_hybrid.log" 2>&1
fi
require_path "${EXP11_S1_HYBRID}" "Exp11 S1 + SFT S2 hybrid"

COMMON=(
  --video_root "${VIDEO_ROOT}"
  --mask_root "${MASK_ROOT}"
  --gt_root "${GT_ROOT}"
  --base_model_path "${BASE_MODEL}"
  --vae_path "${VAE}"
  --propainter_model_dir "${PROP}"
  --pcm_weights_path "${PCM}"
  --input_size "${INPUT_SIZE}"
  --video_length "${VIDEO_LENGTH}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --use_pcm "${USE_PCM}"
  --mask_dilation_iter "${MASK_DILATION_ITER}"
  --raft_model_path "${RAFT_MODEL_PATH}"
)
[[ "${COMPUTE_LPIPS}" == "1" ]] && COMMON+=(--compute_lpips)
[[ "${COMPUTE_VFID}" == "1" ]] && COMMON+=(--compute_vfid --i3d_model_path "${I3D_MODEL_PATH}")
[[ "${COMPUTE_TC}" == "1" ]] && COMMON+=(--compute_tc --tc_model_path "${TC_MODEL_PATH}")
[[ "${COMPUTE_EWARP}" == "1" ]] && COMMON+=(--compute_ewarp)

IFS=',' read -r -a GPUS <<< "${GPU_LIST_CSV}"

run_one() {
  local gpu="$1"
  local label="$2"
  local weights="$3"
  local safe_label
  safe_label=$(echo "${label}" | tr -c 'A-Za-z0-9_.-' '_')
  local out="${OUT_ROOT}/${safe_label}"
  local log="${LOG_ROOT}/${safe_label}.log"
  local save_args=()
  [[ "${SAVE_VIDEOS}" == "1" ]] && save_args+=(--save_videos)
  [[ "${SAVE_COMP_FRAMES}" == "1" ]] && save_args+=(--save_comp_frames)
  echo "[launch] gpu=${gpu} label=${label} out=${out} log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" exp20_autoresearch_scale_adaptive_region_dpo/code/run_exp20_framewise_protocol_eval.py \
    "${COMMON[@]}" \
    --label "${label}" \
    --diffueraser_path "${weights}" \
    --save_path "${out}" \
    --inference_seed 20260619 \
    "${save_args[@]}" > "${log}" 2>&1
  echo "[done] gpu=${gpu} label=${label}"
}

labels=(SFT48000_baseline Exp11_outer_b075_S1_plus_SFT_S2 Exp11_outer_b075_S2)
weights=("${SFT48000}" "${EXP11_S1_HYBRID}" "${EXP11_S2}")

active=0
for i in "${!labels[@]}"; do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  run_one "${gpu}" "${labels[$i]}" "${weights[$i]}" &
  active=$((active + 1))
  if (( active >= MAX_PARALLEL )); then
    wait -n
    active=$((active - 1))
  fi
done
wait

"${PY}" - "${OUT_ROOT}" "${PROJECT_ROOT}/reports" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
reports = Path(sys.argv[2])
rows = []
for path in sorted(root.glob("*/metrics/summary.csv")):
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    row["result_dir"] = str(path.parents[1])
    rows.append(row)
if not rows:
    raise SystemExit("no shadow baseline summary rows found")

out = reports / "exp20_shadow_dev_baselines.csv"
keys = []
for row in rows:
    for key in row:
        if key not in keys:
            keys.append(key)
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)

def f(row, key):
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")

by_label = {row["model_label"]: row for row in rows}
sft = f(by_label.get("SFT48000_baseline", {}), "whole_video_psnr_mean")
s1 = f(by_label.get("Exp11_outer_b075_S1_plus_SFT_S2", {}), "whole_video_psnr_mean")
s2 = f(by_label.get("Exp11_outer_b075_S2", {}), "whole_video_psnr_mean")
(reports / "exp20_shadow_dev_baseline_targets.json").write_text(
    json.dumps(
        {
            "SHADOW_SFT_PSNR": sft,
            "SHADOW_EXP11_S1_PSNR": s1,
            "SHADOW_EXP11_S2_PSNR": s2,
            "eval_root": str(root),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

metric_cols = [
    ("whole_video_psnr_mean", "PSNR"),
    ("whole_video_ssim_mean", "SSIM"),
    ("whole_video_lpips_mean", "LPIPS"),
    ("vfid", "VFID/FVD"),
    ("tc_mean", "TC"),
    ("ewarp_mean", "Ewarp"),
    ("strict_mask_pixel_psnr_mean", "mask PSNR"),
    ("boundary_pixel_psnr_mean", "boundary PSNR"),
]
present_cols = [(col, name) for col, name in metric_cols if any(col in row for row in rows)]
lines = [
    "# Exp20 Shadow-Dev Baselines",
    "",
    f"- eval_root: `{root}`",
    "- protocol: raw6, hard comp, D+G off, no PCM, no mask dilation, no Gaussian blur, frame-wise metrics",
    f"- SHADOW_SFT_PSNR: `{sft:.6f}`",
    f"- SHADOW_EXP11_S1_PSNR: `{s1:.6f}`",
    f"- SHADOW_EXP11_S2_PSNR: `{s2:.6f}`",
    "",
    "| Method | " + " | ".join(name for _, name in present_cols) + " | rows |",
    "|---|" + "|".join(["---:"] * (len(present_cols) + 1)) + "|",
]
for row in rows:
    values = []
    for col, _ in present_cols:
        value = row.get(col, "")
        try:
            values.append(f"{float(value):.6f}")
        except Exception:
            values.append(str(value))
    lines.append(f"| {row.get('model_label', '')} | " + " | ".join(values) + f" | {row.get('num_videos', row.get('rows', ''))} |")
(reports / "exp20_shadow_dev_baselines.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "[done] shadow-dev baselines: reports/exp20_shadow_dev_baselines.csv"
