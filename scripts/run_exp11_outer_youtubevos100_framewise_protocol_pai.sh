#!/usr/bin/env bash
set -euo pipefail

# Fixed YouTubeVOS-100 validation protocol for SFT-48000 vs the current best
# Exp11 boundary outer b0.75 S2 checkpoint. This reuses the DAVIS50 frame-wise
# raw6 hard-comp metric wrapper; only the dataset roots and labels differ.

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp09_10_11_pai_sync}"
cd "${PROJECT_ROOT}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)_youtubevos100_raw6_hardcomp}"
OUT_ROOT="${OUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp11_outer_b075_s2_youtubevos100_${STAMP}}"
LOG_ROOT="${LOG_ROOT:-logs/pipelines/exp11_outer_b075_s2_youtubevos100_${STAMP}}"
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/workspace/hj/nas_hj/data/external/youtubevos_432_240_eval100}"
VIDEO_ROOT="${VIDEO_ROOT:-${DATA_ROOT}/JPEGImages_432_240}"
MASK_ROOT="${MASK_ROOT:-${DATA_ROOT}/test_masks}"
GT_ROOT="${GT_ROOT:-${DATA_ROOT}/JPEGImages_432_240}"

BASE_MODEL="${BASE_MODEL:-/mnt/nas/hj/weights/stable-diffusion-v1-5}"
VAE="${VAE:-/mnt/nas/hj/weights/sd-vae-ft-mse}"
PROP="${PROP:-/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter}"
PCM="${PCM:-/mnt/nas/hj/weights/PCM_Weights}"

SFT48000="${SFT48000:-/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000}"
EXP11_OUTER_B075_S2="${EXP11_OUTER_B075_S2:-/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s2_2000_davis_pai/last_weights}"

VIDEO_LENGTH="${VIDEO_LENGTH:-24}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-6}"
USE_PCM="${USE_PCM:-false}"
MASK_DILATION_ITER="${MASK_DILATION_ITER:-0}"
INPUT_SIZE="${INPUT_SIZE:-432x240}"
GPU_LIST_CSV="${GPU_LIST:-0,1}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"

COMPUTE_LPIPS="${COMPUTE_LPIPS:-1}"
COMPUTE_VFID="${COMPUTE_VFID:-1}"
COMPUTE_TC="${COMPUTE_TC:-1}"
COMPUTE_EWARP="${COMPUTE_EWARP:-0}"
SAVE_VIDEOS="${SAVE_VIDEOS:-1}"
SAVE_COMP_FRAMES="${SAVE_COMP_FRAMES:-1}"
I3D_MODEL_PATH="${I3D_MODEL_PATH:-${PROJECT_ROOT}/weights/i3d_rgb_imagenet.pt}"
TC_MODEL_PATH="${TC_MODEL_PATH:-${PROJECT_ROOT}/weights/open_clip_vit_h14}"
RAFT_MODEL_PATH="${RAFT_MODEL_PATH:-${PROP}/raft-things.pth}"

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
[[ "${COMPUTE_TC}" == "1" ]] && COMMON+=(--compute_tc)
[[ -n "${TC_MODEL_PATH}" ]] && COMMON+=(--tc_model_path "${TC_MODEL_PATH}")
[[ "${COMPUTE_EWARP}" == "1" ]] && COMMON+=(--compute_ewarp)

require_path() {
  local path="$1"
  local label="$2"
  [[ -e "${path}" ]] || { echo "[missing] ${label}: ${path}" >&2; exit 2; }
}

for path in "${VIDEO_ROOT}" "${MASK_ROOT}" "${GT_ROOT}" "${SFT48000}" "${EXP11_OUTER_B075_S2}" "${BASE_MODEL}" "${VAE}" "${PROP}"; do
  require_path "${path}" "${path}"
done
if [[ "${COMPUTE_VFID}" == "1" ]]; then
  require_path "${I3D_MODEL_PATH}" "I3D model"
fi
if [[ "${COMPUTE_TC}" == "1" ]]; then
  require_path "${TC_MODEL_PATH}/open_clip_pytorch_model.bin" "TC model"
fi

video_count=$(find "${VIDEO_ROOT}" -mindepth 1 -maxdepth 1 -type d | wc -l)
mask_count=$(find "${MASK_ROOT}" -mindepth 1 -maxdepth 1 -type d | wc -l)
if [[ "${video_count}" -ne 100 || "${mask_count}" -ne 100 ]]; then
  echo "[bad-data] expected 100 video dirs and 100 mask dirs, got video=${video_count} mask=${mask_count}" >&2
  exit 3
fi

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
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" tools/run_davis50_framewise_protocol_eval.py \
    "${COMMON[@]}" \
    --label "${label}" \
    --diffueraser_path "${weights}" \
    --save_path "${out}" \
    "${save_args[@]}" > "${log}" 2>&1
  echo "[done] gpu=${gpu} label=${label} rc=$?"
}

labels=(SFT48000_baseline Exp11_boundary_outer_b075_S2)
weights=("${SFT48000}" "${EXP11_OUTER_B075_S2}")

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

"${PY}" - "${OUT_ROOT}" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/metrics/summary.csv")):
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    row["result_dir"] = str(path.parents[1])
    rows.append(row)
if not rows:
    raise SystemExit("no summary rows found")

out = root / "metrics"
out.mkdir(parents=True, exist_ok=True)
keys = []
for row in rows:
    for key in row:
        if key not in keys:
            keys.append(key)
with (out / "summary_all.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)

metric_cols = [
    ("whole_video_psnr_mean", "PSNR"),
    ("whole_video_ssim_mean", "SSIM"),
    ("whole_video_lpips_mean", "LPIPS"),
    ("vfid", "VFID"),
    ("tc_mean", "TC"),
    ("strict_mask_pixel_psnr_mean", "strict mask PSNR"),
    ("boundary_pixel_psnr_mean", "boundary PSNR"),
    ("outside_diff_mean_mean", "outside diff mean"),
]
present_cols = [(col, name) for col, name in metric_cols if any(col in row for row in rows)]
lines = [
    "# Exp11 outer b0.75 S2 YouTubeVOS-100 Frame-wise raw6 Summary",
    "",
    "| Method | " + " | ".join(name for _, name in present_cols) + " | rows |",
    "|---|" + "|".join(["---:"] * (len(present_cols) + 1)) + "|",
]
for row in rows:
    values = []
    for col, _ in present_cols:
        value = row.get(col, "")
        if value == "":
            values.append("")
            continue
        try:
            values.append(f"{float(value):.4f}")
        except Exception:
            values.append(str(value))
    lines.append("| {} | {} | {} |".format(row["model_label"], " | ".join(values), row["rows"]))
(out / "summary_all.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

cat > "${OUT_ROOT}/report.md" <<EOF
# Exp11 outer b0.75 S2 YouTubeVOS-100 Frame-wise raw6 Eval

- dataset: YouTubeVOS-100 sampled with fixed seed, see \`${DATA_ROOT}/sample_manifest.csv\`
- protocol: raw6, no PCM, mask dilation 0, no Gaussian blur, hard comp, frame-wise in-memory metrics
- metric backend: inference/metrics.py via tools/run_davis50_framewise_protocol_eval.py
- summary: ${OUT_ROOT}/metrics/summary_all.csv
EOF

echo "OUT_ROOT=${OUT_ROOT}"
echo "LOG_ROOT=${LOG_ROOT}"
