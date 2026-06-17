#!/usr/bin/env bash
set -euo pipefail

# Exp16 Stage1-500 visual sanity eval. This script runs only a DAVIS10 eval;
# it does not start or resume training.

PROJECT_ROOT="${EXP16_PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
cd "${PROJECT_ROOT}"

OUT_ROOT="${OUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp16_stage1_500_visual_sanity_davis10}"
LOG_ROOT="${LOG_ROOT:-logs/pipelines/exp16_stage1_500_visual_sanity_davis10}"
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/workspace/hj/nas_hj/data/external/davis_432_240}"
VIDEO_ROOT="${VIDEO_ROOT:-${DATA_ROOT}/JPEGImages_432_240}"
MASK_ROOT="${MASK_ROOT:-${DATA_ROOT}/test_masks}"
GT_ROOT="${GT_ROOT:-${VIDEO_ROOT}}"
SUBSET_ROOT="${SUBSET_ROOT:-${OUT_ROOT}/davis10_subset}"
SUBSET_VIDEO_ROOT="${SUBSET_ROOT}/JPEGImages_432_240"
SUBSET_MASK_ROOT="${SUBSET_ROOT}/test_masks"

VIDEOS="${VIDEOS:-boat,rhino,dog-agility,blackswan,lucia,bear,dance-jump,soccerball,kite-surf,breakdance}"
VIDEO_LENGTH="${VIDEO_LENGTH:-24}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-6}"
USE_PCM="${USE_PCM:-false}"
MASK_DILATION_ITER="${MASK_DILATION_ITER:-0}"
INPUT_SIZE="${INPUT_SIZE:-432x240}"
GPU_LIST_CSV="${GPU_LIST:-0,1,4}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"

BASE_MODEL="${BASE_MODEL:-/mnt/nas/hj/weights/stable-diffusion-v1-5}"
VAE="${VAE:-/mnt/nas/hj/weights/sd-vae-ft-mse}"
PROP="${PROP:-/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter}"
PCM="${PCM:-/mnt/nas/hj/weights/PCM_Weights}"

SFT48000="${SFT48000:-/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000}"
EXP11_OUTER_B075_S2="${EXP11_OUTER_B075_S2:-/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s2_2000_davis_pai/last_weights}"
# Stage1-only weights do not contain the S2 motion config expected by the
# DiffuEraser DAVIS evaluator. Use the eval-only DPO-S1 + SFT-S2 hybrid.
EXP16_STAGE1_500="${EXP16_STAGE1_500:-/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/hybrid/20260617_exp16_stage1_500_limit100_dpoS1_sftS2/last_weights}"

SAVE_VIDEOS="${SAVE_VIDEOS:-1}"
SAVE_COMP_FRAMES="${SAVE_COMP_FRAMES:-1}"
COMPUTE_LPIPS="${COMPUTE_LPIPS:-0}"
COMPUTE_VFID="${COMPUTE_VFID:-0}"
COMPUTE_TC="${COMPUTE_TC:-0}"
COMPUTE_EWARP="${COMPUTE_EWARP:-0}"
I3D_MODEL_PATH="${I3D_MODEL_PATH:-${PROJECT_ROOT}/weights/i3d_rgb_imagenet.pt}"
TC_MODEL_PATH="${TC_MODEL_PATH:-${PROJECT_ROOT}/weights/open_clip_vit_h14}"
RAFT_MODEL_PATH="${RAFT_MODEL_PATH:-${PROP}/raft-things.pth}"

require_path() {
  local path="$1"
  local label="$2"
  [[ -e "${path}" ]] || { echo "[missing] ${label}: ${path}" >&2; exit 2; }
}

for path in "${VIDEO_ROOT}" "${MASK_ROOT}" "${GT_ROOT}" "${SFT48000}" "${EXP11_OUTER_B075_S2}" "${EXP16_STAGE1_500}" "${BASE_MODEL}" "${VAE}" "${PROP}"; do
  require_path "${path}" "${path}"
done

rm -rf "${SUBSET_ROOT}"
mkdir -p "${SUBSET_VIDEO_ROOT}" "${SUBSET_MASK_ROOT}"
IFS=',' read -r -a VIDEO_ARRAY <<< "${VIDEOS}"
for video in "${VIDEO_ARRAY[@]}"; do
  video="$(echo "${video}" | xargs)"
  require_path "${VIDEO_ROOT}/${video}" "DAVIS video ${video}"
  require_path "${MASK_ROOT}/${video}" "DAVIS mask ${video}"
  ln -s "${VIDEO_ROOT}/${video}" "${SUBSET_VIDEO_ROOT}/${video}"
  ln -s "${MASK_ROOT}/${video}" "${SUBSET_MASK_ROOT}/${video}"
done
printf "%s\n" "${VIDEO_ARRAY[@]}" > "${OUT_ROOT}/selected_videos.txt"

COMMON=(
  --video_root "${SUBSET_VIDEO_ROOT}"
  --mask_root "${SUBSET_MASK_ROOT}"
  --gt_root "${SUBSET_VIDEO_ROOT}"
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

IFS=',' read -r -a GPUS <<< "${GPU_LIST_CSV}"

run_one() {
  local gpu="$1"
  local label="$2"
  local weights="$3"
  local out="${OUT_ROOT}/${label}"
  local log="${LOG_ROOT}/${label}.log"
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
  echo "[done] gpu=${gpu} label=${label}"
}

labels=(SFT48000_baseline Exp11_boundary_outer_b075_S2 Exp16_stage1_500_limit100)
weights=("${SFT48000}" "${EXP11_OUTER_B075_S2}" "${EXP16_STAGE1_500}")

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
    ("strict_mask_pixel_psnr_mean", "strict mask PSNR"),
    ("boundary_pixel_psnr_mean", "boundary PSNR"),
    ("outside_diff_mean_mean", "outside diff mean"),
    ("whole_video_lpips_mean", "LPIPS"),
    ("vfid", "VFID"),
    ("tc_mean", "TC"),
]
present_cols = [(col, name) for col, name in metric_cols if any(str(row.get(col, "")) != "" for row in rows)]
lines = [
    "# Exp16 Stage1-500 DAVIS10 Metric Sanity",
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
        else:
            try:
                values.append(f"{float(value):.4f}")
            except Exception:
                values.append(str(value))
    lines.append("| {} | {} | {} |".format(row["model_label"], " | ".join(values), row["rows"]))
(out / "summary_all.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

"${PY}" exp16_prior_confidence_gated_dpo/code/make_exp16_stage1_500_visual_sanity.py \
  --video_root "${SUBSET_VIDEO_ROOT}" \
  --mask_root "${SUBSET_MASK_ROOT}" \
  --sft_eval_root "${OUT_ROOT}/SFT48000_baseline" \
  --exp11_eval_root "${OUT_ROOT}/Exp11_boundary_outer_b075_S2" \
  --exp16_eval_root "${OUT_ROOT}/Exp16_stage1_500_limit100" \
  --output_root "${OUT_ROOT}" \
  --videos "${VIDEOS}" \
  --video_length "${VIDEO_LENGTH}"

cat > "${OUT_ROOT}/report.md" <<EOF
# Exp16 Stage1-500 DAVIS10 Visual Sanity Eval

- videos: ${VIDEOS}
- protocol: raw6, no PCM, mask dilation 0, no Gaussian blur, hard comp, frame-wise in-memory metrics
- metric backend: inference/metrics.py via tools/run_davis50_framewise_protocol_eval.py
- SFT weights: ${SFT48000}
- Exp11 weights: ${EXP11_OUTER_B075_S2}
- Exp16 weights: ${EXP16_STAGE1_500}
- Exp16 stage composition: Stage1-500 DPO + SFT-48000 Stage2 hybrid for eval loading
- summary: ${OUT_ROOT}/metrics/summary_all.csv
- visuals: ${OUT_ROOT}/side_by_side
EOF

echo "OUT_ROOT=${OUT_ROOT}"
echo "LOG_ROOT=${LOG_ROOT}"
