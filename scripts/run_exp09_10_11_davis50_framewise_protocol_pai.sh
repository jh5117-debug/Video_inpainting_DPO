#!/usr/bin/env bash
set -euo pipefail

# Fixed DAVIS-50 validation protocol for SFT-48000 and Exp9/10/11.
# The protocol intentionally computes metrics on in-memory hard-composited
# frames, not on mp4 outputs.

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp09_10_11_pai_sync}"
cd "$PROJECT_ROOT"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)_framewise_raw6_davis50}"
OUT_ROOT="${OUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp09_10_11_${STAMP}}"
LOG_ROOT="${LOG_ROOT:-logs/pipelines/exp09_10_11_${STAMP}}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
VIDEO_ROOT="${VIDEO_ROOT:-/mnt/workspace/hj/nas_hj/data/external/davis_432_240/JPEGImages_432_240}"
MASK_ROOT="${MASK_ROOT:-/mnt/workspace/hj/nas_hj/data/external/davis_432_240/test_masks}"
GT_ROOT="${GT_ROOT:-/mnt/workspace/hj/nas_hj/data/external/davis_432_240/JPEGImages_432_240}"
BASE_MODEL="${BASE_MODEL:-/mnt/nas/hj/weights/stable-diffusion-v1-5}"
VAE="${VAE:-/mnt/nas/hj/weights/sd-vae-ft-mse}"
PROP="${PROP:-/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter}"
PCM="${PCM:-/mnt/nas/hj/weights/PCM_Weights}"

VIDEO_LENGTH="${VIDEO_LENGTH:-24}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-6}"
USE_PCM="${USE_PCM:-false}"
MASK_DILATION_ITER="${MASK_DILATION_ITER:-0}"
INPUT_SIZE="${INPUT_SIZE:-432x240}"
GPU_LIST_CSV="${GPU_LIST:-0,1,2,3}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

COMPUTE_LPIPS="${COMPUTE_LPIPS:-1}"
COMPUTE_VFID="${COMPUTE_VFID:-1}"
COMPUTE_TC="${COMPUTE_TC:-1}"
COMPUTE_EWARP="${COMPUTE_EWARP:-0}"
SAVE_VIDEOS="${SAVE_VIDEOS:-1}"
I3D_MODEL_PATH="${I3D_MODEL_PATH:-${PROJECT_ROOT}/weights/i3d_rgb_imagenet.pt}"
TC_MODEL_PATH="${TC_MODEL_PATH:-${PROJECT_ROOT}/weights/open_clip_vit_h14}"
RAFT_MODEL_PATH="${RAFT_MODEL_PATH:-${PROP}/raft-things.pth}"

SFT48000="${SFT48000:-/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000}"
EXP9_1="${EXP9_1:-/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/hybrid/20260609_025331_d3n16_val24_exp9_logratio_gap_dpo_s1_2000_davis_pai_dpoS1_sftS2/last_weights}"
EXP9_2="${EXP9_2:-/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260609_025331_d3n16_val24_exp9_logratio_gap_dpo_s2_2000_davis_pai/last_weights}"
EXP10_1="${EXP10_1:-/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/hybrid/20260609_1608_exp10_n16_gpus4_7_scratch_exp10_region_local_dpo_s1_2000_davis_pai_dpoS1_sftS2/last_weights}"
EXP10_2="${EXP10_2:-/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260609_1608_exp10_n16_gpus4_7_scratch_exp10_region_local_dpo_s2_2000_davis_pai/last_weights}"
EXP11_1="${EXP11_1:-/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/hybrid/20260609_2331_exp11_n16_gpus4_7_scratch_exp11_flow_prior_consistency_dpo_s1_2000_davis_pai_dpoS1_sftS2/last_weights}"
EXP11_2="${EXP11_2:-/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260609_2331_exp11_n16_gpus4_7_scratch_exp11_flow_prior_consistency_dpo_s2_2000_davis_pai/last_weights}"

COMMON=(
  --video_root "$VIDEO_ROOT"
  --mask_root "$MASK_ROOT"
  --gt_root "$GT_ROOT"
  --base_model_path "$BASE_MODEL"
  --vae_path "$VAE"
  --propainter_model_dir "$PROP"
  --pcm_weights_path "$PCM"
  --input_size "$INPUT_SIZE"
  --video_length "$VIDEO_LENGTH"
  --num_inference_steps "$NUM_INFERENCE_STEPS"
  --use_pcm "$USE_PCM"
  --mask_dilation_iter "$MASK_DILATION_ITER"
  --raft_model_path "$RAFT_MODEL_PATH"
)

[[ "$COMPUTE_LPIPS" == "1" ]] && COMMON+=(--compute_lpips)
[[ "$COMPUTE_VFID" == "1" ]] && COMMON+=(--compute_vfid --i3d_model_path "$I3D_MODEL_PATH")
[[ "$COMPUTE_TC" == "1" ]] && COMMON+=(--compute_tc)
[[ -n "$TC_MODEL_PATH" ]] && COMMON+=(--tc_model_path "$TC_MODEL_PATH")
[[ "$COMPUTE_EWARP" == "1" ]] && COMMON+=(--compute_ewarp)

IFS=',' read -r -a GPUS <<< "$GPU_LIST_CSV"

run_one() {
  local gpu="$1"
  local label="$2"
  local weights="$3"
  local safe_label
  safe_label=$(echo "$label" | tr -c 'A-Za-z0-9_.-' '_')
  local out="$OUT_ROOT/$safe_label"
  local log="$LOG_ROOT/${safe_label}.log"
  echo "[launch] gpu=$gpu label=$label out=$out log=$log"
  local save_args=()
  [[ "$SAVE_VIDEOS" == "1" ]] && save_args+=(--save_videos)
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" tools/run_davis50_framewise_protocol_eval.py \
    "${COMMON[@]}" \
    --label "$label" \
    --diffueraser_path "$weights" \
    --save_path "$out" \
    "${save_args[@]}" > "$log" 2>&1
  echo "[done] gpu=$gpu label=$label rc=$?"
}

labels=(
  SFT48000_baseline
  Exp9_1_DPO-S1_SFT-S2
  Exp9_2_DPO-S1_DPO-S2
  Exp10_1_DPO-S1_SFT-S2
  Exp10_2_DPO-S1_DPO-S2
  Exp11_1_DPO-S1_SFT-S2
  Exp11_2_DPO-S1_DPO-S2
)

weights=(
  "$SFT48000"
  "$EXP9_1"
  "$EXP9_2"
  "$EXP10_1"
  "$EXP10_2"
  "$EXP11_1"
  "$EXP11_2"
)

for path in "$VIDEO_ROOT" "$MASK_ROOT" "$GT_ROOT" "$SFT48000" "$EXP9_1" "$EXP9_2" "$EXP10_1" "$EXP10_2" "$EXP11_1" "$EXP11_2"; do
  if [[ ! -e "$path" ]]; then
    echo "[missing] $path" >&2
    exit 2
  fi
done
if [[ "$COMPUTE_VFID" == "1" && ! -f "$I3D_MODEL_PATH" ]]; then
  echo "[missing] I3D_MODEL_PATH=$I3D_MODEL_PATH" >&2
  exit 3
fi
if [[ "$COMPUTE_TC" == "1" && ! -f "${TC_MODEL_PATH}/open_clip_pytorch_model.bin" ]]; then
  echo "[missing] TC_MODEL_PATH=${TC_MODEL_PATH}/open_clip_pytorch_model.bin" >&2
  exit 3
fi

active=0
for i in "${!labels[@]}"; do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  run_one "$gpu" "${labels[$i]}" "${weights[$i]}" &
  active=$((active + 1))
  if (( active >= MAX_PARALLEL )); then
    wait -n
    active=$((active - 1))
  fi
done
wait

"$PY" - "$OUT_ROOT" <<'PY'
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

out = root / "metrics"
out.mkdir(parents=True, exist_ok=True)
if not rows:
    raise SystemExit("no summary rows found")

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
    ("mask_region_psnr_mean", "mask PSNR"),
    ("mask_region_ssim_mean", "mask SSIM"),
]
lines = [
    "# Exp9/10/11 DAVIS-50 Frame-wise raw6 Summary",
    "",
    "| Method | " + " | ".join(name for _, name in metric_cols if any(col in row for row in rows)) + " | rows |",
    "|---|" + "|".join(["---:"] * (sum(1 for col, _ in metric_cols if any(col in row for row in rows)) + 1)) + "|",
]
for row in rows:
    values = []
    for col, _ in metric_cols:
        if not any(col in item for item in rows):
            continue
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

cat > "$OUT_ROOT/report.md" <<EOF
# Exp9/10/11 DAVIS-50 Frame-wise raw6 Eval

- protocol: raw6, no PCM, mask dilation 0, no Gaussian blur, hard comp, frame-wise metrics, DAVIS-50, 24 frames per video
- metric backend: inference/metrics.py
- summary: $OUT_ROOT/metrics/summary_all.csv
- videos: $OUT_ROOT/<method>/<video>/comparison_input_gt_current.mp4
EOF

echo "OUT_ROOT=$OUT_ROOT"
