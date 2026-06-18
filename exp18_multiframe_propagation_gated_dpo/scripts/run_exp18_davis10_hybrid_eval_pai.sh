#!/usr/bin/env bash
set -euo pipefail

# Exp18 DAVIS10 metric/visual eval. This script does not train. It evaluates
# Stage1 gates by first building DPO-S1 + SFT-S2 hybrids, mirroring Exp17.

PROJECT_ROOT="${EXP18_PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
cd "${PROJECT_ROOT}"

RUN_VERSION="${RUN_VERSION:-20260618_exp18_gate}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${OUTPUT_ROOT}/experiments}"
OUT_ROOT="${OUT_ROOT:-${OUTPUT_ROOT}/logs/target_eval/exp18_multiframe_propagation_gated_dpo_davis10}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs/pipelines/exp18_multiframe_propagation_gated_dpo_eval}"
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" exp18_multiframe_propagation_gated_dpo/reports

PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
EVAL_GPU="${EVAL_GPU:-0}"

SFT48000="${SFT48000:-/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000}"
EXP11_OUTER_B075_S2="${EXP11_OUTER_B075_S2:-${OUTPUT_ROOT}/experiments/dpo/stage2/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s2_2000_davis_pai/last_weights}"
DAVIS_ROOT="${DAVIS_ROOT:-/mnt/workspace/hj/nas_hj/data/external/davis_432_240}"
VIDEO_ROOT="${VIDEO_ROOT:-${DAVIS_ROOT}/JPEGImages_432_240}"
MASK_ROOT="${MASK_ROOT:-${DAVIS_ROOT}/test_masks}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/mnt/nas/hj/weights}"
BASE_MODEL="${BASE_MODEL:-${WEIGHTS_DIR}/stable-diffusion-v1-5}"
VAE="${VAE:-${WEIGHTS_DIR}/sd-vae-ft-mse}"
PROP="${PROP:-/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter}"
PCM="${PCM:-${WEIGHTS_DIR}/PCM_Weights}"
RAFT_MODEL_PATH="${RAFT_MODEL_PATH:-${PROP}/raft-things.pth}"

VIDEOS="${VIDEOS:-boat,rhino,dog-agility,blackswan,lucia,bear,dance-jump,soccerball,kite-surf,breakdance}"
VIDEO_LENGTH="${VIDEO_LENGTH:-24}"
SUBSET_ROOT="${OUT_ROOT}/davis10_subset"
SUBSET_VIDEO_ROOT="${SUBSET_ROOT}/JPEGImages_432_240"
SUBSET_MASK_ROOT="${SUBSET_ROOT}/test_masks"

require_path() {
  local path="$1"
  local label="$2"
  [[ -e "${path}" ]] || { echo "[missing] ${label}: ${path}" >&2; exit 2; }
}

echo "[exp18-eval] hostname=$(hostname)"
echo "[exp18-eval] project=${PROJECT_ROOT}"
echo "[exp18-eval] run_version=${RUN_VERSION}"
echo "[exp18-eval] out=${OUT_ROOT}"

for path in \
  "${PY}" "${SFT48000}" "${EXP11_OUTER_B075_S2}" "${VIDEO_ROOT}" "${MASK_ROOT}" \
  "${BASE_MODEL}" "${VAE}" "${PROP}" \
  "tools/run_davis50_framewise_protocol_eval.py" \
  "tools/build_diffueraser_dpoS1_sftS2_hybrid.py" \
  "exp18_multiframe_propagation_gated_dpo/code/make_exp18_davis10_visual_cases.py"; do
  require_path "${path}" "${path}"
done

"${PY}" -m py_compile exp18_multiframe_propagation_gated_dpo/code/*.py
bash -n exp18_multiframe_propagation_gated_dpo/scripts/*.sh

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

run_eval() {
  local label="$1"
  local weights="$2"
  local out="${OUT_ROOT}/${label}"
  local log="${LOG_ROOT}/${label}_eval.log"
  if [[ -f "${out}/metrics/summary.csv" ]]; then
    echo "[exp18-eval] skip existing eval ${label}: ${out}"
    return
  fi
  echo "[exp18-eval] eval label=${label} weights=${weights}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PY}" tools/run_davis50_framewise_protocol_eval.py \
    --video_root "${SUBSET_VIDEO_ROOT}" \
    --mask_root "${SUBSET_MASK_ROOT}" \
    --gt_root "${SUBSET_VIDEO_ROOT}" \
    --base_model_path "${BASE_MODEL}" \
    --vae_path "${VAE}" \
    --propainter_model_dir "${PROP}" \
    --pcm_weights_path "${PCM}" \
    --diffueraser_path "${weights}" \
    --save_path "${out}" \
    --label "${label}" \
    --input_size 432x240 \
    --video_length "${VIDEO_LENGTH}" \
    --num_inference_steps 6 \
    --use_pcm false \
    --mask_dilation_iter 0 \
    --raft_model_path "${RAFT_MODEL_PATH}" \
    --save_videos \
    --save_comp_frames > "${log}" 2>&1
}

build_hybrid() {
  local variant="$1"
  local stage_dir="$2"
  local hybrid="${EXPERIMENTS_DIR}/hybrid/${RUN_VERSION}_${variant}_dpoS1_sftS2"
  require_path "${stage_dir}/last_weights" "${variant} Stage1 last_weights"
  if [[ ! -d "${hybrid}/last_weights" ]]; then
    echo "[exp18-eval] build hybrid ${variant}: ${hybrid}" >&2
    "${PY}" tools/build_diffueraser_dpoS1_sftS2_hybrid.py \
      --dpo_stage1_weights "${stage_dir}/last_weights" \
      --sft_stage2_weights "${SFT48000}" \
      --output_dir "${hybrid}" \
      --strict false \
      --report_path "reports/${variant}_hybrid_key_merge_report.md" > "${LOG_ROOT}/${variant}_hybrid.log" 2>&1
  else
    echo "[exp18-eval] reuse hybrid ${variant}: ${hybrid}" >&2
  fi
  require_path "${hybrid}/last_weights" "${variant} hybrid last_weights"
  echo "${hybrid}/last_weights"
}

run_eval "SFT48000_baseline" "${SFT48000}"
run_eval "Exp11_boundary_outer_b075_S2" "${EXP11_OUTER_B075_S2}"

EXP18A_STAGE="${EXPERIMENTS_DIR}/dpo/stage1/${RUN_VERSION}_exp18a_prop_only_s1_500_pai"
EXP18B_STAGE="${EXPERIMENTS_DIR}/dpo/stage1/${RUN_VERSION}_exp18b_prop_gen_s1_500_pai"
EXP18C_STAGE="${EXPERIMENTS_DIR}/dpo/stage1/${RUN_VERSION}_exp18c_oracle_s1_500_pai"

EXP18A_HYBRID="$(build_hybrid exp18a_prop_only_s1_500 "${EXP18A_STAGE}")"
EXP18B_HYBRID="$(build_hybrid exp18b_prop_gen_s1_500 "${EXP18B_STAGE}")"
EXP18C_HYBRID="$(build_hybrid exp18c_oracle_s1_500 "${EXP18C_STAGE}")"

run_eval "Exp18a_prop_only_s1_500" "${EXP18A_HYBRID}"
run_eval "Exp18b_prop_gen_s1_500" "${EXP18B_HYBRID}"
run_eval "Exp18c_oracle_s1_500" "${EXP18C_HYBRID}"

"${PY}" - "${OUT_ROOT}" <<'PY'
import csv
import json
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

cols = [
    ("whole_video_psnr_mean", "PSNR"),
    ("whole_video_ssim_mean", "SSIM"),
    ("strict_mask_pixel_psnr_mean", "strict mask PSNR"),
    ("boundary_pixel_psnr_mean", "boundary PSNR"),
    ("mask_bbox_psnr_mean", "bbox PSNR"),
    ("mask_bbox_ssim_mean", "bbox SSIM"),
]
lines = [
    "# Exp18 DAVIS10 Hybrid Eval Summary",
    "",
    "| Method | " + " | ".join(name for _, name in cols) + " | rows |",
    "|---|" + "|".join(["---:"] * (len(cols) + 1)) + "|",
]
for row in rows:
    vals = []
    for col, _ in cols:
        val = row.get(col, "")
        vals.append("" if val == "" else f"{float(val):.4f}")
    lines.append(f"| {row.get('model_label','')} | " + " | ".join(vals) + f" | {row.get('rows','')} |")
(out / "summary_all.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

by_label = {r.get("model_label", ""): r for r in rows}
exp11 = by_label.get("Exp11_boundary_outer_b075_S2")
candidates = [r for r in rows if r.get("model_label", "").startswith("Exp18")]
decision = {"best_variant": None, "metric_positive": False, "reason": "missing Exp11 or Exp18 candidates"}
if exp11 and candidates:
    def f(row, key):
        try:
            return float(row.get(key, "nan"))
        except Exception:
            return float("nan")
    best = max(candidates, key=lambda r: (f(r, "whole_video_psnr_mean"), f(r, "strict_mask_pixel_psnr_mean")))
    decision["best_variant"] = best.get("model_label")
    decision["metric_positive"] = (
        f(best, "whole_video_psnr_mean") >= f(exp11, "whole_video_psnr_mean")
        and f(best, "whole_video_ssim_mean") >= f(exp11, "whole_video_ssim_mean")
        and f(best, "strict_mask_pixel_psnr_mean") >= f(exp11, "strict_mask_pixel_psnr_mean")
    )
    decision["reason"] = "metric positive on DAVIS10" if decision["metric_positive"] else "no Exp18 variant beats Exp11 on DAVIS10 primary metrics"
(out / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
print("\n".join(lines))
print(json.dumps(decision, indent=2))
PY

"${PY}" exp18_multiframe_propagation_gated_dpo/code/make_exp18_davis10_visual_cases.py \
  --video_root "${SUBSET_VIDEO_ROOT}" \
  --mask_root "${SUBSET_MASK_ROOT}" \
  --method "SFT=${OUT_ROOT}/SFT48000_baseline" \
  --method "Exp11=${OUT_ROOT}/Exp11_boundary_outer_b075_S2" \
  --method "Exp18a=${OUT_ROOT}/Exp18a_prop_only_s1_500" \
  --method "Exp18b=${OUT_ROOT}/Exp18b_prop_gen_s1_500" \
  --method "Exp18c_oracle=${OUT_ROOT}/Exp18c_oracle_s1_500" \
  --reference_label "Exp11" \
  --output_root "${OUT_ROOT}/visual_cases/all_methods" \
  --videos "${VIDEOS}" \
  --video_length "${VIDEO_LENGTH}"

cat > "${OUT_ROOT}/report.md" <<EOF
# Exp18 DAVIS10 Hybrid Eval

- videos: ${VIDEOS}
- protocol: raw6, no PCM, mask dilation 0, no Gaussian blur, hard comp, frame-wise metrics
- metric backend: inference/metrics.py via tools/run_davis50_framewise_protocol_eval.py
- SFT weights: ${SFT48000}
- Exp11 weights: ${EXP11_OUTER_B075_S2}
- Exp18a hybrid: ${EXP18A_HYBRID}
- Exp18b hybrid: ${EXP18B_HYBRID}
- Exp18c oracle hybrid: ${EXP18C_HYBRID}
- summary: ${OUT_ROOT}/metrics/summary_all.csv
- visuals: ${OUT_ROOT}/visual_cases/all_methods
EOF

cp "${OUT_ROOT}/metrics/summary_all.csv" reports/exp18_davis10_metric_summary.csv
cp "${OUT_ROOT}/metrics/summary_all.md" reports/exp18_davis10_metric_summary.md
cp "${OUT_ROOT}/metrics/decision.json" reports/exp18_davis10_metric_decision.json

echo "OUT_ROOT=${OUT_ROOT}"
echo "LOG_ROOT=${LOG_ROOT}"
