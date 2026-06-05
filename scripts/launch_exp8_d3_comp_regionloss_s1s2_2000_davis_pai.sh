#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
cd "${PROJECT_ROOT}"

EXP_NAME="${EXP_NAME:-exp08_d3_comp_regionloss_wingap_lose025_s1s2_2000_davis_pai}"
STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-exp08_d3_comp_regionloss_wingap_lose025_s1_2000_davis_pai}"
STAGE2_RUN_NAME="${STAGE2_RUN_NAME:-exp08_d3_comp_regionloss_wingap_lose025_s2_2000_davis_pai}"
RUN_VERSION="${RUN_VERSION:-$(date +%Y%m%d_%H%M%S)}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${OUTPUT_ROOT}/experiments}"
REG_DIR="${REG_DIR:-experiment_registry/exp08_d3_comp_regionloss_davis_stage1stage2_2000}"
REPORT="${REPORT:-reports/exp08_d3_comp_regionloss_s1s2_2000_davis_report.md}"

D3_ROOT="${D3_ROOT:-${WORKSPACE_ROOT}/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4}"
PREFERENCE_MANIFEST="${PREFERENCE_MANIFEST:-${D3_ROOT}/manifests/selected_primary_comp.repaired.pai_paths.jsonl}"
DAVIS_ROOT="${DAVIS_ROOT:-/mnt/workspace/hj/nas_hj/data/external/davis_432_240}"
DAVIS_VIDEO_ROOT="${DAVIS_VIDEO_ROOT:-${DAVIS_ROOT}/JPEGImages_432_240}"
DAVIS_MASK_ROOT="${DAVIS_MASK_ROOT:-${DAVIS_ROOT}/test_masks}"
DAVIS_GT_ROOT="${DAVIS_GT_ROOT:-${DAVIS_ROOT}/JPEGImages_432_240}"

WEIGHTS_DIR="${WEIGHTS_DIR:-/mnt/nas/hj/weights}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${WEIGHTS_DIR}/stable-diffusion-v1-5}"
VAE_PATH="${VAE_PATH:-${WEIGHTS_DIR}/sd-vae-ft-mse}"
DIFFUERASER_WEIGHT_ROOT="${DIFFUERASER_WEIGHT_ROOT:-/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000}"
SFT_STAGE2_WEIGHTS="${SFT_STAGE2_WEIGHTS:-${DIFFUERASER_WEIGHT_ROOT}}"
PCM_WEIGHTS_PATH="${PCM_WEIGHTS_PATH:-${WEIGHTS_DIR}/PCM_Weights}"
PROPAINTER_WEIGHT_ROOT="${PROPAINTER_WEIGHT_ROOT:-}"
RAFT_MODEL_PATH="${RAFT_MODEL_PATH:-}"

CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/mnt/nas/hj/conda_envs/diffueraser}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PREFIX}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "[Exp8][ERROR] python not found; set PYTHON_BIN or CONDA_ENV_PREFIX" >&2
    exit 2
  fi
fi

NUM_GPUS="${NUM_GPUS:-8}"
TRAIN_HEIGHT="${TRAIN_HEIGHT:-320}"
TRAIN_WIDTH="${TRAIN_WIDTH:-512}"
RESOLUTION="${RESOLUTION:-512}"
NFRAMES="${NFRAMES:-16}"
DAVIS_VIDEO_LENGTH="${DAVIS_VIDEO_LENGTH:-16}"
DAVIS_NUM_QUAL="${DAVIS_NUM_QUAL:-30}"
EVAL_GPU="${EVAL_GPU:-0}"
EVAL_WIDTH="${EVAL_WIDTH:-432}"
EVAL_HEIGHT="${EVAL_HEIGHT:-240}"
COMPUTE_LPIPS="${COMPUTE_LPIPS:-0}"
COMPUTE_EWARP="${COMPUTE_EWARP:-0}"

mkdir -p reports logs/pipelines "${REG_DIR}" "${OUTPUT_ROOT}/logs/target_eval" "${OUTPUT_ROOT}/reports"

die() {
  echo "[Exp8][ERROR] $*" >&2
  {
    echo
    echo "## FAILED"
    echo
    echo "- reason: $*"
  } >> "${REPORT}" 2>/dev/null || true
  exit 2
}

require_path() {
  local path="$1"
  local label="$2"
  [[ -e "${path}" ]] || die "${label} not found: ${path}"
}

find_propainter_root() {
  if [[ -n "${PROPAINTER_WEIGHT_ROOT}" && -d "${PROPAINTER_WEIGHT_ROOT}" ]]; then
    echo "${PROPAINTER_WEIGHT_ROOT}"
    return 0
  fi
  local candidates=(
    "${PROJECT_ROOT}/weights/propainter/current"
    "${PROJECT_ROOT}/weights/propainter"
    "${WEIGHTS_DIR}/propainter"
    "/mnt/workspace/hj/nas_hj/weights/propainter"
    "/mnt/nas/hj/weights/propainter"
    "/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter"
    "/mnt/nas/hj/H20_Video_inpainting_DPO/weights/propainter"
  )
  local c
  for c in "${candidates[@]}"; do
    if [[ -d "${c}" ]]; then
      echo "${c}"
      return 0
    fi
  done
  return 1
}

precheck() {
  echo "[Exp8] precheck start"
  require_path "${REG_DIR}" "experiment registry folder"
  require_path "${PREFERENCE_MANIFEST}" "D3 selected-primary-comp PAI manifest"
  require_path "${DAVIS_ROOT}" "DAVIS root"
  require_path "${DAVIS_VIDEO_ROOT}" "DAVIS video root"
  require_path "${DAVIS_MASK_ROOT}" "DAVIS mask root"
  require_path "${DAVIS_GT_ROOT}" "DAVIS GT root"
  require_path "${BASE_MODEL_PATH}" "stable-diffusion-v1-5"
  require_path "${VAE_PATH}" "sd-vae-ft-mse"
  require_path "${DIFFUERASER_WEIGHT_ROOT}/unet_main/config.json" "SFT-48000 DiffuEraser unet_main"
  require_path "${DIFFUERASER_WEIGHT_ROOT}/brushnet/config.json" "SFT-48000 DiffuEraser brushnet"
  require_path "${PROJECT_ROOT}/training/dpo/scripts/03_dpo_stage1.sbatch" "Stage1 launcher"
  require_path "${PROJECT_ROOT}/training/dpo/scripts/03_dpo_stage2.sbatch" "Stage2 launcher"
  require_path "${PROJECT_ROOT}/tools/build_diffueraser_dpoS1_sftS2_hybrid.py" "hybrid builder"
  require_path "${PROJECT_ROOT}/tools/run_inpainting_metric_eval.py" "metric wrapper"
  require_path "${PROJECT_ROOT}/inference/run_BR.py" "DAVIS BR inference wrapper"

  if grep -m1 -q '/home/nvme01/' "${PREFERENCE_MANIFEST}"; then
    die "D3 PAI manifest still contains H20 /home/nvme01 paths: ${PREFERENCE_MANIFEST}"
  fi

  PROPAINTER_WEIGHT_ROOT="$(find_propainter_root)" || die "ProPainter weights not found; set PROPAINTER_WEIGHT_ROOT"
  export PROPAINTER_WEIGHT_ROOT
  if [[ -z "${RAFT_MODEL_PATH}" && -f "${PROPAINTER_WEIGHT_ROOT}/raft-things.pth" ]]; then
    RAFT_MODEL_PATH="${PROPAINTER_WEIGHT_ROOT}/raft-things.pth"
  fi

  grep -q "build_region_loss_weight_map" training/dpo/train_stage1.py || die "Stage1 region loss helper missing"
  grep -q "build_region_loss_weight_map" training/dpo/train_stage2.py || die "Stage2 region loss helper missing"
  if grep -R "loss_region_mode.*region.*not implemented" training/dpo/train_stage1.py training/dpo/train_stage2.py >/dev/null 2>&1; then
    die "LOSS_REGION_MODE=region still has a not-implemented guard"
  fi
  grep -q -- "--propainter_model_dir" inference/run_BR.py || die "DAVIS inference wrapper does not expose ProPainter prior"

  "${PYTHON_BIN}" -m py_compile training/dpo/train_stage1.py training/dpo/train_stage2.py tools/run_inpainting_metric_eval.py tools/build_diffueraser_dpoS1_sftS2_hybrid.py

  {
    echo "# Exp8 D3 Comp Region-Loss S1/S2 2000 DAVIS Report"
    echo
    date
    echo
    echo "## Precheck"
    echo
    echo "- exp_name: \`${EXP_NAME}\`"
    echo "- manifest: \`${PREFERENCE_MANIFEST}\`"
    echo "- manifest_rows: \`$(wc -l < "${PREFERENCE_MANIFEST}")\`"
    echo "- davis_root: \`${DAVIS_ROOT}\`"
    echo "- diffueraser_weight_path: \`${DIFFUERASER_WEIGHT_ROOT}\`"
    echo "- prior_mode: \`propainter\`"
    echo "- propainter_weight_path: \`${PROPAINTER_WEIGHT_ROOT}\`"
    echo "- raft_model_path: \`${RAFT_MODEL_PATH:-missing_optional}\`"
    echo "- loss_region_mode: \`region\`"
    echo "- region_weights: mask=1.0, boundary=0.5, outside=0.05"
    echo "- metric_wrapper: \`tools/run_inpainting_metric_eval.py\`"
    echo "- metric_backend: \`inference/metrics.py\`"
    echo "- VBench: not used"
  } > "${REPORT}"
  echo "[Exp8] precheck ok"
}

summarize_diag_csv() {
  local csv_path="$1"
  local out_path="$2"
  local stage_label="$3"
  "${PYTHON_BIN}" - "$csv_path" "$out_path" "$stage_label" <<'PY'
import csv, math, sys
from pathlib import Path

csv_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
stage_label = sys.argv[3]
metrics = [
    "dpo_loss", "implicit_acc", "mse_w", "ref_mse_w", "mse_l", "ref_mse_l",
    "win_gap", "lose_gap", "winner_abs_reg", "winner_gap_reg",
    "mse_w_over_ref_mse_w", "mse_l_over_ref_mse_l", "sigma_term",
    "kl_divergence", "loser_dominant_ratio", "grad_norm",
    "region_weighted_mse_w", "region_weighted_mse_l",
    "region_weighted_ref_mse_w", "region_weighted_ref_mse_l",
    "mask_area_ratio", "boundary_area_ratio", "outside_area_ratio",
]

def f(x):
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v

def percentile(vals, q):
    vals = sorted(vals)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals)-1)*q
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo]*(hi-pos)+vals[hi]*(pos-lo)

def fmt(v):
    return "" if v is None else f"{v:.6g}"

rows = []
if csv_path.exists():
    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

lines = [f"# {stage_label} DPO Diagnostics Summary", ""]
lines += [f"csv: `{csv_path}`", f"rows: {len(rows)}", ""]
if not rows:
    lines += ["status: MISSING_DPO_DIAG", ""]
else:
    lines += ["| metric | mean | median | p90 | max | frac key |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for key in metrics:
        vals = [f(r.get(key)) for r in rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        frac = ""
        if key == "dpo_loss":
            frac = sum(v < 1e-3 for v in vals) / len(vals)
        elif key == "implicit_acc":
            frac = sum(v > 0.99 for v in vals) / len(vals)
        elif key == "win_gap":
            frac = sum(v > 0.5 for v in vals) / len(vals)
        elif key == "mse_w_over_ref_mse_w":
            frac = sum(v > 5.0 for v in vals) / len(vals)
        elif key == "sigma_term":
            frac = sum(v > 0.99 for v in vals) / len(vals)
        elif key == "kl_divergence":
            frac = sum(v > 1.0 for v in vals) / len(vals)
        elif key == "loser_dominant_ratio":
            frac = sum(v > 0.9 for v in vals) / len(vals)
        lines.append(
            f"| `{key}` | {fmt(sum(vals)/len(vals))} | {fmt(percentile(vals,0.5))} | {fmt(percentile(vals,0.9))} | {fmt(max(vals))} | {fmt(frac) if frac != '' else ''} |"
        )
    lines.append("")

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(lines), encoding="utf-8")
print(f"[diag-summary] {out_path}")
PY
}

make_pair_manifest_and_side_by_side() {
  local eval_out="$1"
  local current_label="$2"
  local base_pred_root="$3"
  local current_pred_root="$4"
  local pair_manifest="${eval_out}/pair_manifest.csv"
  local side_dir="${eval_out}/side_by_side/${current_label}"
  mkdir -p "${side_dir}"
  "${PYTHON_BIN}" - "$DAVIS_GT_ROOT" "$DAVIS_MASK_ROOT" "$base_pred_root" "$current_pred_root" "$pair_manifest" "$side_dir" "$current_label" "$DAVIS_NUM_QUAL" "$DAVIS_VIDEO_LENGTH" <<'PY'
import csv, os, sys
from pathlib import Path
import cv2
import imageio
import numpy as np

gt_root = Path(sys.argv[1])
mask_root = Path(sys.argv[2])
base_root = Path(sys.argv[3])
cur_root = Path(sys.argv[4])
pair_manifest = Path(sys.argv[5])
side_dir = Path(sys.argv[6])
cur_label = sys.argv[7]
num_qual = int(sys.argv[8])
max_frames = int(sys.argv[9])

def image_files(d):
    return sorted([p for p in d.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}])

def read_frames(path, is_mask=False, size=(432,240)):
    w, h = size
    frames = []
    if path.is_dir():
        for p in image_files(path)[:max_frames]:
            flag = cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR
            arr = cv2.imread(str(p), flag)
            if arr is None:
                continue
            interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
            arr = cv2.resize(arr, (w,h), interpolation=interp)
            if is_mask:
                frames.append((arr > 0).astype(np.uint8) * 255)
            else:
                frames.append(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    else:
        cap = cv2.VideoCapture(str(path))
        while len(frames) < max_frames:
            ok, arr = cap.read()
            if not ok:
                break
            interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
            if is_mask:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                arr = cv2.resize(arr, (w,h), interpolation=interp)
                frames.append((arr > 0).astype(np.uint8) * 255)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                arr = cv2.resize(arr, (w,h), interpolation=interp)
                frames.append(arr)
        cap.release()
    return frames

def label_frame(frame, label):
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0,0,0), -1)
    cv2.putText(out, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    return out

def overlay_mask(gt, mask):
    out = gt.copy()
    red = np.zeros_like(out)
    red[...,0] = 255
    m = mask > 0
    out[m] = (0.55*out[m] + 0.45*red[m]).astype(np.uint8)
    return out

rows = []
names = sorted([p.name for p in gt_root.iterdir() if p.is_dir()])
qual_written = 0
for name in names:
    gt = gt_root / name
    mask = mask_root / name
    base_pred = base_root / name / "diffueraser_comp.mp4"
    cur_pred = cur_root / name / "diffueraser_comp.mp4"
    if not (gt.exists() and mask.exists() and base_pred.exists() and cur_pred.exists()):
        continue
    rows.append({"sample_id": name, "model_label": "DiffuEraser-base", "gt_video_path": str(gt), "prediction_video_path": str(base_pred), "mask_path": str(mask)})
    rows.append({"sample_id": name, "model_label": cur_label, "gt_video_path": str(gt), "prediction_video_path": str(cur_pred), "mask_path": str(mask)})
    if qual_written >= num_qual:
        continue
    gt_frames = read_frames(gt, is_mask=False)
    mask_frames = read_frames(mask, is_mask=True)
    base_frames = read_frames(base_pred, is_mask=False)
    cur_frames = read_frames(cur_pred, is_mask=False)
    n = min(len(gt_frames), len(mask_frames), len(base_frames), len(cur_frames))
    if n == 0:
        continue
    side_frames = []
    for i in range(n):
        columns = [
            label_frame(gt_frames[i], "winner/GT"),
            label_frame(overlay_mask(gt_frames[i], mask_frames[i]), "mask overlay"),
            label_frame(base_frames[i], "DiffuEraser-base"),
            label_frame(cur_frames[i], cur_label),
        ]
        side_frames.append(np.concatenate(columns, axis=1))
    out_path = side_dir / f"{name}.mp4"
    imageio.mimsave(out_path, side_frames, fps=12, codec="libx264", macro_block_size=1)
    qual_written += 1

pair_manifest.parent.mkdir(parents=True, exist_ok=True)
with pair_manifest.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=["sample_id","model_label","gt_video_path","prediction_video_path","mask_path"])
    writer.writeheader()
    writer.writerows(rows)

index = pair_manifest.parent / "index.html"
items = sorted(side_dir.glob("*.mp4"))
html = ["<!doctype html><html><head><meta charset='utf-8'><title>Exp8 DAVIS side-by-side</title></head><body>"]
html.append(f"<h1>Exp8 DAVIS side-by-side: {cur_label}</h1>")
for p in items:
    rel = p.relative_to(index.parent)
    html.append(f"<h3>{p.stem}</h3><video src='{rel.as_posix()}' controls width='1200'></video>")
html.append("</body></html>")
index.write_text("\n".join(html), encoding="utf-8")
print(f"[davis-pairs] rows={len(rows)} pair_manifest={pair_manifest} side_by_side={side_dir}")
PY
}

run_metric_wrapper() {
  local eval_out="$1"
  local pair_manifest="${eval_out}/pair_manifest.csv"
  local flags=()
  case "${COMPUTE_LPIPS,,}" in 1|true|yes|on) flags+=(--compute_lpips) ;; esac
  case "${COMPUTE_EWARP,,}" in 1|true|yes|on) flags+=(--compute_ewarp) ;; esac
  "${PYTHON_BIN}" tools/run_inpainting_metric_eval.py \
    --pair_manifest "${pair_manifest}" \
    --output_dir "${eval_out}" \
    --max_frames "${DAVIS_VIDEO_LENGTH}" \
    --width "${EVAL_WIDTH}" \
    --height "${EVAL_HEIGHT}" \
    "${flags[@]}"
}

run_br_inference() {
  local label="$1"
  local weights_path="$2"
  local out_dir="$3"
  local raft_arg=()
  if [[ -n "${RAFT_MODEL_PATH}" ]]; then
    raft_arg=(--raft_model_path "${RAFT_MODEL_PATH}")
  fi
  mkdir -p "${out_dir}"
  echo "[Exp8][DAVIS] ${label} weights=${weights_path} out=${out_dir}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON_BIN}" inference/run_BR.py \
    --dataset davis \
    --video_root "${DAVIS_VIDEO_ROOT}" \
    --mask_root "${DAVIS_MASK_ROOT}" \
    --gt_root "${DAVIS_GT_ROOT}" \
    --save_path "${out_dir}" \
    --input_size "${EVAL_WIDTH}x${EVAL_HEIGHT}" \
    --video_length "${DAVIS_VIDEO_LENGTH}" \
    --ref_stride 3 \
    --neighbor_length 25 \
    --subvideo_length 80 \
    --mask_dilation_iter 0 \
    --save_comparison \
    --no_metrics \
    --base_model_path "${BASE_MODEL_PATH}" \
    --vae_path "${VAE_PATH}" \
    --diffueraser_path "${weights_path}" \
    --propainter_model_dir "${PROPAINTER_WEIGHT_ROOT}" \
    --pcm_weights_path "${PCM_WEIGHTS_PATH}" \
    "${raft_arg[@]}"
}

run_davis_validation() {
  local stage_label="$1"
  local current_label="$2"
  local current_weights="$3"
  local eval_out="$4"
  mkdir -p "${eval_out}/inference" "${eval_out}/metrics" "${eval_out}/side_by_side"
  echo "[Exp8] ${stage_label} DAVIS validation current=${current_label}"
  run_br_inference "DiffuEraser-base" "${DIFFUERASER_WEIGHT_ROOT}" "${eval_out}/inference/DiffuEraser-base"
  run_br_inference "${current_label}" "${current_weights}" "${eval_out}/inference/${current_label}"
  make_pair_manifest_and_side_by_side \
    "${eval_out}" \
    "${current_label}" \
    "${eval_out}/inference/DiffuEraser-base" \
    "${eval_out}/inference/${current_label}"
  run_metric_wrapper "${eval_out}"
  cat > "${eval_out}/report.md" <<EOF
# ${stage_label} DAVIS Validation

status: complete

- current_label: \`${current_label}\`
- current_weights: \`${current_weights}\`
- baseline_weights: \`${DIFFUERASER_WEIGHT_ROOT}\`
- prior_mode: \`propainter\`
- propainter_weight_path: \`${PROPAINTER_WEIGHT_ROOT}\`
- metric_backend: \`inference/metrics.py\`
- metric_wrapper: \`tools/run_inpainting_metric_eval.py\`
- side_by_side: \`${eval_out}/side_by_side/${current_label}\`
- metrics: \`${eval_out}/metrics/summary.csv\`
- VBench: not used
EOF
}

run_stage1() {
  echo "[Exp8] Stage1 start: ${STAGE1_RUN_NAME}"
  export PROJECT_ROOT OUTPUT_ROOT EXPERIMENTS_DIR
  export DATA="${OUTPUT_ROOT}"
  export WEIGHTS_DIR="${WEIGHTS_DIR}"
  export DPO_DATA_ROOT="${D3_ROOT}"
  export DPO_DATASET_TYPE="generated_loser_manifest"
  export PREFERENCE_MANIFEST
  export TRAIN_MASK_MODE="partial"
  export MASK_FROM_MANIFEST="true"
  export LOSS_REGION_MODE="region"
  export ENABLE_DPO_DIAG="true"
  export DPO_DIAG_LOG_EVERY="${DPO_DIAG_LOG_EVERY:-10}"
  export DPO_DIAG_SAVE_CSV="true"
  export DPO_DIAG_SAVE_WANDB="${DPO_DIAG_SAVE_WANDB:-false}"
  export VAL_DATA_DIR="${DAVIS_ROOT}"
  export REF_MODEL_PATH="${DIFFUERASER_WEIGHT_ROOT}"
  export RUN_NAME="${STAGE1_RUN_NAME}"
  export RUN_VERSION
  export NUM_GPUS
  export MAX_STEPS="2000"
  export CKPT_STEPS="500"
  export CKPT_LIMIT="5"
  export VAL_STEPS="999999"
  export LOGGING_STEPS="${LOGGING_STEPS:-10}"
  export TRAIN_HEIGHT TRAIN_WIDTH RESOLUTION NFRAMES
  export BETA_DPO="10"
  export SFT_REG_WEIGHT="0.0"
  export LOSE_GAP_WEIGHT="0.25"
  export DPO_LOSE_GAP_WEIGHT="0.25"
  export WINNER_ABS_REG_WEIGHT="0.05"
  export WINNER_GAP_REG_WEIGHT="1.0"
  export WINNER_GAP_REG_MARGIN="0.0"
  export REPORT_TO="${REPORT_TO:-none}"
  export WANDB_PROJECT="${WANDB_PROJECT:-DPO_Diffueraser_Exp8}"
  export CONDA_ENV_PREFIX
  bash training/dpo/scripts/03_dpo_stage1.sbatch
}

run_stage2() {
  local stage1_last="$1"
  echo "[Exp8] Stage2 start: ${STAGE2_RUN_NAME}"
  export PROJECT_ROOT OUTPUT_ROOT EXPERIMENTS_DIR
  export DATA="${OUTPUT_ROOT}"
  export WEIGHTS_DIR="${WEIGHTS_DIR}"
  export DPO_DATA_ROOT="${D3_ROOT}"
  export DPO_DATASET_TYPE="generated_loser_manifest"
  export PREFERENCE_MANIFEST
  export TRAIN_MASK_MODE="partial"
  export MASK_FROM_MANIFEST="true"
  export LOSS_REGION_MODE="region"
  export ENABLE_DPO_DIAG="true"
  export DPO_DIAG_LOG_EVERY="${DPO_DIAG_LOG_EVERY:-10}"
  export DPO_DIAG_SAVE_CSV="true"
  export DPO_DIAG_SAVE_WANDB="${DPO_DIAG_SAVE_WANDB:-false}"
  export VAL_DATA_DIR="${DAVIS_ROOT}"
  export PRETRAINED_DPO_S1="${stage1_last}"
  export BASELINE_UNET_PATH="${SFT_STAGE2_WEIGHTS}"
  export REF_MODEL_PATH="${DIFFUERASER_WEIGHT_ROOT}"
  export RUN_NAME="${STAGE2_RUN_NAME}"
  export RUN_VERSION
  export NUM_GPUS
  export MAX_STEPS="2000"
  export CKPT_STEPS="500"
  export CKPT_LIMIT="5"
  export VAL_STEPS="999999"
  export LOGGING_STEPS="${LOGGING_STEPS:-10}"
  export TRAIN_HEIGHT TRAIN_WIDTH RESOLUTION NFRAMES
  export BETA_DPO="10"
  export SFT_REG_WEIGHT="0.0"
  export LOSE_GAP_WEIGHT="0.25"
  export DPO_LOSE_GAP_WEIGHT="0.25"
  export WINNER_ABS_REG_WEIGHT="0.05"
  export WINNER_GAP_REG_WEIGHT="1.0"
  export WINNER_GAP_REG_MARGIN="0.0"
  export REPORT_TO="${REPORT_TO:-none}"
  export WANDB_PROJECT="${WANDB_PROJECT:-DPO_Diffueraser_Exp8}"
  export CONDA_ENV_PREFIX
  bash training/dpo/scripts/03_dpo_stage2.sbatch
}

latest_run_dir() {
  local family="$1"
  local run_name="$2"
  find "${EXPERIMENTS_DIR}/dpo/${family}" -maxdepth 1 -type d -name "*${run_name}*" -printf '%T@ %p\n' 2>/dev/null \
    | sort -n | tail -1 | cut -d' ' -f2-
}

main() {
  precheck
  {
    echo
    echo "## Training Configuration"
    echo
    echo "- Stage1 run_name: \`${STAGE1_RUN_NAME}\`"
    echo "- Stage2 run_name: \`${STAGE2_RUN_NAME}\`"
    echo "- Stage1 steps: 2000"
    echo "- Stage2 steps: 2000"
    echo "- checkpointing_steps: 500"
    echo "- train_mask_mode: partial"
    echo "- mask_from_manifest: true"
    echo "- loss_region_mode: region"
    echo "- prior_mode_validation: propainter"
  } >> "${REPORT}"

  run_stage1
  STAGE1_RUN_DIR="$(latest_run_dir stage1 "${STAGE1_RUN_NAME}")"
  require_path "${STAGE1_RUN_DIR}" "Stage1 run dir"
  STAGE1_LAST="${STAGE1_RUN_DIR}/last_weights"
  require_path "${STAGE1_LAST}/unet_main/config.json" "Stage1 last_weights unet_main"
  require_path "${STAGE1_LAST}/brushnet/config.json" "Stage1 last_weights brushnet"
  require_path "${STAGE1_RUN_DIR}/dpo_diagnostics.csv" "Stage1 dpo_diagnostics.csv"
  summarize_diag_csv "${STAGE1_RUN_DIR}/dpo_diagnostics.csv" "reports/exp08_stage1_dpo_diag_summary.md" "Exp8 Stage1"

  HYBRID_DIR="${OUTPUT_ROOT}/experiments/hybrid/${RUN_VERSION}_${STAGE1_RUN_NAME}_dpoS1_sftS2"
  echo "[Exp8] build Stage1 hybrid: ${HYBRID_DIR}"
  "${PYTHON_BIN}" tools/build_diffueraser_dpoS1_sftS2_hybrid.py \
    --dpo_stage1_weights "${STAGE1_LAST}" \
    --sft_stage2_weights "${SFT_STAGE2_WEIGHTS}" \
    --output_dir "${HYBRID_DIR}" \
    --strict false \
    --report_path "reports/exp08_stage1_hybrid_key_merge_report.md"
  require_path "${HYBRID_DIR}/last_weights/unet_main/config.json" "Stage1 hybrid full weights"

  STAGE1_VAL_DIR="${OUTPUT_ROOT}/logs/target_eval/exp08_stage1_val_davis_${RUN_VERSION}"
  run_davis_validation "Exp8 Stage1 DPO + SFT Stage2" "DPO-S1_SFT-S2" "${HYBRID_DIR}/last_weights" "${STAGE1_VAL_DIR}"

  run_stage2 "${STAGE1_LAST}"
  STAGE2_RUN_DIR="$(latest_run_dir stage2 "${STAGE2_RUN_NAME}")"
  require_path "${STAGE2_RUN_DIR}" "Stage2 run dir"
  STAGE2_LAST="${STAGE2_RUN_DIR}/last_weights"
  require_path "${STAGE2_LAST}/unet_main/config.json" "Stage2 last_weights unet_main"
  require_path "${STAGE2_LAST}/brushnet/config.json" "Stage2 last_weights brushnet"
  require_path "${STAGE2_RUN_DIR}/dpo_diagnostics.csv" "Stage2 dpo_diagnostics.csv"
  summarize_diag_csv "${STAGE2_RUN_DIR}/dpo_diagnostics.csv" "reports/exp08_stage2_dpo_diag_summary.md" "Exp8 Stage2"

  STAGE2_VAL_DIR="${OUTPUT_ROOT}/logs/target_eval/exp08_stage2_val_davis_${RUN_VERSION}"
  run_davis_validation "Exp8 Stage1 DPO + Stage2 DPO" "DPO-S1_DPO-S2" "${STAGE2_LAST}" "${STAGE2_VAL_DIR}"

  {
    echo
    echo "## Outputs"
    echo
    echo "- stage1_run_dir: \`${STAGE1_RUN_DIR}\`"
    echo "- stage1_dpo_diag: \`${STAGE1_RUN_DIR}/dpo_diagnostics.csv\`"
    echo "- stage1_diag_summary: \`reports/exp08_stage1_dpo_diag_summary.md\`"
    echo "- stage1_hybrid: \`${HYBRID_DIR}/last_weights\`"
    echo "- stage1_val: \`${STAGE1_VAL_DIR}\`"
    echo "- stage2_run_dir: \`${STAGE2_RUN_DIR}\`"
    echo "- stage2_dpo_diag: \`${STAGE2_RUN_DIR}/dpo_diagnostics.csv\`"
    echo "- stage2_diag_summary: \`reports/exp08_stage2_dpo_diag_summary.md\`"
    echo "- stage2_val: \`${STAGE2_VAL_DIR}\`"
    echo
    echo "## Decision Questions"
    echo
    echo "1. Does Stage1 DPO + SFT Stage2 improve over SFT-48000 DiffuEraser on DAVIS?"
    echo "2. Does Stage2 DPO improve or degrade relative to Stage1 validation?"
    echo "3. Does region loss improve mask/boundary metrics?"
    echo "4. Does DPO-diag show less win_gap / loser-dominant collapse?"
    echo "5. Should next step be shorter checkpoint, no-lose-gap, target-domain SFT warmup, or stop DPO?"
  } >> "${REPORT}"

  cp "${REPORT}" "${OUTPUT_ROOT}/reports/exp08_d3_comp_regionloss_s1s2_2000_davis_report.md" 2>/dev/null || true
  echo "[Exp8] complete"
  echo "[Exp8] report=${REPORT}"
}

main "$@"
