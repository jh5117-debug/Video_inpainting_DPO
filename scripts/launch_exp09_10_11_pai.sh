#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
cd "${PROJECT_ROOT}"

RUN_EXPERIMENTS="${RUN_EXPERIMENTS:-exp9}"
RUN_VERSION="${RUN_VERSION:-$(date +%Y%m%d_%H%M%S)}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${OUTPUT_ROOT}/experiments}"

SOURCE_D3_ROOT="${SOURCE_D3_ROOT:-${WORKSPACE_ROOT}/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4}"
SOURCE_D3_MANIFEST="${SOURCE_D3_MANIFEST:-${SOURCE_D3_ROOT}/manifests/selected_primary_comp.repaired.pai_paths.jsonl}"
YTBV_ROOT="${YTBV_ROOT:-/mnt/workspace/hj/nas_hj/data/external/ytbv_2019_full_resolution/train}"
AUTO_PREPARE_GTWIN="${AUTO_PREPARE_GTWIN:-1}"

D3_ROOT="${D3_ROOT:-${WORKSPACE_ROOT}/data/generated_losers/exp09_10_11_youtubevos_gtwin_d3comp_pai}"
PREFERENCE_MANIFEST="${PREFERENCE_MANIFEST:-${D3_ROOT}/manifests/selected_primary_comp.gtwin.pai_paths.jsonl}"

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
CHECK_RAFT_LOAD="${CHECK_RAFT_LOAD:-1}"
ALLOW_HOME_NVME01_PATHS="${ALLOW_HOME_NVME01_PATHS:-0}"
LINGBOT_PROCESS_NAME="${LINGBOT_PROCESS_NAME:-lingbot-worldmodel}"
PROCESS_TITLE="${PROCESS_TITLE:-${LINGBOT_PROCESS_NAME}}"
DPO_STAGE1_ENTRYPOINT="${DPO_STAGE1_ENTRYPOINT:-training/dpo/lingbot-worldmodel-stage1.py}"
DPO_STAGE2_ENTRYPOINT="${DPO_STAGE2_ENTRYPOINT:-training/dpo/lingbot-worldmodel-stage2.py}"
export LINGBOT_PROCESS_NAME PROCESS_TITLE DPO_STAGE1_ENTRYPOINT DPO_STAGE2_ENTRYPOINT

CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/mnt/nas/hj/conda_envs/diffueraser}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PREFIX}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "[Exp9-11][ERROR] python not found; set PYTHON_BIN or CONDA_ENV_PREFIX" >&2
    exit 2
  fi
fi

NUM_GPUS="${NUM_GPUS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TRAIN_HEIGHT="${TRAIN_HEIGHT:-320}"
TRAIN_WIDTH="${TRAIN_WIDTH:-512}"
RESOLUTION="${RESOLUTION:-512}"
# Existing D3 generated-loser clips are 16-frame training clips. DiffuEraser /
# ProPainter validation requires effective duration >22, so DAVIS val uses 24.
NFRAMES="${NFRAMES:-16}"
DAVIS_VIDEO_LENGTH="${DAVIS_VIDEO_LENGTH:-24}"
DAVIS_NUM_QUAL="${DAVIS_NUM_QUAL:-30}"
EVAL_GPU="${EVAL_GPU:-0}"
EVAL_WIDTH="${EVAL_WIDTH:-432}"
EVAL_HEIGHT="${EVAL_HEIGHT:-240}"
COMPUTE_LPIPS="${COMPUTE_LPIPS:-0}"
COMPUTE_EWARP="${COMPUTE_EWARP:-0}"

NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-6}"
USE_PCM="${USE_PCM:-false}"
MASK_DILATION="${MASK_DILATION:-0}"
APPLY_GAUSSIAN_BLUR="${APPLY_GAUSSIAN_BLUR:-false}"
HARD_COMP="${HARD_COMP:-true}"

mkdir -p reports logs/pipelines "${OUTPUT_ROOT}/logs/target_eval" "${OUTPUT_ROOT}/reports"

MASTER_REPORT="${MASTER_REPORT:-reports/exp09_10_11_pai_pipeline_${RUN_VERSION}.md}"

die() {
  echo "[Exp9-11][ERROR] $*" >&2
  {
    echo
    echo "## FAILED"
    echo
    echo "- reason: $*"
  } >> "${MASTER_REPORT}" 2>/dev/null || true
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

validate_manifest() {
  "${PYTHON_BIN}" - "${PREFERENCE_MANIFEST}" "${ALLOW_HOME_NVME01_PATHS}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
allow_home_nvme01 = str(sys.argv[2]).lower() in {"1", "true", "yes", "y"}
if not path.exists():
    raise SystemExit(f"manifest missing: {path}")
rows = []
bad_nvme = 0
missing = []
same_paths = []
generated_win = []
sampled_missing_paths = []
required = ["win_video_path", "final_loser_video_path", "mask_path"]

with path.open("r", encoding="utf-8") as fh:
    for line_no, line in enumerate(fh, 1):
        if not line.strip():
            continue
        if "/home/nvme01/" in line:
            bad_nvme += 1
        row = json.loads(line)
        rows.append(row)
        miss = [k for k in required if not row.get(k)]
        if miss:
            missing.append((line_no, miss))
        win = str(row.get("win_video_path", "") or "")
        lose = str(row.get("final_loser_video_path", "") or "")
        if win and lose and win == lose:
            same_paths.append(line_no)
        lowered = win.lower().replace("\\", "/")
        win_source = str(row.get("win_source", "") or "").lower()
        source_is_gt = any(
            marker in win_source
            for marker in ["youtubevos_gt", "gt_aligned", "clean", "ground_truth"]
        )
        if not source_is_gt and any(
            token in lowered
            for token in ["/candidates/", "final_loser", "raw_loser", "comp_loser"]
        ):
            generated_win.append((line_no, win))
        if len(rows) <= 20:
            for key in required:
                value = row.get(key)
                if value and not Path(str(value)).exists():
                    sampled_missing_paths.append((line_no, key, str(value)))

if not rows:
    raise SystemExit(f"manifest has zero rows: {path}")
if bad_nvme and not allow_home_nvme01:
    raise SystemExit(f"manifest contains /home/nvme01 paths: count={bad_nvme}")
if missing:
    raise SystemExit(f"manifest missing required fields; examples={missing[:5]}")
if same_paths:
    raise SystemExit(f"win_video_path equals final_loser_video_path; examples={same_paths[:5]}")
if generated_win:
    raise SystemExit(
        "win_video_path looks generated instead of GT/clean; "
        f"examples={generated_win[:3]}"
    )
if sampled_missing_paths:
    raise SystemExit(f"sampled manifest paths missing; examples={sampled_missing_paths[:5]}")

print(f"[manifest-check] rows={len(rows)}")
print("[manifest-check] win_video_path accepted as GT/clean by field contract and path heuristics")
PY
}

prepare_gtwin_manifest_if_needed() {
  if [[ "${AUTO_PREPARE_GTWIN}" != "1" ]]; then
    return 0
  fi
  if [[ -f "${PREFERENCE_MANIFEST}" ]]; then
    local ok
    ok="$("${PYTHON_BIN}" - "${PREFERENCE_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
ok = 0
total = 0
with path.open("r", encoding="utf-8") as fh:
    for line in fh:
        if not line.strip():
            continue
        row = json.loads(line)
        total += 1
        source = str(row.get("win_source", "") or "").lower()
        win = str(row.get("win_video_path", "") or "").lower().replace("\\", "/")
        if ("youtubevos_gt" in source or "gt_aligned" in source or "ground_truth" in source) and "/candidates/" not in win:
            ok += 1
        if total >= 20:
            break
print("1" if total and ok == total else "0")
PY
)"
    if [[ "${ok}" == "1" ]]; then
      echo "[gtwin] existing GT-win manifest accepted: ${PREFERENCE_MANIFEST}"
      return 0
    fi
    echo "[gtwin] existing manifest is not GT-win; regenerating: ${PREFERENCE_MANIFEST}"
  fi

  require_path "${SOURCE_D3_MANIFEST}" "source D3 repaired PAI manifest"
  require_path "${YTBV_ROOT}" "YouTube-VOS train root"
  require_path "${PROJECT_ROOT}/tools/prepare_exp8c_gtwin_manifest.py" "GT-win manifest preparation tool"
  mkdir -p "${D3_ROOT}/manifests" reports
  echo "[gtwin] preparing GT-win manifest"
  echo "[gtwin] source=${SOURCE_D3_MANIFEST}"
  echo "[gtwin] output=${PREFERENCE_MANIFEST}"
  "${PYTHON_BIN}" tools/prepare_exp8c_gtwin_manifest.py \
    --source_manifest "${SOURCE_D3_MANIFEST}" \
    --youtubevos_train_root "${YTBV_ROOT}" \
    --output_root "${D3_ROOT}" \
    --output_manifest "${PREFERENCE_MANIFEST}" \
    --cache_root "${D3_ROOT}/gt_win_cache" \
    --link_mode symlink \
    --strict \
    --report_path "reports/exp09_10_11_gtwin_manifest_prepare_${RUN_VERSION}.md"
}

precheck_common() {
  echo "[Exp9-11] common precheck start"
  prepare_gtwin_manifest_if_needed
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
  require_path "${PROJECT_ROOT}/inference/metrics.py" "DAVIS metric backend"

  validate_manifest

  PROPAINTER_WEIGHT_ROOT="$(find_propainter_root)" || die "ProPainter weights not found; set PROPAINTER_WEIGHT_ROOT"
  export PROPAINTER_WEIGHT_ROOT
  if [[ -z "${RAFT_MODEL_PATH}" && -f "${PROPAINTER_WEIGHT_ROOT}/raft-things.pth" ]]; then
    RAFT_MODEL_PATH="${PROPAINTER_WEIGHT_ROOT}/raft-things.pth"
  fi
  require_path "${PROPAINTER_WEIGHT_ROOT}" "ProPainter prior weight root"
  require_path "${RAFT_MODEL_PATH}" "RAFT prior/metric weight"

  if [[ "${CHECK_RAFT_LOAD}" == "1" ]]; then
    "${PYTHON_BIN}" - "${RAFT_MODEL_PATH}" <<'PY'
import sys
import torch
from pathlib import Path
p = Path(sys.argv[1])
torch.load(p, map_location="cpu")
print(f"[raft-check] load_ok path={p} size={p.stat().st_size}")
PY
  fi

  grep -q "gap_normalization" training/dpo/train_stage1.py || die "Stage1 gap_normalization support missing"
  grep -q "gap_normalization" training/dpo/train_stage2.py || die "Stage2 gap_normalization support missing"
  grep -q "build_region_loss_weight_map" training/dpo/train_stage1.py || die "Stage1 region loss helper missing"
  grep -q "build_region_loss_weight_map" training/dpo/train_stage2.py || die "Stage2 region loss helper missing"
  grep -q -- "--propainter_model_dir" inference/run_BR.py || die "DAVIS inference wrapper does not expose ProPainter prior"
  grep -q -- "--num_inference_steps" inference/run_BR.py || die "DAVIS inference wrapper does not expose raw6 steps"
  grep -q -- "--use_pcm" inference/run_BR.py || die "DAVIS inference wrapper does not expose no-PCM switch"

  "${PYTHON_BIN}" -m py_compile \
    training/dpo/train_stage1.py \
    training/dpo/train_stage2.py \
    training/dpo/scripts/run_stage1.py \
    training/dpo/scripts/run_stage2.py \
    tools/run_inpainting_metric_eval.py \
    tools/build_diffueraser_dpoS1_sftS2_hybrid.py \
    inference/run_BR.py \
    inference/metrics.py
  bash -n training/dpo/scripts/03_dpo_stage1.sbatch
  bash -n training/dpo/scripts/03_dpo_stage2.sbatch

  cat > "${MASTER_REPORT}" <<EOF
# Exp9 / Exp10 / Exp11 PAI Pipeline Report

- generated_at: $(date)
- run_version: \`${RUN_VERSION}\`
- run_experiments: \`${RUN_EXPERIMENTS}\`
- default_run: \`exp9\`
- manifest: \`${PREFERENCE_MANIFEST}\`
- source_d3_manifest: \`${SOURCE_D3_MANIFEST}\`
- gtwin_auto_prepare: \`${AUTO_PREPARE_GTWIN}\`
- youtubevos_train_root: \`${YTBV_ROOT}\`
- manifest_rows: \`$(wc -l < "${PREFERENCE_MANIFEST}")\`
- winner_contract: \`win_video_path must be GT/clean and not generated\`
- loser_contract: \`final_loser_video_path\`
- train_mask_mode: \`partial\`
- mask_from_manifest: \`true\`
- validation_prior_mode: \`propainter\`
- propainter_weight_root: \`${PROPAINTER_WEIGHT_ROOT}\`
- raft_model_path: \`${RAFT_MODEL_PATH}\`
- diffueraser_sft_48000: \`${DIFFUERASER_WEIGHT_ROOT}\`
- davis_root: \`${DAVIS_ROOT}\`
- metric_wrapper: \`tools/run_inpainting_metric_eval.py\`
- metric_backend: \`inference/metrics.py\`
- vbench: \`not used\`
- process_name: \`${LINGBOT_PROCESS_NAME}\`
- stage1_entrypoint: \`${DPO_STAGE1_ENTRYPOINT}\`
- stage2_entrypoint: \`${DPO_STAGE2_ENTRYPOINT}\`
- eval_num_inference_steps: \`${NUM_INFERENCE_STEPS}\`
- eval_use_pcm: \`${USE_PCM}\`
- eval_mask_dilation: \`${MASK_DILATION}\`
- eval_apply_gaussian_blur: \`${APPLY_GAUSSIAN_BLUR}\`
- eval_hard_comp: \`${HARD_COMP}\`

EOF
}

latest_run_dir() {
  local stage="$1"
  local run_name="$2"
  find "${EXPERIMENTS_DIR}/dpo/${stage}" -maxdepth 1 -type d -name "*_${run_name}" -printf "%T@ %p\n" 2>/dev/null \
    | sort -n | tail -1 | cut -d' ' -f2-
}

summarize_diag_csv() {
  local csv_path="$1"
  local out_path="$2"
  local title="$3"
  "${PYTHON_BIN}" - "$csv_path" "$out_path" "$title" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
title = sys.argv[3]
rows = []
with csv_path.open(newline="", encoding="utf-8", errors="ignore") as fh:
    for row in csv.DictReader(fh):
        rows.append(row)

keys = [
    "dpo_loss", "anchored_total_loss", "implicit_acc", "raw_win_gap", "raw_lose_gap",
    "norm_win_gap", "norm_lose_gap", "norm_lose_gap_clipped", "mse_w", "ref_mse_w",
    "mse_l", "ref_mse_l", "mse_w_over_ref_mse_w", "mse_l_over_ref_mse_l",
    "loser_dominant_ratio", "sigma_term", "kl_divergence", "grad_norm",
    "mask_region_mse", "boundary_region_mse", "outside_region_mse",
    "mask_area_ratio", "boundary_area_ratio", "region_weight_sum",
]

def as_float(row, key):
    try:
        value = row.get(key, "")
        if value in {"", None}:
            return None
        return float(value)
    except Exception:
        return None

def percentile(vals, q):
    if not vals:
        return None
    vals = sorted(vals)
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac

def fmt(value):
    if value is None:
        return ""
    return f"{value:.6g}"

lines = [f"# {title} DPO Diagnostic Summary", "", f"- source: `{csv_path}`", f"- rows: `{len(rows)}`", ""]
if rows:
    lines.append("| metric | mean | p50 | p90 | max |")
    lines.append("|---|---:|---:|---:|---:|")
    for key in keys:
        vals = [v for row in rows for v in [as_float(row, key)] if v is not None]
        if not vals:
            continue
        lines.append(f"| `{key}` | {fmt(sum(vals)/len(vals))} | {fmt(percentile(vals, 0.5))} | {fmt(percentile(vals, 0.9))} | {fmt(max(vals))} |")
    lines.append("")
else:
    lines.append("status: empty")

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
import csv
import sys
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
    return sorted([p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])

def read_frames(path, is_mask=False, size=(432, 240)):
    w, h = size
    frames = []
    if path.is_dir():
        for p in image_files(path)[:max_frames]:
            flag = cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR
            arr = cv2.imread(str(p), flag)
            if arr is None:
                continue
            interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
            arr = cv2.resize(arr, (w, h), interpolation=interp)
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
                arr = cv2.resize(arr, (w, h), interpolation=interp)
                frames.append((arr > 0).astype(np.uint8) * 255)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                arr = cv2.resize(arr, (w, h), interpolation=interp)
                frames.append(arr)
        cap.release()
    return frames

def label_frame(frame, label):
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out

def overlay_mask(gt, mask):
    out = gt.copy()
    red = np.zeros_like(out)
    red[..., 0] = 255
    m = mask > 0
    out[m] = (0.55 * out[m] + 0.45 * red[m]).astype(np.uint8)
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
    try:
        imageio.mimsave(out_path, side_frames, fps=12, codec="libx264")
    except Exception as imageio_error:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        h, w = side_frames[0].shape[:2]
        writer = cv2.VideoWriter(str(out_path), fourcc, 12, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open cv2 VideoWriter for {out_path}") from imageio_error
        for frame in side_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
    qual_written += 1

pair_manifest.parent.mkdir(parents=True, exist_ok=True)
with pair_manifest.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=["sample_id", "model_label", "gt_video_path", "prediction_video_path", "mask_path"])
    writer.writeheader()
    writer.writerows(rows)

index = pair_manifest.parent / "index.html"
items = sorted(side_dir.glob("*.mp4"))
html = ["<!doctype html><html><head><meta charset='utf-8'><title>DAVIS side-by-side</title></head><body>"]
html.append(f"<h1>DAVIS side-by-side: {cur_label}</h1>")
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
  mkdir -p "${out_dir}"
  echo "[Exp9-11][DAVIS] ${label} weights=${weights_path} out=${out_dir}"
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
    --mask_dilation_iter "${MASK_DILATION}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --use_pcm "${USE_PCM}" \
    --apply_gaussian_blur "${APPLY_GAUSSIAN_BLUR}" \
    --hard_comp "${HARD_COMP}" \
    --save_comparison \
    --no_metrics \
    --base_model_path "${BASE_MODEL_PATH}" \
    --vae_path "${VAE_PATH}" \
    --diffueraser_path "${weights_path}" \
    --propainter_model_dir "${PROPAINTER_WEIGHT_ROOT}" \
    --pcm_weights_path "${PCM_WEIGHTS_PATH}" \
    --raft_model_path "${RAFT_MODEL_PATH}"
}

run_davis_validation() {
  local exp_tag="$1"
  local stage_label="$2"
  local current_label="$3"
  local current_weights="$4"
  local eval_out="$5"
  mkdir -p "${eval_out}/inference" "${eval_out}/metrics" "${eval_out}/side_by_side"
  echo "[${exp_tag}] ${stage_label} DAVIS validation current=${current_label}"
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
- num_inference_steps: \`${NUM_INFERENCE_STEPS}\`
- use_pcm: \`${USE_PCM}\`
- mask_dilation: \`${MASK_DILATION}\`
- apply_gaussian_blur: \`${APPLY_GAUSSIAN_BLUR}\`
- hard_comp: \`${HARD_COMP}\`
- VBench: not used
EOF
}

write_exp11_audit() {
  mkdir -p reports
  cat > reports/exp11_flow_prior_implementation_audit.md <<'EOF'
# Exp11 Flow / Prior Consistency Implementation Audit

status: blocked

Exp11 requires differentiable train-time flow/prior consistency on top of Exp10.
The current training loops consume winner/loser/condition/mask tensors and do
not expose a safe differentiable RAFT flow tensor or a verified x0/image-space
ProPainter-prior consistency target inside the diffusion DPO step.

Allowed implemented pieces:
- Exp10 region-local log-ratio DPO
- boundary-region weighted MSE diagnostics
- DAVIS inference with ProPainter prior

Blocked pieces:
- `L_flow` as train-time differentiable temporal warp consistency
- `L_prior` as train-time model-output vs ProPainter-prior consistency
- flow confidence statistics

Decision: do not launch Exp11 training until these pieces are explicitly
implemented and re-audited.
EOF
  echo "[Exp11] blocked; see reports/exp11_flow_prior_implementation_audit.md"
}

configure_experiment() {
  local exp="$1"
  MASK_REGION_WEIGHT="${MASK_REGION_WEIGHT:-1.0}"
  BOUNDARY_REGION_WEIGHT="${BOUNDARY_REGION_WEIGHT:-0.5}"
  OUTSIDE_REGION_WEIGHT="${OUTSIDE_REGION_WEIGHT:-0.05}"
  GAP_EPS="${GAP_EPS:-1e-6}"
  LOSE_GAP_CLIP_TAU="${LOSE_GAP_CLIP_TAU:-1.0}"
  BETA_DPO="${BETA_DPO:-10}"
  SFT_REG_WEIGHT="${SFT_REG_WEIGHT:-0.0}"
  LOSE_GAP_WEIGHT="${LOSE_GAP_WEIGHT:-0.25}"
  WINNER_ABS_REG_WEIGHT="${WINNER_ABS_REG_WEIGHT:-0.05}"
  WINNER_GAP_REG_WEIGHT="${WINNER_GAP_REG_WEIGHT:-1.0}"
  WINNER_GAP_REG_MARGIN="${WINNER_GAP_REG_MARGIN:-0.0}"
  WANDB_PROJECT="${WANDB_PROJECT:-DPO_Diffueraser_Exp09_10_11}"

  case "${exp}" in
    exp9)
      EXP_TAG="Exp9"
      EXP_DIR="exp9_logratio_gap_dpo"
      REG_DIR="experiment_registry/exp09_logratio_gap_dpo"
      EXP_NAME="exp9_logratio_gap_dpo_s1s2_2000_davis_pai"
      STAGE1_RUN_NAME="exp9_logratio_gap_dpo_s1_2000_davis_pai"
      STAGE2_RUN_NAME="exp9_logratio_gap_dpo_s2_2000_davis_pai"
      LOSS_REGION_MODE="full"
      GAP_NORMALIZATION="log_ratio"
      CURRENT_STAGE1_LABEL="Exp9_DPO-S1_SFT-S2"
      CURRENT_STAGE2_LABEL="Exp9_DPO-S1_DPO-S2"
      ;;
    exp10)
      EXP_TAG="Exp10"
      EXP_DIR="exp10_region_local_dpo"
      REG_DIR="experiment_registry/exp10_region_local_dpo"
      EXP_NAME="exp10_region_local_dpo_s1s2_2000_davis_pai"
      STAGE1_RUN_NAME="exp10_region_local_dpo_s1_2000_davis_pai"
      STAGE2_RUN_NAME="exp10_region_local_dpo_s2_2000_davis_pai"
      LOSS_REGION_MODE="region"
      GAP_NORMALIZATION="log_ratio"
      CURRENT_STAGE1_LABEL="Exp10_DPO-S1_SFT-S2"
      CURRENT_STAGE2_LABEL="Exp10_DPO-S1_DPO-S2"
      ;;
    exp11)
      EXP_TAG="Exp11"
      EXP_DIR="exp11_flow_prior_consistency_dpo"
      REG_DIR="experiment_registry/exp11_flow_prior_consistency_dpo"
      EXP_NAME="exp11_flow_prior_consistency_dpo_s1s2_2000_davis_pai"
      STAGE1_RUN_NAME="exp11_flow_prior_consistency_dpo_s1_2000_davis_pai"
      STAGE2_RUN_NAME="exp11_flow_prior_consistency_dpo_s2_2000_davis_pai"
      LOSS_REGION_MODE="region"
      GAP_NORMALIZATION="log_ratio"
      CURRENT_STAGE1_LABEL="Exp11_DPO-S1_SFT-S2"
      CURRENT_STAGE2_LABEL="Exp11_DPO-S1_DPO-S2"
      ;;
    *)
      die "unknown experiment '${exp}', expected exp9, exp10, or exp11"
      ;;
  esac
  require_path "${EXP_DIR}" "${EXP_TAG} experiment folder"
  require_path "${REG_DIR}" "${EXP_TAG} registry folder"
}

run_stage1() {
  echo "[${EXP_TAG}] Stage1 start: ${STAGE1_RUN_NAME}"
  local stage_dir="${EXPERIMENTS_DIR}/dpo/stage1/${RUN_VERSION}_${STAGE1_RUN_NAME}"
  export PROJECT_ROOT OUTPUT_ROOT EXPERIMENTS_DIR
  export DATA="${OUTPUT_ROOT}"
  export WEIGHTS_DIR DPO_DATA_ROOT="${D3_ROOT}" DPO_DATASET_TYPE="generated_loser_manifest"
  export PREFERENCE_MANIFEST TRAIN_MASK_MODE="partial" MASK_FROM_MANIFEST="true" LOSS_REGION_MODE
  export GAP_NORMALIZATION GAP_EPS LOSE_GAP_CLIP_TAU
  export MASK_REGION_WEIGHT BOUNDARY_REGION_WEIGHT OUTSIDE_REGION_WEIGHT
  export DPO_GAP_TRACE_CSV="${stage_dir}/dpo_gap_trace.csv"
  export DPO_GAP_SAMPLES_JSONL_GZ="${stage_dir}/dpo_gap_samples.jsonl.gz"
  export ENABLE_DPO_DIAG="true" DPO_DIAG_LOG_EVERY="${DPO_DIAG_LOG_EVERY:-10}" DPO_DIAG_SAVE_CSV="true" DPO_DIAG_SAVE_WANDB="${DPO_DIAG_SAVE_WANDB:-false}"
  export VAL_DATA_DIR="${DAVIS_ROOT}" REF_MODEL_PATH="${DIFFUERASER_WEIGHT_ROOT}"
  export RUN_NAME="${STAGE1_RUN_NAME}" RUN_VERSION NUM_GPUS CUDA_VISIBLE_DEVICES
  export MAX_STEPS="${MAX_STEPS:-2000}" CKPT_STEPS="${CKPT_STEPS:-500}" CKPT_LIMIT="${CKPT_LIMIT:-5}" VAL_STEPS="${VAL_STEPS:-999999}" LOGGING_STEPS="${LOGGING_STEPS:-10}"
  export TRAIN_HEIGHT TRAIN_WIDTH RESOLUTION NFRAMES
  export BETA_DPO SFT_REG_WEIGHT LOSE_GAP_WEIGHT DPO_LOSE_GAP_WEIGHT="${LOSE_GAP_WEIGHT}"
  export WINNER_ABS_REG_WEIGHT WINNER_GAP_REG_WEIGHT WINNER_GAP_REG_MARGIN
  export REPORT_TO="${REPORT_TO:-none}" WANDB_PROJECT CONDA_ENV_PREFIX
  export LINGBOT_PROCESS_NAME PROCESS_TITLE DPO_STAGE1_ENTRYPOINT DPO_STAGE2_ENTRYPOINT
  export MIXED_PRECISION="${MIXED_PRECISION:-bf16}" POLICY_DTYPE="${POLICY_DTYPE:-auto}" VAE_DTYPE="${VAE_DTYPE:-fp32}" REF_DTYPE="${REF_DTYPE:-bf16}" TEXT_DTYPE="${TEXT_DTYPE:-bf16}" SPLIT_POS_NEG_FORWARD="${SPLIT_POS_NEG_FORWARD:-1}"
  bash training/dpo/scripts/03_dpo_stage1.sbatch
}

run_stage2() {
  local stage1_last="$1"
  echo "[${EXP_TAG}] Stage2 start: ${STAGE2_RUN_NAME}"
  local stage_dir="${EXPERIMENTS_DIR}/dpo/stage2/${RUN_VERSION}_${STAGE2_RUN_NAME}"
  export PROJECT_ROOT OUTPUT_ROOT EXPERIMENTS_DIR
  export DATA="${OUTPUT_ROOT}"
  export WEIGHTS_DIR DPO_DATA_ROOT="${D3_ROOT}" DPO_DATASET_TYPE="generated_loser_manifest"
  export PREFERENCE_MANIFEST TRAIN_MASK_MODE="partial" MASK_FROM_MANIFEST="true" LOSS_REGION_MODE
  export GAP_NORMALIZATION GAP_EPS LOSE_GAP_CLIP_TAU
  export MASK_REGION_WEIGHT BOUNDARY_REGION_WEIGHT OUTSIDE_REGION_WEIGHT
  export DPO_GAP_TRACE_CSV="${stage_dir}/dpo_gap_trace.csv"
  export DPO_GAP_SAMPLES_JSONL_GZ="${stage_dir}/dpo_gap_samples.jsonl.gz"
  export ENABLE_DPO_DIAG="true" DPO_DIAG_LOG_EVERY="${DPO_DIAG_LOG_EVERY:-10}" DPO_DIAG_SAVE_CSV="true" DPO_DIAG_SAVE_WANDB="${DPO_DIAG_SAVE_WANDB:-false}"
  export VAL_DATA_DIR="${DAVIS_ROOT}" PRETRAINED_DPO_S1="${stage1_last}" BASELINE_UNET_PATH="${DIFFUERASER_WEIGHT_ROOT}" REF_MODEL_PATH="${DIFFUERASER_WEIGHT_ROOT}"
  export RUN_NAME="${STAGE2_RUN_NAME}" RUN_VERSION NUM_GPUS CUDA_VISIBLE_DEVICES
  export MAX_STEPS="${MAX_STEPS:-2000}" CKPT_STEPS="${CKPT_STEPS:-500}" CKPT_LIMIT="${CKPT_LIMIT:-5}" VAL_STEPS="${VAL_STEPS:-999999}" LOGGING_STEPS="${LOGGING_STEPS:-10}"
  export TRAIN_HEIGHT TRAIN_WIDTH RESOLUTION NFRAMES
  export BETA_DPO SFT_REG_WEIGHT LOSE_GAP_WEIGHT DPO_LOSE_GAP_WEIGHT="${LOSE_GAP_WEIGHT}"
  export WINNER_ABS_REG_WEIGHT WINNER_GAP_REG_WEIGHT WINNER_GAP_REG_MARGIN
  export REPORT_TO="${REPORT_TO:-none}" WANDB_PROJECT CONDA_ENV_PREFIX
  export LINGBOT_PROCESS_NAME PROCESS_TITLE DPO_STAGE1_ENTRYPOINT DPO_STAGE2_ENTRYPOINT
  export MIXED_PRECISION="${MIXED_PRECISION:-bf16}" POLICY_DTYPE="${POLICY_DTYPE:-auto}" VAE_DTYPE="${VAE_DTYPE:-fp32}" REF_DTYPE="${REF_DTYPE:-bf16}" TEXT_DTYPE="${TEXT_DTYPE:-bf16}" SPLIT_POS_NEG_FORWARD="${SPLIT_POS_NEG_FORWARD:-1}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  bash training/dpo/scripts/03_dpo_stage2.sbatch
}

run_one_experiment() {
  local exp="$1"
  configure_experiment "${exp}"
  echo "[${EXP_TAG}] configured: ${EXP_NAME}"
  if [[ "${exp}" == "exp11" ]]; then
    write_exp11_audit
    if [[ "${EXP11_ENABLE_TRAINING:-0}" != "1" ]]; then
      die "Exp11 train-time flow/prior consistency audit is blocked; set EXP11_ENABLE_TRAINING=1 only after implementing audited losses"
    fi
  fi

  {
    echo
    echo "## ${EXP_TAG}"
    echo
    echo "- exp_name: \`${EXP_NAME}\`"
    echo "- stage1_run_name: \`${STAGE1_RUN_NAME}\`"
    echo "- stage2_run_name: \`${STAGE2_RUN_NAME}\`"
    echo "- loss_region_mode: \`${LOSS_REGION_MODE}\`"
    echo "- gap_normalization: \`${GAP_NORMALIZATION}\`"
    echo "- lose_gap_clip_tau: \`${LOSE_GAP_CLIP_TAU}\`"
    echo "- winner_abs_reg_weight: \`${WINNER_ABS_REG_WEIGHT}\`"
    echo "- winner_gap_reg_weight: \`${WINNER_GAP_REG_WEIGHT}\`"
  } >> "${MASTER_REPORT}"

  run_stage1
  STAGE1_RUN_DIR="$(latest_run_dir stage1 "${STAGE1_RUN_NAME}")"
  require_path "${STAGE1_RUN_DIR}" "${EXP_TAG} Stage1 run dir"
  STAGE1_LAST="${STAGE1_RUN_DIR}/last_weights"
  require_path "${STAGE1_LAST}/unet_main/config.json" "${EXP_TAG} Stage1 last_weights unet_main"
  require_path "${STAGE1_LAST}/brushnet/config.json" "${EXP_TAG} Stage1 last_weights brushnet"
  require_path "${STAGE1_RUN_DIR}/dpo_diagnostics.csv" "${EXP_TAG} Stage1 dpo_diagnostics.csv"
  summarize_diag_csv "${STAGE1_RUN_DIR}/dpo_diagnostics.csv" "${EXP_DIR}/dpo_diag_summary.md" "${EXP_TAG} Stage1"
  cp "${EXP_DIR}/dpo_diag_summary.md" "${REG_DIR}/dpo_diag_summary.md"

  HYBRID_DIR="${OUTPUT_ROOT}/experiments/hybrid/${RUN_VERSION}_${STAGE1_RUN_NAME}_dpoS1_sftS2"
  echo "[${EXP_TAG}] build Stage1 hybrid: ${HYBRID_DIR}"
  "${PYTHON_BIN}" tools/build_diffueraser_dpoS1_sftS2_hybrid.py \
    --dpo_stage1_weights "${STAGE1_LAST}" \
    --sft_stage2_weights "${SFT_STAGE2_WEIGHTS}" \
    --output_dir "${HYBRID_DIR}" \
    --strict false \
    --report_path "reports/${EXP_NAME}_stage1_hybrid_key_merge_report.md"
  require_path "${HYBRID_DIR}/last_weights/unet_main/config.json" "${EXP_TAG} Stage1 hybrid full weights"

  STAGE1_VAL_DIR="${OUTPUT_ROOT}/logs/target_eval/${EXP_NAME}_stage1_val_davis_${RUN_VERSION}"
  run_davis_validation "${EXP_TAG}" "${EXP_TAG} Stage1 DPO + SFT Stage2" "${CURRENT_STAGE1_LABEL}" "${HYBRID_DIR}/last_weights" "${STAGE1_VAL_DIR}"
  cp "${STAGE1_VAL_DIR}/metrics/summary.md" "${EXP_DIR}/metric_summary.md" 2>/dev/null || true
  cp "${STAGE1_VAL_DIR}/metrics/summary.md" "${REG_DIR}/metric_summary.md" 2>/dev/null || true

  run_stage2 "${STAGE1_LAST}"
  STAGE2_RUN_DIR="$(latest_run_dir stage2 "${STAGE2_RUN_NAME}")"
  require_path "${STAGE2_RUN_DIR}" "${EXP_TAG} Stage2 run dir"
  STAGE2_LAST="${STAGE2_RUN_DIR}/last_weights"
  require_path "${STAGE2_LAST}/unet_main/config.json" "${EXP_TAG} Stage2 last_weights unet_main"
  require_path "${STAGE2_LAST}/brushnet/config.json" "${EXP_TAG} Stage2 last_weights brushnet"
  require_path "${STAGE2_RUN_DIR}/dpo_diagnostics.csv" "${EXP_TAG} Stage2 dpo_diagnostics.csv"
  summarize_diag_csv "${STAGE2_RUN_DIR}/dpo_diagnostics.csv" "reports/${EXP_NAME}_stage2_dpo_diag_summary.md" "${EXP_TAG} Stage2"

  STAGE2_VAL_DIR="${OUTPUT_ROOT}/logs/target_eval/${EXP_NAME}_stage2_val_davis_${RUN_VERSION}"
  run_davis_validation "${EXP_TAG}" "${EXP_TAG} Stage1 DPO + Stage2 DPO" "${CURRENT_STAGE2_LABEL}" "${STAGE2_LAST}" "${STAGE2_VAL_DIR}"

  cat > "${EXP_DIR}/status.md" <<EOF
# ${EXP_TAG} Status

status: complete
updated_at: $(date)

- stage1_run_dir: \`${STAGE1_RUN_DIR}\`
- stage1_dpo_diag: \`${STAGE1_RUN_DIR}/dpo_diagnostics.csv\`
- stage1_gap_trace: \`${STAGE1_RUN_DIR}/dpo_gap_trace.csv\`
- stage1_gap_samples: \`${STAGE1_RUN_DIR}/dpo_gap_samples.jsonl.gz\`
- stage1_val: \`${STAGE1_VAL_DIR}\`
- stage2_run_dir: \`${STAGE2_RUN_DIR}\`
- stage2_dpo_diag: \`${STAGE2_RUN_DIR}/dpo_diagnostics.csv\`
- stage2_gap_trace: \`${STAGE2_RUN_DIR}/dpo_gap_trace.csv\`
- stage2_gap_samples: \`${STAGE2_RUN_DIR}/dpo_gap_samples.jsonl.gz\`
- stage2_val: \`${STAGE2_VAL_DIR}\`
EOF
  cp "${EXP_DIR}/status.md" "${REG_DIR}/status.md"

  {
    echo
    echo "### ${EXP_TAG} outputs"
    echo
    echo "- stage1_run_dir: \`${STAGE1_RUN_DIR}\`"
    echo "- stage1_dpo_diag: \`${STAGE1_RUN_DIR}/dpo_diagnostics.csv\`"
    echo "- stage1_gap_trace: \`${STAGE1_RUN_DIR}/dpo_gap_trace.csv\`"
    echo "- stage1_gap_samples: \`${STAGE1_RUN_DIR}/dpo_gap_samples.jsonl.gz\`"
    echo "- stage1_val: \`${STAGE1_VAL_DIR}\`"
    echo "- stage2_run_dir: \`${STAGE2_RUN_DIR}\`"
    echo "- stage2_dpo_diag: \`${STAGE2_RUN_DIR}/dpo_diagnostics.csv\`"
    echo "- stage2_gap_trace: \`${STAGE2_RUN_DIR}/dpo_gap_trace.csv\`"
    echo "- stage2_gap_samples: \`${STAGE2_RUN_DIR}/dpo_gap_samples.jsonl.gz\`"
    echo "- stage2_val: \`${STAGE2_VAL_DIR}\`"
  } >> "${MASTER_REPORT}"
}

main() {
  precheck_common
  IFS=',' read -r -a experiments <<< "${RUN_EXPERIMENTS}"
  local exp
  for exp in "${experiments[@]}"; do
    exp="$(echo "${exp}" | xargs)"
    [[ -z "${exp}" ]] && continue
    run_one_experiment "${exp}"
  done
  echo >> "${MASTER_REPORT}"
  echo "status: complete" >> "${MASTER_REPORT}"
  cp "${MASTER_REPORT}" "${OUTPUT_ROOT}/reports/exp09_10_11_pai_pipeline_${RUN_VERSION}.md" 2>/dev/null || true
  echo "[Exp9-11] complete"
  echo "[Exp9-11] report=${MASTER_REPORT}"
}

main "$@"
