#!/usr/bin/env bash
set -uo pipefail

# Generate Exp7-fix small-mask ProPainter-prior data on H20 GPUs 0-5.
# This script uses isolated shard output directories and merges manifests at
# the end; do not replace it with multiple writers to one candidates_all.jsonl.

PROJECT_ROOT="${PROJECT_ROOT:-/home/nvme01/H20_Video_inpainting_DPO}"
cd "$PROJECT_ROOT" || exit 1

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export OPENCV_NUM_THREADS="${OPENCV_NUM_THREADS:-1}"

PY="${PY:-/home/nvme01/conda_envs/diffueraser/bin/python}"
OUT_ROOT="${OUTPUT_ROOT:-data/generated_losers/exp07_fix_videodpo_smallmask15_20_prior_k4}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SHARDS_ROOT="${SHARDS_ROOT:-${OUT_ROOT}/_shards_gpu0_5_${RUN_TAG}}"
REPORT_PATH="${REPORT_PATH:-${OUT_ROOT}/reports/generated_loser_full_generation_gpus0_5_${RUN_TAG}.md}"
TRAIN_DATA_YAML="${VIDEO_DPO_TRAIN_DATA_YAML:-${PROJECT_ROOT}/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml}"
MASK_POLICY_CONFIG="${MASK_POLICY_CONFIG:-configs/generation/videodpo_partialmask_policy_v2_smallmask15_20_k4.yaml}"
SELECTION_CONFIG="${SELECTION_CONFIG:-configs/generation/medium_hard_balanced_selection_v1.yaml}"

GPUS_CSV="${GPUS:-0,1,2,3,4,5}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
SHARD_SIZE="${SHARD_SIZE:-5}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-1000}"
SEED="${SEED:-20260604}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"
MODELS="${MODELS:-diffueraser}"

export BASE_MODEL_PATH="${BASE_MODEL_PATH:-${PROJECT_ROOT}/weights/stable-diffusion-v1-5}"
export VAE_PATH="${VAE_PATH:-${PROJECT_ROOT}/weights/sd-vae-ft-mse}"
export DIFFUERASER_WEIGHT_ROOT="${DIFFUERASER_WEIGHT_ROOT:-${PROJECT_ROOT}/weights/diffuEraser/converted_weights_step48000}"
export PROPAINTER_WEIGHT_ROOT="${PROPAINTER_WEIGHT_ROOT:-${PROJECT_ROOT}/weights/propainter}"
export PCM_WEIGHTS_PATH="${PCM_WEIGHTS_PATH:-${PROJECT_ROOT}/weights/PCM_Weights}"
export DIFFUERASER_PYTHON="${DIFFUERASER_PYTHON:-$PY}"
export PROPAINTER_PYTHON="${PROPAINTER_PYTHON:-$PY}"
export VIDEO_DPO_TRAIN_DATA_YAML="$TRAIN_DATA_YAML"

mkdir -p "$OUT_ROOT/manifests" "$OUT_ROOT/reports" "$OUT_ROOT/logs" "$SHARDS_ROOT" logs/pipelines reports

for path in "$PY" "$TRAIN_DATA_YAML" "$MASK_POLICY_CONFIG" "$SELECTION_CONFIG" "$BASE_MODEL_PATH" "$VAE_PATH" "$DIFFUERASER_WEIGHT_ROOT" "$PROPAINTER_WEIGHT_ROOT" "$PCM_WEIGHTS_PATH"; do
  if [ ! -e "$path" ]; then
    echo "[exp07-fix-data-gpus0-5][ERROR] missing required path: $path" >&2
    exit 2
  fi
done

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
TOTAL_WORKERS=$((${#GPUS[@]} * WORKERS_PER_GPU))
if [ "$TOTAL_WORKERS" -lt 1 ]; then
  echo "[exp07-fix-data-gpus0-5][ERROR] no workers requested" >&2
  exit 2
fi

echo "===== H20 EXP07 FIX SMALLMASK DATA GENERATION GPUS 0-5 ====="
echo "project_root=$PROJECT_ROOT"
echo "out_root=$OUT_ROOT"
echo "shards_root=$SHARDS_ROOT"
echo "pair_range=[$START_INDEX, $END_INDEX)"
echo "gpus=$GPUS_CSV workers_per_gpu=$WORKERS_PER_GPU total_workers=$TOTAL_WORKERS shard_size=$SHARD_SIZE"
echo "models=$MODELS"
echo "mask_policy=$MASK_POLICY_CONFIG"
echo "selection_policy=$SELECTION_CONFIG"
echo "train_data_yaml=$TRAIN_DATA_YAML"
echo "diffueraser_weight_root=$DIFFUERASER_WEIGHT_ROOT"
echo "report=$REPORT_PATH"

run_shard() {
  local gpu="$1"
  local shard_start="$2"
  local shard_end="$3"
  local shard_name shard_root stdout_log done_file failed_file
  shard_name="$(printf 'shard_%06d_%06d' "$shard_start" "$shard_end")"
  shard_root="$SHARDS_ROOT/$shard_name"
  stdout_log="$shard_root/stdout.log"
  done_file="$shard_root/.done"
  failed_file="$shard_root/.failed"

  if [ -f "$done_file" ]; then
    echo "[skip] $shard_name gpu=$gpu already done"
    return 0
  fi

  rm -rf "$shard_root"
  mkdir -p "$shard_root"
  echo "[launch] $shard_name gpu=$gpu"
  (
    export DIFFUERASER_GPU="$gpu"
    export PROPAINTER_GPU="$gpu"
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PY" tools/videodpo_generated_loser_calibration.py \
      --output_root "$shard_root" \
      --models "$MODELS" \
      --limit 0 \
      --start_index "$shard_start" \
      --end_index "$shard_end" \
      --seed "$SEED" \
      --timeout_sec "$TIMEOUT_SEC" \
      --train_data_yaml "$TRAIN_DATA_YAML" \
      --mask_policy_config "$MASK_POLICY_CONFIG" \
      --selection_config "$SELECTION_CONFIG" \
      --calibration_report "$shard_root/generated_loser_calibration_report.md"
  ) >"$stdout_log" 2>&1

  local rc=$?
  if [ "$rc" -eq 0 ]; then
    touch "$done_file"
    echo "[done] $shard_name gpu=$gpu"
    return 0
  fi
  echo "$rc" > "$failed_file"
  echo "[failed] $shard_name gpu=$gpu rc=$rc log=$stdout_log" >&2
  tail -80 "$stdout_log" >&2 || true
  return "$rc"
}

worker_loop() {
  local worker_id="$1"
  local gpu="$2"
  local stride=$((TOTAL_WORKERS * SHARD_SIZE))
  local shard_start=$((START_INDEX + worker_id * SHARD_SIZE))
  while [ "$shard_start" -lt "$END_INDEX" ]; do
    local shard_end=$((shard_start + SHARD_SIZE))
    if [ "$shard_end" -gt "$END_INDEX" ]; then
      shard_end="$END_INDEX"
    fi
    run_shard "$gpu" "$shard_start" "$shard_end" || return $?
    shard_start=$((shard_start + stride))
  done
}

pids=()
worker_id=0
for gpu in "${GPUS[@]}"; do
  for _slot in $(seq 1 "$WORKERS_PER_GPU"); do
    worker_loop "$worker_id" "$gpu" &
    pids+=("$!")
    worker_id=$((worker_id + 1))
  done
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  echo "[exp07-fix-data-gpus0-5][ERROR] one or more shards failed; inspect:"
  find "$SHARDS_ROOT" -name .failed -print | sort
  exit 1
fi

echo "===== MERGE MANIFESTS ====="
find "$SHARDS_ROOT" -path "*/manifests/candidates_all.jsonl" -print | sort > "$OUT_ROOT/manifests/shard_candidate_manifests.txt"
if [ ! -s "$OUT_ROOT/manifests/shard_candidate_manifests.txt" ]; then
  echo "[exp07-fix-data-gpus0-5][ERROR] no shard candidates_all.jsonl found" >&2
  exit 1
fi

xargs cat < "$OUT_ROOT/manifests/shard_candidate_manifests.txt" > "$OUT_ROOT/manifests/candidates_all.jsonl"
cp "$OUT_ROOT/manifests/candidates_all.jsonl" "$OUT_ROOT/manifests/candidates_all.scored.jsonl"

"$PY" tools/videodpo_loser_candidate_selection.py \
  --candidates_manifest "$OUT_ROOT/manifests/candidates_all.jsonl" \
  --selection_config "$SELECTION_CONFIG" \
  --output_dir "$OUT_ROOT/manifests" \
  --mode partial \
  --calibration_report "$REPORT_PATH"

echo "===== SUMMARY ====="
"$PY" - <<PY
from pathlib import Path
root = Path("$OUT_ROOT/manifests")
for name in [
    "candidates_all.jsonl",
    "candidates_all.scored.jsonl",
    "selected_primary_comp.jsonl",
    "selected_primary_nocomp.jsonl",
    "selected_secondary_comp.jsonl",
    "selected_secondary_nocomp.jsonl",
    "selection_events.jsonl",
]:
    path = root / name
    print(name, sum(1 for _ in path.open()) if path.exists() else "MISSING")
PY
