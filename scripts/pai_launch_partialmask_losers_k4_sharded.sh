#!/usr/bin/env bash
set -uo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root" || exit 1

export LINGBOT_PROCESS_NAME="${LINGBOT_PROCESS_NAME:-lingbot-phy}"

if [ -f configs/paths/pai.detected.env ]; then
  # shellcheck disable=SC1091
  source configs/paths/pai.detected.env
fi

out_root="${OUTPUT_ROOT:-$repo_root/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4}"
shards_root="${SHARDS_ROOT:-$out_root/_shards}"
report_path="${REPORT_PATH:-$out_root/reports/generated_loser_full_generation_report.md}"
mask_policy_config="${MASK_POLICY_CONFIG:-configs/generation/videodpo_partialmask_policy_v1_medium_hard_k4.yaml}"
selection_config="${SELECTION_CONFIG:-configs/generation/medium_hard_balanced_selection_v1.yaml}"
train_data_yaml="${VIDEO_DPO_TRAIN_DATA_YAML:-/mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.pai.yaml}"
models_csv="${MODELS:-all}"

sd15_full="${SD15_FULL:-/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/downloads/sd_inpaint_hf_extract/stable-diffusion-inpainting}"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-$sd15_full}"
export PCM_WEIGHTS_PATH="${PCM_WEIGHTS_PATH:-/mnt/nas/hj/weights/PCM_Weights}"
export COCOCO_SD_INPAINT_ROOT="${COCOCO_SD_INPAINT_ROOT:-$sd15_full}"

default_diffueraser_python="/mnt/nas/hj/conda_envs/diffueraser/bin/python"
default_videodpo_python="/mnt/nas/hj/conda_envs/videodpo/bin/python"
default_minimax_python="/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/envs/minimax/bin/python"

if [ "${RESPECT_EXISTING_MODEL_PYTHONS:-0}" = "1" ]; then
  export DIFFUERASER_PYTHON="${DIFFUERASER_PYTHON:-$default_diffueraser_python}"
  export PROPAINTER_PYTHON="${PROPAINTER_PYTHON:-$default_videodpo_python}"
  export COCOCO_PYTHON="${COCOCO_PYTHON:-$default_videodpo_python}"
  export MINIMAX_REMOVER_PYTHON="${MINIMAX_REMOVER_PYTHON:-$default_minimax_python}"
else
  export DIFFUERASER_PYTHON="$default_diffueraser_python"
  export PROPAINTER_PYTHON="$default_videodpo_python"
  export COCOCO_PYTHON="$default_videodpo_python"
  export MINIMAX_REMOVER_PYTHON="$default_minimax_python"
fi

orchestration_python="${PYTHON:-$default_videodpo_python}"
if [ ! -x "$orchestration_python" ]; then
  orchestration_python="$(command -v python 2>/dev/null || command -v python3 2>/dev/null || true)"
fi
if [ -z "$orchestration_python" ] || [ ! -x "$orchestration_python" ]; then
  echo "[error] ORCHESTRATION_PYTHON not found/executable" >&2
  exit 2
fi

gpus_csv="${GPUS:-0,1,2,3,4,5,6}"
workers_per_gpu="${WORKERS_PER_GPU:-2}"
shard_size="${SHARD_SIZE:-10}"
start_index="${START_INDEX:-0}"
end_index="${END_INDEX:-}"
seed="${SEED:-20260524}"
timeout_sec="${TIMEOUT_SEC:-3600}"

# DiffuEraser/torch/opencv subprocesses otherwise fan out many CPU threads per
# worker. On PAI this can create thousands of runnable threads and leave GPUs
# idle while the host thrashes.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export OPENCV_NUM_THREADS="${OPENCV_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export DIFFUERASER_INFERENCE_STACK="${DIFFUERASER_INFERENCE_STACK:-or}"
export DIFFUERASER_PRIOR_MODE="${DIFFUERASER_PRIOR_MODE:-propainter}"

IFS=',' read -r -a gpus <<< "$gpus_csv"
total_workers=$((${#gpus[@]} * workers_per_gpu))

if [ "$total_workers" -lt 1 ]; then
  echo "[error] no workers requested" >&2
  exit 2
fi

if [ -z "$end_index" ]; then
  end_index="$("$orchestration_python" - <<'PY'
import os
from pathlib import Path
from tools.pai_videodpo_single_sample_generation_smoke import read_json, resolve_videodpo_roots

train_yaml = Path(os.environ.get("VIDEO_DPO_TRAIN_DATA_YAML", "/mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.pai.yaml")).resolve()
root = resolve_videodpo_roots(train_yaml)[0]
print(len(read_json(root / "pair.json")))
PY
)"
fi

mkdir -p "$out_root/manifests" "$out_root/reports" "$out_root/logs" "$shards_root"

echo "===== PAI PARTIALMASK K4 FULL GENERATION ====="
echo "out_root=$out_root"
echo "shards_root=$shards_root"
echo "pair_range=[$start_index, $end_index)"
echo "gpus=$gpus_csv workers_per_gpu=$workers_per_gpu total_workers=$total_workers shard_size=$shard_size"
echo "models=$models_csv"
echo "mask_mode=partial num_masks_per_video=4 comp=true"
echo "diffueraser_inference_stack=$DIFFUERASER_INFERENCE_STACK"
echo "diffueraser_prior_mode=$DIFFUERASER_PRIOR_MODE"
echo "mask_policy=$mask_policy_config"
echo "selection_policy=$selection_config"
echo "cpu_threads=OMP:$OMP_NUM_THREADS MKL:$MKL_NUM_THREADS OPENBLAS:$OPENBLAS_NUM_THREADS NUMEXPR:$NUMEXPR_NUM_THREADS OPENCV:$OPENCV_NUM_THREADS"
echo "ORCHESTRATION_PYTHON=$orchestration_python"
echo "DIFFUERASER_PYTHON=$DIFFUERASER_PYTHON"
echo "PROPAINTER_PYTHON=$PROPAINTER_PYTHON"
echo "COCOCO_PYTHON=$COCOCO_PYTHON"
echo "MINIMAX_REMOVER_PYTHON=$MINIMAX_REMOVER_PYTHON"

run_shard() {
  local gpu="$1"
  local shard_start="$2"
  local shard_end="$3"
  local shard_name
  shard_name="$(printf 'shard_%06d_%06d' "$shard_start" "$shard_end")"
  local shard_root="$shards_root/$shard_name"
  local stdout_log="$shard_root/stdout.log"
  local done_file="$shard_root/.done"
  local failed_file="$shard_root/.failed"

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
    export COCOCO_GPU="$gpu"
    export MINIMAX_REMOVER_GPU="$gpu"
    "$orchestration_python" tools/videodpo_generated_loser_calibration.py \
      --output_root "$shard_root" \
      --models "$models_csv" \
      --limit 0 \
      --start_index "$shard_start" \
      --end_index "$shard_end" \
      --seed "$seed" \
      --timeout_sec "$timeout_sec" \
      --train_data_yaml "$train_data_yaml" \
      --mask_policy_config "$mask_policy_config" \
      --selection_config "$selection_config" \
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
  local stride=$((total_workers * shard_size))
  local shard_start=$((start_index + worker_id * shard_size))
  while [ "$shard_start" -lt "$end_index" ]; do
    local shard_end=$((shard_start + shard_size))
    if [ "$shard_end" -gt "$end_index" ]; then
      shard_end="$end_index"
    fi
    run_shard "$gpu" "$shard_start" "$shard_end" || return $?
    shard_start=$((shard_start + stride))
  done
}

pids=()
worker_id=0
for gpu in "${gpus[@]}"; do
  for _slot in $(seq 1 "$workers_per_gpu"); do
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
  echo "[error] one or more shards failed; inspect:"
  find "$shards_root" -name .failed -print | sort
  exit 1
fi

echo "===== MERGE MANIFESTS ====="
find "$shards_root" -path "*/manifests/candidates_all.jsonl" -print | sort > "$out_root/manifests/shard_candidate_manifests.txt"
if [ ! -s "$out_root/manifests/shard_candidate_manifests.txt" ]; then
  echo "[error] no shard candidates_all.jsonl found" >&2
  exit 1
fi

xargs cat < "$out_root/manifests/shard_candidate_manifests.txt" > "$out_root/manifests/candidates_all.jsonl"
cp "$out_root/manifests/candidates_all.jsonl" "$out_root/manifests/candidates_all.scored.jsonl"

"$orchestration_python" tools/videodpo_loser_candidate_selection.py \
  --candidates_manifest "$out_root/manifests/candidates_all.jsonl" \
  --selection_config "$selection_config" \
  --output_dir "$out_root/manifests" \
  --mode partial \
  --calibration_report "$report_path"

echo "===== SUMMARY ====="
"$orchestration_python" - <<PY
from pathlib import Path
root = Path("$out_root/manifests")
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
echo "report=$report_path"
