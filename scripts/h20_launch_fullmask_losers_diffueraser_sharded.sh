#!/usr/bin/env bash
set -uo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root" || exit 1

export LINGBOT_PROCESS_NAME="${LINGBOT_PROCESS_NAME:-lingbot-world}"

if [ -f configs/paths/pai.detected.env ]; then
  # shellcheck disable=SC1091
  source configs/paths/pai.detected.env
fi

while [ "$#" -gt 0 ]; do
  case "$1" in
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --start_index|--start-index)
      START_INDEX="$2"
      shift 2
      ;;
    --end_index|--end-index)
      END_INDEX="$2"
      shift 2
      ;;
    --shard_size|--shard-size)
      SHARD_SIZE="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --workers_per_gpu|--workers-per-gpu)
      WORKERS_PER_GPU="$2"
      shift 2
      ;;
    --output_root|--output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    *)
      echo "[error] unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

first_existing() {
  for path in "$@"; do
    if [ -n "$path" ] && [ -e "$path" ]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

out_root="${OUTPUT_ROOT:-$repo_root/data/generated_losers/official_videodpo_diffueraser_data_fullmask_loser}"
shards_root="${SHARDS_ROOT:-$out_root/_shards}"
report_path="${REPORT_PATH:-$out_root/reports/generated_loser_fullmask_generation_report.md}"
selection_config="${SELECTION_CONFIG:-configs/generation/medium_hard_balanced_selection_v1.yaml}"
train_data_yaml="${VIDEO_DPO_TRAIN_DATA_YAML:-}"
if [ -z "$train_data_yaml" ]; then
  train_data_yaml="$(first_existing \
    /home/nvme01/H20_Video_inpainting_DPO/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml \
    /home/nvme01/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.pai.yaml \
    /home/nvme01/VideoDPO/configs/vc2_dpo/vidpro/train_data.yaml \
    /home/nvme01/H20_Video_inpainting_DPO/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.pai.yaml \
    /home/nvme01/H20_Video_inpainting_DPO/external/VideoDPO/configs/vc2_dpo/vidpro/train_data.yaml \
    /mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.pai.yaml || true)"
fi

models_csv="${MODELS:-diffueraser}"
if [ "$models_csv" != "diffueraser" ]; then
  echo "[error] D1省时版只允许 MODELS=diffueraser；不要在当前主线恢复 all-model generation。" >&2
  exit 2
fi

third_party_root="$(first_existing \
  "${THIRD_PARTY_VIDEO_INPAINTING_ROOT:-}" \
  /home/nvme01/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting \
  "$repo_root/third_party_video_inpainting" \
  /mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting || true)"
if [ -n "$third_party_root" ]; then
  export THIRD_PARTY_VIDEO_INPAINTING_ROOT="${THIRD_PARTY_VIDEO_INPAINTING_ROOT:-$third_party_root}"
fi

diffueraser_python="$(first_existing \
  "${DIFFUERASER_PYTHON:-}" \
  /home/nvme01/conda_envs/diffueraser/bin/python \
  "$(dirname "$repo_root")/conda_envs/diffueraser/bin/python" \
  /mnt/nas/hj/conda_envs/diffueraser/bin/python || true)"
export DIFFUERASER_PYTHON="${DIFFUERASER_PYTHON:-$diffueraser_python}"

orchestration_python="$(first_existing \
  "${PYTHON:-}" \
  "${VIDEODPO_PYTHON:-}" \
  "${PROPAINTER_PYTHON:-}" \
  /home/nvme01/conda_envs/videodpo/bin/python \
  /mnt/nas/hj/conda_envs/videodpo/bin/python \
  "$(command -v python 2>/dev/null || true)" \
  "$(command -v python3 2>/dev/null || true)" || true)"

base_model="$(first_existing \
  "${BASE_MODEL_PATH:-}" \
  "${third_party_root:-}/downloads/sd_inpaint_hf_extract/stable-diffusion-inpainting" \
  /home/nvme01/weights/stable-diffusion-inpainting \
  /mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/downloads/sd_inpaint_hf_extract/stable-diffusion-inpainting || true)"
if [ -n "$base_model" ]; then
  export BASE_MODEL_PATH="${BASE_MODEL_PATH:-$base_model}"
fi

vae_path="$(first_existing \
  "${VAE_PATH:-}" \
  /home/nvme01/weights/sd-vae-ft-mse \
  "$repo_root/weights/sd-vae-ft-mse" \
  /mnt/nas/hj/weights/sd-vae-ft-mse || true)"
if [ -n "$vae_path" ]; then
  export VAE_PATH="${VAE_PATH:-$vae_path}"
fi

diffueraser_weights="$(first_existing \
  "${DIFFUERASER_WEIGHT_ROOT:-}" \
  "${third_party_root:-}/weights/diffueraser/Orign_Diffueraser" \
  "${third_party_root:-}/weights/diffuEraser" \
  "$repo_root/weights/diffueraser" \
  /mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/diffueraser/Orign_Diffueraser || true)"
if [ -n "$diffueraser_weights" ]; then
  export DIFFUERASER_WEIGHT_ROOT="${DIFFUERASER_WEIGHT_ROOT:-$diffueraser_weights}"
fi

propainter_weights="$(first_existing \
  "${PROPAINTER_WEIGHT_ROOT:-}" \
  "${third_party_root:-}/weights/propainter" \
  /home/nvme01/data/third_party_video_inpainting/weights/propainter \
  /mnt/nas/hj/data/third_party_video_inpainting/weights/propainter || true)"
if [ -n "$propainter_weights" ]; then
  export PROPAINTER_WEIGHT_ROOT="${PROPAINTER_WEIGHT_ROOT:-$propainter_weights}"
fi

pcm_weights="$(first_existing \
  "${PCM_WEIGHTS_PATH:-}" \
  /home/nvme01/weights/PCM_Weights \
  "$repo_root/weights/PCM_Weights" \
  "${third_party_root:-}/weights/PCM_Weights" \
  "${third_party_root:-}/downloads/PCM_Weights" \
  /mnt/nas/hj/weights/PCM_Weights || true)"
if [ -n "$pcm_weights" ]; then
  export PCM_WEIGHTS_PATH="${PCM_WEIGHTS_PATH:-$pcm_weights}"
fi

gpus_csv="${GPUS:-0,1,2,3}"
workers_per_gpu="${WORKERS_PER_GPU:-1}"
shard_size="${SHARD_SIZE:-1}"
start_index="${START_INDEX:-0}"
end_index="${END_INDEX:-}"
limit="${LIMIT:-}"
seed="${SEED:-20260524}"
timeout_sec="${TIMEOUT_SEC:-7200}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export OPENCV_NUM_THREADS="${OPENCV_NUM_THREADS:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [ -z "$train_data_yaml" ] || [ ! -f "$train_data_yaml" ]; then
  echo "[error] VIDEO_DPO_TRAIN_DATA_YAML not found; run scripts/h20_audit_fullmask_generation_readiness.sh first" >&2
  exit 2
fi
if [ -z "${DIFFUERASER_PYTHON:-}" ] || [ ! -x "$DIFFUERASER_PYTHON" ]; then
  echo "[error] DIFFUERASER_PYTHON not found/executable; run scripts/h20_audit_fullmask_generation_readiness.sh first" >&2
  exit 2
fi
if [ -z "$orchestration_python" ] || [ ! -x "$orchestration_python" ]; then
  echo "[error] ORCHESTRATION_PYTHON not found/executable; run scripts/h20_audit_fullmask_generation_readiness.sh first" >&2
  exit 2
fi

IFS=',' read -r -a gpus <<< "$gpus_csv"
total_workers=$((${#gpus[@]} * workers_per_gpu))
if [ "$total_workers" -lt 1 ]; then
  echo "[error] no workers requested" >&2
  exit 2
fi

if [ -n "$limit" ]; then
  end_index=$((start_index + limit))
fi

if [ -z "$end_index" ]; then
  end_index="$(VIDEO_DPO_TRAIN_DATA_YAML="$train_data_yaml" "$orchestration_python" - <<'PY'
import os
from pathlib import Path
from tools.pai_videodpo_single_sample_generation_smoke import read_json, resolve_videodpo_roots
train_yaml = Path(os.environ["VIDEO_DPO_TRAIN_DATA_YAML"]).resolve()
root = resolve_videodpo_roots(train_yaml)[0]
print(len(read_json(root / "pair.json")))
PY
)"
fi

mkdir -p "$out_root/manifests" "$out_root/reports" "$out_root/logs" "$shards_root"

echo "===== H20 FULLMASK D1 DIFFUERASER-ONLY GENERATION ====="
echo "output_root=$out_root"
echo "shards_root=$shards_root"
echo "pair_range=[$start_index, $end_index)"
echo "gpus=$gpus_csv workers_per_gpu=$workers_per_gpu total_workers=$total_workers shard_size=$shard_size"
echo "models=$models_csv"
echo "mask_mode=full num_masks_per_video=1 comp=false"
echo "generation_source=diffueraser_only"
echo "process_name=$LINGBOT_PROCESS_NAME"
echo "train_data_yaml=$train_data_yaml"
echo "selection_policy=$selection_config"
echo "cpu_threads=OMP:$OMP_NUM_THREADS MKL:$MKL_NUM_THREADS OPENBLAS:$OPENBLAS_NUM_THREADS NUMEXPR:$NUMEXPR_NUM_THREADS OPENCV:$OPENCV_NUM_THREADS"
echo "ORCHESTRATION_PYTHON=$orchestration_python"
echo "DIFFUERASER_PYTHON=$DIFFUERASER_PYTHON"
echo "BASE_MODEL_PATH=${BASE_MODEL_PATH:-}"
echo "VAE_PATH=${VAE_PATH:-}"
echo "DIFFUERASER_WEIGHT_ROOT=${DIFFUERASER_WEIGHT_ROOT:-}"
echo "PROPAINTER_WEIGHT_ROOT=${PROPAINTER_WEIGHT_ROOT:-}"
echo "PCM_WEIGHTS_PATH=${PCM_WEIGHTS_PATH:-}"

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
    exec -a "$LINGBOT_PROCESS_NAME" "$orchestration_python" tools/videodpo_generated_loser_calibration.py \
      --output_root "$shard_root" \
      --models "$models_csv" \
      --mask_mode full \
      --limit 0 \
      --start_index "$shard_start" \
      --end_index "$shard_end" \
      --seed "$seed" \
      --timeout_sec "$timeout_sec" \
      --train_data_yaml "$train_data_yaml" \
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
  --mode full \
  --calibration_report "$report_path"

echo "===== SUMMARY ====="
"$orchestration_python" - <<PY
from pathlib import Path
root = Path("$out_root/manifests")
for name in [
    "candidates_all.jsonl",
    "candidates_all.scored.jsonl",
    "selected_primary_fullmask.jsonl",
    "selected_secondary_fullmask.jsonl",
    "selection_events.jsonl",
]:
    path = root / name
    print(name, sum(1 for _ in path.open()) if path.exists() else "MISSING")
PY
echo "report=$report_path"
