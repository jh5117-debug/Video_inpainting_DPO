#!/usr/bin/env bash
set -uo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root" || exit 1

export LINGBOT_PROCESS_NAME="${LINGBOT_PROCESS_NAME:-lingbot-phy}"

if [ -f configs/paths/pai.detected.env ]; then
  # shellcheck disable=SC1091
  source configs/paths/pai.detected.env
fi

stamp="$(date +%Y%m%d_%H%M%S)"
out_root="${OUT_ROOT:-$repo_root/outputs/asset_smoke_tests/parallel_generation_smoke_${stamp}}"
mask_modes="${MASK_MODES:-full,partial}"
models_csv="${MODELS:-diffueraser,propainter,cococo,minimax_remover}"
timeout_sec="${TIMEOUT_SEC:-0}"

mkdir -p "$out_root"

echo "===== PARALLEL SINGLE-SAMPLE GENERATION SMOKE ====="
echo "out_root=$out_root"
echo "models=$models_csv"
echo "mask_modes=$mask_modes"
echo "timeout_sec=$timeout_sec"
echo

IFS=',' read -r -a models <<< "$models_csv"

gpu_for_model() {
  case "$1" in
    diffueraser) echo "${DIFFUERASER_GPU:-1}" ;;
    propainter) echo "${PROPAINTER_GPU:-2}" ;;
    cococo) echo "${COCOCO_GPU:-3}" ;;
    minimax_remover) echo "${MINIMAX_REMOVER_GPU:-4}" ;;
    *) echo "${DEFAULT_SMOKE_GPU:-1}" ;;
  esac
}

declare -A pids
declare -A logs
declare -A dirs

for raw_model in "${models[@]}"; do
  model="$(echo "$raw_model" | xargs | tr '-' '_')"
  [ -z "$model" ] && continue
  gpu="$(gpu_for_model "$model")"
  model_dir="$out_root/$model"
  stdout_log="$model_dir/stdout.log"
  mkdir -p "$model_dir"
  dirs["$model"]="$model_dir"
  logs["$model"]="$stdout_log"

  echo "[launch] model=$model gpu=$gpu log=$stdout_log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python tools/pai_videodpo_single_sample_generation_smoke.py \
      --models "$model" \
      --mask_modes "$mask_modes" \
      --output_root "$model_dir" \
      --report_path "$model_dir/report.md" \
      --manifest_path "$model_dir/smoke_manifest.jsonl" \
      --run_generation \
      --timeout_sec "$timeout_sec" \
      --print_log_tail_lines "${PRINT_LOG_TAIL_LINES:-80}"
  ) > "$stdout_log" 2>&1 &
  pids["$model"]=$!
done

echo
echo "===== WAIT ====="
overall=0
for model in "${!pids[@]}"; do
  pid="${pids[$model]}"
  if wait "$pid"; then
    echo "[done] model=$model status=OK"
  else
    rc=$?
    overall=1
    echo "[done] model=$model status=FAILED rc=$rc"
  fi
done

echo
echo "===== RESULT TABLES ====="
for raw_model in "${models[@]}"; do
  model="$(echo "$raw_model" | xargs | tr '-' '_')"
  model_dir="${dirs[$model]:-$out_root/$model}"
  stdout_log="${logs[$model]:-$model_dir/stdout.log}"
  echo
  echo "### $model"
  if [ -f "$model_dir/report.md" ]; then
    grep -E '^\| (diffueraser|propainter|cococo|minimax_remover) \|' "$model_dir/report.md" || true
  else
    echo "report missing: $model_dir/report.md"
  fi
  echo "-- stdout tail --"
  tail -80 "$stdout_log" 2>/dev/null || true
done

echo
echo "===== OUTPUTS ====="
find "$out_root" -maxdepth 3 -type f \( -name 'report.md' -o -name 'smoke_manifest.jsonl' -o -name 'stdout.log' -o -name '*.log' \) | sort

exit "$overall"
