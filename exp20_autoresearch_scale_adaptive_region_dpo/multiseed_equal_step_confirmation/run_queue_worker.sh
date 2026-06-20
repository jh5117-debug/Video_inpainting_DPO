#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${1:?gpu}"
PORT="${2:?port}"
BASE="exp20_autoresearch_scale_adaptive_region_dpo/multiseed_equal_step_confirmation"
RESULTS="$BASE/results_raw.tsv"
mkdir -p "$BASE/running" "$BASE/done" "$BASE/crash" "$BASE/launcher_logs"
while true; do
  cfg=$(find "$BASE/queue" -maxdepth 1 -type f -name '*.json' | sort | head -n 1 || true)
  [[ -n "$cfg" ]] || break
  name=$(basename "$cfg")
  if mv "$cfg" "$BASE/running/$name" 2>/dev/null; then
    echo "[worker gpu=$GPU_ID] running $name"
    set +e
    /mnt/nas/hj/conda_envs/diffueraser/bin/python exp20_autoresearch_scale_adaptive_region_dpo/code/trial_runner.py \
      --config "$BASE/running/$name" \
      --gpu-id "$GPU_ID" \
      --main-process-port "$PORT" \
      --results-tsv "$RESULTS" \
      > "$BASE/launcher_logs/${name%.json}.log" 2>&1
    rc=$?
    set -e
    if [[ "$rc" -eq 0 ]]; then mv "$BASE/running/$name" "$BASE/done/$name"; else mv "$BASE/running/$name" "$BASE/crash/$name"; fi
  fi
done
