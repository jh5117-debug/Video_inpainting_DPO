#!/usr/bin/env bash
set -u
GPU="$1"
PORT="$2"
BASE="exp20_autoresearch_scale_adaptive_region_dpo/region_balanced_search"
LOGROOT="/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp20/region_balanced_search"
mkdir -p "$BASE/running" "$BASE/done" "$BASE/crash" "$LOGROOT"
LOCK="/tmp/exp20_region_balanced_gpu_${GPU}.flock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date -Is)] GPU $GPU lock busy" | tee -a "$LOGROOT/worker_gpu${GPU}.log"
  exit 0
fi
while true; do
  picked=""
  for q in "$BASE"/queue/*.json; do
    [ -e "$q" ] || break
    name=$(basename "$q")
    if mv "$q" "$BASE/running/${name}.gpu${GPU}.running" 2>/dev/null; then
      picked="$name"
      break
    fi
  done
  [ -n "$picked" ] || break
  cfg="$BASE/configs/$picked"
  tid="${picked%.json}"
  launch_log="$LOGROOT/${tid}.gpu${GPU}.launcher.log"
  echo "[$(date -Is)] START trial=$tid gpu=$GPU cfg=$cfg log=$launch_log" | tee -a "$LOGROOT/worker_gpu${GPU}.log"
  /mnt/nas/hj/conda_envs/diffueraser/bin/python exp20_autoresearch_scale_adaptive_region_dpo/code/trial_runner.py \
    --config "$cfg" \
    --results-tsv "$BASE/results_raw.tsv" \
    --gpu-id "$GPU" \
    --main-process-port "$PORT" \
    > "$launch_log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    mv "$BASE/running/${picked}.gpu${GPU}.running" "$BASE/done/${picked}.done" 2>/dev/null || true
    echo "[$(date -Is)] DONE trial=$tid rc=$rc" | tee -a "$LOGROOT/worker_gpu${GPU}.log"
  else
    mv "$BASE/running/${picked}.gpu${GPU}.running" "$BASE/crash/${picked}.crash" 2>/dev/null || true
    echo "[$(date -Is)] CRASH trial=$tid rc=$rc" | tee -a "$LOGROOT/worker_gpu${GPU}.log"
  fi
done
