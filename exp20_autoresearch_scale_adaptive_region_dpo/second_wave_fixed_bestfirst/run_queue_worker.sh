#!/usr/bin/env bash
set -u
GPU_ID="$1"
PORT="$2"
ROOT="exp20_autoresearch_scale_adaptive_region_dpo/second_wave_fixed_bestfirst"
LOG_ROOT="/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp20/second_wave_fixed_bestfirst"
RESULTS="$ROOT/results_raw.tsv"
mkdir -p "$LOG_ROOT" "$ROOT/running" "$ROOT/done" "$ROOT/crash"
while true; do
  q=$(find "$ROOT/queue" -maxdepth 1 -name "*.todo" | sort | head -1)
  if [ -z "$q" ]; then echo "[$(date -Is)] EMPTY_QUEUE gpu=$GPU_ID"; exit 0; fi
  running="$ROOT/running/$(basename "$q" .todo).running"
  if ! mv "$q" "$running" 2>/dev/null; then sleep 2; continue; fi
  cfg=$(cat "$running")
  trial=$(basename "$cfg" .json)
  log="$LOG_ROOT/${trial}.gpu${GPU_ID}.launcher.log"
  echo "[$(date -Is)] START trial=$trial gpu=$GPU_ID cfg=$cfg log=$log"
  flock -n "/tmp/exp20_gpu_${GPU_ID}.lock" /mnt/nas/hj/conda_envs/diffueraser/bin/python exp20_autoresearch_scale_adaptive_region_dpo/code/trial_runner.py --config "$cfg" --gpu-id "$GPU_ID" --main-process-port "$PORT" --results-tsv "$RESULTS" > "$log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then mv "$running" "$ROOT/done/$(basename "$running" .running).done"; echo "[$(date -Is)] DONE trial=$trial rc=$rc"; else echo "$rc" > "$ROOT/crash/$(basename "$running" .running).rc"; mv "$running" "$ROOT/crash/$(basename "$running" .running).crash"; echo "[$(date -Is)] CRASH trial=$trial rc=$rc"; fi
done
