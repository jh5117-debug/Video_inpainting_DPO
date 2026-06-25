#!/usr/bin/env bash
set -euo pipefail

# Exp26 right-side only shadow-dev confirmatory validation.
# This script never touches the left CLI runtime except read-only audit.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CODE_DIR="$PROJECT_ROOT/exp26_videopainter_dpo_v2/code"
MANIFEST_DIR="$PROJECT_ROOT/exp26_videopainter_dpo_v2/manifests"
RUN_ROOT="${RUN_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/shadowdev_confirmatory_20260625}"
TRAIN_ROOT="${TRAIN_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp26_videopainter_dpo_v2/vp_primary32_50step_20260625_171032}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp26_shadowdev_confirmatory_20260625}"
LEFT_RUNTIME="${LEFT_RUNTIME:-/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/cli4}"
VP_ROOT="${VP_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter}"
VP_BASE="${VP_BASE:-$VP_ROOT/ckpt/CogVideoX-5b-I2V}"
STEP0_CKPT="${STEP0_CKPT:-$VP_ROOT/ckpt/VideoPainter/checkpoints/branch}"
SHADOW_SOURCE_MANIFEST="${SHADOW_SOURCE_MANIFEST:-$MANIFEST_DIR/vp2_vor_bg_shadow_dev_32.jsonl}"
SEED="${SEED:-20260619}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$RUN_ROOT" "$RUNTIME_ROOT" "$PROJECT_ROOT/reports"

MONITOR="$RUN_ROOT/monitor_5min.csv"
PROCESS_REGISTRY="$RUN_ROOT/process_registry.csv"
LEFT_AUDIT_MD="$PROJECT_ROOT/reports/exp26_shadowdev_left_cli_protection_audit.md"
LEFT_AUDIT_CSV="$PROJECT_ROOT/reports/exp26_shadowdev_left_cli_protection_audit.csv"

log() {
  printf '[%s] %s\n' "$(date -Ins)" "$*" | tee -a "$RUN_ROOT/controller.log"
}

append_monitor() {
  local task="$1"; shift
  local gpu="$1"; shift
  local pid="$1"; shift
  local msg="$1"; shift || true
  if [[ ! -f "$MONITOR" ]]; then
    echo "time,left_reserved_gpus,right_eligible_gpus,right_running_gpus,other_occupied_gpus,task,gpu,pid,pgid,vram_used,utilization,heartbeat,last_error,next_action,nas_free" > "$MONITOR"
  fi
  local pgid=""
  if [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1; then
    pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
  fi
  local query
  query="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';' || true)"
  local free
  free="$(df -h /mnt/nas/hj/H20_Video_inpainting_DPO 2>/dev/null | awk 'NR==2{print $4}' || true)"
  echo "$(date -Ins),\"$LEFT_RESERVED_GPUS\",\"$RIGHT_ELIGIBLE_GPUS\",\"$RIGHT_RUNNING_GPUS\",\"$OTHER_OCCUPIED_GPUS\",\"$task\",\"$gpu\",\"$pid\",\"$pgid\",\"$query\",\"\",\"$RUNTIME_ROOT\",\"$msg\",\"$free\"" >> "$MONITOR"
}

left_cli_audit() {
  log "left CLI read-only audit"
  {
    echo "# Exp26 Shadow-Dev Left CLI Protection Audit"
    echo
    echo "- hostname: \`$(hostname)\`"
    echo "- date: \`$(date -Ins)\`"
    echo "- left_runtime: \`$LEFT_RUNTIME\`"
    echo
    echo "## Locks"
    echo
    find "$LEFT_RUNTIME" -maxdepth 2 -type f -printf '%p\n' 2>/dev/null | sort | sed 's/^/- `/' | sed 's/$/`/' || true
    echo
    echo "## GPU Snapshot"
    echo
    echo '```'
    nvidia-smi || true
    echo '```'
  } > "$LEFT_AUDIT_MD"

  {
    echo "pid,ppid,pgid,sid,gpu,cwd,cmdline,left_cli_match"
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u | while read -r pid; do
      [[ -z "$pid" ]] && continue
      local cmd cwd ppid pgid sid gpu match
      cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
      cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
      ppid="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
      pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
      sid="$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
      gpu="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null | awk -v p="$pid" '$0 ~ p {print $1}' | paste -sd ';' -)"
      match="false"
      if [[ "$cmd $cwd" == *cli4* || "$cmd $cwd" == *exp25_cli4* || "$cmd $cwd" == *exp27_cli4* || "$cmd $cwd" == *exp28_fine_inner_boundary_sweep* ]]; then
        match="true"
      fi
      printf '"%s","%s","%s","%s","%s","%s","%s","%s"\n' "$pid" "$ppid" "$pgid" "$sid" "$gpu" "$cwd" "$cmd" "$match"
    done
  } > "$LEFT_AUDIT_CSV"
}

gpu_snapshot_file() {
  local out="$1"
  {
    echo "date=$(date -Ins)"
    nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv
    echo "--- compute apps ---"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv || true
    echo "--- pmon ---"
    nvidia-smi pmon -c 1 || true
    echo "--- dmesg xid ---"
    dmesg 2>/dev/null | grep -Ei 'NVRM|Xid|CUDA' | tail -50 || true
  } > "$out"
}

eligible_gpus_once() {
  local out="$1"
  : > "$out"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | while IFS=',' read -r idx mem util; do
    idx="$(echo "$idx" | xargs)"
    mem="$(echo "$mem" | xargs)"
    util="$(echo "$util" | xargs)"
    case "$idx" in
      0|5|6|7) ;;
      *) continue ;;
    esac
    if [[ "$mem" -gt 1024 || "$util" -gt 5 ]]; then
      continue
    fi
    local has_pid="false"
    local uuid
    uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F',' -v i="$idx" '$1 ~ i {gsub(/ /,"",$2); print $2}')"
    if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null | grep -q "$uuid"; then
      has_pid="true"
    fi
    if [[ "$has_pid" == "false" ]]; then
      echo "$idx" >> "$out"
    fi
  done
}

refresh_eligible_gpus() {
  gpu_snapshot_file "$RUN_ROOT/gpu_snapshot_a.txt"
  eligible_gpus_once "$RUN_ROOT/eligible_a.txt"
  sleep 60
  gpu_snapshot_file "$RUN_ROOT/gpu_snapshot_b.txt"
  eligible_gpus_once "$RUN_ROOT/eligible_b.txt"
  RIGHT_EXP26_ELIGIBLE_GPUS="$(comm -12 <(sort "$RUN_ROOT/eligible_a.txt") <(sort "$RUN_ROOT/eligible_b.txt") | paste -sd',' -)"
  LEFT_RESERVED_GPUS="1,2,3,4"
  RIGHT_ELIGIBLE_GPUS="$RIGHT_EXP26_ELIGIBLE_GPUS"
  RIGHT_RUNNING_GPUS=""
  OTHER_OCCUPIED_GPUS="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '$2+0>1024 {gsub(/ /,"",$1); print $1}' | grep -v -E '^(1|2|3|4)$' | paste -sd',' - || true)"
  log "RIGHT_EXP26_ELIGIBLE_GPUS=${RIGHT_EXP26_ELIGIBLE_GPUS:-none}"
}

prepare_shadow_manifest() {
  log "prepare shadow-dev extraction/materialization/masks"
  if [[ ! -s "$RUN_ROOT/gate64_mask_ready.jsonl" ]]; then
    if [[ ! -s "$RUN_ROOT/shadowdev_extracted.jsonl" ]]; then
      "$PYTHON_BIN" "$CODE_DIR/extract_vp2_gate64_vor_bg.py" \
        --manifest "$SHADOW_SOURCE_MANIFEST" \
        --output-root "$RUN_ROOT/extracted" \
        --output-manifest "$RUN_ROOT/shadowdev_extracted.jsonl" \
        --status-csv "$RUN_ROOT/shadowdev_extraction_status.csv" \
        --limit 32
    fi
    if [[ ! -s "$RUN_ROOT/shadowdev_materialized_49f.jsonl" ]]; then
      "$PYTHON_BIN" "$CODE_DIR/materialize_vp2_49f_sources.py" \
        --manifest "$RUN_ROOT/shadowdev_extracted.jsonl" \
        --source-root "$RUN_ROOT/extracted" \
        --output-root "$RUN_ROOT/materialized_49f" \
        --output-manifest "$RUN_ROOT/shadowdev_materialized_49f.jsonl" \
        --status-csv "$RUN_ROOT/shadowdev_materialized_49f_status.csv" \
        --num-frames 49
    fi
    "$PYTHON_BIN" "$CODE_DIR/generate_vp2_moving_br_masks.py" \
      --materialized-manifest "$RUN_ROOT/shadowdev_materialized_49f.jsonl" \
      --output-root "$RUN_ROOT/masks" \
      --output-manifest "$RUN_ROOT/gate64_mask_ready.jsonl" \
      --status-csv "$RUN_ROOT/gate64_mask_status.csv" \
      --seed 20260623 \
      --first-frame-gt
  fi
  "$PYTHON_BIN" "$CODE_DIR/shadowdev_confirmatory_analysis.py" --run-root "$RUN_ROOT" preregister
}

checkpoint_path_for() {
  case "$1" in
    step0) echo "$STEP0_CKPT" ;;
    step10) echo "$TRAIN_ROOT/checkpoint-10" ;;
    step30) echo "$TRAIN_ROOT/checkpoint-30" ;;
    step50) echo "$TRAIN_ROOT/checkpoint-50" ;;
    *) return 2 ;;
  esac
}

run_checkpoint() {
  local step="$1"
  local gpu="$2"
  local ckpt
  ckpt="$(checkpoint_path_for "$step")"
  local step_root="$RUN_ROOT/$step"
  local out="$step_root/official_generation"
  local lock="$RUNTIME_ROOT/exp26_shadowdev_${step}.lock"
  local log_path="$RUN_ROOT/${step}.log"
  mkdir -p "$step_root" "$RUNTIME_ROOT"
  if [[ -s "$out/gate64_generation_summary.json" ]] && grep -q '"status": "passed"' "$out/gate64_generation_summary.json"; then
    log "$step already complete"
    return 0
  fi
  (
    flock -n 9 || { echo "lock busy: $lock"; exit 75; }
    echo "$BASHPID" > "$RUNTIME_ROOT/${step}.pid"
    echo "$gpu" > "$RUNTIME_ROOT/${step}.gpu"
    echo "running" > "$RUNTIME_ROOT/${step}.state"
    while true; do
      date -Ins > "$RUNTIME_ROOT/${step}.heartbeat"
      sleep 30
    done &
    hb=$!
    trap 'kill "$hb" 2>/dev/null || true' EXIT
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$CODE_DIR/run_vp2_gate64_official_generation.py" \
      --videopainter-root "$VP_ROOT" \
      --base-model "$VP_BASE" \
      --branch-checkpoint "$ckpt" \
      --manifest "$RUN_ROOT/gate64_mask_ready.jsonl" \
      --output-dir "$out" \
      --limit 32 \
      --height 480 \
      --width 720 \
      --num-frames 49 \
      --num-inference-steps 20 \
      --guidance-scale 6.0 \
      --seed "$SEED" \
      --dtype bf16 \
      --device cuda
    "$PYTHON_BIN" "$CODE_DIR/review_gate64_official_outputs.py" \
      --run-root "$step_root" \
      --manifest "$RUN_ROOT/gate64_mask_ready.jsonl" \
      --output-dir "$step_root/${step}_review" \
      --num-frames 49
    echo "completed" > "$RUNTIME_ROOT/${step}.state"
  ) 9>"$lock" >"$log_path" 2>&1 &
  local pid=$!
  echo "$pid" > "$RUNTIME_ROOT/${step}.launcher.pid"
  local pgid
  pgid="$(ps -o pgid= -p "$pid" | tr -d ' ' || true)"
  if [[ ! -f "$PROCESS_REGISTRY" ]]; then
    echo "time,checkpoint,gpu,pid,pgid,checkpoint_path,log_path,output_root" > "$PROCESS_REGISTRY"
  fi
  echo "$(date -Ins),$step,$gpu,$pid,$pgid,$ckpt,$log_path,$step_root" >> "$PROCESS_REGISTRY"
  log "started $step on GPU$gpu pid=$pid"
}

wait_for_checkpoints() {
  local pids=("$@")
  local still=1
  while [[ "$still" -eq 1 ]]; do
    still=0
    RIGHT_RUNNING_GPUS=""
    for pid in "${pids[@]}"; do
      if ps -p "$pid" >/dev/null 2>&1; then
        still=1
      fi
    done
    append_monitor "checkpoint_inference" "${RIGHT_RUNNING_GPUS:-}" "" "waiting"
    sleep 300
  done
  local failed=0
  for step in step0 step10 step30 step50; do
    if [[ ! -s "$RUN_ROOT/$step/official_generation/gate64_generation_summary.json" ]] || ! grep -q '"status": "passed"' "$RUN_ROOT/$step/official_generation/gate64_generation_summary.json"; then
      log "FAILED checkpoint generation: $step"
      failed=1
    fi
  done
  return "$failed"
}

run_metric_and_audits() {
  log "run leakage, metrics, paired stats, dynamics"
  "$PYTHON_BIN" "$CODE_DIR/shadowdev_confirmatory_analysis.py" --run-root "$RUN_ROOT" leakage || true
  CUDA_VISIBLE_DEVICES="${METRIC_GPU:-0}" "$PYTHON_BIN" "$CODE_DIR/shadowdev_confirmatory_analysis.py" --run-root "$RUN_ROOT" --device cuda metrics
  "$PYTHON_BIN" "$CODE_DIR/shadowdev_confirmatory_analysis.py" --run-root "$RUN_ROOT" stats
  "$PYTHON_BIN" "$CODE_DIR/shadowdev_confirmatory_analysis.py" --train-root "$TRAIN_ROOT" dynamics
}

main() {
  cd "$PROJECT_ROOT"
  log "start Exp26 shadow-dev confirmatory"
  left_cli_audit
  "$PYTHON_BIN" "$CODE_DIR/shadowdev_confirmatory_analysis.py" --run-root "$RUN_ROOT" --train-root "$TRAIN_ROOT" --step0-checkpoint "$STEP0_CKPT" readback
  "$PYTHON_BIN" "$CODE_DIR/shadowdev_confirmatory_analysis.py" --run-root "$RUN_ROOT" integrity
  prepare_shadow_manifest
  "$PYTHON_BIN" "$CODE_DIR/shadowdev_confirmatory_analysis.py" --run-root "$RUN_ROOT" --train-root "$TRAIN_ROOT" --step0-checkpoint "$STEP0_CKPT" checkpoint-identity
  refresh_eligible_gpus
  if [[ -z "${RIGHT_EXP26_ELIGIBLE_GPUS:-}" ]]; then
    log "no eligible right-side GPU; CPU-side prep complete, sleeping for monitor retry"
    exit 66
  fi
  IFS=',' read -r -a GPUS <<< "$RIGHT_EXP26_ELIGIBLE_GPUS"
  local pids=()
  local steps=()
  case "${#GPUS[@]}" in
    1) steps=(step0 step50 step30 step10) ;;
    2) steps=(step0 step50 step30 step10) ;;
    3) steps=(step0 step50 step30 step10) ;;
    *) steps=(step0 step50 step30 step10) ;;
  esac
  local idx=0
  for step in "${steps[@]}"; do
    while true; do
      local active=0
      for pid in "${pids[@]}"; do
        ps -p "$pid" >/dev/null 2>&1 && active=$((active + 1))
      done
      if [[ "$active" -lt "${#GPUS[@]}" ]]; then
        break
      fi
      append_monitor "queue_wait" "" "" "waiting_for_slot"
      sleep 300
    done
    local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
    run_checkpoint "$step" "$gpu"
    pids+=("$(cat "$RUNTIME_ROOT/${step}.launcher.pid")")
    idx=$((idx + 1))
    METRIC_GPU="${GPUS[0]}"
    sleep 120
  done
  wait_for_checkpoints "${pids[@]}"
  run_metric_and_audits
  log "Exp26 shadow-dev confirmatory completed"
}

main "$@"
