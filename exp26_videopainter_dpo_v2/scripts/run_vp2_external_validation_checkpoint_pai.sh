#!/usr/bin/env bash
set -euo pipefail

# Run one Exp26 post-confirmation external-validation checkpoint on one GPU.
# This is task-level parallelism only. It does not modify training state,
# checkpoint selection, search-dev, shadow-dev, or left-side CLI runtime.

CHECKPOINT_STEP="${CHECKPOINT_STEP:?set CHECKPOINT_STEP to step0, step10, step30, or step50}"
GPU_INDEX="${GPU_INDEX:?set GPU_INDEX to the single allowed GPU index}"

RUN_ROOT="${RUN_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/postconfirmation_20260626/external_validation}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp26_external_validation_20260626}"
CODE_DIR="${CODE_DIR:-/tmp/exp26_external_validation_code}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp26_videopainter}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"

VP_ROOT="${VP_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter}"
VP_BASE="${VP_BASE:-$VP_ROOT/ckpt/CogVideoX-5b-I2V}"
STEP0_CKPT="${STEP0_CKPT:-$VP_ROOT/ckpt/VideoPainter/checkpoints/branch}"
TRAIN_ROOT="${TRAIN_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp26_videopainter_dpo_v2/vp_primary32_50step_20260625_171032}"
MANIFEST="${MANIFEST:-$RUN_ROOT/preregistered/manifests/vp2_external_validation_preregistered.jsonl}"

case "$CHECKPOINT_STEP" in
  step0) BRANCH_CKPT="$STEP0_CKPT" ;;
  step10) BRANCH_CKPT="$TRAIN_ROOT/checkpoint-10" ;;
  step30) BRANCH_CKPT="$TRAIN_ROOT/checkpoint-30" ;;
  step50) BRANCH_CKPT="$TRAIN_ROOT/checkpoint-50" ;;
  *) echo "unknown CHECKPOINT_STEP: $CHECKPOINT_STEP" >&2; exit 2 ;;
esac

STEP_ROOT="$RUN_ROOT/$CHECKPOINT_STEP"
OUT_DIR="$STEP_ROOT/official_generation"
REVIEW_DIR="$STEP_ROOT/${CHECKPOINT_STEP}_review"
LOCK="$RUNTIME_ROOT/exp26_external_${CHECKPOINT_STEP}.lock"
PID_FILE="$RUNTIME_ROOT/${CHECKPOINT_STEP}.pid"
GPU_FILE="$RUNTIME_ROOT/${CHECKPOINT_STEP}.gpu"
STATE_FILE="$RUNTIME_ROOT/${CHECKPOINT_STEP}.state"
HEARTBEAT="$RUNTIME_ROOT/${CHECKPOINT_STEP}.heartbeat"
LOG_FILE="$RUN_ROOT/${CHECKPOINT_STEP}.log"
REGISTRY="$RUN_ROOT/process_registry.csv"

mkdir -p "$STEP_ROOT" "$RUNTIME_ROOT" "$RUN_ROOT"

if [[ -s "$OUT_DIR/gate64_generation_summary.json" ]] && grep -q '"status": "passed"' "$OUT_DIR/gate64_generation_summary.json"; then
  echo "$CHECKPOINT_STEP already has passed generation output: $OUT_DIR"
  exit 0
fi

(
  flock -n 9 || { echo "lock busy: $LOCK" >&2; exit 75; }
  echo "$$" > "$PID_FILE"
  echo "$GPU_INDEX" > "$GPU_FILE"
  echo "running" > "$STATE_FILE"
  if [[ ! -f "$REGISTRY" ]]; then
    echo "time,checkpoint,gpu,pid,pgid,checkpoint_path,manifest,log_path,output_root" > "$REGISTRY"
  fi
  PGID="$(ps -o pgid= -p "$$" | tr -d ' ' || true)"
  echo "$(date -Ins),$CHECKPOINT_STEP,$GPU_INDEX,$$,$PGID,$BRANCH_CKPT,$MANIFEST,$LOG_FILE,$STEP_ROOT" >> "$REGISTRY"
  (
    while true; do
      date -Ins > "$HEARTBEAT"
      sleep 30
    done
  ) &
  HB_PID="$!"
  trap 'kill "$HB_PID" 2>/dev/null || true' EXIT

  export PYTHONPATH="$CODE_DIR:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
  export PYTHONUNBUFFERED=1

  "$PYTHON_BIN" "$CODE_DIR/run_vp2_gate64_official_generation.py" \
    --videopainter-root "$VP_ROOT" \
    --base-model "$VP_BASE" \
    --branch-checkpoint "$BRANCH_CKPT" \
    --manifest "$MANIFEST" \
    --output-dir "$OUT_DIR" \
    --limit 32 \
    --height 480 \
    --width 720 \
    --num-frames 49 \
    --num-inference-steps 20 \
    --guidance-scale 6.0 \
    --seed 20260619 \
    --dtype bf16 \
    --device cuda

  "$PYTHON_BIN" "$CODE_DIR/review_gate64_official_outputs.py" \
    --run-root "$STEP_ROOT" \
    --manifest "$MANIFEST" \
    --output-dir "$REVIEW_DIR" \
    --num-frames 49

  echo "completed" > "$STATE_FILE"
) 9>"$LOCK" >"$LOG_FILE" 2>&1
