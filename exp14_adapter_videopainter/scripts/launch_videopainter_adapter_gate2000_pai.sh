#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
VP_ROOT="${VIDEO_PAINTER_ROOT:-$ROOT/third_party/VideoPainter}"
ADAPTER_TRAIN="${ADAPTER_TRAIN:-$ROOT/exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py}"
SMOKE20_DIR="$ROOT/exp14_adapter_videopainter/runs/smoke20"
LOG_DIR="$ROOT/logs/pipelines"
LOG="$LOG_DIR/exp14_videopainter_adapter_gate2000.log"

mkdir -p "$LOG_DIR"
exec > >(tee "$LOG") 2>&1

echo "===== Exp14 VideoPainter Adapter Gate2000 Guard ====="
date

if [ ! -d "$VP_ROOT/.git" ]; then
  echo "BLOCKED: VideoPainter repo missing: $VP_ROOT"
  exit 2
fi

if [ ! -f "$ADAPTER_TRAIN" ]; then
  echo "BLOCKED: adapter train script is not implemented yet."
  exit 2
fi

if [ ! -d "$SMOKE20_DIR" ]; then
  echo "BLOCKED: Smoke20 output dir missing: $SMOKE20_DIR"
  echo "Gate2000 not launched."
  exit 2
fi

if [ "${EXP14_CONFIRM_GATE2000:-false}" != "true" ]; then
  echo "BLOCKED: user confirmation required."
  echo "Set EXP14_CONFIRM_GATE2000=true only after explicit approval."
  exit 2
fi

python -m py_compile "$ADAPTER_TRAIN"
python "$ADAPTER_TRAIN" --max_train_steps 2000 --output_dir "$ROOT/exp14_adapter_videopainter/runs/gate2000"
