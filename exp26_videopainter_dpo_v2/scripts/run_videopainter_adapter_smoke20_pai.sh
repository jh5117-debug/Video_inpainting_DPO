#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
VP_ROOT="${VIDEO_PAINTER_ROOT:-$ROOT/third_party/VideoPainter}"
ADAPTER_TRAIN="${ADAPTER_TRAIN:-$ROOT/exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py}"
SMOKE1_DIR="$ROOT/exp14_adapter_videopainter/runs/smoke1"
LOG_DIR="$ROOT/logs/pipelines"
LOG="$LOG_DIR/exp14_videopainter_adapter_smoke20.log"

mkdir -p "$LOG_DIR"
exec > >(tee "$LOG") 2>&1

echo "===== Exp14 VideoPainter Adapter Smoke20 Guard ====="
date

if [ ! -d "$VP_ROOT/.git" ]; then
  echo "BLOCKED: VideoPainter repo missing: $VP_ROOT"
  echo "Run Smoke1 precheck first so it can clone/audit the repo."
  exit 2
fi

if [ ! -f "$ADAPTER_TRAIN" ]; then
  echo "BLOCKED: adapter train script is not implemented yet."
  exit 2
fi

if [ ! -d "$SMOKE1_DIR" ]; then
  echo "BLOCKED: Smoke1 output dir missing: $SMOKE1_DIR"
  echo "Smoke20 not run."
  exit 2
fi

if [ "${EXP14_ALLOW_SMOKE20:-false}" != "true" ]; then
  echo "BLOCKED: set EXP14_ALLOW_SMOKE20=true only after Smoke1 passes."
  exit 2
fi

python -m py_compile "$ADAPTER_TRAIN"
python "$ADAPTER_TRAIN" --max_train_steps 20 --smoke --tiny_val_videos 3 --output_dir "$ROOT/exp14_adapter_videopainter/runs/smoke20"
