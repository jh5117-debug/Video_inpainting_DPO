#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
VP_ROOT="${VIDEO_PAINTER_ROOT:-$ROOT/third_party/VideoPainter}"
ADAPTER_TRAIN="${ADAPTER_TRAIN:-$ROOT/exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py}"
LOG_DIR="$ROOT/logs/pipelines"
LOG="$LOG_DIR/exp14_videopainter_adapter_smoke1.log"

mkdir -p "$LOG_DIR"
exec > >(tee "$LOG") 2>&1

echo "===== Exp14 VideoPainter Adapter Smoke1 Guard ====="
date
echo "ROOT=$ROOT"
echo "VP_ROOT=$VP_ROOT"
echo "ADAPTER_TRAIN=$ADAPTER_TRAIN"

test -d "$ROOT"
if [ ! -d "$VP_ROOT/.git" ]; then
  echo "VideoPainter repo missing at $VP_ROOT"
  echo "Cloning official repo for audit/smoke precheck only."
  mkdir -p "$(dirname "$VP_ROOT")"
  git clone --depth 1 https://github.com/TencentARC/VideoPainter.git "$VP_ROOT"
fi
test -f "$VP_ROOT/train/train_cogvideox_inpainting_i2v_video.py"
test -f "$VP_ROOT/train/VideoPainter.sh"

echo "VideoPainter upstream training code exists."

if [ ! -f "$ADAPTER_TRAIN" ]; then
  echo "BLOCKED: adapter train script is not implemented yet."
  echo "Expected: $ADAPTER_TRAIN"
  echo "Smoke1 not run."
  exit 2
fi

echo "Adapter train script found, but this guard will not launch without explicit env."
if [ "${EXP14_ALLOW_SMOKE1:-false}" != "true" ]; then
  echo "BLOCKED: set EXP14_ALLOW_SMOKE1=true only after code review."
  exit 2
fi

python -m py_compile "$ADAPTER_TRAIN"
python "$ADAPTER_TRAIN" --max_train_steps 1 --smoke --output_dir "$ROOT/exp14_adapter_videopainter/runs/smoke1"
