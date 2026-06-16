#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate}"
cd "$PROJECT_ROOT"

PY="${PY:-/mnt/nas/hj/conda_envs/diffueraser/bin/python}"
if [ ! -x "$PY" ]; then
  PY="${PYTHON:-python}"
fi

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODE="${1:-full}"
OUT_ROOT="${OUT_ROOT:-logs/target_eval/exp14_videopainter_adapter_gate2000_davis}"
LOG_ROOT="${LOG_ROOT:-logs/pipelines}"
mkdir -p "$LOG_ROOT" "$OUT_ROOT" reports exp14_adapter_videopainter/reports

COMMON_ARGS=(
  --project_root "$PROJECT_ROOT"
  --videopainter_root "$PROJECT_ROOT/third_party/VideoPainter"
  --base_model "$PROJECT_ROOT/third_party/VideoPainter/ckpt/CogVideoX-5b-I2V"
  --baseline_branch "$PROJECT_ROOT/third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch"
  --adapter_checkpoint "$PROJECT_ROOT/exp14_adapter_videopainter/runs/gate2000/last_weights"
  --davis_root /mnt/workspace/hj/nas_hj/data/external/davis_432_240
  --height "${HEIGHT:-480}"
  --width "${WIDTH:-720}"
  --num_frames "${NUM_FRAMES:-49}"
  --guidance_scale "${GUIDANCE_SCALE:-6.0}"
  --seed "${SEED:-42}"
  --dtype "${DTYPE:-bf16}"
  --device "${DEVICE:-cuda}"
)

echo "===== VideoPainter adapter DAVIS eval ====="
date
hostname
pwd
echo "mode=$MODE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

echo
echo "===== precheck ====="
test -f exp14_adapter_videopainter/code/eval_videopainter_adapter_davis.py
test -d exp14_adapter_videopainter/runs/gate2000/last_weights
test -f exp14_adapter_videopainter/dpo_diag/dpo_diagnostics.csv
test -d third_party/VideoPainter/ckpt/CogVideoX-5b-I2V
test -d third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch
test -d /mnt/workspace/hj/nas_hj/data/external/davis_432_240
test -f inference/metrics.py

"$PY" -m py_compile exp14_adapter_videopainter/code/eval_videopainter_adapter_davis.py

if [ "$MODE" = "debug" ]; then
  "$PY" exp14_adapter_videopainter/code/eval_videopainter_adapter_davis.py \
    "${COMMON_ARGS[@]}" \
    --output_dir "${OUT_ROOT}_debug" \
    --limit_videos "${DEBUG_LIMIT_VIDEOS:-3}" \
    --num_inference_steps "${DEBUG_NUM_INFERENCE_STEPS:-6}" \
    --debug
else
  "$PY" exp14_adapter_videopainter/code/eval_videopainter_adapter_davis.py \
    "${COMMON_ARGS[@]}" \
    --output_dir "$OUT_ROOT" \
    --num_inference_steps "${NUM_INFERENCE_STEPS:-50}"
fi

echo "===== eval finished ====="
date
