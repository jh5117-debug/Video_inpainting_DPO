#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp26_videopainter}"
VP_ROOT="${VIDEO_PAINTER_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter}"
VP_BASE_MODEL="${VIDEO_PAINTER_BASE_MODEL:-$VP_ROOT/ckpt/CogVideoX-5b-I2V}"
VP_CKPT="${VIDEO_PAINTER_CHECKPOINT_ROOT:-$VP_ROOT/ckpt/VideoPainter/checkpoints/branch}"
VP_REF_CKPT="${VIDEO_PAINTER_REFERENCE_CHECKPOINT_ROOT:-$VP_CKPT}"
OFFICIAL_TRAIN="${OFFICIAL_TRAIN_FILE:-$VP_ROOT/train/train_cogvideox_inpainting_i2v_video.py}"
PAIR_MANIFEST="${PAIR_MANIFEST:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4/manifests/selected_primary_comp.repaired.pai_paths.jsonl}"
DAVIS_ROOT="${DAVIS_ROOT:-/mnt/workspace/hj/nas_hj/data/external/davis_432_240}"
OUT_DIR="${OUT_DIR:-/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp26_videopainter_dpo_v2/l0_l4_gates_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26}"
LOG="$LOG_DIR/vp2_l0_l4_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/vp2_l0_l4.pid"
HEARTBEAT="$LOG_DIR/vp2_l0_l4_heartbeat.json"
RUNNER="$ROOT/exp26_videopainter_dpo_v2/code/run_vp2_l0_l4_gates.py"

mkdir -p "$LOG_DIR" "$OUT_DIR" "$ROOT/reports"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  # Exp26 policy for this run: prefer GPU2, then GPU6 only if genuinely free.
  if nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '$1 ~ /2/ && $2 + 0 < 5000 {found=1} END{exit !found}'; then
    export CUDA_VISIBLE_DEVICES=2
  elif nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '$1 ~ /6/ && $2 + 0 < 5000 {found=1} END{exit !found}'; then
    export CUDA_VISIBLE_DEVICES=6
  else
    echo "BLOCKED: neither GPU2 nor GPU6 is free enough for Exp26 L0-L4." >&2
    exit 2
  fi
fi

cat > "$HEARTBEAT" <<EOF
{"status":"starting","pid":null,"cuda_visible_devices":"$CUDA_VISIBLE_DEVICES","out_dir":"$OUT_DIR","log":"$LOG"}
EOF

setsid nohup bash -lc "
  set -euo pipefail
  cd '$ROOT'
  export CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'
  export PYTHONUNBUFFERED=1
  python '$RUNNER' \
    --videopainter_root '$VP_ROOT' \
    --pretrained_model_name_or_path '$VP_BASE_MODEL' \
    --policy_checkpoint '$VP_CKPT' \
    --reference_checkpoint '$VP_REF_CKPT' \
    --official_train_file '$OFFICIAL_TRAIN' \
    --plumbing_pair_manifest '$PAIR_MANIFEST' \
    --davis_root '$DAVIS_ROOT' \
    --output_dir '$OUT_DIR' \
    --mixed_precision bf16 \
    --height 240 \
    --width 432 \
    --l4_steps 10
" > "$LOG" 2>&1 < /dev/null &

PID="$!"
echo "$PID" > "$PID_FILE"
cat > "$HEARTBEAT" <<EOF
{"status":"running","pid":$PID,"cuda_visible_devices":"$CUDA_VISIBLE_DEVICES","out_dir":"$OUT_DIR","log":"$LOG"}
EOF

echo "PID=$PID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "LOG=$LOG"
echo "OUT_DIR=$OUT_DIR"
echo "PID_FILE=$PID_FILE"
echo "HEARTBEAT=$HEARTBEAT"
