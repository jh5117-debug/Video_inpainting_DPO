#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/runtime_code_snapshots/exp26_current}"
EXP_ROOT="$ROOT/exp26_videopainter_dpo_v2"
VP_ROOT="${VIDEO_PAINTER_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter}"
VP_BASE_MODEL="${VIDEO_PAINTER_BASE_MODEL:-$VP_ROOT/ckpt/CogVideoX-5b-I2V}"
VP_CKPT="${VIDEO_PAINTER_CHECKPOINT_ROOT:-$VP_ROOT/ckpt/VideoPainter/checkpoints/branch}"
ARCHIVE_DIR="${VOR_ARCHIVE_DIR:-/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/downloads/fa09dc61128ca0418a4a13364d97a08018ea9cc7}"
MANIFEST="${GATE64_SOURCE_MANIFEST:-$EXP_ROOT/manifests/vp2_gate64_source_manifest.jsonl}"
RUN_ID="${RUN_ID:-gate64_official_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26/$RUN_ID}"
LOG_DIR="${LOG_DIR:-/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26}"
LOG="$LOG_DIR/${RUN_ID}.log"
PID_FILE="$LOG_DIR/${RUN_ID}.pid"
HEARTBEAT="$LOG_DIR/${RUN_ID}_heartbeat.json"
LIMIT="${LIMIT:-64}"
SEED="${SEED:-20260624}"

mkdir -p "$LOG_DIR" "$OUT_ROOT"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES="$(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F, '$1 + 0 <= 6 && $2 + 0 < 5000 {gsub(/ /, "", $1); print $1; exit}'
  )"
  if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "BLOCKED: no free GPU0-6 found for Exp26 Gate64 generation." >&2
    exit 2
  fi
  export CUDA_VISIBLE_DEVICES
fi

cat > "$HEARTBEAT" <<EOF
{"status":"starting","pid":null,"cuda_visible_devices":"$CUDA_VISIBLE_DEVICES","out_root":"$OUT_ROOT","log":"$LOG","limit":$LIMIT}
EOF

setsid nohup bash -lc "
  set -euo pipefail
  cd '$ROOT'
  export CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'
  export PYTHONUNBUFFERED=1
  export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
  echo '[stage] extract Gate64 BG videos'
  python '$EXP_ROOT/code/extract_vp2_gate64_vor_bg.py' \
    --manifest '$MANIFEST' \
    --archive-dir '$ARCHIVE_DIR' \
    --output-root '$OUT_ROOT/source_videos' \
    --output-manifest '$OUT_ROOT/gate64_extracted_sources.jsonl' \
    --status-csv '$OUT_ROOT/gate64_extraction_status.csv' \
    --limit '$LIMIT'

  echo '[stage] materialize first 49 real frames'
  python '$EXP_ROOT/code/materialize_vp2_49f_sources.py' \
    --manifest '$OUT_ROOT/gate64_extracted_sources.jsonl' \
    --source-root '$OUT_ROOT/source_videos' \
    --output-root '$OUT_ROOT/materialized_49f' \
    --output-manifest '$OUT_ROOT/gate64_materialized_49f.jsonl' \
    --status-csv '$OUT_ROOT/gate64_materialized_49f_status.csv' \
    --num-frames 49 \
    --stride 1 \
    --offset 0 \
    --limit '$LIMIT'

  echo '[stage] generate locked mixed moving masks'
  python '$EXP_ROOT/code/generate_vp2_moving_br_masks.py' \
    --materialized-manifest '$OUT_ROOT/gate64_materialized_49f.jsonl' \
    --output-root '$OUT_ROOT/masks' \
    --output-manifest '$OUT_ROOT/gate64_mask_ready.jsonl' \
    --status-csv '$OUT_ROOT/gate64_mask_status.csv' \
    --seed '$SEED' \
    --first-frame-gt \
    --limit '$LIMIT'

  echo '[stage] official VideoPainter Gate64 generation'
  python '$EXP_ROOT/code/run_vp2_gate64_official_generation.py' \
    --videopainter-root '$VP_ROOT' \
    --base-model '$VP_BASE_MODEL' \
    --branch-checkpoint '$VP_CKPT' \
    --manifest '$OUT_ROOT/gate64_mask_ready.jsonl' \
    --output-dir '$OUT_ROOT/official_generation' \
    --limit '$LIMIT' \
    --height 480 \
    --width 720 \
    --num-frames 49 \
    --num-inference-steps 20 \
    --guidance-scale 6.0 \
    --seed '$SEED' \
    --dtype bf16 \
    --device cuda
  echo '[stage] done'
" > "$LOG" 2>&1 < /dev/null &

PID="$!"
echo "$PID" > "$PID_FILE"
cat > "$HEARTBEAT" <<EOF
{"status":"running","pid":$PID,"cuda_visible_devices":"$CUDA_VISIBLE_DEVICES","out_root":"$OUT_ROOT","log":"$LOG","limit":$LIMIT}
EOF

echo "PID=$PID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "LOG=$LOG"
echo "OUT_ROOT=$OUT_ROOT"
echo "PID_FILE=$PID_FILE"
echo "HEARTBEAT=$HEARTBEAT"
