#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
EXP="exp14_adapter_videopainter_gate2000"

VP_ROOT="${VIDEO_PAINTER_ROOT:-$ROOT/third_party/VideoPainter}"
ADAPTER_TRAIN="${ADAPTER_TRAIN:-$ROOT/exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py}"
VP_CKPT="${VIDEO_PAINTER_CHECKPOINT_ROOT:-$VP_ROOT/ckpt/VideoPainter/checkpoints/branch}"
VP_REF_CKPT="${VIDEO_PAINTER_REFERENCE_CHECKPOINT_ROOT:-$VP_CKPT}"

YTVOS_ROOT="${YTVOS_ROOT:-/mnt/workspace/hj/nas_hj/data/external/youtubevos_432_240}"
DAVIS_ROOT="${DAVIS_ROOT:-/mnt/workspace/hj/nas_hj/data/external/davis_432_240}"
PAIR_MANIFEST="${PAIR_MANIFEST:-$ROOT/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4/manifests/selected_primary_comp.repaired.pai_paths.jsonl}"

RUN_DIR="${RUN_DIR:-$ROOT/exp14_adapter_videopainter/runs/gate2000}"
LOG_DIR="$ROOT/logs/pipelines"
LOG="$LOG_DIR/${EXP}.log"
PID_FILE="$LOG_DIR/${EXP}.pid"
PRECHECK_REPORT="$ROOT/reports/videopainter_adapter_gate2000_precheck.md"

mkdir -p "$LOG_DIR" "$RUN_DIR" "$ROOT/reports" "$ROOT/exp14_adapter_videopainter/dpo_diag" "$ROOT/exp14_adapter_videopainter/logs"

exec > >(tee "$LOG") 2>&1

echo "===== Exp14 VideoPainter Adapter Gate2000 Precheck + Launch ====="
date
echo "ROOT=$ROOT"
echo "VP_ROOT=$VP_ROOT"
echo "ADAPTER_TRAIN=$ADAPTER_TRAIN"
echo "RUN_DIR=$RUN_DIR"
echo "LOG=$LOG"

fail() {
  local msg="$1"
  {
    echo "# VideoPainter Adapter Gate2000 Precheck"
    echo
    echo "status: blocked"
    echo "reason: $msg"
    echo "time: $(date)"
    echo
    echo "root: $ROOT"
    echo "videopainter_repo: $VP_ROOT"
    echo "adapter_train: $ADAPTER_TRAIN"
    echo "policy_checkpoint: $VP_CKPT"
    echo "reference_checkpoint: $VP_REF_CKPT"
    echo "pair_manifest: $PAIR_MANIFEST"
    echo "youtubevos_root: $YTVOS_ROOT"
    echo "davis_root: $DAVIS_ROOT"
  } > "$PRECHECK_REPORT"
  echo "BLOCKED: $msg"
  echo "PRECHECK_REPORT=$PRECHECK_REPORT"
  exit 2
}

test -d "$ROOT" || fail "project root missing"
test -d "$VP_ROOT/.git" || fail "VideoPainter repo missing or not a git repo"
test -f "$VP_ROOT/train/VideoPainter.sh" || fail "VideoPainter train/VideoPainter.sh missing"
test -f "$VP_ROOT/train/train_cogvideox_inpainting_i2v_video.py" || fail "VideoPainter Python training entry missing"
test -f "$VP_ROOT/infer/inpaint.py" || fail "VideoPainter inference entry missing"
test -f "$VP_ROOT/evaluate/eval_inpainting.py" || fail "VideoPainter evaluation entry missing"

test -f "$ADAPTER_TRAIN" || fail "isolated VideoPainter DPO adapter trainer is not implemented"
test -e "$VP_CKPT" || fail "VideoPainter policy checkpoint root missing"
test -e "$VP_REF_CKPT" || fail "VideoPainter reference checkpoint root missing"
test -d "$YTVOS_ROOT" || fail "YouTube-VOS train root missing"
test -d "$DAVIS_ROOT" || fail "DAVIS eval root missing"
test -f "$PAIR_MANIFEST" || fail "DPO pair manifest missing"

if grep -q "/home/nvme01" "$PAIR_MANIFEST"; then
  fail "pair manifest contains /home/nvme01 paths and is not PAI-safe"
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES="$(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F, '$2 + 0 < 5000 {gsub(/ /, "", $1); print $1}' \
      | head -4 \
      | paste -sd, -
  )"
  export CUDA_VISIBLE_DEVICES
fi

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  fail "no GPU with <5GB used memory found"
fi

GPU_COUNT="$(printf '%s\n' "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')"
if [ "$GPU_COUNT" -lt 1 ]; then
  fail "CUDA_VISIBLE_DEVICES did not contain a valid GPU list"
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

python -m py_compile "$ADAPTER_TRAIN" || fail "adapter trainer py_compile failed"

{
  echo "# VideoPainter Adapter Gate2000 Precheck"
  echo
  echo "status: passed"
  echo "time: $(date)"
  echo "root: $ROOT"
  echo "videopainter_repo: $VP_ROOT"
  echo "adapter_train: $ADAPTER_TRAIN"
  echo "policy_checkpoint: $VP_CKPT"
  echo "reference_checkpoint: $VP_REF_CKPT"
  echo "pair_manifest: $PAIR_MANIFEST"
  echo "youtubevos_root: $YTVOS_ROOT"
  echo "davis_root: $DAVIS_ROOT"
  echo "cuda_visible_devices: $CUDA_VISIBLE_DEVICES"
  echo "run_dir: $RUN_DIR"
  echo "log: $LOG"
} > "$PRECHECK_REPORT"

echo "===== Launching gate2000 ====="
setsid nohup python "$ADAPTER_TRAIN" \
  --max_train_steps 2000 \
  --checkpointing_steps 500 \
  --checkpoints_total_limit 5 \
  --mixed_precision bf16 \
  --report_to none \
  --dpo_diag_log_every 10 \
  --dpo_diag_csv "$ROOT/exp14_adapter_videopainter/dpo_diag/dpo_diagnostics.csv" \
  --videopainter_root "$VP_ROOT" \
  --policy_checkpoint "$VP_CKPT" \
  --reference_checkpoint "$VP_REF_CKPT" \
  --pair_manifest "$PAIR_MANIFEST" \
  --youtubevos_root "$YTVOS_ROOT" \
  --davis_root "$DAVIS_ROOT" \
  --output_dir "$RUN_DIR" \
  --beta_dpo 10 \
  --lose_gap_weight 0.25 \
  --lose_gap_clip_tau 1.0 \
  --winner_abs_reg_weight 0.05 \
  --winner_gap_reg_weight 1.0 \
  --winner_gap_reg_margin 0.0 \
  --boundary_mode outer \
  --mask_weight 1.0 \
  --boundary_weight 0.75 \
  --outside_weight 0.05 \
  > "$RUN_DIR/train.log" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
echo "PID=$(cat "$PID_FILE")"
echo "PID_FILE=$PID_FILE"
echo "TRAIN_LOG=$RUN_DIR/train.log"
