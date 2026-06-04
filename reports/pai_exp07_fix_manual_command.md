# PAI Exp07 Fix Manual Command

This command is manual-only. Codex must not execute it on PAI.

```bash
cd /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO
set -euo pipefail

TS=$(date +%Y%m%d_%H%M%S)
REPORT=reports/pai_exp07_fix_smallmask_prior_manual_preflight_${TS}.md
EXP=exp07_fix_smallmask_prior_wingap_lose025_stage1_gate1500
DATA_ROOT=/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/exp07_fix_videodpo_smallmask15_20_prior_k4
MANIFEST=${DATA_ROOT}/manifests/selected_primary_comp.repaired.jsonl
REG_DIR=experiment_registry/exp07_fix_smallmask_prior
PLOG=logs/pipelines/${EXP}.log
PID_FILE=logs/pipelines/${EXP}.pid

mkdir -p reports logs/pipelines "$REG_DIR"

{
  echo "# PAI Exp07 Fix Smallmask Prior Manual Preflight"
  date
  echo
  echo "## Registry"
  test -d "$REG_DIR" && echo "registry_exists=$REG_DIR" || echo "registry_missing=$REG_DIR"
  sed -n '1,160p' "$REG_DIR/config.yaml" 2>/dev/null || true
  echo
  echo "## Required paths"
  echo "DATA_ROOT=$DATA_ROOT"
  echo "MANIFEST=$MANIFEST"
  ls -ld "$DATA_ROOT" "$DATA_ROOT/manifests" 2>/dev/null || true
  ls -lh "$MANIFEST" 2>/dev/null || true
  echo
  echo "## SFT-48000 weights"
  ls -ld /mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000 \
         /mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000/unet_main \
         /mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000/brushnet 2>/dev/null || true
  echo
  echo "## Manifest checks"
  if [ -f "$MANIFEST" ]; then
    echo "rows=$(wc -l < "$MANIFEST")"
    if grep -m1 -q '/home/nvme01/' "$MANIFEST"; then
      echo "ERROR: H20 path found in PAI manifest"
    else
      echo "path_check=ok_no_h20_paths"
    fi
  else
    echo "manifest_missing"
  fi
  echo
  echo "## Matching processes"
  ps -eo pid,ppid,etime,stat,cmd | grep -E 'exp07_fix_smallmask|train_stage1.py|accelerate|lingbot-worldphy' | grep -v grep || true
  echo
  echo "## GPU"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
} | tee "$REPORT"

echo
if [ "${LAUNCH:-0}" != "1" ]; then
  echo "Preflight only. To launch after data exists and report is clean, rerun with: LAUNCH=1 bash <this block/script>"
  echo "Report: $REPORT"
  exit 0
fi

if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: cannot launch; manifest missing: $MANIFEST" >&2
  exit 1
fi
if grep -m1 -q '/home/nvme01/' "$MANIFEST"; then
  echo "ERROR: cannot launch; manifest contains H20 paths" >&2
  exit 1
fi
if [ ! -d /mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000 ]; then
  echo "ERROR: cannot launch; SFT-48000 weights missing" >&2
  exit 1
fi

unset PREFERENCE_MANIFEST RUN_NAME MAX_STEPS STAGE1_MAX_STEPS CKPT_STEPS CKPT_LIMIT CHECKPOINTING_STEPS CHECKPOINTS_TOTAL_LIMIT EXP_NAME TRAIN_MASK_MODE MASK_FROM_MANIFEST LOSS_REGION_MODE

nohup bash scripts/launch_exp07_fix_smallmask_prior_wingap_stage1_gate_pai.sh > "$PLOG" 2>&1 &
echo $! > "$PID_FILE"
sleep 30

echo "PID=$(cat "$PID_FILE")"
ps -fp "$(cat "$PID_FILE")" || true

grep -a -nE 'exp07-fix-pai|manifest=|weights=|Stage1 only|train_mask_mode|loss_region_mode|beta=|max_steps|Traceback|ERROR|OutOfMemory|SIGFPE' "$PLOG" | tail -120 || true
```
