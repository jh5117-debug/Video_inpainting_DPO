# Commands

Codex cannot execute PAI commands. This registry entry records the current PAI manual run from handoff/user-pasted audit context.

## Safe PAI Read-Only Monitor

```bash
cd /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO
set +e

PLOG=$(ls -t logs/pipelines/exp08_d3_comp_fullloss_continue_after_s1_fixsafety_len24_pai_*.log 2>/dev/null | head -1)
PID_FILE=logs/pipelines/exp08_d3_comp_fullloss_continue_after_s1_fixsafety_len24_pai.pid
VAL1=$(find /mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval \
  -maxdepth 1 -type d -name '*exp08a_fullloss_stage1_val_davis*fixsafety_len24*' \
  -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

echo DATE=$(date)
echo PLOG=$PLOG
echo PID_FILE=$PID_FILE
[ -f "$PID_FILE" ] && echo PID=$(cat "$PID_FILE")
[ -f "$PID_FILE" ] && ps -fp "$(cat "$PID_FILE")" || true

echo GPU
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

echo KEY_LOG
grep -a -nE 'Exp8|Exp8a|Stage1|Stage2|DAVIS|DPO-S1_SFT-S2|DiffuEraser-base|metrics|summary.csv|side_by_side|Total optimization steps|Epoch|global_step=|checkpoint|last_weights|Traceback|FAILED|ERROR|OutOfMemory|SIGFPE|ValueError|RuntimeError|OSError|No such file' "$PLOG" | tail -220 || true

echo VAL1=$VAL1
if [ -n "$VAL1" ]; then
  echo sample_mp4_count=$(find "$VAL1/inference" -name '*.mp4' 2>/dev/null | wc -l)
  echo side_by_side_count=$(find "$VAL1/side_by_side" -name '*.mp4' 2>/dev/null | wc -l)
  ls -lh "$VAL1/metrics/summary.csv" "$VAL1/pair_manifest.csv" "$VAL1/index.html" "$VAL1/report.md" 2>/dev/null || true
fi

echo STAGE2_DIRS
find /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2 \
  -maxdepth 1 -type d -name '*exp08_d3_comp_fullloss*' \
  -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort | tail -5

echo AUDIT_DONE
```
