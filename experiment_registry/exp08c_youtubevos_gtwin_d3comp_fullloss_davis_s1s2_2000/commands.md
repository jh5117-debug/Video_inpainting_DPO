# Commands

H20 formal launcher:

```bash
cd /home/nvme01/H20_Video_inpainting_DPO_exp8c_gtwin
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
MAIN_PROCESS_PORT=29541 \
MIXED_PRECISION=no \
POLICY_DTYPE=fp32 \
VAE_DTYPE=fp32 \
REF_DTYPE=fp32 \
TEXT_DTYPE=fp32 \
SPLIT_POS_NEG_FORWARD=0 \
nohup bash scripts/launch_exp8c_youtubevos_gtwin_d3comp_fullloss_s1s2_2000_davis_h20.sh \
  > logs/pipelines/exp08c_youtubevos_gtwin_d3comp_fullloss_wingap_lose025_s1s2_2000_davis_h20_fp32_nosplit_20260606_114413.log 2>&1 &
```

Monitor:

```bash
cd /home/nvme01/H20_Video_inpainting_DPO_exp8c_gtwin
grep -a -nE 'global_step=|dpo_diag|checkpoint|Stage2|Traceback|FAILED|ERROR|OutOfMemory|SIGFPE' \
  logs/pipelines/exp08c_youtubevos_gtwin_d3comp_fullloss_wingap_lose025_s1s2_2000_davis_h20_fp32_nosplit_20260606_114413.log | tail -120
```

PAI launch from git-tracked code:

```bash
cd /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO || exit 1

TS=$(date +%Y%m%d_%H%M%S)
EXP=exp08c_youtubevos_gtwin_d3comp_fullloss_wingap_lose025_s1s2_2000_davis_pai
LOG=logs/pipelines/${EXP}_bf16_${TS}.log
PID_FILE=logs/pipelines/${EXP}_bf16.pid
mkdir -p logs/pipelines

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NUM_GPUS=8 \
MAIN_PROCESS_PORT=29561 \
MIXED_PRECISION=bf16 \
POLICY_DTYPE=auto \
VAE_DTYPE=fp32 \
REF_DTYPE=bf16 \
TEXT_DTYPE=bf16 \
SPLIT_POS_NEG_FORWARD=1 \
YTBV_ROOT=/mnt/workspace/hj/nas_hj/data/external/ytbv_2019_full_resolution/train \
nohup bash scripts/launch_exp8c_youtubevos_gtwin_d3comp_fullloss_s1s2_2000_davis_pai.sh \
  > "$LOG" 2>&1 &

echo $! > "$PID_FILE"
sleep 30
echo "PID=$(cat "$PID_FILE")"
echo "LOG=$LOG"
ps -fp "$(cat "$PID_FILE")" || true
grep -a -nE 'Exp8c|precheck|Stage1|Stage2|DAVIS|bf16|Total optimization steps|global_step=|dpo_diag|checkpoint|last_weights|Traceback|FAILED|ERROR|OutOfMemory|SIGFPE' "$LOG" | tail -160 || true
```

PAI compact monitor:

```bash
cd /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO || exit 1
PID_FILE=logs/pipelines/exp08c_youtubevos_gtwin_d3comp_fullloss_wingap_lose025_s1s2_2000_davis_pai_bf16.pid
LOG=$(ls -t logs/pipelines/exp08c_youtubevos_gtwin_d3comp_fullloss_wingap_lose025_s1s2_2000_davis_pai_bf16_*.log 2>/dev/null | head -1)
echo "PID=$(cat "$PID_FILE" 2>/dev/null)"
echo "LOG=$LOG"
ps -fp "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
if [ -f "$LOG" ]; then
  grep -a -nE 'Exp8c|Stage1|Stage2|DAVIS|global_step=|dpo_diag|checkpoint|last_weights|Traceback|FAILED|ERROR|OutOfMemory|SIGFPE' "$LOG" | tail -120 || true
fi
```
