# Commands

## Current command status

H20 smallmask data exists and the active restart is a two-stage run using the
git-tracked registry code below. Top-level `scripts/` paths are compatibility
wrappers only.

## H20 data generation command

Launched 2026-06-05 CST with `tools/videodpo_generated_loser_calibration.py --models diffueraser --limit 1000 --mask_policy_config configs/generation/videodpo_partialmask_policy_v2_smallmask15_20_k4.yaml`. See report `reports/h20_exp07_fix_smallmask_prior_status.md`.

## H20 Stage1+Stage2 command

Run from a clean git-synced H20 worktree, while pointing data/weights/output to
the H20 main artifact tree:

```bash
cd /home/nvme01/H20_Video_inpainting_DPO_exp8c_pai_sync || exit 1
set +e

TS=$(date +%Y%m%d_%H%M%S)
EXP=exp07_fix_smallmask_prior_wingap_lose025_s1s2_2000_h20
LOG=/home/nvme01/H20_Video_inpainting_DPO/logs/pipelines/${EXP}_${TS}.log
PID_FILE=/home/nvme01/H20_Video_inpainting_DPO/logs/pipelines/${EXP}.pid
mkdir -p /home/nvme01/H20_Video_inpainting_DPO/logs/pipelines

RUN_VERSION=$TS \
OUTPUT_ROOT=/home/nvme01/H20_Video_inpainting_DPO \
DATA_ROOT=/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/exp07_fix_videodpo_smallmask15_20_prior_k4 \
WEIGHTS_DIR=/home/nvme01/H20_Video_inpainting_DPO/weights \
CONDA_ENV_PREFIX=/home/nvme01/conda_envs/diffueraser \
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
NUM_GPUS=7 \
MIXED_PRECISION=no \
POLICY_DTYPE=fp32 \
VAE_DTYPE=fp32 \
REF_DTYPE=fp32 \
TEXT_DTYPE=fp32 \
SPLIT_POS_NEG_FORWARD=0 \
MAIN_PROCESS_PORT=29571 \
nohup bash experiment_registry/exp07_fix_smallmask_prior/code/launch_s1s2_h20.sh > "$LOG" 2>&1 &

echo $! > "$PID_FILE"
```
