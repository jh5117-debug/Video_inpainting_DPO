# PAI Runbook

## Before Any Run

1. Confirm host and repo:

```bash
set -euo pipefail
date
hostname
whoami
pwd
git rev-parse --show-toplevel
git status --short
```

2. Confirm GPU/Python:

```bash
nvidia-smi
which python
python --version
conda info --envs
```

3. Run the PAI audit and asset preparation helper:

```bash
bash scripts/pai_audit_and_prepare_assets.sh
```

This writes `PRD/pai_audit_pai_node_<timestamp>.md`, creates `current`
symlinks under `data/` and `weights/` for confirmed assets, and writes
`configs/paths/pai.detected.env`. Any unconfirmed root remains commented or
missing in that env file.

4. Load path config:

```bash
source configs/paths/pai.detected.env  # or configs/paths/pai.example.env copied to a private local env
```

5. Check critical env vars:

```bash
for v in VIDEO_DPO_DATA_ROOT YOUTUBE_VOS_ROOT GENERATED_LOSER_ROOT \
  DIFFUERASER_WEIGHT_ROOT PROPAINTER_WEIGHT_ROOT COCOCO_WEIGHT_ROOT \
  MINIMAX_REMOVER_WEIGHT_ROOT OFFICIAL_VIDEODPO_ROOT VC2_WEIGHT_ROOT EXP_OUTPUT_ROOT
do
  printf '%s=%s\n' "$v" "${!v:-}"
done
```

## Four Loser Generation Models

Before Experiment 1/2A/2B data generation, search and verify each model independently:

```bash
find . /home/hj /mnt/workspace /mnt/data \
  -maxdepth 6 \
  \( -iname "*diffueraser*" -o -iname "*propainter*" -o -iname "*cococo*" -o -iname "*minimax*" -o -iname "*remover*" \) \
  2>/dev/null | sort | head -500

grep -RInE "DiffuEraser|diffueraser|ProPainter|propainter|CoCoCo|cococo|MiniMax|minimax|Remover|remover" . 2>/dev/null | head -500 || true
```

Record for each model:

- code path;
- conda/env path;
- weight path;
- generation script;
- README/runbook;
- mask convention: which value means masked/hole and which value means known/keep.

If anything is missing or ambiguous, write `未找到` or `未确认`; do not guess.

## Process Name Convention

All new bash launchers default to:

```bash
export LINGBOT_PROCESS_NAME="${LINGBOT_PROCESS_NAME:-lingbot-phy}"
```

Use `scripts/lingbot_process.sh` for new wrappers. W&B names should remain experiment-specific.

Note: `nvidia-smi` reports the CUDA process executable, so DiffuEraser jobs are expected to show as paths such as `/mnt/nas/hj/conda_envs/diffueraser/bin/python`. This does not mean the run is using the wrong project; verify the command line with `ps -eo pid,etime,cmd`.

## DiffuEraser-Only Partial-Mask Generation

Active production direction as of 2026-05-25:

- generated-loser data: `official_videodpo_diffueraser_data_partialmask_loser_k4`
- model set: `MODELS=diffueraser`
- generation_source: `diffueraser_only`
- masks: K=4 per VideoDPO winner
- expected candidate rows for 100 winners: `100 * 4 = 400`
- expected candidate rows for full 10k winners: `10000 * 4 = 40000`

Before restarting after an overloaded or exploratory run, stop active shard
processes and archive test shards instead of mixing them with production data:

```bash
OUT=/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4

pkill -TERM -f 'pai_launch_partialmask_losers_k4_sharded.sh' || true
pkill -TERM -f "$OUT/_shards" || true
sleep 15
pkill -KILL -f "$OUT/_shards" || true

ARCHIVE="$OUT/_shards_overload_$(date +%Y%m%d_%H%M%S)"
mv "$OUT/_shards" "$ARCHIVE"
mkdir -p "$OUT/_shards"
echo "archived=$ARCHIVE"
```

Recommended restart command for the 100-pair validation run:

```bash
MODELS=diffueraser \
GPUS=0,1,2,3,4,5,6 \
WORKERS_PER_GPU=4 \
SHARD_SIZE=1 \
END_INDEX=100 \
TIMEOUT_SEC=7200 \
bash scripts/pai_launch_partialmask_losers_k4_sharded.sh
```

The launcher caps CPU thread fan-out by default:

```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
OPENCV_NUM_THREADS=1
```

Do not chase full GPU memory occupancy. DiffuEraser generation is often limited
by CPU preprocessing, model loading, and NAS IO. The bad high-concurrency probe
showed GPU memory near 32-47 GiB per card but GPU util near 0 with host load
above 3000, which means too many runnable host threads/processes.

Tuning rule:

- start with `WORKERS_PER_GPU=4`, `SHARD_SIZE=1`;
- if `rows` advances steadily and host load is reasonable, try `WORKERS_PER_GPU=5`;
- if GPU util stays at 0 and load remains very high, reduce to `WORKERS_PER_GPU=3`;
- do not use `WORKERS_PER_GPU=8` for this workload without a new successful IO/CPU probe.

Progress monitor:

```bash
export OUT=/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4

watch -n 60 '
date
echo done=$(find "$OUT/_shards" -name .done | wc -l)
echo failed=$(find "$OUT/_shards" -name .failed | wc -l)
echo rows=$(find "$OUT/_shards" -path "*/manifests/candidates_all.jsonl" -exec wc -l {} \; | awk "{s+=\$1} END{print s+0}")
uptime
ps -eo stat,cmd | grep -E "run_OR|infer_diffueraser|videodpo_generated" | grep -v grep | awk "{print \$1}" | sort | uniq -c
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
'
```

Run IO diagnostics only when throughput stalls or before changing concurrency;
continuous high-frequency IO probing adds noise:

```bash
iostat -xm 5 12
pidstat -d 5 12
```

## H20 D1 Fullmask DiffuEraser-Only Preparation

Do not stop or modify the PAI D2 run for this. H20 is only for preparing D1:

```text
D1 = VideoDPO fullmask loser data
generation_source = diffueraser_only
mask_mode = full
num_masks_per_video = 1
comp = false
process_name = lingbot-world
```

First run the H20 audit:

```bash
cd /home/nvme01/H20_Video_inpainting_DPO
bash scripts/h20_audit_fullmask_generation_readiness.sh 2>&1 | tee /tmp/h20_fullmask_audit.log
```

Then run small samples only:

```bash
MODELS=diffueraser \
LINGBOT_PROCESS_NAME=lingbot-world \
DIFFUERASER_INFERENCE_STACK=br \
DIFFUERASER_PRIOR_MODE=noise \
GPUS=0,1,2,3 \
WORKERS_PER_GPU=1 \
SHARD_SIZE=1 \
START_INDEX=0 \
END_INDEX=20 \
TIMEOUT_SEC=7200 \
bash scripts/h20_launch_fullmask_losers_diffueraser_sharded.sh

MODELS=diffueraser \
LINGBOT_PROCESS_NAME=lingbot-world \
DIFFUERASER_INFERENCE_STACK=br \
DIFFUERASER_PRIOR_MODE=noise \
GPUS=0,1,2,3 \
WORKERS_PER_GPU=1 \
SHARD_SIZE=1 \
START_INDEX=0 \
END_INDEX=100 \
TIMEOUT_SEC=7200 \
bash scripts/h20_launch_fullmask_losers_diffueraser_sharded.sh
```

Current result: the D1 BR/no-prior 100-sample technical gate passed, but the
quality gate failed (`too_bad=95/100`, median q=`0.1947`). Do not run full D1
generation from this setting. The following audit command is retained for
diagnostics:

```bash
OUT=/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_fullmask_loser_br_noise

wc -l "$OUT/manifests/candidates_all.jsonl" \
      "$OUT/manifests/selected_primary_fullmask.jsonl"

python tools/inspect_generated_loser_manifest_videos.py \
  --manifest "$OUT/manifests/selected_primary_fullmask.jsonl" \
  --expect_frames 16 \
  --expect_height 320 \
  --expect_width 512 \
  --warn_prefix /home/nvme01/H20_Video_inpainting_DPO
```

If manifests contain `/home/nvme01/...` paths and data will be copied to PAI,
rewrite path prefixes with:

```bash
python tools/rewrite_generated_loser_manifest_paths.py \
  --input data/generated_losers/official_videodpo_diffueraser_data_fullmask_loser/manifests/selected_primary_fullmask.jsonl \
  --output /tmp/selected_primary_fullmask.pai.jsonl \
  --map /home/nvme01/H20_Video_inpainting_DPO=/mnt/nas/hj/H20_Video_inpainting_DPO
```

## Completed Official Experiments

Re-check these on PAI before moving/deleting anything:

- VC2 training: `logs/videodpo_vc2_dpo_official_clean/pai-vc2-dpo-official-full-gpu0-3-gb8-step3000-20260521_061414`
- VC2 full VBench: `logs/vbench_full/pai-vc2-official-step3000-full-vbench-20260521_141824`
- DiffuEraser stage1: `logs/official_diffueraser_stage1/pai-official-diffueraser-stage1-full-gpu4-7-gb8-step3000-20260521_072559`
- DiffuEraser stage2: `logs/official_diffueraser_stage2/pai-official-diffueraser-stage2-full-gpu4-7-gb8-step3000-20260521_150540`
- DiffuEraser fullmask VBench: `logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926`

## Smoke Tests Only During Refactor

Do not start full training during structural cleanup. Use:

```bash
python -m tools.offline_loser_generation --help
bash official_videodpo_diffueraser_data_fullmask_loser/scripts/run_generate_losers.sh --help
python - <<'PY'
import importlib
for name in ["official_videodpo_diffueraser", "official_videodpo_vc2"]:
    importlib.import_module(name)
    print("OK", name)
PY
```

## Common Failure Checks

- Missing NAS mount: `/mnt/nas` or `/mnt/workspace` not visible.
- Conda solver issue: `GLIBCXX_3.4.31` missing for libmamba.
- Missing runtime packages in active env: `moviepy`, `av`, `omegaconf`.
- Stale GPU processes with old `lingbotworld-phy` names from previous runs.
- Prompt strings beginning with `-` must be passed as `--prompt=<text>`, not `--prompt <text>`, otherwise argparse treats the prompt as an option.
- Host overload: high load average, many `R`/`S` processes, GPU memory occupied, and GPU util near 0. Stop the run, archive `_shards`, restart with lower `WORKERS_PER_GPU`, and keep CPU thread env vars capped.
