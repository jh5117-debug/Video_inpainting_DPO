# D2 Training Entry Readiness

## Data Asset

D2 is complete on PAI:

```text
root = /mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4
candidate_rows = 40000 / 40000
selected_primary_comp = 10000
selected_primary_nocomp = 10000
failed_shards = 0
generation_source = diffueraser_only after repair
```

Before training, run:

```bash
python tools/d2_post_generation_audit_and_repair.py \
  --output_root /mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4
```

Then run the final training-readiness check:

```bash
OUT=/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4
python tools/d2_training_readiness_check.py --output_root "$OUT"
```

Expected output:

```text
reports/d2_training_readiness_report.md
selected_primary_comp.repaired.jsonl = 10000
selected_primary_nocomp.repaired.jsonl = 10000
selected_secondary_comp.repaired.jsonl = 10000
selected_secondary_nocomp.repaired.jsonl = 10000
```

The readiness check samples 100 selected comp rows and verifies path existence,
frame decode, 16-frame count, 512x320 storage / canonical 320x512 resolution,
mask readability, and outside-mask comp difference. It also rejects
`/home/nvme01/...` training paths on PAI and writes `.pai_paths.jsonl` copies if
an H20-only prefix is detected.

Use repaired manifests only:

```text
Experiment 5: manifests/selected_primary_comp.repaired.jsonl
Experiment 6: manifests/selected_primary_nocomp.repaired.jsonl
Experiment 7: manifests/selected_primary_comp.repaired.jsonl + mask_path
Experiment 8: manifests/selected_primary_comp.repaired.jsonl + mask_path
```

## Current Training-Code Support

Current code now supports the D2 generated-loser manifest interface through:

```text
dpo_dataset_type = generated_loser_manifest
adapter = training/dpo/dataset/generated_loser_manifest_dataset.py
smoke = tools/smoke_manifest_dataset.py
```

Supported arguments:

```text
--preference_manifest PATH
--mask_from_manifest true|false
--train_mask_mode full|partial
--loss_region_mode full|region
--enable_dpo_diag true|false
--dpo_diag_log_every 10
--dpo_diag_save_csv true
--dpo_diag_save_wandb true
```

`loss_region_mode=region` is intentionally blocked in training for now. It has
a dataset/CLI entrypoint for experiment 8, but the region-weighted loss wrapper
has not been implemented yet. Do not train experiment 8 until that wrapper is
added around the existing `compute_dpo_loss`.

## PAI Smoke Results On 2026-05-26

### Dataset Smoke

Command:

```bash
OUT=/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4

python tools/smoke_manifest_dataset.py \
  --d2_root "$OUT" \
  --sample_size 8 \
  --report "$OUT/reports/manifest_dataset_smoke_report.md"
```

Result:

```text
exp5_comp_full rows=10000 issues=0
exp6_nocomp_full rows=10000 issues=0
exp7_comp_partial rows=10000 issues=0
exp8_comp_partial_region_dataset_only rows=10000 issues=0
total_issues=0
```

Report:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4/reports/manifest_dataset_smoke_report.md
```

### Stage1 Five-Step Smoke

All Stage1 smoke runs used:

```bash
export PROJECT_ROOT=/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO
export CONDA_ENV_PREFIX=/mnt/nas/hj/conda_envs/diffueraser
export WEIGHTS_DIR=/mnt/nas/hj/weights
export EXPERIMENTS_DIR=/mnt/nas/hj/H20_Video_inpainting_DPO/experiments
export LINGBOT_PROCESS_NAME=lingbot-worldphy
export DPO_DATASET_TYPE=generated_loser_manifest
export ENABLE_DPO_DIAG=true
export NUM_GPUS=8
export MAX_STEPS=5
export REPORT_TO=none
export LOGGING_STEPS=1
export DPO_DIAG_LOG_EVERY=1
export NUM_WORKERS=0
export TRAIN_HEIGHT=320
export TRAIN_WIDTH=512
export RESOLUTION=512
export NFRAMES=16
```

Required PAI weight paths were confirmed:

```text
/mnt/nas/hj/weights/stable-diffusion-v1-5
/mnt/nas/hj/weights/sd-vae-ft-mse
/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000
```

Important note: when `DPO_DATASET_TYPE=generated_loser_manifest`, the printed
`DPO Data Root` may still show the old SC default path, but the dataset actually
reads `PREFERENCE_MANIFEST`. This is acceptable for these smoke runs.

Results:

| Experiment | Manifest | train_mask_mode | mask_from_manifest | loss_region_mode | Result |
| --- | --- | --- | --- | --- | --- |
| Exp5 | `selected_primary_comp.repaired.jsonl` | `full` | `false` | `full` | 5/5 steps passed, `dpo=0.6658`, `acc=1.0000` |
| Exp6 | `selected_primary_nocomp.repaired.jsonl` | `full` | `false` | `full` | 5/5 steps passed, `dpo=0.6521`, `acc=1.0000` |
| Exp7 | `selected_primary_comp.repaired.jsonl` | `partial` | `true` | `full` | 5/5 steps passed, `dpo=0.6533`, `acc=1.0000` |
| Exp8 | `selected_primary_comp.repaired.jsonl` | `partial` | `true` | `region` | not trained; region loss wrapper pending |

Smoke log paths on PAI:

```text
logs/smoke/exp5_stage1_smoke5.log
logs/smoke/exp6_stage1_smoke5.log
logs/smoke/exp7_stage1_smoke5.log
```

Smoke output roots:

```text
/sc-projects/sc-proj-cc09-repair/hongyou/dev/data/Video_inpainting_DPO/experiments/dpo/stage1/20260526_061331_exp5_d2_comp_k4_stage1_smoke5
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260526_062023_exp6_d2_nocomp_k4_stage1_smoke5
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260526_062339_exp7_d2_comp_partialmask_stage1_smoke5
```

Exp5 was first run before `EXPERIMENTS_DIR` was redirected to NAS, so its smoke
output landed under `/sc-projects/...`. This is acceptable as smoke evidence,
but official training should set:

```bash
export EXPERIMENTS_DIR=/mnt/nas/hj/H20_Video_inpainting_DPO/experiments
```

## Current Recommendation

Experiment 5 is ready for formal Stage1 training. Experiment 6 and 7 are also
smoke-ready, but should wait until Exp5 is running or has completed the first
planned checkpoint. Experiment 8 remains blocked on region-weighted loss.

Formal Exp5 Stage1 launch baseline:

```bash
cd /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO

export PROJECT_ROOT=/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO
export CONDA_ENV_PREFIX=/mnt/nas/hj/conda_envs/diffueraser
export WEIGHTS_DIR=/mnt/nas/hj/weights
export EXPERIMENTS_DIR=/mnt/nas/hj/H20_Video_inpainting_DPO/experiments

export LINGBOT_PROCESS_NAME=lingbot-worldphy
export DPO_DATASET_TYPE=generated_loser_manifest
export PREFERENCE_MANIFEST=/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4/manifests/selected_primary_comp.repaired.jsonl
export TRAIN_MASK_MODE=full
export MASK_FROM_MANIFEST=false
export LOSS_REGION_MODE=full
export ENABLE_DPO_DIAG=true

export RUN_NAME=exp5_d2_comp_k4_stage1_full
export NUM_GPUS=8
export MAX_STEPS=20000
export REPORT_TO=wandb
export LOGGING_STEPS=10
export DPO_DIAG_LOG_EVERY=10
export CKPT_STEPS=2000
export CKPT_LIMIT=2
export VAL_STEPS=2000
export NUM_WORKERS=0
export TRAIN_HEIGHT=320
export TRAIN_WIDTH=512
export RESOLUTION=512
export NFRAMES=16

mkdir -p logs/train
nohup bash training/dpo/scripts/03_dpo_stage1.sbatch \
  > logs/train/exp5_d2_comp_k4_stage1_full.log 2>&1 &
echo $! > logs/train/exp5_d2_comp_k4_stage1_full.pid
```

Monitor:

```bash
tail -f logs/train/exp5_d2_comp_k4_stage1_full.log
watch -n 30 'date; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits'
```

Expected tensor keys from the adapter remain:

```text
pixel_values_pos
pixel_values_neg
conditioning_pixel_values
masks
input_ids
```
