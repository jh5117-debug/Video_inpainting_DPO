# Project Structure

This repository separates code, external inputs, external weights, and generated outputs.

## Code

```text
training/sft/        SFT Stage 1/2 trainers and launchers
training/dpo/        DPO Stage 1/2 trainers, launchers, and DPO dataset
training/common/     shared experiment/output helpers
dataset/             SFT dataset code
diffueraser/         pipelines and inference wrappers
libs/                model definitions
propainter/          local ProPainter modules used by post-weight evaluation
inference/           visualization, metric, report, and sweep scripts
tools/               utility scripts
```

Compatibility wrappers are kept in the historical locations so existing commands still work.

## Inputs

```text
data/                training inputs, ignored by Git
data_val/            validation inputs, ignored by Git
weights/             pretrained/external weights, ignored by Git
archives/            compressed data archives, ignored by Git
```

These directories should be treated as read-only inputs during training.

## Outputs

```text
experiments/sft/stage1/<version>_<run_name>/
experiments/sft/stage2/<version>_<run_name>/
experiments/dpo/stage1/<version>_<run_name>/
experiments/dpo/stage2/<version>_<run_name>/
experiments/evaluation/weight_sweep/
```

Every launcher-created run directory contains:

```text
run_manifest.json    command, inputs, params, git metadata
checkpoint-*         accelerator checkpoints
converted_weights/   exported SFT weights when produced
best_weights/        best DPO weights when produced
last_weights/        last DPO weights when produced
console_logs/        captured process logs when enabled
```

Each stage directory keeps:

```text
latest               symlink to latest run when the filesystem allows it
LATEST               plain-text fallback with latest run path
```

This is weak versioning: enough to keep runs separated and reproducible at the command/input level without introducing DVC or weight/data management.

## Evaluation

After exporting a new DiffuEraser checkpoint into `weights/`, run:

```bash
bash inference/run_weight_sweep.sh
```

The runner compares available weights named `Orign`, `FT_S2_8K`, `FT_S2_26K`,
`FT_S2_34K`, and `FT_S2_48K` across the OR/BR configurations from the previous
test workspace. Missing weight directories are skipped, outputs stay under
`experiments/evaluation/weight_sweep/`, and `inference/generate_report.py`
aggregates completed `summary.json` files into an experiment report.
