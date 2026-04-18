# H20 Video Inpainting DPO

DiffuEraser video inpainting finetuning project with two isolated training tracks:

1. SFT finetunes DiffuEraser in two stages.
2. DPO adapts VideoDPO-style preference learning to DiffuEraser.

## Directory Layout

```text
H20_Video_inpainting_DPO/
├── training/
│   ├── sft/                    # SFT Stage 1/2 training implementation
│   ├── dpo/                    # DPO Stage 1/2 training + DPO dataset
│   └── common/                 # shared paths, manifests, validation helpers
├── scripts/                    # compatibility launchers for SFT
├── DPO_finetune/               # compatibility launchers + DPO docs
├── dataset/                    # SFT dataset code only
├── diffueraser/                # DiffuEraser pipelines and inference wrappers
├── libs/                       # model definitions
├── propainter/                 # local ProPainter support code for evaluation
├── inference/                  # visualization, metrics, report, and sweep tools
├── tools/                      # repository tools
├── docs/                       # engineering notes
├── PRD/                        # reports and project notes
├── data/                       # external input datasets, ignored
├── data_val/                   # external validation inputs, ignored
├── weights/                    # external model weights, ignored
└── experiments/                # generated run outputs, ignored
```

The old entrypoints remain as thin wrappers:

```text
train_DiffuEraser_stage1.py      -> training/sft/train_stage1.py
train_DiffuEraser_stage2.py      -> training/sft/train_stage2.py
DPO_finetune/train_dpo_stage1.py -> training/dpo/train_stage1.py
DPO_finetune/train_dpo_stage2.py -> training/dpo/train_stage2.py
```

## Output Isolation

New launcher defaults write generated outputs to:

```text
experiments/
├── sft/
│   ├── stage1/<version>_<run_name>/
│   └── stage2/<version>_<run_name>/
└── dpo/
    ├── stage1/<version>_<run_name>/
    └── stage2/<version>_<run_name>/
```

Each run directory gets a `run_manifest.json` with command, inputs, key params, git branch/commit/status, and output path. Each stage also maintains a best-effort `latest` symlink plus a `LATEST` text file.

## Quick Start

SFT:

```bash
python scripts/run_train_stage1.py
python scripts/run_train_stage2.py
```

DPO:

```bash
python DPO_finetune/scripts/run_dpo_stage1.py --chunk_aligned
python DPO_finetune/scripts/run_dpo_stage2.py --chunk_aligned
```

H20 server, no SLURM:

```bash
cd /home/nvme01/H20_Video_inpainting_DPO
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/h20_run_dpo_stage1.sh
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/h20_run_dpo_stage2.sh
```

The H20 launchers keep all caches and generated files inside the project by
default. Runtime paths resolve to:

```text
project root:   /home/nvme01/H20_Video_inpainting_DPO
DPO data:       data/external/DPO_Finetune_data
validation:     data/external/davis_432_240
weights:        weights/
outputs:        experiments/dpo/
```

Override `CUDA_VISIBLE_DEVICES`, `NUM_GPUS`, `DPO_DATA_ROOT`, `WEIGHTS_DIR`,
`VAL_DATA_DIR`, `EXPERIMENTS_DIR`, `RUN_NAME`, or `RUN_VERSION` from the shell
when needed.

Evaluation / visualization after a new weight is exported:

```bash
bash inference/run_weight_sweep.sh
bash inference/start_weight_sweep.sh
```

The historical entrypoints are kept as wrappers:

```bash
bash run_24exp.sh
bash start_24exp.sh
```

The sweep runner reads DAVIS inputs from `data/external/`, model assets from `weights/`,
and writes generated videos, metrics, logs, and reports to
`experiments/evaluation/weight_sweep/`. BR runs compute pixel metrics by default;
OR runs generate videos by default and can enable VBench with `OR_ENABLE_VBENCH=1`.

SLURM compatibility entrypoints are still available:

```bash
sbatch scripts/02_train_all.sbatch
sbatch DPO_finetune/scripts/03_dpo_stage1.sbatch
sbatch DPO_finetune/scripts/03_dpo_stage2.sbatch
```

Useful environment overrides:

```bash
EXPERIMENTS_DIR=/path/to/experiments
RUN_NAME=my_ablation
RUN_VERSION=20260418_a
DATA_DIR=/path/to/data
WEIGHTS_DIR=/path/to/weights
```

Weights and datasets are intentionally not managed by this repo.

## External Assets

Large datasets, pretrained weights, and compressed backups are kept out of Git.
Use the project-local ignored directories for runtime:

```text
data/external/     unpacked datasets
weights/           model weights
archives/          temporary compressed archives before/after HF transfer
```

The canonical compressed archives live on Hugging Face:

```text
https://huggingface.co/datasets/JiaHuang01/DPO_Finetune_Data/blob/main/DPO_Finetune_data.tar.gz
https://huggingface.co/datasets/JiaHuang01/DPO_Finetune_Data/blob/main/DiffuEraser_DAVIS_YouTubeVOS_datasets_20260418.tar.zst
https://huggingface.co/datasets/JiaHuang01/DPO_Finetune_Data/blob/main/DiffuEraser_runtime_weights_20260418.tar.zst
```

Do not commit unpacked data, weights, generated experiments, or local archives to
GitHub. The repository should contain only code, launchers, lightweight docs, and
small configuration files.
