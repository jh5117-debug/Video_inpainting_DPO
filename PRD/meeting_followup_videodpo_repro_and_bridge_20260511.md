# Meeting Follow-Up: VideoDPO Reproduction and DiffuEraser Bridge

Date: 2026-05-11

This note records the new direction after the advisor meeting.  It should be read together with:

- `PRD/NEXT_CHAT_FULL_CONTEXT_20260509.md`
- `PRD/DPO_Training_Metrics_Explained.md`
- `PRD/dpo_metric_regularization_prd_20260505.md`

## 1. Advisor Feedback

The previous metric-panel figure is useful for diagnosing DPO training behavior, but it is not enough to claim that VideoDPO was reproduced.  For the VideoDPO baseline, the final benchmark must include the metrics reported by the VideoDPO paper, especially VBench.  Without matching or at least carefully comparing against the paper's reported VBench setting, the reproduction is incomplete.

The second strategic point is to stop treating the current DiffuEraser-DPO failure as the only starting point.  VideoDPO is the known working reference.  The new plan is to start from VideoDPO and move toward DiffuEraser by changing one variable at a time.

## 2. Task 1: VideoDPO Reproduction With VBench

### 2.1 What is released by the official VideoDPO repo

From `/home/hj/VideoDPO/readme.md`:

- The released preference dataset is `vidpro-vc2-dataset`, i.e. the VideoCrafter2/VC2 data.
- The TODO list still says T2V-Turbo training dataset will be released later.
- The TODO list also says CogVideoX code will be released later.
- The repo contains code/configs for VC2 and T2V-Turbo.

Current interpretation:

- The open-source reproduction should first target VC2.
- T2V-Turbo can be tested because code exists, but its training data is not released in the same way as VC2.
- CogVideoX cannot be fully reproduced from this repo alone unless extra official code/checkpoints are obtained.

### 2.2 What is T2V-Turbo

T2V-Turbo is a latent consistency / distilled text-to-video model based on VideoCrafter2.  In this repo it is implemented by `lvdm.models.ddpm3d.T2VTurboDPO` and loads `checkpoints/t2v-turbo/unet_lora.pt`.  It is still a text-to-video generation model, but it uses a turbo/LCM-style few-step inference path and LoRA-style UNet weights.

### 2.3 What to do if there is no official test set

VBench is a prompt-based benchmark, not a private GT-video test set.  If the paper does not release an additional exact test set, the honest reproduction protocol is:

1. Use the official VBench prompt suite and official VBench code.
2. Generate videos from the same prompt suite with the same model family.
3. Evaluate the generated videos with VBench.
4. Record any mismatch from the paper setting: checkpoint availability, seeds, number of samples per prompt, resolution, fps, ddim steps, and whether the exact final DPO checkpoint is available.

Using the released `vidpro-vc2-dataset` training prompts as the final benchmark is not enough for a paper-level reproduction; it can be used to inspect the data format and sanity-check training, but VBench should be the reported reproduction metric.

### 2.4 SC code added for Task 1

New files:

- `DPO_finetune/scripts/sc_prepare_videodpo_vc2_assets.sbatch`
- `DPO_finetune/scripts/sc_prepare_videodpo_vc2_checkpoints.sh`
- `DPO_finetune/scripts/videodpo_env_smoke_and_export.sh`
- `DPO_finetune/scripts/sc_videodpo_vc2_train.sbatch`
- `DPO_finetune/scripts/sc_videodpo_vc2_vbench.sbatch`
- `DPO_finetune/scripts/sc_videodpo_health_check.sh`
- `DPO_finetune/scripts/sc_videodpo_pull_submodules_and_health_check.sh`
- `tools/prepare_videodpo_vc2_dataset.py`
- `tools/videodpo_prepare_vbench_standard.py`
- `tools/summarize_vbench_results.py`

External dependencies are repo-local submodules:

- `external/VideoDPO`: official VideoDPO code
- `external/VBench`: official VBench code

Do not maintain sibling naked clones like `${PROJECT_DEV}/VideoDPO` or `${PROJECT_DEV}/VBench` for this workflow.  After pulling this repo on SC, initialize/update submodules from inside this repo:

```bash
source ~/.bashrc
cd "$PROJECT_DEV/Video_inpainting_DPO"
git pull --ff-only origin main
bash DPO_finetune/scripts/sc_videodpo_pull_submodules_and_health_check.sh
```

Prepare or reuse a VideoDPO-compatible conda environment:

```bash
# Official-style fresh env.  This normalizes the typo `tenosrboard` to `tensorboard`.
CREATE_ENV=1 INSTALL_REQUIREMENTS=1 CONDA_ENV=videodpo \
bash DPO_finetune/scripts/videodpo_env_smoke_and_export.sh
```

If SC must reuse the existing DiffuEraser-DPO environment temporarily, run:

```bash
CONDA_ENV=diffueraser \
bash DPO_finetune/scripts/videodpo_env_smoke_and_export.sh
```

On HAL, the existing `diffueraser` env passed a CPU-only VideoDPO smoke on 2026-05-11: key packages imported, `configs/vc2_dpo/config.yaml` loaded, and `lvdm.models.ddpm3d` imported.  It is a compatibility fallback, not the clean official VideoDPO environment.

The SC script:

- runs VC2 inference from `external/VideoDPO`;
- repeats inference for `SAMPLES_PER_PROMPT` seeds;
- renames/symlinks `0001.mp4` style outputs into VBench standard filenames `<prompt>-<sample_idx>.mp4`;
- runs VBench standard-mode evaluation;
- writes `summary.json` and `summary.csv`.

Before training/evaluation on SC, prepare the released VC2 preference dataset and check VBench:

```bash
source ~/.bashrc
cd "$PROJECT_DEV/Video_inpainting_DPO"
git pull --ff-only origin main
mkdir -p logs
bash DPO_finetune/scripts/sc_videodpo_pull_submodules_and_health_check.sh

CONDA_ENV=videodpo \
bash DPO_finetune/scripts/sc_prepare_videodpo_vc2_checkpoints.sh

sbatch --export=ALL DPO_finetune/scripts/sc_prepare_videodpo_vc2_assets.sbatch
```

This writes an absolute VideoDPO train yaml by default:

```text
${PROJECT_DATA}/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml
```

and places the dataset by default under:

```text
${PROJECT_DATA}/VideoDPO/data/vidpro-vc2-dpo-dataset
```

VBench code now comes from `external/VBench`.  VBench metric model weights may still be downloaded lazily during the first real evaluation; keep HF cache on project storage if SC home quota is tight.

Train VC2-DPO on SC:

```bash
RUN_NAME=sc-vc2-dpo-beta5000 \
MAX_OPT_STEPS=10000 \
CKPT_EVERY=1000 \
BETA_DPO=5000 \
VC2_DATA_YAML="${PROJECT_DATA}/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml" \
sbatch --export=ALL DPO_finetune/scripts/sc_videodpo_vc2_train.sbatch
```

The trained checkpoints should be under:

```text
${PROJECT_DATA}/experiments/videodpo_vc2_dpo/${RUN_NAME}/checkpoints
```

Minimal SC command:

```bash
source ~/.bashrc
cd "$PROJECT_DEV/Video_inpainting_DPO"
git pull --ff-only origin main
mkdir -p logs

sbatch --export=ALL DPO_finetune/scripts/sc_videodpo_vc2_vbench.sbatch
```

Smoke command with fewer prompts:

```bash
PROMPT_LIMIT=20 SAMPLES_PER_PROMPT=1 RUN_VBENCH=0 \
sbatch --export=ALL DPO_finetune/scripts/sc_videodpo_vc2_vbench.sbatch
```

`PROMPT_LIMIT` is inference-only smoke mode.  Full VBench standard evaluation expects the full prompt suite, so remove `PROMPT_LIMIT` when `RUN_VBENCH=1`.

Full reproduction-style command with base and DPO checkpoints:

```bash
CKPT_SPECS="vc2_base:${PROJECT_DEV}/Video_inpainting_DPO/external/VideoDPO/checkpoints/vc2/model.ckpt,vc2_dpo:${PROJECT_DATA}/experiments/videodpo_vc2_dpo/sc-vc2-dpo-beta5000/checkpoints/last.ckpt" \
SAMPLES_PER_PROMPT=5 \
sbatch --export=ALL DPO_finetune/scripts/sc_videodpo_vc2_vbench.sbatch
```

Important uncertainty:

- I do not currently know the exact SC path of the final trained VC2-DPO checkpoint.  The script therefore accepts it through `CKPT_SPECS`.
- If the exact official VideoDPO final checkpoint is not released, the result should be reported as a best-effort reproduction, not an exact reproduction.

## 3. Task 2: Move From VideoDPO Toward DiffuEraser With Minimum Variable Changes

Advisor's proposed variable isolation:

1. Keep VideoDPO data.
2. Keep VideoDPO text-to-video preference task.
3. Change only the model toward DiffuEraser.

The key complication is that DiffuEraser is not a pure text-to-video model.  It is a video inpainting model with extra BrushNet conditioning:

```text
noisy_latents + text_embedding + brushnet_cond
brushnet_cond = concat(masked_image_latent, mask)
```

To make it behave like a generation model, use a full-image mask:

- all pixels are treated as unknown/hole;
- masked image becomes black everywhere;
- DiffuEraser receives no visible region and has to synthesize the full video from the prompt.

In this codebase's current mask convention, the original DAVIS white-hole mask is inverted before being passed to BrushNet.  Therefore full-hole generation uses:

```text
conditioning_pixel_values = black image normalized to -1
masks = 0 everywhere
```

## 4. Code Added for Task 2

New dataset adapter:

- `training/dpo/dataset/videodpo_fullmask_dataset.py`

New dataset factory:

- `training/dpo/dataset/factory.py`

Updated training files:

- `training/dpo/train_stage1.py`
- `training/dpo/train_stage2.py`
- `training/dpo/scripts/run_stage1.py`
- `training/dpo/scripts/run_stage2.py`
- `training/dpo/scripts/03_dpo_stage1.sbatch`
- `training/dpo/scripts/03_dpo_stage2.sbatch`
- `DPO_finetune/scripts/03_dpo_stage1.sbatch`
- `DPO_finetune/scripts/03_dpo_stage2.sbatch`
- `scripts/sc_submit_dpo_stage1.sh`

New SC scripts:

- `DPO_finetune/scripts/sc_videodpo_fullmask_dataset_smoke.sbatch`
- `DPO_finetune/scripts/sc_videodpo_fullmask_diffueraser_stage1.sbatch`

Local smoke helper:

- `tools/create_videodpo_tiny_dataset.py`

The default dataset remains unchanged:

```text
--dpo_dataset_type diffueraser_inpainting
```

The new bridge experiment is enabled only when:

```text
--dpo_dataset_type videodpo_fullmask
```

## 5. SC Commands for Task 2

First, smoke-test the dataset adapter:

```bash
source ~/.bashrc
cd "$PROJECT_DEV/Video_inpainting_DPO"
git pull --ff-only origin main
mkdir -p logs

sbatch --export=ALL DPO_finetune/scripts/sc_videodpo_fullmask_dataset_smoke.sbatch
```

Then run a short stage1 smoke training:

```bash
RUN_NAME=sc-videodpo-fullmask-diffueraser-stage1-smoke \
MAX_STEPS=1000 \
BETA_DPO=10 \
DPO_DATA_ROOT="${PROJECT_DATA}/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml" \
sbatch --export=ALL DPO_finetune/scripts/sc_videodpo_fullmask_diffueraser_stage1.sbatch
```

If the VideoDPO dataset is not located under `${PROJECT_DATA}/VideoDPO/data/...`, set:

```bash
VIDEODPO_DATA_BASE=/path/that/contains/the/META/relative/data/root
```

For a longer run after the smoke passes:

```bash
RUN_NAME=sc-videodpo-fullmask-diffueraser-stage1-beta10 \
MAX_STEPS=10000 \
CKPT_STEPS=2000 \
VAL_STEPS=2000 \
BETA_DPO=10 \
DPO_DATA_ROOT="${PROJECT_DATA}/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml" \
sbatch --export=ALL DPO_finetune/scripts/sc_videodpo_fullmask_diffueraser_stage1.sbatch
```

## 5.1 HAL Local Smoke Before SC Queue

Because SC queue time is expensive, the full-mask bridge should be checked on HAL first.
HAL currently has local DiffuEraser weights under `/home/hj/Video_inpainting_DPO/weights` and can run a tiny synthetic VideoDPO-format smoke without downloading the full HF dataset.

Create a tiny local VideoDPO-style dataset:

```bash
cd /home/hj/Video_inpainting_DPO
python tools/create_videodpo_tiny_dataset.py \
  --output_dir /home/hj/Video_inpainting_DPO/smoke_outputs/videodpo_tiny_fullmask \
  --num_frames 20 \
  --size 96
```

Check the dataset adapter in the same `diffueraser` environment used by training:

```bash
source /home/hj/miniconda/etc/profile.d/conda.sh
conda activate diffueraser

python tools/smoke_videodpo_fullmask_dataset.py \
  --dpo_data_root /home/hj/Video_inpainting_DPO/smoke_outputs/videodpo_tiny_fullmask/train_data.yaml \
  --base_model_name_or_path /home/hj/Video_inpainting_DPO/weights/stable-diffusion-v1-5 \
  --resolution 64 \
  --nframes 16 \
  --index 0
```

For a one-step training smoke on one 3090, use 8-bit Adam.  Plain AdamW can OOM at optimizer-state initialization even at 64px because the smoke still loads policy/ref UNet and BrushNet.

```bash
cd /home/hj/Video_inpainting_DPO

CUDA_VISIBLE_DEVICES=3 \
PROJECT_DEV=/home/hj \
PROJECT_DATA=/home/hj \
CONDA_ENV=/home/hj/miniconda/envs/diffueraser \
WEIGHTS_DIR=/home/hj/Video_inpainting_DPO/weights \
DPO_DATA_ROOT=/home/hj/Video_inpainting_DPO/smoke_outputs/videodpo_tiny_fullmask/train_data.yaml \
DPO_DATASET_TYPE=videodpo_fullmask \
VIDEODPO_DATA_BASE=/home/hj/Video_inpainting_DPO/smoke_outputs/videodpo_tiny_fullmask \
VIDEODPO_REPO=/home/hj/Video_inpainting_DPO/external/VideoDPO \
VAL_DATA_DIR=/home/hj/Video_inpainting_DPO/data/external/davis_432_240 \
EXPERIMENTS_DIR=/home/hj/Video_inpainting_DPO/smoke_outputs/experiments \
RUN_NAME=hal-videodpo-fullmask-diffueraser-stage1-smoke \
RUN_VERSION=local_smoke_20260511_codex_8bit \
MAX_STEPS=1 \
CKPT_STEPS=9999 \
VAL_STEPS=9999 \
RESOLUTION=64 \
NFRAMES=4 \
NUM_GPUS=1 \
BATCH_SIZE=1 \
GRAD_ACCUM=1 \
BETA_DPO=10 \
MIXED_PRECISION=bf16 \
VAE_DTYPE=fp32 \
GRADIENT_CHECKPOINTING=0 \
SPLIT_POS_NEG_FORWARD=1 \
CHUNK_ALIGNED=0 \
USE_8BIT_ADAM=1 \
WANDB_MODE=offline \
WANDB_PROJECT=DPO_Diffueraser_smoke \
bash DPO_finetune/scripts/03_dpo_stage1.sbatch
```

Observed HAL smoke result on 2026-05-11: dataset adapter passed, one-step training completed, DPO loss was about `0.6918`, and `last_weights` were saved under `smoke_outputs/experiments/dpo/stage1/local_smoke_20260511_codex_8bit_hal-videodpo-fullmask-diffueraser-stage1-smoke/last_weights`.

## 6. Important Uncertainties and Safety Rules

Do not hide these uncertainties:

- I do not know the actual SC path for the downloaded HF `JiaHuang01/vidpro10k-vc2-dataset`; scripts therefore use `DPO_DATA_ROOT` and `VIDEODPO_DATA_BASE`.
- Official VideoDPO and VBench code should come from repo-local submodules under `external/`; run `git submodule update --init --recursive` after pulling.
- I do not know whether the official final VC2-DPO checkpoint is available.  Pass it through `CKPT_SPECS`.
- Full-mask DiffuEraser generation is a deliberate distribution shift; it is a smoke experiment, not yet a validated replacement for VC2.

Do not do these:

- Do not hardcode H20 paths like `/home/nvme01/...` into SC scripts.
- Do not overwrite existing DiffuEraser DPO defaults.
- Do not claim paper-level VideoDPO reproduction before VBench is computed.
- Do not compare the full-mask DiffuEraser experiment against DiffuEraser-BR DPO as if only the model changed unless the data/task settings are explicitly documented.

## 7. Recommended This-Week Order

1. Run HAL local `videodpo_fullmask` dataset and one-step training smoke before spending SC queue time.
2. Run `sc_videodpo_vc2_vbench.sbatch` smoke with `PROMPT_LIMIT=20`.
3. Run full VC2 VBench for base VC2 and the best available VC2-DPO checkpoint.
4. Run short `MAX_STEPS=1000` full-mask DiffuEraser DPO on SC.
5. Inspect diagnostics: `implicit_acc`, `win_gap`, `lose_gap`, `mse_w/ref_mse_w`, `mse_l/ref_mse_l`, and generated samples.
6. Only then scale to 10k steps.
