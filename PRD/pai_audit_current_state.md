# PAI / Current State Audit

Date: 2026-05-23

This audit was run before any refactor or experiment scaffold changes. The current shell is **not seeing the previous PAI NAS workspace** used in the May 21-22 runs. The repository is available locally at `/home/hj/Video_inpainting_DPO`, but `/mnt/nas` and `/mnt/workspace` are empty or not mounted in this session. Therefore PAI artifacts recorded below are split into:

- **Verified in current session**: local code, local weights, local datasets, local docs, current git/env/GPU.
- **Recorded but not directly verified in current session**: PAI `/mnt/nas` and `/mnt/workspace` experiment outputs referenced by PRD logs.

## Basic Environment

| Item | Value |
| --- | --- |
| Host | `hal-9000` |
| User | `hj` |
| Repo path | `/home/hj/Video_inpainting_DPO` |
| Current shell cwd during audit | `/home/hj/Video_inpainting_DPO` |
| Date command | `Sat May 23 06:55:18 CEST 2026` |

This differs from the earlier PAI session paths:

- `/mnt/nas/hj/H20_Video_inpainting_DPO`
- `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO`

Those paths were not visible/mounted during this audit.

## Git State

Repository:

```text
/home/hj/Video_inpainting_DPO
branch: main
last commit: 70393a0 Document PAI DiffuEraser stage1 and stage2 status
```

Dirty worktree at audit time:

```text
 M DPO_finetune/scripts/sc_videodpo_fullmask_diffueraser_stage2.sbatch
 M PRD/DPO_Project_Complete_Summary.md
 D PRD/First_Finetuning_Summary.md
 M PRD/PAI_H20_VideoDPO_Migration_Runbook_20260515.md
 D PRD/PPT.pptx
 D PRD/stage2_motion_module_init.md
 D PRD/update_ppt_tables.py
 D PRD/validation_optimization.md
?? videodpo-vc2-dpo-diag-log/
```

Important: these changes pre-existed this audit and should not be reverted during cleanup.

## GPU / Python / Conda

Current host GPU state:

```text
0, NVIDIA GeForce RTX 3090, 1939 MiB / 24576 MiB, 0%
1, NVIDIA GeForce RTX 3090, 2131 MiB / 24576 MiB, 22%
2, NVIDIA GeForce RTX 3090, 1059 MiB / 24576 MiB, 0%
3, NVIDIA GeForce RTX 3090, 1070 MiB / 24576 MiB, 0%
4, NVIDIA GeForce RTX 3090, 1059 MiB / 24576 MiB, 0%
5, NVIDIA GeForce RTX 3090, 1057 MiB / 24576 MiB, 0%
6, NVIDIA GeForce RTX 3090, 12733 MiB / 24576 MiB, 98%
```

This is **not** the earlier PAI L20X environment.

Python:

```text
python: /home/hj/miniconda/bin/python
Python: 3.13.9
pip: /home/hj/miniconda/bin/pip, pip 25.1
```

Conda envs found:

```text
base
cococo
diffueraser
floed
minimax
vace
videopainter2
```

Conda warning:

```text
conda-libmamba-solver failed to load because GLIBCXX_3.4.31 was not found.
```

Important Python package status in the active base env:

| Package | Status |
| --- | --- |
| `torch` | OK `2.9.1+cu128` |
| `torchvision` | OK `0.24.1+cu128` |
| `diffusers` | OK `0.35.2` |
| `transformers` | OK `4.57.3` |
| `accelerate` | OK `1.12.0` |
| `decord` | OK `0.6.0` |
| `cv2` | OK `4.12.0` |
| `imageio` | OK `2.37.2` |
| `numpy` | OK `2.2.6` |
| `PIL` | OK `12.0.0` |
| `wandb` | OK `0.25.1` |
| `einops` | OK `0.8.1` |
| `moviepy` | missing |
| `av` | missing |
| `omegaconf` | missing |

## Main Code Directories

Verified local project structure includes:

| Path | Role |
| --- | --- |
| `DPO_finetune/` | Historical and current DiffuEraser / VideoDPO bridge scripts, multimodel loser generation wrappers, PAI launchers. |
| `training/dpo/` | Project-native DPO training code, stage1/stage2, dataset factory, diagnostics. |
| `training/sft/` | SFT training code. |
| `official_videodpo_diffueraser/` | Minimal official VideoDPO -> DiffuEraser bridge package. |
| `diffueraser/` | DiffuEraser model/pipeline implementation. |
| `propainter/` | ProPainter implementation. |
| `tools/` | Utility scripts including DiffuEraser VBench generation. |
| `external/VideoDPO/` | Vendored/external VideoDPO code. |
| `external/VBench/` | Vendored VBench code. |
| `patches/videodpo/` | Historical VideoDPO patches and diagnostics patch notes. |
| `videodpo-vc2-dpo-diag-log/` | Imported VC2 DPO diagnostics logs. |

`official_videodpo_diffueraser` current structure:

```text
official_videodpo_diffueraser/
  __init__.py
  data.py
  models.py
```

Key imports:

- `official_videodpo_diffueraser/data.py` imports `VideoDPOFullMaskDiffuEraserDataset`.
- `official_videodpo_diffueraser/models.py` imports `BrushNetModel`, `UNet2DConditionModel`, `UNetMotionModel`, `compute_dpo_loss`, `forward_stage1_pair_member`, and `forward_stage2_pair_member`.

This confirms the DiffuEraser official-VideoDPO bridge is implemented by importing project-native training/model code, not by copying all training logic.

## Training / DPO Diagnostics Structure

Verified core files:

```text
training/dpo/train_stage1.py
training/dpo/train_stage2.py
training/dpo/dataset/dpo_dataset.py
training/dpo/dataset/factory.py
training/dpo/dataset/videodpo_fullmask_dataset.py
training/dpo/scripts/run_stage1.py
training/dpo/scripts/run_stage2.py
```

DPO diagnostics found in current code and historical patches:

- `implicit_acc`
- `win_gap`
- `lose_gap`
- `mse_w`
- `ref_mse_w`
- `mse_l`
- `ref_mse_l`
- `loser_dominant_ratio` / `loser_degrade_ratio`
- `sigma_term`
- `grad_norm`
- `sft_reg_weight`
- `lose_gap_weight`

Primary code locations:

- `training/dpo/train_stage1.py`
- `training/dpo/train_stage2.py`
- `patches/videodpo/h20_videoinpaint_dpo_adapter.patch`
- `patches/videodpo/README.md`
- `videodpo-vc2-dpo-diag-log/loginfo/*`
- `PRD/assets/dpo_metric_analysis_20260505/*`

## Data Paths

Verified local data paths:

```text
data/external/davis_2017_full_resolution/
data/external/davis_432_240/
data/external/youtubevos_432_240/
data/external/ytbv_2019_full_resolution/
```

`data/external/youtubevos_432_240/JPEGImages_432_240/` exists and contains many sequence directories.

Common PAI/data mount checks:

```text
/mnt/nas: not mounted or empty in current session
/mnt/workspace: not mounted or empty in current session
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO: not found
/mnt/nas/hj/H20_Video_inpainting_DPO: not found
/home/hj/data: not found
/home/hj/datasets: not found
/mnt/data: not found
/mnt/datasets: not found
/root/autodl-tmp: not found
/root/autodl-fs: not found
```

Conclusion: YouTube-VOS-style local data exists under the repo; the previous PAI NAS data root is not directly verified in this session.

## Weight Paths

Verified local weight paths:

| Path | Status |
| --- | --- |
| `weights/diffuEraser/Orign_Diffueraser/brushnet/diffusion_pytorch_model.safetensors` | found |
| `weights/diffuEraser/Orign_Diffueraser/unet_main/diffusion_pytorch_model.safetensors` | found |
| `weights/diffuEraser/converted_weights_step34000/{brushnet,unet_main}/diffusion_pytorch_model.safetensors` | found |
| `weights/diffuEraser/converted_weights_step48000/{brushnet,unet_main}/diffusion_pytorch_model.safetensors` | found |
| `weights/propainter/ProPainter.pth` | found |
| `weights/propainter/raft-things.pth` | found |
| `weights/propainter/recurrent_flow_completion.pth` | found |
| `weights/Qwen2.5-VL-7B-Instruct/model-00001-of-00005.safetensors` through `00005` | found |
| `weights/stable-diffusion-v1-5/` | found |
| `weights/sd-vae-ft-mse/` | found |
| `weights/animatediff-motion-adapter-v1-5-2/` | found |
| `weights/PCM_Weights/` | found |
| `weights/i3d_rgb_imagenet.pt` | found |
| `weights/metrics/sa_0_4_vit_l_14_linear.pth` | found |

Hugging Face cache also contains:

```text
/home/hj/.cache/huggingface/hub/models--lixiaowen--diffuEraser
/home/hj/.cache/huggingface/hub/models--zibojia--minimax-remover
```

No obvious local CoCoCo model weight directory was confirmed during this audit. CoCoCo may rely on Qwen caption generation plus external code/config, but the actual generation checkpoint is **unconfirmed**.

## Four Inpainting / Loser Generation Models

### DiffuEraser

Status: code and weights found locally.

Code:

- `diffueraser/`
- `DPO_finetune/infer_diffueraser_candidate.py`
- `tools/generate_diffueraser_fullmask_vbench.py`
- `DPO_finetune/scripts/h20_diffueraser_fullmask_vbench.sh`

Weights:

- `weights/diffuEraser/Orign_Diffueraser`
- `weights/diffuEraser/converted_weights_step34000`
- `weights/diffuEraser/converted_weights_step48000`

Env:

- `diffueraser` conda env exists.

Generation scripts: found.

### ProPainter

Status: code and weights found locally.

Code:

- `propainter/`
- `DPO_finetune/infer_propainter_candidate.py`
- `inference/run_OR.py`

Weights:

- `weights/propainter/ProPainter.pth`
- `weights/propainter/raft-things.pth`
- `weights/propainter/recurrent_flow_completion.pth`

Generation scripts: found.

### CoCoCo

Status: code wrappers and conda env found; model weights not confirmed.

Code:

- `DPO_finetune/infer_cococo_candidate.py`
- `DPO_finetune/generate_cococo_captions_qwen.py`
- `DPO_finetune/scripts/generate_cococo_captions_h20.sh`

Env:

- `cococo` conda env exists.

Related weights:

- `weights/Qwen2.5-VL-7B-Instruct/` found for captioning.

Unconfirmed:

- The actual CoCoCo video inpainting model checkpoint / runnable generation environment was not clearly found in the current session.

### MiniMax-Remover

Status: code wrapper and conda env found; Hugging Face cache exists; runnable state not confirmed.

Code:

- `DPO_finetune/infer_minimax_candidate.py`

Env:

- `minimax` conda env exists.

Weights/cache:

- `/home/hj/.cache/huggingface/hub/models--zibojia--minimax-remover` found.

Unconfirmed:

- Whether the cached MiniMax-Remover model is complete and directly runnable.
- Whether any additional preprocessing/runtime dependency is required.

## Existing Official Experiment Outputs

Current session cannot verify the PAI artifact directories because `/mnt/nas` and `/mnt/workspace` are not mounted. The following are recorded in PRD files and prior terminal logs, but should be rechecked on the PAI node before deleting or moving anything there.

### official-VideoDPO VC2

Recorded paths:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/videodpo_vc2_dpo_official_clean/pai-vc2-dpo-official-full-gpu0-3-gb8-step3000-20260521_061414/checkpoints/last.ckpt
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai-vc2-official-step3000-full-vbench-20260521_141824
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/qual_sbs_30/vc2_and_diffueraser_20260522/tables/vc2_4row_paper_style.md
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/logs/qual_sbs_30/vc2_and_diffueraser_20260522/vc2_base_vs_official_videodpo_samplemix
```

Recorded final score:

| Model | Total | Quality | Semantic | MeanRaw |
| --- | ---: | ---: | ---: | ---: |
| Official VC2 VideoDPO step3000 | 80.5997 | 82.8055 | 71.7763 | 0.6596 |

### official-VideoDPO DiffuEraser

Recorded paths:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/official_diffueraser_stage1/pai-official-diffueraser-stage1-full-gpu4-7-gb8-step3000-20260521_072559/last_weights
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/official_diffueraser_stage2/pai-official-diffueraser-stage2-full-gpu4-7-gb8-step3000-20260521_150540/last_weights
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926
```

Recorded final VBench:

| Model | Total | Quality | Semantic | MeanRaw |
| --- | ---: | ---: | ---: | ---: |
| DiffuEraser-Base-Fullmask | 64.6162 | 74.4651 | 25.2204 | 0.3935 |
| DiffuEraser-Stage2-Fullmask | 73.6463 | 78.4804 | 54.3099 | 0.5560 |
| Delta | +9.0301 | +4.0153 | +29.0894 | +0.1625 |

Paper-style dimensions recorded:

| Model | Total | Motion smooth. | Dynamic degree | Aesthetic quality | Object class | Multiple objects | Human action | Spatial relation. | Scene | Appear. style | Subject consist. | Back. consist. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DiffuEraser Base Fullmask | 64.62 | 98.33 | 0.28 | 36.28 | 14.18 | 2.15 | 6.20 | 9.14 | 0.47 | 22.78 | 99.44 | 99.09 |
| DiffuEraser Stage2 Fullmask | 73.65 | 97.30 | 44.72 | 51.77 | 69.08 | 24.59 | 66.20 | 26.03 | 27.49 | 23.79 | 95.87 | 98.34 |
| Delta | +9.03 | -1.03 | +44.44 | +15.49 | +54.91 | +22.44 | +60.00 | +16.89 | +27.02 | +1.01 | -3.57 | -0.76 |

## Existing PRD / Docs

Active or still useful PRD files:

- `PRD/DPO_Project_Complete_Summary.md`
- `PRD/PAI_H20_VideoDPO_Migration_Runbook_20260515.md`
- `PRD/DPO_Training_Metrics_Explained.md`
- `PRD/dpo_metric_regularization_prd_20260505.md`
- `PRD/CURRENT_STATUS_20260512.md`
- `PRD/code_structure_review.md`
- `PRD/meeting_followup_videodpo_repro_and_bridge_20260511.md`

Likely archive candidates:

- `PRD/NEXT_CHAT_FULL_CONTEXT_20260509.md`
- `PRD/NEXT_CHAT_PROMPT_20260512.md`
- `PRD/PROJECT_HANDOFF_20260509.md`
- `PRD/Project_Complete_Report.md`
- `PRD/README_FOR_NEXT_CHAT.md`
- `PRD/sft_pipeline_enhancement.md`
- `PRD/video_inpainting_literature_review.md`
- `PRD/video_inpainting_papers_summary.md`
- Deleted-but-still-in-git old files shown by `git status`, if they still need historical preservation before final cleanup.

Do not delete old PRD files directly. Move historical docs into `PRD/archive/` only after extracting still-current information into the new active PRD set.

## Risks / Unconfirmed Items

1. This audit was not run on the visible PAI L20X node. It ran on `hal-9000` with RTX 3090 GPUs.
2. `/mnt/nas` and `/mnt/workspace` PAI paths were not mounted/verified here.
3. Official VC2 and DiffuEraser final outputs are recorded in PRDs, but not directly accessible in this session.
4. CoCoCo generation checkpoint and exact runnable environment are not confirmed.
5. MiniMax-Remover has a Hugging Face cache and env, but direct runnable state is not confirmed.
6. The base conda env is missing `moviepy`, `av`, and `omegaconf`; those may exist in task-specific envs, but this was not exhaustively tested.
7. Conda reports a `GLIBCXX_3.4.31` issue for `conda-libmamba-solver`; package installs/env solves may need the classic solver or a fixed libstdc++ setup.
8. Git worktree is dirty before refactor; cleanup must preserve user changes and avoid reverting deleted/modified PRDs accidentally.

## Recommendation Before Refactor

Proceed with refactor/scaffold only after deciding whether the next phase should happen in this local `/home/hj/Video_inpainting_DPO` checkout or on the PAI-mounted `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO` checkout. If the goal is to clean PAI experiment outputs and preserve PAI artifacts, reconnecting to the PAI node or mounting `/mnt/nas` is required before any destructive operation.
