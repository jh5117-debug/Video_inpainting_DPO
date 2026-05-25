# official_videodpo_diffueraser_youtubevos_partialmask_data

Purpose: data-source ablation that extends the partial-mask preference-data construction to YouTube-VOS.

## Definition

Changed:

- Data source changes from original VideoDPO preference pairs to YouTube-VOS-derived video/mask data.
- Partial-mask inpainting preference data should follow the comp setup from `official_videodpo_diffueraser_data_partialmask_loser_comp`.
- Training is based on the `official_videodpo_diffueraser_task_partialmask` setting, so DiffuEraser receives partial masks during training.

Not changed:

- First implementation should reuse the same manifest schema.
- DPO loss and metrics are not changed.
- First version should use comp. No-comp is a later diagnostic.

## Current Status

H20-2 data root is confirmed:

```text
/home/nvme01/H20_Video_inpainting_DPO/data/external/ytbv_2019_full_resolution/train
```

It contains `JPEGImages/` and `Annotations/` with 3471 video directories each.
The current implementation supports DiffuEraser-only partial-mask K=4
generation with OR + ProPainter prior, matching the active D2 partialmask data
policy.

Do not launch this as final full generation until prompt construction and a
small H20 quality gate pass.

## Data Contract

| Field | Value |
| --- | --- |
| data source | YouTube-VOS |
| win | clean / target YouTube-VOS clip |
| loser | partial-mask inpainting generated loser |
| comp | true in first version |
| mask for loser generation | partial |
| mask for training | partial |
| changed variable | data source |

Clip sampling should reuse or audit existing YouTube-VOS SFT / DiffuEraser data code before implementation.

## Prompt Policy

YouTube-VOS has no VideoDPO-style text prompt. Prompt construction must be
cached before full generation and recorded in manifests.

Recommended first policy:

- use an open-weight video/image VLM captioner to generate one concise English
  scene prompt per YouTube-VOS video;
- store the result as JSON, keyed by YouTube-VOS video id;
- pass the JSON through `--caption_json`;
- record `prompt_source` and `prompt_model` in every candidate row.

Practical H20 defaults:

- preferred: Qwen3-VL or Qwen2.5-VL if weights and dependency versions are
  already available;
- fallback for pipeline smoke only: deterministic video-id prompt.

Fallback prompts are not acceptable for final D3 training data.

## H20 Entrypoints

Small smoke:

```bash
cd /home/nvme01/H20_Video_inpainting_DPO
MODELS=diffueraser \
LINGBOT_PROCESS_NAME=lingbot-world \
GPUS=0,1,2,3 \
WORKERS_PER_GPU=1 \
SHARD_SIZE=1 \
START_INDEX=0 \
END_INDEX=20 \
TIMEOUT_SEC=7200 \
bash scripts/h20_launch_youtubevos_partialmask_losers_k4_sharded.sh
```

With prompt cache:

```bash
CAPTION_JSON=/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4/prompts/youtubevos_prompts.json \
PROMPT_MODEL=Qwen3-VL-or-Qwen2.5-VL \
bash scripts/h20_launch_youtubevos_partialmask_losers_k4_sharded.sh --limit 20
```

## Metrics / Diagnostics

Use the same PSNR, SSIM, VBench, SBS, and DPO diagnostics policy as the VideoDPO-source partial-mask experiments.
