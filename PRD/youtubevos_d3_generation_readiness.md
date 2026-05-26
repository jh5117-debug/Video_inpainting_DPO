# YouTube-VOS D3 Generation Readiness

## Scope

D3 is the later data-source extension for experiment 9:

```text
source_dataset = youtubevos
generation_source = diffueraser_only
generation_model = diffueraser
mask_mode = partial
num_masks_per_video = 4
comp = true
diffueraser_inference_stack = or
diffueraser_prior_mode = propainter
```

Current plan: do not run D3 full generation until experiments 7/8 determine the
preferred partial-mask task/loss setting.

## Source Root

H20-2 source root:

```text
/home/nvme01/H20_Video_inpainting_DPO/data/external/ytbv_2019_full_resolution/train
```

Observed structure:

```text
JPEGImages/<video_id>/*.jpg
Annotations/<video_id>/*.png
```

Previous readiness scan found:

```text
JPEGImages video dirs = 3471
Annotations video dirs = 3471
valid videos with >=16 frames = 3327
```

## Clip Sampling

Use the same canonical storage setting as D2:

```text
num_frames = 16
height = 320
width = 512
frame_selection = seeded_random
frame_stride = 1
```

The D3 tool samples frame directories directly and writes canonical PNG frame
directories for `win`, `mask`, `raw`, and `comp`.

## Prompt Policy

Manual smoke inspection accepted D3 generation quality with no semantic prompt.
Therefore the current lightweight policy is:

```text
PROMPT_MODE=none
prompt = ""
prompt_source = no_prompt
```

This avoids adding a new caption model dependency before D3 is actually needed.
If later results show that D3 quality is too weak, add a separate prompt-cache
ablation instead of silently changing the official D3 policy.

Fallback video-id prompts are allowed only for smoke tests:

```text
prompt_source = fallback_video_id
```

They must not be used for final D3 training data.

## Mask And Selection Policy

Reuse D2 policies:

```text
mask_policy_config = configs/generation/videodpo_partialmask_policy_v1_medium_hard_k4.yaml
selection_config = configs/generation/medium_hard_balanced_selection_v1.yaml
```

## Output Root Plan

Official future D3 output root:

```text
/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4
```

Do not reuse the interrupted scratch root:

```text
/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4_noprompt
```

That scratch root was accidentally started before the latest plan was clarified
and was stopped at:

```text
partial_rows = 1124
done_shards = 276
failed_shards = 0
```

It is not an official D3 artifact.

## Disk Estimate

D2 PAI full data is 10000 videos x 4 masks. D3 H20 valid count is 3327 videos x
4 masks, so raw candidate count is:

```text
3327 * 4 = 13308 candidate rows
```

Disk footprint should be roughly one third of D2 if frame storage and work dirs
are comparable. Keep at least several hundred GB free before full generation,
because temporary `work/` directories can temporarily dominate usage.

## Smoke Command

Use this only after D2 audit/repair and experiment 7/8 training-readiness work:

```bash
cd /home/nvme01/H20_Video_inpainting_DPO
MODELS=diffueraser \
LINGBOT_PROCESS_NAME=lingbot-world \
PROMPT_MODE=none \
GPUS=0,1,2,3 \
WORKERS_PER_GPU=1 \
SHARD_SIZE=1 \
START_INDEX=0 \
END_INDEX=20 \
TIMEOUT_SEC=7200 \
bash scripts/h20_launch_youtubevos_partialmask_losers_k4_sharded.sh
```

Smoke gate:

- `candidates_all.jsonl = 80`
- `selected_primary_comp.jsonl = 20`
- all statuses OK
- `generation_source = diffueraser_only`
- `prompt_source = no_prompt`
- frame count 16
- frame size 512x320 storage / canonical 320x512
- comp outside-mask diff zero or near zero

## Current Blocker

D3 is not blocked technically. It is schedule-blocked: official generation waits
until experiments 7/8 decide the best task/loss setting.
