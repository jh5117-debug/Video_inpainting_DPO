# Exp25 DiffuEraser OR Stack Audit v2

Date: 2026-06-23

## Key Clarification

PCM mode and ProPainter prior are separate variables:

- PCM mode controls DiffuEraser's inference acceleration adapter.
- ProPainter prior is the video inpainting prior fed into the DiffuEraser OR pipeline.

The current DiffuEraser core checkpoint is still the full `brushnet/` + `unet_main/` model, not LoRA. The previous smoke failure is the PCM inference adapter load path, not the DPO model.

## Candidate Stacks

### A. DE_OFFICIAL_PCM2_PROP_PRIOR

- PCM: `official_pcm2`
- prior: `propainter`
- steps: 2
- scheduler: TCD/PCM path
- policy identity: accelerated-policy variant, not strict on-policy
- status: `OFFICIAL_PINNED_ENV_SMOKE_PENDING`

### B. DE_NO_PCM_PROP_PRIOR

- PCM: `none`
- prior: `propainter`
- steps: 6
- scheduler: UniPC no-PCM path
- policy identity: strict on-policy candidate because the loser generator uses the same core checkpoint without extra PCM adapter
- status: `CONFIG_READY_SMOKE_PENDING`

### C. DE_NO_PCM_NO_PRIOR_DIAGNOSTIC

- PCM: `none`
- prior: `none`
- status: `NOT_IMPLEMENTED_FOR_PRIMARY`
- reason: this changes a second variable and should not be used as primary without separate quality evidence.

## Exp25 Wrapper

New wrapper:

`exp25_vor_or_preference_data/scripts/infer_diffueraser_or_exp25.py`

It creates an overlay project and patches only the overlay copy of `diffueraser_OR.py`.

Supported explicit modes:

- `--pcm_mode official_pcm2`
- `--pcm_mode none`

The no-PCM path does not read PCM weights and writes a separate generator identity. It is not a silent fallback.

Smoke launcher:

`exp25_vor_or_preference_data/scripts/run_vor_or_model_smoke.py`

now records `pcm_mode`, `prior_mode`, and unique `generator_id`.

## Current Status

No DiffuEraser stack has passed the fixed six-sample smoke yet.

Primary stack remains:

`PRIMARY_STACK_PENDING`

Expected default decision if smoke passes:

`DE_NO_PCM_PROP_PRIOR` becomes the primary self-model loser stack; `DE_OFFICIAL_PCM2_PROP_PRIOR` remains an official accelerated baseline/diagnostic unless OR deployment explicitly requires PCM2.
