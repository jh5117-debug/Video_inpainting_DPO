# Exp29 EffectErase Trainable Forward Audit

Date: 2026-06-27

Status: `EFFECTERASE_BASELINE_ONLY_FOR_NOW`

## Scope

This audit was run after the official 81-frame EffectErase inference smoke
passed. It did not run training, DPO, zero-gap, one-step, 10-step, RC-FPO, or
any optimizer update.

## Code Evidence

Repository:

`/home/hj/video_inpainting_third_party/EffectErase`

Commit:

`bcee0a5da5ef387c2ba39390dc4d579503669fb8`

Official removal inference entry:

`examples/remove_wan/infer_remove_wan.py`

Removal pipeline:

`diffsynth/pipelines/wan_video.py::WanRemovePipeline`

The official removal inference path loads:

- `remove_condition_adapter`
- `insert_condition_adapter`
- `task_fg_tokenizer`
- `CrossAttHeadProb`
- EffectErase LoRA through `ModelManager.load_lora_v2(...)`

The removal call uses `WanRemovePipeline.__call__(..., task="remove", ...)`.
That call is decorated with `torch.no_grad()` and performs denoising directly.
The class has no `training_loss` method.

Generic Wan training exists in:

`examples/wanvideo/model_training/train.py`

and uses:

`diffsynth/pipelines/wan_video_new.py::WanVideoPipeline.training_loss`

That generic path trains a WanVideoPipeline flow-matching objective with
`input_image,end_image` extra inputs. It is not wired to the EffectErase
removal-specific adapters, task token, or affected-attention head used by
`WanRemovePipeline`.

## Decision

EffectErase is now verified as an official 81-frame OR strong baseline /
diagnostic. It is not a true-adapter-ready backend in Exp29 because the audited
official code does not expose a removal-specific trainable forward suitable for:

- policy/reference clone parity;
- removal-task zero-gap;
- one-step strict reload;
- DPO or Linear-DPO loss over winner/loser removal preferences.

Final adapter-feasibility status for this audit:

`EFFECTERASE_BASELINE_ONLY_FOR_NOW`

Preserved baseline status:

`EFFECTERASE_OR_BASELINE_READY`

## Forbidden Claims

The following remain unsupported:

- `EFFECTERASE_TRUE_ADAPTER_FEASIBILITY_CONFIRMED`
- `EFFECTERASE_ZERO_GAP_PASSED`
- `EFFECTERASE_ONE_STEP_STRICT_RELOAD_PASSED`
- `SCIENTIFIC_POSITIVE`
- `UNIVERSAL_ADAPTER`
