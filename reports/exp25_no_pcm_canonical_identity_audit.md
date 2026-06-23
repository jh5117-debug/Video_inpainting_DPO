# Exp25 DiffuEraser No-PCM Canonical Identity Audit

Date: 2026-06-23

## Decision

`DE_CANONICAL_RAW6_NO_PCM_PROP_PRIOR` is now locked as the only Exp25
DiffuEraser no-PCM identity that may feed Gate32/Gate128 calibration.

The existing Smoke6 result
`diffueraser_or_none_propainter_62d00ca9c76a` remains a valid technical smoke,
but it is **not** the canonical raw6 identity because the smoke launcher passed
`--mask_dilation_iter 8` and the old `generator_identity.json` did not record
that field.

## Canonical Raw6 No-PCM Fields

- Scheduler: `UniPCMultistepScheduler`.
- Steps: `6`.
- Guidance: `0.0`.
- PCM: disabled; no PCM LoRA weights are read.
- Prior: ProPainter prior is used.
- Mask dilation: `0`.
- Output: raw generator frames, no hard comp.
- Hard comp: allowed only for metric diagnostics, not for loser generation.
- Condition: VOR `FG_BG` / `V_obj`.
- Winner: VOR `BG` / `V_bg`.
- VOR-Eval: never used.

Config:

`exp25_vor_or_preference_data/configs/diffueraser_or_canonical_no_pcm.json`

## Evidence

Project DAVIS/BR raw6 evaluation code uses no PCM and no eval dilation:

- `tools/run_davis50_framewise_protocol_eval.py` defaults
  `--num_inference_steps 6`, `--use_pcm false`, `--mask_dilation_iter 0`.
- Exp11/Exp12/Exp23 validation launchers use `val_num_inference_steps=6`,
  `val_mask_dilation_iter=0`, and `guidance_scale=0.0`.

Official DiffuEraser OR example is a different accelerated/default stack:

- `run_diffueraser.py` defaults `mask_dilation_iter=8`.
- Official OR path uses ProPainter prior.
- Official PCM2 path maps to a TCD/PCM two-step setup.

Therefore Exp25 must keep two names separate:

- `DE_OFFICIAL_PCM2_PROP_PRIOR`: official accelerated OR baseline, still
  blocked by PCM runtime compatibility in the active environment.
- `DE_CANONICAL_RAW6_NO_PCM_PROP_PRIOR`: strict project raw6 no-PCM self-model
  identity, pending fresh canonical Smoke6 with dilation 0.

## Code Fix

The Exp25 smoke launcher and wrapper now include `mask_dilation_iter` in the
generator identity payload. Future results cannot silently inherit the old
launcher default without making it visible in the identity hash and report.

## Gate Status

`READY_GATE32 = false`

Reason: the old Smoke6 was a technical no-PCM pass under OR-style dilation 8.
A fresh canonical Smoke6 must be run with `--mask-dilation-iter 0` and visually
reviewed before Gate32 or larger source calibration starts.
