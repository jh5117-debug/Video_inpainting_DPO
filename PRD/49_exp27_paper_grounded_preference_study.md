# PRD 49: Exp27 Paper-Grounded Preference Study

## Goal
Build a paper-grounded research track that compares LocalDPO, Diffusion-SDPO, and Linear-DPO against the current LoVI-style video inpainting DPO pipeline, then selects a primary and fallback method only after independent review and exact reproduction gates.

## Guardrails
- Do not modify Exp1-Exp26, shared trainers, or inference/metrics.py.
- Do not start long training before method decision and micro gates.
- Do not claim novelty that is already covered by LocalDPO, Diffusion-SDPO, or Linear-DPO.
- VOR-Eval is final-only and forbidden for selection/threshold/checkpoint choice.

## Current State

Status: `PAPER_REVIEW_COMPLETE`

Status: `EXACT_BASELINE_REPRODUCTION_IN_PROGRESS`

Status: `NO_LONG_TRAINING`

Branch: `research/exp27-paper-grounded-preference-study`.

## 2026-06-23 Update

- Downloaded and hashed LocalDPO, Diffusion-SDPO, and Linear-DPO PDFs.
- Cloned official repositories and recorded commits/licenses.
- Completed five independent review passes A-E.
- Completed CPU parity helpers:
  - SDPO scalar safe-lambda parity passed.
  - Linear-DPO utility and EMA parity passed.
  - LocalDPO official random-mask code is blocked by a runtime error in the official code path.
- Selected primary candidate:
  `RC-FPO`, Restoration-Critical Failure-Structured Preference Optimization.
- Selected fallback:
  `ST-Pref`, Stage-Aware Spatial/Temporal Preference Decomposition.
- Paused Region-SDPO and Linear-DPO as baselines/ablations rather than primary novelty.

No long Exp27 training has been started.

## 2026-06-23 LocalDPO Runtime Diagnosis

The official LocalDPO random mask generator fails in the current dependency
environment even under the official default image size. Root cause: official
`random_mask_gen.py` reads a 4-channel `tostring_argb()` matplotlib canvas
buffer and reshapes it as 3-channel RGB. After that is fixed, the file also
uses `cv2.resize` without importing `cv2`.

Exp27 now adds an isolated compatibility wrapper that does not edit the
official clone:

`exp27_paper_grounded_preference_study/code/localdpo_compat.py`

Mask-only compatibility status:

`OFFICIAL_CODE_COMPATIBILITY_PATCH_MASK_ONLY_PASSED`

Faithful LocalDPO baseline remains incomplete until progressive corruption,
outside latent reinjection/fusion, six-video pair smoke, and 1/10-step training
smoke pass.

SDPO and Linear-DPO remain toy parity only; real DiffuEraser-batch parity is
pending.

## 2026-06-23 LocalDPO Fusion Primitive

Added:

`exp27_paper_grounded_preference_study/code/localdpo_full_adapter.py`

This isolates LocalDPO's core outside-preservation semantics for the
DiffuEraser adaptation:

- task mask, corruption mask, and restoration-critical region are distinct;
- corruption-mask inside uses the denoised/current latent;
- outside the corruption mask reinjects the re-noised original latent at every
  progressive denoising step.

This is an algorithm-primitive parity step, not a full LocalDPO baseline. The
remaining required gates are single-video local corruption, six-video pair
generation, real DiffuEraser-batch SDPO parity, real DiffuEraser-batch
Linear-DPO parity, and 1/10-step micro training.
