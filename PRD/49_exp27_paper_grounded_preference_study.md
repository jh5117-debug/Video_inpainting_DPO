# PRD 49: Exp27 Paper-Grounded Preference Study

## Goal
Build a paper-grounded research track that compares LocalDPO, Diffusion-SDPO, and Linear-DPO against the current LoVI-style video inpainting DPO pipeline, then selects a primary and fallback method only after independent review and exact reproduction gates.

## Guardrails
- Do not modify Exp1-Exp26, shared trainers, or inference/metrics.py.
- Do not start long training before method decision and micro gates.
- Do not claim novelty that is already covered by LocalDPO, Diffusion-SDPO, or Linear-DPO.
- VOR-Eval is final-only and forbidden for selection/threshold/checkpoint choice.

## Current State
Initialized on branch `research/exp27-paper-grounded-preference-study`.

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
