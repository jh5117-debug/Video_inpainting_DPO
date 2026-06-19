# PRD 42: Exp20 Scale-Adaptive Region-Balanced DPO

Date: 2026-06-19

## Numbering Note

The requested PRD number `41` is already occupied by
`PRD/41_exp19_flow_strength_and_warp_supervision.md`. Following the safety rule,
Exp20 uses the next available PRD number: `42`.

## Scope

Exp20 is DiffuEraser-only. It does not modify Exp9-Exp19, shared DPO training,
`inference/metrics.py`, or current Exp11 source-of-truth.

Current best remains:

```text
Exp11 boundary outer b0.75 S2
```

## Motivation

Exp11 validated a latent-space one-cell outer boundary ring with weight `0.75`,
but it did not systematically search image-space boundary radius. Exp20 tests:

- image-space outer boundary radius;
- adaptive radius from mask scale;
- region-balanced aggregation of mask / boundary / outside log-ratios;
- safe, config-only best-first autoresearch.

## Fixed Invariants

- model: DiffuEraser;
- base: SFT-48000 for Stage1 fast search;
- win: GT clean video;
- lose: generated loser;
- mask convention: 1 = hole;
- beta: 10;
- lose gap weight: 0.25;
- loser clip tau: 1.0;
- winner abs anchor: 0.05;
- winner gap anchor: 1.0;
- outside contribution: 0.05;
- evaluation: raw6 hard-comp, no PCM, no dilation, no blur, frame-wise metrics.

## New Components

```text
exp20_autoresearch_scale_adaptive_region_dpo/code/boundary_maps.py
exp20_autoresearch_scale_adaptive_region_dpo/code/region_balanced_loss.py
exp20_autoresearch_scale_adaptive_region_dpo/code/search_controller.py
exp20_autoresearch_scale_adaptive_region_dpo/search_space.yaml
```

## Current Status

```text
PRECHECK_IMPLEMENTED
```

HAL and PAI clean worktrees have been created. Heavy PAI search is blocked
until:

1. legacy exact parity passes;
2. locked dev split is audited;
3. SFT and Exp11 dev baselines are recomputed by the same evaluator.

## Do Not Claim Yet

No Exp20 score is available yet. This PRD records implementation readiness, not
a positive result.
