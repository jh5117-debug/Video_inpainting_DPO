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
FIRST_WAVE_COMPLETED
```

HAL and PAI worktrees are synchronized on:

```text
research/exp20-adaptive-region-autoresearch-20260619
64febe8122b0f67d9f5d982c7b0eba49e628ced3
```

Completed gates:

1. known prototype correctness fixes;
2. isolated real DiffuEraser Stage1 trainer wiring;
3. Exp11 `legacy_latent_exact` region/loss/prediction-gradient parity;
4. real-data 10-step smoke and checkpoint strict reload;
5. locked internal dev split with no training / DAVIS50 / YouTubeVOS100 overlap;
6. same-evaluator fixed-seed dev baselines;
7. first fixed-boundary 30-minute PAI pilot P0-P5.

Locked dev baseline:

```text
SFT_DEV_PSNR       = 29.173336
EXP11_S1_DEV_PSNR = 29.333541
EXP11_S2_DEV_PSNR = 29.355372
TARGET_DEV_PSNR   = 29.523336
```

First-wave best fixed config:

```text
P4: fixed_image_px radius=16, boundary_weight=2.0
PSNR = 29.390553
SSIM = 0.969074
LPIPS = 0.018198
Ewarp = 11.994790
strict mask PSNR = 17.085242
boundary PSNR = 22.946999
```

Interpretation:

- P4 exceeds SFT dev, Exp11-S1 dev, and Exp11-S2 dev by PSNR.
- P4 does not reach `TARGET_DEV_PSNR`.
- P4 slightly worsens LPIPS and Ewarp relative to Exp11-S1, so this is a
  dev pilot candidate, not a final success.
- No adaptive search, region-balanced search, Stage2, DAVIS50, or
  YouTubeVOS100 final evaluation has been started.

## Do Not Claim Yet

Exp20 has a first fixed-boundary dev result. Do not claim final improvement or
paper result until the next gate validates whether P4 generalizes beyond the
internal dev split.

## 2026-06-20 Fast Search + Equal-Step Outcome

Status: COMPLETED_NEGATIVE for this search budget.

Completed:
- VFID/FVD-style and TC metric assets restored and used for first-wave, region-balanced, adaptive, and equal-step summaries.
- Shadow-dev manifest locked, but not used for promotion because no candidate passed the search/equal-step gates.
- P0/P4 three-seed validation completed earlier: P4 mean PSNR exceeded P0 by about +0.0207 dB and exceeded Exp11-S1, but did not reach TARGET_DEV_PSNR.
- Second fixed-boundary roots + best-first completed.
- Region-balanced search completed.
- Calibrated adaptive search completed.
- Equal-step confirmation completed for P4, P0, BF07, RB08, AD04 at 112 optimizer steps.

Best equal-step PSNR candidate:
- EQ_BF07: fixed_image_px radius 28, boundary weight 5.0, legacy_global_weighted_mean.
- PSNR 29.393079, SSIM 0.968993, LPIPS 0.018441, VFID/FVD-style 0.232887, TC 0.975930, Ewarp 11.967787.

Decision:
- No candidate reached TARGET_DEV_PSNR = 29.523336.
- EQ_BF07 is a Pareto candidate, not a clean promotion, because the small PSNR gain comes with worse LPIPS/VFID/TC tradeoffs relative to P0/P4.
- No 500/1000/2000-step long training, Stage2, DAVIS50, or YouTubeVOS100 final evaluation was launched.

Next recommended action:
- If continuing Exp20, run a narrow BF07-vs-P4 multi-seed + shadow-dev check before any long training, or revisit the objective to reduce loser-dominant behavior rather than broadening boundary radius further.

## 2026-06-21 Multiseed Shadow Confirmation

Status: `COMPLETED_NEGATIVE_AFTER_MULTISEED_SHADOW`.

Completed:
- Candidate checkpoint identity audit for EQ_P0, EQ_P4, EQ_BF07, EQ_AD04.
- Shadow-dev baselines for SFT-48000, Exp11-S1+SFT-S2, Exp11-S2.
- P0/P4/BF07 equal-step 112-step three-seed confirmation on search-dev and shadow-dev.
- Full metric backfill for PSNR, SSIM, LPIPS, VFID/FVD-style, TC, Ewarp, strict mask PSNR, boundary PSNR.
- Paired bootstrap/sign-test/leave-one-video-out statistics.
- Codex visual review of all 20 shadow-dev contact sheets.

Shadow-dev baselines:

```text
SHADOW_SFT_PSNR       = 29.199451
SHADOW_EXP11_S1_PSNR  = 29.500778
SHADOW_EXP11_S2_PSNR  = 29.521728
```

Three-seed PSNR:

```text
search-dev:
  P0   = 29.356682 +/- 0.018347
  P4   = 29.377911 +/- 0.012939
  BF07 = 29.380092 +/- 0.010602

shadow-dev:
  P0   = 29.429970 +/- 0.055299
  P4   = 29.416158 +/- 0.057094
  BF07 = 29.371829 +/- 0.048729
```

Decision:
- P4 fails Stage1 promotion because shadow-dev mean is 0.084620 dB below Exp11-S1 and bootstrap probability is only 0.0666.
- BF07 fails Stage1 promotion because shadow-dev mean is 0.128949 dB below Exp11-S1.
- BF07 fails to replace P4: shadow mean is 0.044330 dB below P4, seed win count is 0/3, per-video win rate is 40%, and bootstrap P(delta>0) is 0.0577.
- AD04 remains a single-seed adaptive ablation; it was not expanded because it did not cleanly satisfy the conditional promotion requirements.
- Visual review found 0 BF07-better cases, 4 P4-better cases, and 16 ties.

Final:

```text
NO_CANDIDATE_PROMOTED
NO_500_STEP_GATE
NO_STAGE2
NO_DAVIS50
NO_YOUTUBE_VOS100
```

Exp20 boundary parameter search should stop here. The next useful direction is objective-level work on loser-dominant/saturation behavior, not broader radius or boundary-weight search.
