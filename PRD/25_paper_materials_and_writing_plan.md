# PRD 25: Paper Materials And Writing Plan

Date: 2026-06-15

## Current Writing Goal

Close the experimental story around the current best method:

```text
Exp11 boundary outer b0.75 S2
```

This method is currently best under the fixed DAVIS50 / YouTubeVOS100 raw6 hard-comp protocol.

## Core Story

### Problem

Global video DPO is not directly suitable for video inpainting. The objective can over-optimize preference gaps while ignoring where the actual reconstruction error matters. In practice this produced mask-local artifacts such as purple fog, paste-like blobs, grid patterns, and boundary discontinuities.

### Observations

1. GT winner helps because dirty rollout winners inject the wrong preference signal.
2. Raw `win_gap` / `lose_gap` scale is unstable; old raw DPO runs showed saturation and winner-gap explosion.
3. Full-frame DPO optimizes too much background and not enough hole / boundary region.
4. Stage2 DPO is useful only after the Stage1 objective is stabilized; otherwise it can amplify artifacts.

### Method

The current method combines:

- winner-anchor regularization
- log-ratio normalized gap
- clipped loser gap
- region-local loss
- boundary-aware weighting, with the best current setting using an outer boundary ring at weight `0.75`

In short: push DPO back to the actual inpainting region and boundary instead of letting global preference loss dominate the whole frame.

### Current Best

```text
Exp11 outer b0.75 S2
DPO-S1 + DPO-S2
boundary mode = outer
boundary weight = 0.75
outside weight = 0.05
```

## Evidence Chain

### 1. Main Metrics

Use the fixed protocol only:

```text
DAVIS50 / YouTubeVOS100
raw6
D+G off
no PCM
no mask dilation
no Gaussian blur
hard comp
frame-wise in-memory metric
```

Main metric report:

```text
reports/exp11_outer_b075_s2_youtubevos100_davis50_eval.md
reports/exp11_outer_b075_s2_youtubevos100_davis50_eval.csv
```

Result summary:

- DAVIS50 PSNR: SFT-48000 `32.7314` -> Exp11 outer b0.75 S2 `33.0140`
- YouTubeVOS100 PSNR: SFT-48000 `33.3968` -> Exp11 outer b0.75 S2 `33.7238`
- LPIPS and VFID improve on both datasets.
- Mask PSNR improves on both datasets.

### 2. Qualitative Evidence

Final 20 paper/PPT-ready cases:

```text
/home/hj/dpo-2-1-exp/final_20_visual_cases_for_paper
reports/final_20_visual_cases_for_paper_summary.md
reports/final_20_visual_cases_for_paper_summary.csv
```

Selected visual evidence:

```text
/home/hj/dpo-2-1-exp/this_week_exp11_exp12/visual_evidence/exp11_outer_b075_s2_selected_visuals
reports/exp11_outer_b075_s2_visual_evidence_report.md
```

Lead figure candidate:

- `boat`: strongest positive case; baseline has white fog / patch around wake and hull, while Exp11 keeps cleaner water texture and boundary continuity.

Other positive cases:

- `rhino`
- `dog-agility`
- `lucia`
- `blackswan`

Caution / failure cases:

- `dance-jump`
- `soccerball`

These two should not be used as positive evidence because their per-video metric is below baseline.

### 3. DPO Diagnostics

DPO diagnostic summary:

```text
reports/exp11_outer_b075_s2_dpo_diag_summary.md
```

The key interpretation is balanced:

- no old-style raw-DPO winner-gap explosion
- Stage2 remains stable enough to improve metrics
- diagnostics are still loser-dominant, so the paper/PPT should report metric + visual + dpo-diag rather than scores alone

## Paper Material Order

1. Introduction draft
2. Related Work taxonomy
3. Method overview figure
4. Experiment setting table
5. Main result table
6. Ablation table

## Introduction Draft Points

- Video inpainting is different from generic video generation because the known context must remain stable and the hole boundary must be consistent.
- Directly applying video DPO to inpainting can improve preference scores but create local artifacts in the masked region.
- The main issue is spatial mismatch: global DPO rewards are not localized to the mask and boundary.
- We propose a boundary-aware, region-local DPO formulation with normalized gaps and winner anchoring.
- The final setting improves over the strong SFT-48000 DiffuEraser baseline on DAVIS50 and YouTubeVOS100 under a fixed raw6 hard-comp evaluation protocol.

## Related Work Taxonomy

1. Traditional / propagation-based video inpainting
2. Diffusion / DiT video inpainting
3. Preference optimization for generative video
4. Region-aware and mask-aware training objectives
5. Evaluation protocols for video inpainting

## Method Figure Plan

Figure panels:

1. Input video + binary mask
2. GT winner and generated loser pair
3. Frozen SFT reference and DPO policy
4. Log-ratio normalized gap and clipped loser gap
5. Region weighting: mask / outer boundary / outside
6. Final DPO loss and diagnostics

## Experiment Setting Table

Required rows:

- datasets: D3 YouTube-VOS generated loser, GT winner, partial-mask inpainting
- baseline: SFT-48000 DiffuEraser
- eval datasets: DAVIS50 and YouTubeVOS100
- protocol: raw6, D+G off, no PCM, no dilation, no blur, hard comp, frame-wise metrics
- metrics: PSNR, SSIM, LPIPS, VFID, TC, mask PSNR, mask SSIM

## Main Result Table

Use:

```text
reports/exp11_outer_b075_s2_youtubevos100_davis50_eval.md
```

Rows:

- SFT-48000 baseline on DAVIS50
- Exp11 outer b0.75 S2 on DAVIS50
- SFT-48000 baseline on YouTubeVOS100
- Exp11 outer b0.75 S2 on YouTubeVOS100

## Ablation Table

Rows should include:

- Exp9 normalized gap
- Exp10 region-local
- Exp11 inner b0.75
- Exp11 outer b0.75 S1
- Exp11 outer b0.75 S2
- Exp11 both b0.75
- Exp11 both b1.0
- Exp12 adaptive normalization

Main conclusion:

- normalization helps stabilize the gap scale
- region-local improves the inpainting region
- outer boundary b0.75 is currently the best boundary definition
- adaptive normalization did not beat the boundary-aware method

## What Not To Claim

Do not claim:

- general SOTA without a broader benchmark table
- real optical-flow prior consistency for the old Exp11-proxy
- MiniMax-Remover adapter training unless training code/data are actually available
- improvements based only on whole-frame PSNR

## Immediate Next Work

1. Draft Introduction and Method overview.
2. Convert final 20 cases into PPT/paper figures.
3. Keep adapter work as feasibility only until the user explicitly starts a gate.
