# Exp29 MiniMax Next Micro Plan

Date: 2026-06-26

Status: `MINIMAX_NEXT_MICRO_PLAN_LOCKED`

This plan replaces the previous tiny 4-train/2-heldout plumbing setup. It does
not authorize long training.

## Step 1: Preference Data Quality Gate

Build a MiniMax micro preference set before any new optimizer step.

- Source candidates: at least 32.
- Per source: at most 3 pre-registered inference seeds.
- No VOR-Eval rows for training, tuning, or selection.
- Save every candidate, including rejected ones.
- Raw OR output only; no hard comp for loser construction.
- Required labels:
  - `MEDIUM_HARD_ELIGIBLE`
  - `HARD_BUT_PLAUSIBLE`
  - `TOO_CLOSE`
  - `TRIVIAL_BAD`
  - `TECHNICAL_INVALID`
  - `WRAPPER_FAILURE`
- Defect vector:
  - object residual
  - effect residual
  - texture mismatch
  - boundary seam
  - flicker
  - ghosting
  - outside damage
  - color shift
  - temporal instability
  - too-close
  - trivial-bad

Promotion to recipe gate requires:

- train rows: 16
- heldout rows: 16
- train/heldout scene-group overlap: 0
- VOR-Eval overlap: 0
- medium-hard + hard-plausible: at least 24/32
- trivial-bad: at most 25%
- technical-invalid: 0
- real per-video review for all selected and rejected candidates

If this fails, mark `MINIMAX_DATA_YIELD_INSUFFICIENT` and do not train.

## Step 2: Optimizer / Precision Recipe Gate

Only after `MINIMAX_MICRO_DATA_READY`.

Try at most four 10-step recipes:

- `R0`: previous conservative `SGD(lr=1e-7)` baseline.
- `R1`: AdamW small LR with finite-loss checks and strict grad clipping.
- `R2`: AdamW smaller LR plus stricter grad clip / safer precision handling.
- `R3`: Linear-DPO frozen-reference objective.
- `R4`: Linear-DPO EMA reference only if R1/R2 are stable; keep total recipes
  at four.

Each recipe must use the same train16/heldout16, seed, flow time rule, noise
rule, target velocity, and inference protocol. Step10 outputs must be generated
on heldout16 and reviewed video-by-video.

Recipe selection criteria:

1. no NaN/Inf
2. strict reload passed
3. output visibly changes but no collapse
4. heldout not worse
5. at least one local metric improves
6. no systematic outside damage
7. stable gradients
8. not merely a trivial pixel change

If no recipe passes, mark `MINIMAX_RECIPE_NOT_READY` and do not run 30-step.

## Step 3: 30-Step Confirmatory Micro

Only one selected recipe can run, and only to 30 optimizer steps.

Required checkpoints:

- step0
- step1
- step10
- step20
- step30

Required heldout16 evaluation:

- Step0 vs Step10
- Step0 vs Step30

Required metrics:

- PSNR
- SSIM
- LPIPS
- TC
- Ewarp
- strict mask PSNR
- boundary PSNR
- outside PSNR/LPIPS
- affected-region metric if available
- object/effect residual diagnostics

Primary pass condition:

- no NaN/Inf
- strict reload pass
- reference unchanged unless EMA recipe
- heldout strict mask PSNR mean delta positive or boundary/effect metric improves
- LPIPS not worse by more than `0.0005`
- Ewarp not worse by more than `0.03`
- no systematic outside damage
- visual review: better at least 8/16, worse or new-artifact at most 4/16
- not one-video dominated

If positive, the allowed claim is:

`THIRD_BACKBONE_MICRO_POSITIVE_EVIDENCE`

The forbidden claims remain:

- `UNIVERSAL_ADAPTER`
- `ALL_MODELS_SUPPORTED`
- `FINAL_SOTA`
- `TOP_CONFERENCE_NOVELTY_CONFIRMED`

