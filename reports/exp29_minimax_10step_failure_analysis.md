# Exp29 MiniMax 10-Step Failure Analysis

Date: 2026-06-26

Status: `MINIMAX_10STEP_FAILURE_ANALYZED`

This milestone explains why the previous MiniMax 10-step micro gate produced
almost no visible heldout change. No new MiniMax training was launched for this
analysis.

## Readback

- branch: `research/exp29-minimax-effecterase-adapter-feasibility-20260626`
- HEAD at milestone start: `bd44f67d53a3430fef27917b6f96f4962bd824f8`
- previous gate report: `reports/exp29_minimax_10step_micro.md`
- gate diagnostics: `reports/exp29_minimax_adapter_gates.json`
- heldout metrics: `reports/exp29_minimax_10step_metrics.csv`
- heldout visual review: `reports/exp29_minimax_10step_visual_review.csv`
- left CLI: read-only protection maintained; no signal and no file mutation.

## Key Observations

- Zero-gap was correct: DPO loss `0.6931471825`, with win/lose gaps equal to 0.
- One-step strict reload passed.
- The successful 10-step recovery path used conservative `SGD(lr=1e-7)` after
  the initial fp16 AdamW attempt produced NaNs.
- Step10 strict reload passed, but the parameter delta probe was only
  `1.1061271569642785e-10`.
- Reference delta probe was `0.0`.
- Gradients were finite: mean preclip grad norm `0.7237282794`, max
  `1.2341757971`.
- DPO loss stayed essentially at `log(2)`:
  - min `0.6931452155`
  - max `0.6931496859`
  - mean `0.6931473672`
- Preference margin stayed tiny:
  - min `-5.0142407417e-06`
  - max `3.9562582970e-06`
  - mean `-3.9152801037e-07`
- Heldout PSNR deltas were negligible:
  - `davis_hockey`: `-0.0008006331`
  - `davis_koala`: `+0.0024137239`
- Heldout visual review classified both videos as ties.

## Root-Cause Assessment

| Question | Answer | Evidence |
| --- | --- | --- |
| Optimizer too conservative? | Yes. | `SGD(lr=1e-7)` produced only `1.1e-10` step10 parameter-probe delta. |
| SGD too weak? | Yes for quality movement. | Finite gradients existed, but output videos were visually unchanged. |
| LR too small? | Yes for this recipe. | The run was stable but parameter and output changes were near zero. |
| Trainable params actually updated? | Technically yes, practically tiny. | Policy delta was nonzero but at numerical-probe scale; reference delta stayed zero. |
| Output insensitive? | Likely secondary. | With such tiny parameter deltas, inference cannot be expected to move. |
| Loser too trivial? | Yes. | Previous smoke had 3/4 training losers classified trivial-bad. |
| Heldout rows too few? | Yes. | Only 2 heldout rows; adequate for plumbing, not quality promotion. |
| Flow target wired correctly? | Yes. | Zero-gap, finite forward/backward, and target `epsilon - z0` were confirmed. |
| Objective too weak? | In this recipe, yes. | Margins stayed within about `5e-6`; DPO loss remained around `log(2)`. |
| Inference seed / bad-noise mismatch? | Plausible. | MiniMax paper motivates bad-noise/minimax selection, while this gate used a fixed small diagnostic setup. |
| Wrong modules frozen? | No evidence. | Full transformer trainable path produced finite gradients; VAE/reference were frozen intentionally. |

## Decision

The previous 10-step result should remain:

`MINIMAX_10STEP_PARETO_MIXED`

It is useful adapter-plumbing evidence, but it is not a quality-positive third
backbone result. The next step must not be a longer run of the same recipe.
MiniMax must first pass a data-quality gate with medium-hard or hard-plausible
preference pairs, then a small optimizer/precision recipe gate.

## Reports Produced

- `reports/exp29_minimax_10step_failure_analysis.md`
- `reports/exp29_minimax_10step_failure_analysis.csv`
- `reports/exp29_minimax_next_micro_plan.md`

