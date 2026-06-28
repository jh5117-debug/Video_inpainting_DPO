# Exp37 MiniMax Paper Positioning

Status: `TWO_BACKBONE_PLUS_MINIMAX_PLUMBING_ONLY`

Exp37 tested whether a cleaner LocalDPO-style corruption pool plus
outside-sane bad-noise states could rescue MiniMax Flow-DPO at a bounded
10-step scale.

## What Exp37 Adds

- MiniMax still has plumbing-positive evidence:
  - inference uses trained adapter weights;
  - checkpoint/load is not the failure mode;
  - winner-SFT can lower train loss and move outputs;
  - LocalDPO-style corruption data can be built cleanly;
  - bad-noise states can be mined without using VOR-Eval.
- The Exp37 LocalDPO-badnoise recipes produced nonzero output changes.
- R1 produced mixed numeric local movement, including mean mask PSNR
  `+0.161946`, but only `1/16` heldout videos was visibly better.

## What Exp37 Does Not Add

- It does not make MiniMax third-backbone adapter evidence.
- It does not unlock 30-step.
- It does not support universal-adapter language.
- It does not show heldout visual quality improvement at the preregistered
  threshold.

## Paper Language

Allowed:

- DiffuEraser and VideoPainter provide cross-backbone LoVI-DPO adapter evidence.
- MiniMax is a promising but unresolved flow-style adapter candidate.
- MiniMax failure is now narrowed to data/objective/update-scale/generalization,
  not checkpoint loading or ignored adapter weights.

Not allowed:

- `UNIVERSAL_ADAPTER`
- `ALL_MODELS_SUPPORTED`
- `FINAL_SOTA`
- `TOP_CONFERENCE_NOVELTY_CONFIRMED`
- MiniMax as third-backbone adapter success

## Next Minimal Experiment

The next MiniMax step should not be longer training. The smallest useful next
experiment is to improve the objective/data signal before training again:

1. inspect whether the heldout LocalDPO corruptions align with MiniMax's own
   removal failure modes;
2. design a stronger positive-control that must improve heldout local residuals
   before DPO;
3. consider a narrower trainable scope or loss normalization that targets the
   visible local residual rather than broad low-amplitude texture shifts.

Until that exists, MiniMax remains:

`MINIMAX_PLUMBING_POSITIVE_RECIPE_NOT_READY`.
