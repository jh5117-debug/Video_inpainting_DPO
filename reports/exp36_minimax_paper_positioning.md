# Exp36 MiniMax Paper Positioning

Final MiniMax status: `MINIMAX_PLUMBING_POSITIVE_RECIPE_NOT_READY`.

Paper claim status: `TWO_BACKBONE_PLUS_MINIMAX_PLUMBING_ONLY`.

## Evidence Reviewed

Exp36 completed four bounded milestones:

1. Readback from Exp30/Exp35: previous MiniMax recipes failed by tie/slight degradation, not collapse.
2. No-change forensic audit: dominant cause remains utility/objective/update-scale weakness rather than checkpoint fallback.
3. Inference sensitivity: MiniMax inference responds to weight perturbation, so trained weights are not ignored.
4. Trainable scope audit: S1 LoRA attention/projection scope is implemented and strict roundtrip tested.
5. Winner-SFT positive-control: S0/S1 10-step controls reduce train loss and move outputs, but heldout quality does not improve.

## Final Scientific Interpretation

MiniMax is not a failed integration: it loads, trains, strict-reloads, and affects inference. The rescue failure is also not black/purple collapse. The current blocker is that bounded supervised and preference-style objectives do not produce visible heldout repair quality.

Therefore MiniMax remains:

- plumbing-positive;
- trainability-positive;
- inference-sensitivity-positive;
- not quality-positive;
- not third-backbone adapter evidence.

## Allowed Language

Allowed:

- DiffuEraser and VideoPainter provide cross-backbone adapter evidence.
- MiniMax is a flow-style third-backbone candidate with trainable-forward and inference-sensitivity evidence.
- Current MiniMax objective recipes are not quality-positive.

Forbidden:

- universal adapter;
- all models supported;
- MiniMax third-backbone success;
- final SOTA;
- top-conference novelty confirmed from MiniMax.

## Stop Decision

Do not run Exp36 bad-noise mining, objective rescue, or 30-step confirmatory from this state. Winner-SFT did not establish heldout quality-positive learning, so the prerequisite for DPO rescue is not met.

## Next Minimal Experiment

The smallest useful next step is not more steps. It is a targeted objective/data redesign outside Exp36: either a stronger non-trivial controlled-corruption MiniMax heldout split or a MiniMax-native loss derivation that directly optimizes affected-region flow consistency without relying on near-constant DPO utility. That should be preregistered before any new training.
