# Exp35 MiniMax Rescue 10-Step Recipes

Status: `MINIMAX_RESCUE_RECIPE_NOT_READY`

This milestone ran only the preregistered 10-step recipes `R1`, `R2`, and `R3` on the locked Exp30 Gate64 train32/heldout16 split and fixed `hard_state_A`. No 30-step, long training, RC-FPO, or protected-lane action was launched.

Codex visual review completed `48/48` heldout evidence strips. The review found no recipe with a reliable visible quality improvement; most rows were ties or subtle texture/luminance drift, and the metric direction was not supportive.

## Recipe Metrics

| Recipe | Mean full PSNR delta | Mean mask PSNR delta | Mean boundary PSNR delta | Mean outside PSNR delta | Mean temporal-diff MAE delta | Mask win rate | Boundary win rate | Visual counts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R1 | 0.065600 | -0.048611 | -0.423993 | -0.307885 | 0.303775 | 0.375 | 0.188 | STEP10_SLIGHTLY_WORSE=5, TIE=9, TIE_METRIC_MIXED=2 |
| R2 | 0.057080 | -0.053910 | -0.434234 | -0.321102 | 0.305560 | 0.375 | 0.188 | STEP10_SLIGHTLY_WORSE=5, TIE=9, TIE_METRIC_MIXED=2 |
| R3 | 0.002126 | -0.081454 | -0.493050 | -0.419038 | 0.299251 | 0.375 | 0.188 | STEP10_SLIGHTLY_WORSE=6, TIE=8, TIE_METRIC_MIXED=2 |

## Visual Review

- R1 LoVI-Linear-Frozen-HardNoise: no clearly/slightly better rows; changes were mostly near-identical or subtle brightness/texture drift.
- R2 LoVI-Linear-EMA-HardNoise: no improvement over R1; EMA did not produce a quality-positive heldout effect.
- R3 WinnerAnchor-Linear-Hybrid: no quality-positive rows; winner/outside anchors did not prevent negative boundary/outside metric drift.
- New collapse/black-purple artifacts: `0`, but there was no usable repair signal.

## Gate Decision

The preregistered 10-step gate required no NaN/Inf, strict reload, visible change without collapse, heldout not worse, at least one local metric improvement, no systematic outside damage, stable gradients, and non-trivial visual improvement. The metrics and visual review failed the heldout-quality requirements.

Therefore 30-step confirmatory micro remains locked. MiniMax remains a trainable/plumbing-positive flow backbone candidate, but it is not third-backbone quality-positive evidence.
