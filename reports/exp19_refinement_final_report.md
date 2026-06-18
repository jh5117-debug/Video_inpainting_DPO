# Exp19 Flow Refinement Final Report

Date: 2026-06-18

## Status

```text
COMPLETED_NEGATIVE_GATE
```

This run completed the requested Exp19 refinement sequence:

1. inference parity repair;
2. zero-training residual scale / confidence exponent sweep;
3. real/zero/shuffled/reversed flow causality audit;
4. Exp19c light latent-warp continuation sweep;
5. DAVIS10 metric and visual judgement;
6. positive-gate decision.

## R0 Parity

The calibrated wrapper fixed the disabled-output mismatch:

```text
disabled_vs_Exp11_MAE = 0.0
```

The earlier `~0.009878` error was caused by evaluator protocol/RNG mismatch,
not by the hook wrapper itself.

## R0 Strength Sweep

Best non-degrading zero-training setting:

```text
residual_scale = 0.5
confidence_exponent = 2.0
```

It produced only tiny movement:

```text
PSNR delta = +0.000355
Ewarp delta = -0.000124
```

## Flow Causality

Real flow was better than zero/shuffled/reversed controls, but the effect was
extremely small. This justified testing Exp19c-light, but did not by itself
support expansion.

## Exp19c DAVIS10

| Method | PSNR | SSIM | LPIPS | Ewarp | strict mask PSNR | boundary PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exp11 outer b0.75 S2 | 29.829309 | 0.963257 | 0.02065550 | 8.330730 | 18.531525 | 24.657501 |
| Exp19b Stage2-500 | 29.829470 | 0.963257 | 0.02065455 | 8.330525 | 18.531685 | 24.657372 |
| lambda000 | 29.829031 | 0.963255 | 0.02065269 | 8.330644 | 18.531247 | 24.657456 |
| lambda005 | 29.829262 | 0.963255 | 0.02065183 | 8.330690 | 18.531478 | 24.657507 |
| lambda010 | 29.829166 | 0.963256 | 0.02065105 | 8.330801 | 18.531382 | 24.657439 |
| lambda020 | 29.829368 | 0.963257 | 0.02065228 | 8.330675 | 18.531584 | 24.657357 |

TC was not computed because the TC backend requires an OpenCLIP download that
was unavailable from PAI. Ewarp was computed with local RAFT.

## Motion-Bin Result

High-motion subset:

```text
Exp11 Ewarp      = 14.125456
Exp19b Ewarp     = 14.125256
lambda000 Ewarp  = 14.125176
lambda020 Ewarp  = 14.125236
```

The best high-motion Ewarp is the lambda=0 continuation control, not a positive
warp-loss variant.

## Visual Review

Inspected contact sheets for dog-agility, car-roundabout, dance-jump, camel,
soccerball, boat, blackswan, rhino, lucia, and flamingo.

```text
better: 0
tie: 10
worse: 0
```

No clear temporal stabilization, moving-boundary improvement, or strict
mask/boundary improvement was visible. No major new ghosting or deformation was
introduced either.

## Positive Gate

```text
FAIL
```

Reasons:

- no lambda>0 warp variant beats lambda000 continuation control on Ewarp;
- Ewarp changes are far below the 1% gate;
- no visually clear better cases;
- high-motion subset does not improve with positive warp loss;
- TC unavailable, so no alternate temporal metric supports expansion.

## Decision

```text
Do not start Exp19d.
Do not run DAVIS50.
Do not continue to 1000 / 2000 steps.
```

Exp19 remains a useful engineering/ablation result: the flow adapter reads real
flow and can be trained safely without degrading spatial quality, but the
current adapter and light latent warp supervision do not create a meaningful
temporal benefit over Exp11 outer b0.75 S2.
