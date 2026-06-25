# Exp26 VideoPainter Shadow-Dev Final Decision

- status: `VIDEOPAINTER_SHADOWDEV_CONFIRMED`
- scope: independent pre-registered shadow-dev confirmation for fixed Step50 vs fixed Step0
- no retraining, no checkpoint reselection, no 100-step, no RC-FPO
- left CLI untouched: no signals sent, no left files modified

## Identity

- primary32 SHA256: `82f661f2f30a581a213972533817624217eabb97eba7aaeedc00ee2109e4e716`
- search-dev SHA256: `41c6571d26e4a5130818dd50fbbe1314c9d953284561a3cd20f630572f7c2a71`
- shadow-dev SHA256: `0338dba1513cfe0e5dd85cbf793b3782902b981ac9610b0e472c6a048f738c02`
- shadow rows / scene groups: `32 / 32`
- train/search/shadow overlaps: `0 / 0 / 0`
- checkpoint trajectory: fixed `vp_primary32_50step_20260625_171032`, Step0/10/30/50 only

## Primary Metrics

Primary endpoint is frame1-48 comp, Step50 - Step0.

| metric | mean delta | win rate | bootstrap 95% CI | probability improved |
| --- | ---: | ---: | ---: | ---: |
| strict mask PSNR | `+5.186942` | `0.781250` | `[+2.781118, +7.818869]` | `1.000000` |
| boundary PSNR | `+12.175098` | `1.000000` | `[+10.184673, +14.212251]` | `1.000000` |
| LPIPS | `-0.040142` | `0.937500` | `[-0.052638, -0.028834]` | `1.000000` |
| Ewarp | `-8.378847` | `0.968750` | `[-13.581173, -4.369433]` | `1.000000` |

Whole comp PSNR frame1-48 improved by `+5.160739` dB. Leave-one-out ranges
remained positive for strict mask and boundary PSNR.

## TC / VFID

Primary no-first-frame TC/VFID was computed through the existing
`inference.metrics.py` backend using the audited I3D and OpenCLIP assets.

| step | variant | TC | VFID/FVD-style |
| --- | --- | ---: | ---: |
| Step0 | raw | `0.987396` | `0.525803` |
| Step0 | comp | `0.986760` | `0.531078` |
| Step50 | raw | `0.991770` | `0.512713` |
| Step50 | comp | `0.991139` | `0.499650` |

## Leakage And Visual Review

- leakage audit: `NO_UNEXPECTED_WINNER_LEAKAGE_DETECTED`
- exact-copy flagged rows: `0`
- visual review: `32 / 32`
- Step50 clearly better: `12`
- Step50 slightly better: `13`
- tie: `3`
- Step0 better or Step50 new artifact: `4`
- Step50 new artifact count: `3`

Step50 removes many translucent oval/ring residuals from Step0. Remaining
failures are local, mainly water/grass cases with green or purple patches.
No systematic frame-order bug, GT leakage, first-frame failure, or outside
comp artifact was observed.

## Seed Robustness

The 16-row robustness subset is fixed at SHA256
`4b3b18bc275eabcdc591ddf18173e34f544811fb2d7a206b014136befb243db2`.

| seed | rows | strict mask PSNR delta | boundary PSNR delta | LPIPS delta | Ewarp delta | pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 20260619 | 16 | `+7.019990` | `+13.411947` | `-0.054398` | `-10.220416` | true |
| 20260620 | 16 | `+7.871717` | `+13.562566` | `-0.054063` | `-8.354823` | true |
| 20260621 | 16 | `+9.124383` | `+13.908393` | `-0.060119` | `-9.401894` | true |

Seed robustness status: `SEED_ROBUSTNESS_PASS`.

## Dynamics

- max grad norm: `471.683581`
- p95 grad norm: `136.259904`
- grad > 10 / 50 / 100: `41 / 7 / 4`
- implicit accuracy mean / last: `0.28 / 0.0`
- loser-dominant mean / last: `0.12 / 0.0`
- recommendation: `NO_100STEP_BY_PROTOCOL`

The large gradient spikes did not correspond to a Step50 shadow-dev quality
collapse. Step30 to Step50 still improves local, perceptual, temporal, and
VFID-style metrics, but protocol forbids continuing to 100 steps in this turn.

## Decision

`VIDEOPAINTER_SHADOWDEV_CONFIRMED`.

The held-out shadow-dev result confirms that the Step50 VideoPainter adapter
improves over fixed Step0 for this VOR-BG distribution. This supports
`CROSS_BACKBONE_ADAPTER_EVIDENCE_CONFIRMED` across DiffuEraser and
VideoPainter, but it is not a universal-adapter or final SOTA claim. External
cross-dataset benchmarking is still required before any broader claim.
