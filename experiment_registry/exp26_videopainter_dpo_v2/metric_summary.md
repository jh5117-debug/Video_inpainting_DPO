# Exp26 Metric Summary

## Gate64 Final Temporal Review

Status: `GATE64_DATA_READY`

This milestone did not run VideoPainter DPO training. Metrics here summarize
preference-data quality only.

| Item | Value |
| --- | ---: |
| Gate64 formal-valid | 64 / 64 |
| Final temporal reviewed | 64 / 64 |
| Medium-hard | 37 |
| Hard-plausible | 18 |
| Too-close | 1 |
| Trivial-bad | 8 |
| Technical-invalid | 0 |
| Eligible | 55 |
| Primary rows | 32 |
| Primary medium-hard | 16 |
| Primary hard-plausible | 16 |
| Primary path/frame validation | 32 / 32 |

Primary manifest SHA256:
`82f661f2f30a581a213972533817624217eabb97eba7aaeedc00ee2109e4e716`

## Primary-32 L0/L1

| Item | Value |
| --- | ---: |
| L0 DPO loss | 0.6931471824645996 |
| L0 policy grad norm | 14.379858399808493 |
| L0 reference has grad | false |
| L1 policy delta norm | 1.6732795822542237 |
| L1 reference delta norm | 0.0 |
| strict reload max abs diff | 0.0 |

## Search-Dev Step0 Baseline

Status: `VP_STEP0_BASELINE_LOCKED`

Temporal evidence review: `32 / 32` reviewer pass.

| Variant | PSNR | SSIM | LPIPS | Ewarp | Mask PSNR | Boundary PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 22.829909342886836 | 0.8492259011345125 | 0.09040529020302882 | 7.401047145404978 | 15.996989577661143 | 15.445102895985176 |
| comp | 24.301897366442233 | 0.871557953992803 | 0.07080062118242197 | 8.04273951919754 | 16.012427341260924 | 16.01195236353609 |

## Primary-32 10-Step Gate

Status: `VIDEOPAINTER_10STEP_GATE_PASSED`

| step | PSNR | SSIM | LPIPS | Ewarp | Mask PSNR | Boundary PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| step0 comp | 24.301897366442233 | 0.871557953992803 | 0.07080062118242197 | 8.04273951919754 | 16.012427341260924 | 16.01195236353609 |
| step1 comp | 27.18641296098709 | 0.9212518619353416 | 0.07871026372082937 | 1.728870088675135 | 18.94667263526916 | 22.795687530028623 |
| step10 comp | 25.279149648052446 | 0.9041987144484136 | 0.06630134451485475 | 6.741282913869403 | 16.987619106855355 | 21.094158585504704 |

Step10 minus step0:

- PSNR `+0.977252`
- SSIM `+0.032641`
- LPIPS `-0.004499`
- Ewarp `-1.301457`
- mask PSNR `+0.975192`
- boundary PSNR `+5.082206`

Dense evidence visual review covered `32/32` rows and did not block the
conditional 50-step gate.

## Primary-32 50-Step Gate

Status: `VIDEOPAINTER_ADAPTER_POSITIVE`

Scope: locked search-dev micro-training gate only; not `SCIENTIFIC_POSITIVE`.

| step | PSNR | SSIM | LPIPS | Ewarp | Mask PSNR | Boundary PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| step0 comp | 24.301897366442233 | 0.871557953992803 | 0.07080062118242197 | 8.04273951919754 | 16.012427341260924 | 16.01195236353609 |
| step10 comp | 25.435825499433605 | 0.9031576620291053 | 0.06795676456571602 | 6.790276521977981 | 17.14805054427446 | 21.176618178372053 |
| step30 comp | 29.03247933591733 | 0.9560208106174282 | 0.031781330745597726 | 1.9569201151495481 | 20.85899678409783 | 25.27817230953844 |
| step50 comp | 29.11806557328507 | 0.959440551089185 | 0.026741246365195457 | 0.9876170219917195 | 20.95467319099135 | 28.123841505101794 |

Step50 minus step0:

- PSNR `+4.8161682068428355`
- SSIM `+0.08788259709638202`
- LPIPS `-0.04405937481722652`
- Ewarp `-7.0551224972058195`
- mask PSNR `+4.942245849730426`
- boundary PSNR `+12.111889141565705`

Paired statistics:

- PSNR win rate `0.718750`
- PSNR bootstrap 95% CI `[+2.648960, +7.234666]`
- PSNR probability(delta > 0) `1.000000`
- LPIPS probability improved `1.000000`
- Ewarp probability improved `1.000000`

## Shadow-Dev Confirmatory Validation

Status: `VIDEOPAINTER_SHADOWDEV_CONFIRMED`

Primary endpoint: frame1-48 comp, fixed Step50 - fixed Step0.

| Metric | Mean delta | Win rate | Bootstrap 95% CI | Probability improved |
| --- | ---: | ---: | ---: | ---: |
| strict mask PSNR | `+5.186942` | `0.781250` | `[+2.781118, +7.818869]` | `1.000000` |
| boundary PSNR | `+12.175098` | `1.000000` | `[+10.184673, +14.212251]` | `1.000000` |
| LPIPS | `-0.040142` | `0.937500` | `[-0.052638, -0.028834]` | `1.000000` |
| Ewarp | `-8.378847` | `0.968750` | `[-13.581173, -4.369433]` | `1.000000` |

No-first-frame comp trajectory:

| step | PSNR | SSIM | LPIPS | Ewarp | strict mask PSNR | boundary PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Step0 | `23.703743` | `0.875636` | `0.066593` | `9.430909` | `16.165117` | `15.740496` |
| Step10 | `24.184792` | `0.896096` | `0.064336` | `6.602979` | `16.615746` | `20.743876` |
| Step30 | `28.617596` | `0.951033` | `0.032386` | `1.356535` | `21.093739` | `25.298020` |
| Step50 | `28.864482` | `0.956006` | `0.026451` | `1.052062` | `21.352059` | `27.915594` |

TC/VFID-style no-first-frame comp:

- Step0: TC `0.986760`, VFID `0.531078`
- Step50: TC `0.991139`, VFID `0.499650`

Seed robustness over fixed 16-row subset:

- status: `SEED_ROBUSTNESS_PASS`
- primary direction pass: `3 / 3`
- mean strict mask PSNR delta: `+8.005364`
- mean boundary PSNR delta: `+13.627635`
- mean LPIPS delta: `-0.056193`
- mean Ewarp delta: `-9.325711`

## 2026-06-26 Post-Confirmation Sanity Audit

Status: `EXP26_POSTCONFIRMATION_SANITY_AUDIT_PASSED`

The audit re-read the completed held-out evidence and found no metric or
identity inconsistency. Primary no-first-frame comp deltas remain:

| Metric | Step50 - Step0 |
| --- | ---: |
| strict mask PSNR | `+5.186942` |
| boundary PSNR | `+12.175098` |
| LPIPS | `-0.040142` |
| Ewarp | `-8.378847` |
| TC | `+0.004378` |
| VFID/FVD-style | `-0.031428` |

No external-validation metric has been run yet.

## 2026-06-26 External 49F Inventory

Status: `EXP26_EXTERNAL_49F_INVENTORY_COMPLETE`

No model metrics were computed in this milestone. The source inventory found
`54` valid clean 49F DAVIS-derived sequences and locked `32` rows for the next
pre-registered external Step0-vs-Step50 validation.

## 2026-06-26 External Validation Preregistration

Status: `EXP26_EXTERNAL_VALIDATION_PREREGISTERED`

No external Step0/Step50 model metrics were computed in this milestone. The
external validation protocol is now fixed at `32` DAVIS-derived exact-49F rows,
first-frame GT, mask seed `20260623`, inference seed `20260619`, `720x480`, 20
inference steps, guidance `6.0`, and bf16.

Locked manifest SHA256 values:

- preregistered rows: `69ecd96d4b25da702229df2d45bf1343ad5e7ef5385cbd32d24ce61644e4bc2c`
- masks: `f646792469f53a8122fe341be5988344ba7b32d33b3a53593d558e227aed138b`

The next metric milestone is fixed external Step50-vs-Step0 inference and
frame1-48 paired statistics. Step10/30 remain trajectory-only diagnostics.
