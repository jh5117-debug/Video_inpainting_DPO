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
