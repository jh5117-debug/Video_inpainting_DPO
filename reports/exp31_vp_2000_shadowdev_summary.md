# Exp31 VideoPainter 2000-Step shadow-dev Summary

Status: `VIDEOPAINTER_2000_SHADOWDEV_EVALUATED`

Step2000 is the preregistered long-run endpoint. This split was not used to choose a new checkpoint.

## Step Metrics

| step | rows | ok | full_psnr | mask_psnr | full_ssim | medium-hard | hard-plausible | trivial-bad | invalid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 32 | 32 | 21.5685 | 15.2466 | 0.8009 | 26 | 5 | 1 | 0 |
| 50 | 32 | 32 | 21.3544 | 24.0494 | 0.7819 | 32 | 0 | 0 | 0 |
| 2000 | 32 | 32 | 27.8317 | 26.1325 | 0.8375 | 31 | 0 | 0 | 0 |

## Paired Deltas

| comparison | rows | win_rate | full_psnr_delta | mask_psnr_delta | boundary_delta_sampled4 | outside_l1_delta_sampled4 | temporal_delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| step2000_vs_step0 | 32 | 1.0000 | 6.2632 | 10.8860 | 12.2343 | 0.7666 | -0.5314 |
| step2000_vs_step50 | 32 | 1.0000 | 6.4772 | 2.0832 | 3.9405 | -10.5232 | 0.2140 |

## Visual Review

- Opened all-32 evidence and crop montage pages for Step0, Step50, and Step2000.
- Step0 is visibly weak: gray/white local fills, poor mask fidelity, and boundary errors.
- Step50 improves the edited region but shows broad outside/color pollution in many rows.
- Step2000 is visibly cleaner than Step50 in both full-frame and crop sheets; residual local texture/darkening remains in a small number of rows.
- LPIPS and Ewarp were not computed in this fast summary, so this split alone cannot satisfy the formal `VIDEOPAINTER_2000_POSITIVE` gate.

CSV: `reports/exp31_vp_2000_shadowdev_visual_review.csv`
