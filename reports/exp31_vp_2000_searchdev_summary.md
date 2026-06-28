# Exp31 VideoPainter 2000-Step search-dev Summary

Status: `VIDEOPAINTER_2000_SEARCHDEV_EVALUATED`

Step2000 is the preregistered long-run endpoint. This split was not used to choose a new checkpoint.

## Step Metrics

| step | rows | ok | full_psnr | mask_psnr | full_ssim | medium-hard | hard-plausible | trivial-bad | invalid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 32 | 32 | 22.6866 | 16.1617 | 0.8375 | 27 | 4 | 1 | 0 |
| 50 | 32 | 32 | 22.1228 | 24.2617 | 0.8233 | 32 | 0 | 0 | 0 |
| 2000 | 32 | 32 | 28.2567 | 26.1364 | 0.8703 | 32 | 0 | 0 | 0 |

## Paired Deltas

| comparison | rows | win_rate | full_psnr_delta | mask_psnr_delta | boundary_delta_sampled4 | outside_l1_delta_sampled4 | temporal_delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| step2000_vs_step0 | 32 | 0.9688 | 5.5701 | 9.9747 | 12.0920 | 0.7533 | -0.5468 |
| step2000_vs_step50 | 32 | 1.0000 | 6.1338 | 1.8747 | 3.7226 | -10.0351 | 0.2022 |

## Visual Review

- Opened all-32 evidence and crop montage pages for Step0, Step50, and Step2000.
- Step0 is visibly weak: gray/white local fills, poor mask fidelity, and boundary errors.
- Step50 improves the edited region but shows broad outside/color pollution in many rows.
- Step2000 is visibly cleaner than Step50 in both full-frame and crop sheets; residual local texture/darkening remains in a small number of rows.
- LPIPS and Ewarp were not computed in this fast summary, so this split alone cannot satisfy the formal `VIDEOPAINTER_2000_POSITIVE` gate.

CSV: `reports/exp31_vp_2000_searchdev_visual_review.csv`
