# Exp23 Pair001 Final Decision

pair_id: `phaseA_scale1_pair001_outer2_corrected_outer_control_seed20260619_gpus2456`
updated_at: `2026-06-22T16:38:45+08:00`

final_status: `PAIR001_PARETO_MIXED`

## Boundary Audit

- Original pair `phaseA_scale1_pair001_outer2_gpus2456` is invalid for scientific comparison because fresh control did not provide reliable runtime evidence of `boundary_mode=outer` and legacy path could default to `both`.
- Corrected pair retrained from scratch with explicit runtime configs.
- Fresh Stage1/Stage2 corrected config: `legacy_exact=true`, `boundary_mode=outer`, `inner_pool_steps=0`, `outer_pool_steps=1`, `mask/boundary/outside=1.0/0.75/0.05`.
- Candidate Stage1/Stage2 corrected config: `legacy_exact=false`, `boundary_mode=outer`, `pool_grid_scale=1`, `inner_pool_steps=0`, `outer_pool_steps=2`, `inner_weight=0.0`, `outer_weight=0.75`.

## Main S2-2000 Result

| metric | fresh | candidate | delta |
|---|---|---|---|
| PSNR | 32.902394 | 32.912649 | 0.010255 |
| SSIM | 0.971324 | 0.971804 | 0.000480 |
| LPIPS | 0.015577 | 0.015717 | 0.000140 |
| VFID | 0.200350 | 0.193077 | -0.007272 |
| TC | 0.971529 | 0.971810 | 0.000281 |
| Ewarp | 7.090408 | 7.099068 | 0.008660 |
| strict_mask_PSNR | 21.258944 | 21.269199 | 0.010255 |
| boundary_PSNR | 26.449689 | 26.427846 | -0.021842 |

## Positive Gate

| gate | passed | value |
|---|---:|---|
| paired PSNR delta > +0.02 dB | NO | 0.010255 |
| per-video PSNR win rate >= 55% | YES | 0.560000 |
| PSNR bootstrap probability(delta>0) >= 0.90 | NO | 0.562900 |
| strict mask or boundary PSNR clearly improves | NO | mask +0.010255, boundary -0.021842 |
| LPIPS degradation <= 0.0003 | YES | 0.000140 |
| VFID degradation <= 0.005 | YES | -0.007272 |
| TC drop <= 0.0002 | YES | 0.000281 |
| Ewarp degradation <= 0.03 | YES | 0.008660 |
| no obvious new visual artifacts | NO | 8 candidate penalty flags |
| not single-video dominated | NO | LOO -0.017312203825388027..0.04422780145343155 |
| checkpoint identity passed | YES | passed for evaluated exports/checkpoints |

Decision rationale: candidate S2-2000 has a small PSNR/strict-mask gain (+0.0103 dB) and better VFID/TC, but it misses the +0.02 dB PSNR gate, has weak bootstrap support (P(delta>0)=0.5629), worsens official boundary PSNR, has slightly worse LPIPS/Ewarp, and visual review is mixed. This is not a positive result and should not trigger the next morphology pair automatically.

## Visual Summary

- candidate better: 18
- tie: 16
- fresh better: 16
- candidate artifact/perceptual penalty flags: 8

## Required Stop

Do not launch Phase A next pair until the user explicitly decides how to handle this Pareto-mixed outcome.
