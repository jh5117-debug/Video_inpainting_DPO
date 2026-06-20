# Exp20 Search Status After Fast Budget + Equal-Step

TARGET_DEV_PSNR = 29.523336
Best overall by search-dev PSNR: EQ_BF07 (equal_step) PSNR=29.393079.
Best fixed/global: EQ_BF07 PSNR=29.393079.
Best region-balanced: EQ_RB08 PSNR=29.361332.
Best adaptive: EQ_AD04 PSNR=29.387359.

Decision: FAST_SEARCH_BUDGET_REACHED_WITHOUT_TARGET. No automatic long training, Stage2, DAVIS50, or YouTubeVOS100 final eval was launched.

Rationale:
- 48 fast-trial budget was reached (6 first-wave + 18 fixed roots + 8 best-first + 10 region-balanced + 6 adaptive).
- Equal-step confirmation was run for P4/P0/BF07/RB08/AD04 at 112 optimizer steps.
- EQ_BF07 has the highest PSNR but worsens LPIPS/VFID and TC relative to P0/P4, so it is a Pareto candidate rather than a clean promotion.
- No candidate reached TARGET_DEV_PSNR.

Top search/equal-step candidates:

| rank | trial | family | config | PSNR | SSIM | LPIPS | VFID/FVD | TC | Ewarp | mask PSNR | boundary PSNR |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | EQ_BF07 | equal_step | fixed_image_px r=28.0 k=0.0 b=5.0 legacy_global_weighted_mean | 29.393079 | 0.968993 | 0.018441 | 0.232887 | 0.975930 | 11.967787 | 17.087768 | 22.947160 |
| 2 | P4 | first_wave | fixed_image_px r=16 k= b=  | 29.390553 | 0.969074 | 0.018198 | 0.232074 | 0.976040 | 11.994790 | 17.085242 | 22.946999 |
| 3 | EQ_P4 | equal_step | fixed_image_px r=16.0 k=0.0 b=2.0 legacy_global_weighted_mean | 29.389916 | 0.969032 | 0.018072 | 0.227931 | 0.976041 | 11.993817 | 17.084604 | 22.966567 |
| 4 | EQ_AD04 | equal_step | adaptive_sqrt_area r=0.0 k=0.103897 b=2.0 legacy_global_weighted_mean | 29.387359 | 0.969029 | 0.017982 | 0.226749 | 0.976091 | 12.002448 | 17.082048 | 22.969489 |
| 5 | P5 | first_wave | fixed_image_px r=24 k= b=  | 29.385103 | 0.968993 | 0.018433 | 0.234728 | 0.975934 | 11.976952 | 17.079791 | 22.930627 |
| 6 | EQ_P0 | equal_step | legacy_latent_exact r=0.0 k=0.0 b=0.75 legacy_global_weighted_mean | 29.374471 | 0.968993 | 0.017839 | 0.223601 | 0.976136 | 12.016644 | 17.069160 | 22.947473 |
| 7 | P3 | first_wave | fixed_image_px r=12 k= b=  | 29.374159 | 0.969043 | 0.018020 | 0.229008 | 0.976174 | 12.016267 | 17.068848 | 22.923562 |
| 8 | P0 | first_wave | legacy_latent_exact r=0 k= b=  | 29.373852 | 0.968990 | 0.017842 | 0.223621 | 0.976131 | 12.016763 | 17.068540 | 22.947118 |
| 9 | BF07 | second_fixed_bestfirst | fixed_image_px r=28.0 k=0.0 b=5.0 legacy_global_weighted_mean | 29.369880 | 0.968801 | 0.018465 | 0.230950 | 0.975952 | 11.957151 | 17.064569 | 22.951337 |
| 10 | P2 | first_wave | fixed_image_px r=8 k= b=  | 29.366833 | 0.969021 | 0.017936 | 0.227789 | 0.976126 | 12.025352 | 17.061521 | 22.911305 |
| 11 | BF08 | second_fixed_bestfirst | fixed_image_px r=24.0 k=0.0 b=3.5 legacy_global_weighted_mean | 29.366428 | 0.968772 | 0.018372 | 0.230838 | 0.975907 | 11.963386 | 17.061117 | 22.954126 |
| 12 | BF03 | second_fixed_bestfirst | fixed_image_px r=24.0 k=0.0 b=5.0 legacy_global_weighted_mean | 29.363525 | 0.968835 | 0.018364 | 0.230150 | 0.975968 | 11.979235 | 17.058214 | 22.908234 |
