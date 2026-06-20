# Equal-step confirmation

TARGET_DEV_PSNR = 29.523336

Best: EQ_BF07 PSNR=29.393079, SSIM=0.968993, LPIPS=0.018441, VFID=0.232887, TC=0.975930, Ewarp=11.967787.

| trial | family | config | PSNR | SSIM | LPIPS | VFID/FVD | TC | Ewarp | mask PSNR | boundary PSNR | loser degrade | target |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| EQ_BF07 | equal_step | fixed_image_px r=28.0 k=0.0 b=5.0 legacy_global_weighted_mean | 29.393079 | 0.968993 | 0.018441 | 0.232887 | 0.975930 | 11.967787 | 17.087768 | 22.947160 | 1.0 | no |
| EQ_P4 | equal_step | fixed_image_px r=16.0 k=0.0 b=2.0 legacy_global_weighted_mean | 29.389916 | 0.969032 | 0.018072 | 0.227931 | 0.976041 | 11.993817 | 17.084604 | 22.966567 | 1.0 | no |
| EQ_AD04 | equal_step | adaptive_sqrt_area r=0.0 k=0.103897 b=2.0 legacy_global_weighted_mean | 29.387359 | 0.969029 | 0.017982 | 0.226749 | 0.976091 | 12.002448 | 17.082048 | 22.969489 | 1.0 | no |
| EQ_P0 | equal_step | legacy_latent_exact r=0.0 k=0.0 b=0.75 legacy_global_weighted_mean | 29.374471 | 0.968993 | 0.017839 | 0.223601 | 0.976136 | 12.016644 | 17.069160 | 22.947473 | 1.0 | no |
| EQ_RB08 | equal_step | fixed_image_px r=24.0 k=0.0 b=2.0 region_balanced | 29.361332 | 0.968911 | 0.017950 | 0.226904 | 0.976039 | 11.998562 | 17.056020 | 22.925615 | 0.0 | no |
