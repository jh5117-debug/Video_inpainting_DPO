# Adaptive search

TARGET_DEV_PSNR = 29.523336

Best: AD04 PSNR=29.359881, SSIM=0.968924, LPIPS=0.017984, VFID=0.227926, TC=0.976143, Ewarp=12.014500.

| trial | family | config | PSNR | SSIM | LPIPS | VFID/FVD | TC | Ewarp | mask PSNR | boundary PSNR | loser degrade | target |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AD04 | adaptive | adaptive_sqrt_area r=0.0 k=0.103897 b=2.0 legacy_global_weighted_mean | 29.359881 | 0.968924 | 0.017984 | 0.227926 | 0.976143 | 12.014500 | 17.054570 | 22.917904 | 1.0 | no |
| AD06 | adaptive | adaptive_sqrt_area r=0.0 k=0.207794 b=2.0 legacy_global_weighted_mean | 29.359513 | 0.968887 | 0.018168 | 0.228898 | 0.976085 | 11.998575 | 17.054202 | 22.906608 | 1.0 | no |
| AD03 | adaptive | adaptive_area_perimeter r=0.0 k=0.510769 b=2.0 legacy_global_weighted_mean | 29.359069 | 0.968895 | 0.018111 | 0.228540 | 0.976099 | 12.004064 | 17.053758 | 22.910063 | 1.0 | no |
| AD02 | adaptive | adaptive_area_perimeter r=0.0 k=0.340513 b=2.0 legacy_global_weighted_mean | 29.358743 | 0.968916 | 0.018003 | 0.228086 | 0.976130 | 12.013275 | 17.053431 | 22.916690 | 1.0 | no |
| AD05 | adaptive | adaptive_sqrt_area r=0.0 k=0.138529 b=2.0 legacy_global_weighted_mean | 29.357826 | 0.968900 | 0.018046 | 0.228550 | 0.976085 | 12.009945 | 17.052515 | 22.912990 | 1.0 | no |
| AD01 | adaptive | adaptive_area_perimeter r=0.0 k=0.255385 b=2.0 legacy_global_weighted_mean | 29.357017 | 0.968920 | 0.017931 | 0.227119 | 0.976119 | 12.020403 | 17.051705 | 22.916456 | 1.0 | no |
