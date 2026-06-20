# Region-balanced search

TARGET_DEV_PSNR = 29.523336

Best: RB08 PSNR=29.359323, SSIM=0.968904, LPIPS=0.017981, VFID=0.227438, TC=0.976056, Ewarp=11.999266.

| trial | family | config | PSNR | SSIM | LPIPS | VFID/FVD | TC | Ewarp | mask PSNR | boundary PSNR | loser degrade | target |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RB08 | region_balanced | fixed_image_px r=24.0 k=0.0 b=2.0 region_balanced | 29.359323 | 0.968904 | 0.017981 | 0.227438 | 0.976056 | 11.999266 | 17.054011 | 22.919800 | 0.0 | no |
| RB10 | region_balanced | fixed_image_px r=28.0 k=0.0 b=2.5 region_balanced | 29.356687 | 0.968867 | 0.018082 | 0.228790 | 0.975989 | 11.991496 | 17.051376 | 22.914620 | 0.0 | no |
| RB09 | region_balanced | fixed_image_px r=28.0 k=0.0 b=1.5 region_balanced | 29.353812 | 0.968889 | 0.017886 | 0.225863 | 0.976064 | 12.008453 | 17.048501 | 22.912110 | 0.0 | no |
| RB05 | region_balanced | fixed_image_px r=16.0 k=0.0 b=2.5 region_balanced | 29.341867 | 0.968748 | 0.018063 | 0.225225 | 0.976104 | 11.992294 | 17.036556 | 22.911410 | 0.0 | no |
| RB04 | region_balanced | fixed_image_px r=16.0 k=0.0 b=2.0 region_balanced | 29.339192 | 0.968752 | 0.017976 | 0.225315 | 0.976095 | 11.999816 | 17.033881 | 22.908671 | 0.0 | no |
| RB03 | region_balanced | fixed_image_px r=16.0 k=0.0 b=1.5 region_balanced | 29.336762 | 0.968762 | 0.017864 | 0.224356 | 0.976087 | 12.009635 | 17.031451 | 22.904697 | 0.0 | no |
| RB02 | region_balanced | fixed_image_px r=16.0 k=0.0 b=1.0 region_balanced | 29.330825 | 0.968752 | 0.017712 | 0.222688 | 0.976094 | 12.023062 | 17.025514 | 22.898580 | 0.0 | no |
| RB01 | region_balanced | fixed_image_px r=16.0 k=0.0 b=0.75 region_balanced | 29.327651 | 0.968747 | 0.017626 | 0.221882 | 0.976071 | 12.031673 | 17.022339 | 22.893952 | 0.0 | no |
| RB06 | region_balanced | fixed_image_px r=24.0 k=0.0 b=1.0 region_balanced | 29.322438 | 0.968698 | 0.017714 | 0.222671 | 0.976100 | 12.022642 | 17.017127 | 22.887099 | 0.0 | no |
| RB07 | region_balanced | fixed_image_px r=24.0 k=0.0 b=1.5 region_balanced | 29.322029 | 0.968642 | 0.017963 | 0.223824 | 0.976091 | 12.000329 | 17.016718 | 22.889239 | 1.0 | no |
