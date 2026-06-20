# Second fixed best-first

TARGET_DEV_PSNR = 29.523336

Best: BF07 PSNR=29.369880, SSIM=0.968801, LPIPS=0.018465, VFID=0.230950, TC=0.975952, Ewarp=11.957151.

| trial | family | config | PSNR | SSIM | LPIPS | VFID/FVD | TC | Ewarp | mask PSNR | boundary PSNR | loser degrade | target |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BF07 | second_fixed_bestfirst | fixed_image_px r=28.0 k=0.0 b=5.0 legacy_global_weighted_mean | 29.369880 | 0.968801 | 0.018465 | 0.230950 | 0.975952 | 11.957151 | 17.064569 | 22.951337 | 1.0 | no |
| BF08 | second_fixed_bestfirst | fixed_image_px r=24.0 k=0.0 b=3.5 legacy_global_weighted_mean | 29.366428 | 0.968772 | 0.018372 | 0.230838 | 0.975907 | 11.963386 | 17.061117 | 22.954126 | 1.0 | no |
| BF03 | second_fixed_bestfirst | fixed_image_px r=24.0 k=0.0 b=5.0 legacy_global_weighted_mean | 29.363525 | 0.968835 | 0.018364 | 0.230150 | 0.975968 | 11.979235 | 17.058214 | 22.908234 | 0.0 | no |
| BF05 | second_fixed_bestfirst | fixed_image_px r=32.0 k=0.0 b=5.0 legacy_global_weighted_mean | 29.361559 | 0.968802 | 0.018469 | 0.230818 | 0.975975 | 11.972037 | 17.056248 | 22.898742 | 0.0 | no |
| BF02 | second_fixed_bestfirst | fixed_image_px r=22.0 k=0.0 b=4.0 legacy_global_weighted_mean | 29.359382 | 0.968829 | 0.018278 | 0.229598 | 0.976047 | 11.986451 | 17.054071 | 22.908672 | 0.0 | no |
| BF06 | second_fixed_bestfirst | fixed_image_px r=28.0 k=0.0 b=3.5 legacy_global_weighted_mean | 29.354710 | 0.968798 | 0.018317 | 0.229208 | 0.976010 | 11.985880 | 17.049399 | 22.898726 | 0.0 | no |
| BF04 | second_fixed_bestfirst | fixed_image_px r=32.0 k=0.0 b=3.5 legacy_global_weighted_mean | 29.354699 | 0.968787 | 0.018369 | 0.229785 | 0.975960 | 11.982223 | 17.049388 | 22.894874 | 0.0 | no |
| BF01 | second_fixed_bestfirst | fixed_image_px r=14.0 k=0.0 b=2.0 legacy_global_weighted_mean | 29.344260 | 0.968809 | 0.017976 | 0.227433 | 0.976203 | 12.015804 | 17.038948 | 22.904445 | 1.0 | no |
