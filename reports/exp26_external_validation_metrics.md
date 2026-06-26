# Exp26 External Validation Metrics

- status: `EXP26_EXTERNAL_VALIDATION_NOT_CONFIRMED`
- whole comp PSNR delta no-first-frame: `-2.563047`

## Primary Step50 - Step0 Comp Frame1-48

| metric | n | mean_delta | median_delta | win_rate | bootstrap_ci_low | bootstrap_ci_high | probability_improved | leave_one_out_min | leave_one_out_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| whole_video_lpips | 32 | 0.002466 | 0.004605 | 0.437500 | -0.008718 | 0.013012 | 0.323800 | 0.000967 | 0.005189 |
| strict_mask_pixel_psnr | 32 | -2.610576 | -3.378906 | 0.218750 | -4.434436 | -0.624556 | 0.006500 | -3.239878 | -2.316287 |
| boundary_pixel_psnr | 32 | 0.662358 | -0.287562 | 0.437500 | -1.275275 | 2.649467 | 0.742700 | 0.128141 | 1.002071 |
| ewarp_mask_region | 32 | -3.602171 | -0.039578 | 0.500000 | -16.682015 | 6.197310 | 0.699300 | -5.344481 | 1.557997 |


Step10/Step30 are trajectory diagnostics only and are not used for checkpoint reselection.
