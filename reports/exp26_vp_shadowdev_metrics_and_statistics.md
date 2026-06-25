# Exp26 VideoPainter Shadow-Dev Metrics And Statistics

- status: `VIDEOPAINTER_SHADOWDEV_METRIC_GATE_PASSED_PENDING_VISUAL_AND_SEED`
- whole_comp_psnr_delta_no_first_frame: +5.160739

## Primary Step50 - Step0 Comp Frame1-48

| metric | n | mean_delta | median_delta | win_rate | bootstrap_ci_low | bootstrap_ci_high | probability_improved | leave_one_out_min | leave_one_out_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| whole_video_lpips | 32 | -0.040142 | -0.033544 | 0.937500 | -0.052638 | -0.028834 | 1.000000 | -0.041673 | -0.036627 |
| strict_mask_pixel_psnr | 32 | 5.186942 | 4.784493 | 0.781250 | 2.781118 | 7.818869 | 1.000000 | 4.413658 | 5.560689 |
| boundary_pixel_psnr | 32 | 12.175098 | 12.151278 | 1.000000 | 10.184673 | 14.212251 | 1.000000 | 11.713893 | 12.537872 |
| ewarp_mask_region | 32 | -8.378847 | -2.861780 | 0.968750 | -13.581173 | -4.369433 | 1.000000 | -8.650984 | -6.448122 |


Step10/Step30 are trajectory diagnostics only and are not used for checkpoint reselection.
