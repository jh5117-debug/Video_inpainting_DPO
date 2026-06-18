# Exp19b Exploratory 2000 DAVIS50 Result

- status: complete
- run_dir: /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/exp19b_boundary_flow_adapter_s2_2000_exploratory_from500_limit100
- adapter: /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/exp19b_boundary_flow_adapter_s2_2000_exploratory_from500_limit100/last_weights/flow_adapter.pt
- eval_dir: /mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp19b_exploratory_s2_2000_davis50
- note: evaluator label still prints Exp19b_stage2_500, but EXP19_ADAPTER points to the 2000 exploratory adapter above.

## Summary

DAVIS50 summary:

```csv
model_label,rows,frames_mean,frames_median,whole_video_psnr_mean,whole_video_psnr_median,whole_video_ssim_mean,whole_video_ssim_median,mask_region_psnr_mean,mask_region_psnr_median,mask_region_ssim_mean,mask_region_ssim_median,strict_mask_pixel_psnr_mean,strict_mask_pixel_psnr_median,boundary_pixel_psnr_mean,boundary_pixel_psnr_median,outside_diff_mean_mean,outside_diff_mean_median,outside_diff_max_mean,outside_diff_max_median,whole_video_lpips_mean,whole_video_lpips_median,ewarp_mean,ewarp_median,flow_conf_mean_mean,flow_conf_mean_median,valid_flow_ratio_mean,valid_flow_ratio_median,mean_flow_magnitude_mean,mean_flow_magnitude_median,gate_mean_mean,gate_mean_median,gate_p10_mean,gate_p10_median,gate_p50_mean,gate_p50_median,gate_p90_mean,gate_p90_median,nonzero_gate_ratio_mean,nonzero_gate_ratio_median
SFT48000_baseline,50,24.0,24.0,32.66533014766,32.659994579047385,0.9710623860154729,0.9790511111726816,23.81886321799523,23.585146103636646,0.7959685279224651,0.8328522370191286,21.02188005711482,20.421875706705343,26.194571345974246,25.732078455223977,0.0,0.0,0.0,0.0,0.01622243576256248,0.014673170730626831,7.214799257015567,4.924778897396248,,,,,,,,,,,,,,,,
Exp11_boundary_outer_b075_S2,50,24.0,24.0,32.84021345665417,33.038394183451246,0.9718181384357395,0.9798867046761713,23.99374652698941,23.565652071000937,0.802023148507393,0.8363504823137511,21.196763366108993,20.728429055398333,26.441316368183966,26.012912758659684,0.0,0.0,0.0,0.0,0.015339268120005727,0.013726946044092378,7.1817817833051905,4.934992975271914,,,,,,,,,,,,,,,,
Exp19b_stage2_500,50,24.0,24.0,32.840121525915784,33.0377474484263,0.9718176683105323,0.9798852788369848,23.993654596251023,23.56616518891031,0.8020217018206953,0.8363450067784253,21.196671435370607,20.728482707219825,26.44122417345095,26.013121440925893,0.0,0.0,0.0,0.0,0.015340347875996183,0.01372836371107648,7.181850381387883,4.935021375548905,0.7247146761417389,0.7430773079395294,0.8434480857849121,0.8576794564723969,5.98978155747056,4.144049167633057,0.0039867746154777705,0.003481472027488053,0.0,0.0,0.0,0.0,0.0,0.0,0.008373625646345317,0.006981618236750364
```

Compact table:

| Method | PSNR | SSIM | LPIPS | Ewarp | strict mask PSNR | boundary PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT-48000 | 32.665330 | 0.971062 | 0.016222 | 7.214799 | 21.021880 | 26.194571 |
| Exp11 outer b0.75 S2 | 32.840213 | 0.971818 | 0.015339 | 7.181782 | 21.196763 | 26.441316 |
| Exp19b exploratory 2000 | 32.840122 | 0.971818 | 0.015340 | 7.181850 | 21.196671 | 26.441224 |

Delta versus Exp11:

| Metric | Delta |
| --- | ---: |
| PSNR | -0.000092 |
| SSIM | -0.000000 |
| LPIPS | +0.000001 |
| Ewarp | +0.000069 |
| strict mask PSNR | -0.000092 |
| boundary PSNR | -0.000092 |

Representative contact sheets were copied to HAL at:

```text
/home/hj/exp19b_exploratory_davis50_contact_sheets/
```

Reviewed `dog-agility`, `car-roundabout`, and `boat`; Exp19b exploratory 2000
was visually tied with Exp11 and did not show a reliable temporal or boundary
improvement.

## Decision

Exploratory 2000 does not validate the tiny DAVIS10 Ewarp trend. Compared with Exp11, PSNR, SSIM, LPIPS, strict-mask PSNR, boundary PSNR, and Ewarp are all slightly worse. Do not continue Exp19b to longer training under this setup.
