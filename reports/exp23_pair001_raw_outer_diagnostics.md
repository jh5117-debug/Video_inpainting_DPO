# Exp23 Pair001 Raw Outer-Ring Diagnostics

pair_id: `phaseA_scale1_pair001_outer2_corrected_outer_control_seed20260619_gpus2456`

Computed from saved raw DiffuEraser frames and DAVIS masks with explicit `mask > 127`, without rerunning model inference. Positive PSNR/SSIM delta means candidate is better; positive LPIPS delta means worse.

| region | metric | fresh | candidate | delta |
|---|---:|---:|---:|---:|
| mask_core | raw_psnr | 21.221588 | 21.286828 | 0.065240 |
| mask_core | raw_ssim | 0.803824 | 0.804022 | 0.000198 |
| mask_core | raw_lpips | 0.015567 | 0.015546 | -0.000021 |
| mask_core | hard_psnr | 21.221588 | 21.286828 | 0.065240 |
| mask_core | hard_ssim | 0.803824 | 0.804022 | 0.000198 |
| mask_core | hard_lpips | 0.015567 | 0.015546 | -0.000021 |
| outer1 | raw_psnr | nan | nan | nan |
| outer1 | raw_ssim | 1.000000 | 1.000000 | 0.000000 |
| outer1 | raw_lpips | 0.000000 | 0.000000 | 0.000000 |
| outer1 | hard_psnr | nan | nan | nan |
| outer1 | hard_ssim | 1.000000 | 1.000000 | 0.000000 |
| outer1 | hard_lpips | 0.000000 | 0.000000 | 0.000000 |
| outer2_cumulative | raw_psnr | nan | nan | nan |
| outer2_cumulative | raw_ssim | 1.000000 | 1.000000 | 0.000000 |
| outer2_cumulative | raw_lpips | 0.000000 | 0.000000 | 0.000000 |
| outer2_cumulative | hard_psnr | nan | nan | nan |
| outer2_cumulative | hard_ssim | 1.000000 | 1.000000 | 0.000000 |
| outer2_cumulative | hard_lpips | 0.000000 | 0.000000 | 0.000000 |
| outer2_band | raw_psnr | nan | nan | nan |
| outer2_band | raw_ssim | 1.000000 | 1.000000 | 0.000000 |
| outer2_band | raw_lpips | 0.000000 | 0.000000 | 0.000000 |
| outer2_band | hard_psnr | nan | nan | nan |
| outer2_band | hard_ssim | 1.000000 | 1.000000 | 0.000000 |
| outer2_band | hard_lpips | 0.000000 | 0.000000 | 0.000000 |

Key observation: raw outer rings are measurable after explicit postprocessing. Candidate improves raw mask core but does not produce a clean outer-ring win; official hard-comp boundary PSNR is lower for candidate at S2-2000.
