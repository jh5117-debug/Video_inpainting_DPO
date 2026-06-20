# Exp20 Shadow-Dev Baselines

- eval_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/20260620_233144_dev_boundary_shadow_v1_baselines`
- protocol: raw6, hard comp, D+G off, no PCM, no mask dilation, no Gaussian blur, frame-wise metrics
- SHADOW_SFT_PSNR: `29.199451`
- SHADOW_EXP11_S1_PSNR: `29.500778`
- SHADOW_EXP11_S2_PSNR: `29.521728`

| Method | PSNR | SSIM | LPIPS | VFID/FVD | TC | Ewarp | mask PSNR | boundary PSNR | rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Exp11_outer_b075_S1_plus_SFT_S2 | 29.500778 | 0.968900 | 0.018102 | 0.217299 | 0.975506 | 8.890011 | 18.207907 | 25.376861 | 20 |
| Exp11_outer_b075_S2 | 29.521728 | 0.968988 | 0.018183 | 0.217858 | 0.975675 | 8.878917 | 18.228857 | 25.390760 | 20 |
| SFT48000_baseline | 29.199451 | 0.966516 | 0.019383 | 0.240703 | 0.974590 | 8.955795 | 17.906581 | 25.031478 | 20 |
