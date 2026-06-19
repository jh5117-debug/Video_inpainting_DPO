# Exp20 Dev Baselines

- eval_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/20260620_0112_dev_boundary_search_v1_24f_fixedseed_baselines`
- protocol: raw6, hard comp, D+G off, no PCM, no mask dilation, no Gaussian blur, frame-wise metrics
- SFT_DEV_PSNR: `29.173336`
- EXP11_S1_DEV_PSNR: `29.333541`
- EXP11_S2_DEV_PSNR: `29.355372`
- TARGET_DEV_PSNR: `29.523336`
- VFID / TC: `not available in this run` because the PAI worktree has no local I3D or OpenCLIP TC weights; the evaluator kept PSNR/SSIM/LPIPS/Ewarp and did not fabricate missing metrics.

| Method | PSNR | SSIM | LPIPS | Ewarp | strict mask PSNR | boundary PSNR | rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exp11_outer_b075_S1_plus_SFT_S2 | 29.3335 | 0.9684 | 0.0181 | 11.9570 | 17.0282 | 22.9355 | 16 |
| Exp11_outer_b075_S2 | 29.3554 | 0.9686 | 0.0182 | 11.9395 | 17.0501 | 22.9532 | 16 |
| Exp20_legacy_exact_untrained_control | 29.1733 | 0.9677 | 0.0189 | 11.9973 | 16.8680 | 22.5302 | 16 |
| SFT48000_baseline | 29.1733 | 0.9677 | 0.0189 | 11.9972 | 16.8680 | 22.5302 | 16 |
