# Exp11 Outer B0.75 S2 DAVIS50 + YouTubeVOS100 Eval

Protocol: raw6, D+G off, no PCM, no mask dilation, no Gaussian blur, hard comp, frame-wise in-memory metrics.

| Dataset | Method | Rows | PSNR | SSIM | LPIPS | VFID | TC | Mask PSNR | Mask SSIM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DAVIS50 | SFT-48000 baseline | 50 | 32.7314 | 0.9705 | 0.0167 | 0.2018 | 0.9712 | 23.8849 | 0.7976 |
| DAVIS50 | Exp11 outer b0.75 S2 | 50 | 33.0140 | 0.9723 | 0.0154 | 0.1754 | 0.9711 | 24.1675 | 0.8099 |
| YouTubeVOS100 | SFT-48000 baseline | 100 | 33.3968 | 0.9701 | 0.0176 | 0.2007 | 0.9819 | 24.4262 | 0.7935 |
| YouTubeVOS100 | Exp11 outer b0.75 S2 | 100 | 33.7238 | 0.9711 | 0.0168 | 0.1925 | 0.9821 | 24.7532 | 0.7990 |

## Deltas

| Dataset | PSNR Delta | SSIM Delta | LPIPS Delta | VFID Delta | TC Delta | Mask PSNR Delta |
|---|---:|---:|---:|---:|---:|---:|
| DAVIS50 | +0.2826 | +0.0018 | -0.0013 | -0.0264 | -0.0001 | +0.2826 |
| YouTubeVOS100 | +0.3270 | +0.0009 | -0.0008 | -0.0083 | +0.0001 | +0.3270 |

## Paths

- DAVIS50 summary: `/home/hj/dpo-2-1-exp/this_week_exp11_exp12/metrics/summary_all.csv`
- YouTubeVOS100 summary: `/home/hj/dpo-2-1-exp/this_week_exp11_exp12/youtubevos100_eval/exp11_outer_b075_s2_youtubevos100_20260615_194218_youtubevos100_raw6_hardcomp/metrics/summary_all.csv`
- YouTubeVOS100 PAI output: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp11_outer_b075_s2_youtubevos100_20260615_194218_youtubevos100_raw6_hardcomp`
