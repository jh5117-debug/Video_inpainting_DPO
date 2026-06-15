# Metrics

Protocol:

```text
raw6, D+G off, no PCM, no mask dilation, no Gaussian blur, hard comp,
frame-wise in-memory metric via tools/run_davis50_framewise_protocol_eval.py
```

| Dataset | Method | Rows | PSNR | SSIM | LPIPS | VFID | TC | Mask PSNR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DAVIS50 | SFT-48000 | 50 | 32.7314 | 0.9705 | 0.0167 | 0.2018 | 0.9712 | 23.8849 |
| DAVIS50 | Exp11 outer b0.75 S2 | 50 | 33.0140 | 0.9723 | 0.0154 | 0.1754 | 0.9711 | 24.1675 |
| YouTubeVOS100 | SFT-48000 | 100 | 33.3968 | 0.9701 | 0.0176 | 0.2007 | 0.9819 | 24.4262 |
| YouTubeVOS100 | Exp11 outer b0.75 S2 | 100 | 33.7238 | 0.9711 | 0.0168 | 0.1925 | 0.9821 | 24.7532 |

Reports:

```text
reports/exp11_outer_b075_s2_youtubevos100_davis50_eval.md
reports/exp11_outer_b075_s2_youtubevos100_davis50_eval.csv
```
