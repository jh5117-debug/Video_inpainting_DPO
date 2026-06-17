# Metric Summary

DAVIS10 sanity eval completed with the fixed protocol:

```text
raw6, no PCM, no mask dilation, no Gaussian blur, hard comp, frame-wise metric, no VBench
```

Exp16 was evaluated as `DPO-S1 Stage1-500 + SFT-48000 Stage2 hybrid`.

| Method | PSNR | SSIM | strict mask PSNR | boundary PSNR | bbox PSNR | bbox SSIM |
|---|---:|---:|---:|---:|---:|---:|
| SFT-48000 | 29.8193 | 0.9625 | 18.2894 | 24.2926 | 21.3016 | 0.7380 |
| Exp11 outer b0.75 S2 | 30.1736 | 0.9644 | 18.6437 | 24.5907 | 21.6559 | 0.7513 |
| Exp16 Stage1-500 | 29.9460 | 0.9642 | 18.4161 | 24.5280 | 21.4284 | 0.7562 |

Conclusion: Exp16 Stage1-500 improves over SFT-48000 on this subset, but it
does not exceed Exp11 outer b0.75 S2 on the primary PSNR / strict-mask /
boundary-PSNR metrics.

Reports:

```text
reports/exp16_stage1_500_davis10_metric_summary.md
reports/exp16_stage1_500_visual_case_judgement.md
```
