# Exp16 Stage1-500 DAVIS10 Metric Sanity

| Method | PSNR | SSIM | strict mask PSNR | boundary PSNR | bbox PSNR | bbox SSIM | rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT48000_baseline | 29.8193 | 0.9625 | 18.2894 | 24.2926 | 21.3016 | 0.7380 | 10 |
| Exp11_boundary_outer_b075_S2 | 30.1736 | 0.9644 | 18.6437 | 24.5907 | 21.6559 | 0.7513 | 10 |
| Exp16_stage1_500_limit100 | 29.9460 | 0.9642 | 18.4161 | 24.5280 | 21.4284 | 0.7562 | 10 |

Protocol: raw6, no PCM, no mask dilation, no Gaussian blur, hard comp,
frame-wise metrics, no VBench.

Exp16 Stage1-500 is evaluated as a DPO-S1 + SFT-S2 hybrid because the
Stage1-only checkpoint does not contain the Stage2 motion config required by the
DAVIS evaluator.

Conclusion: Exp16 Stage1-500 improves over SFT-48000 on this DAVIS10 subset,
but it does not exceed Exp11 outer b0.75 S2 on the primary PSNR / strict-mask /
boundary-PSNR metrics.
