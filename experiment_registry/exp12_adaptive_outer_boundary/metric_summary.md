# Metrics

Protocol: DAVIS50 raw6 hard-comp, D+G off, no PCM, frame-wise in-memory metric.

| Method | Stage | PSNR | SSIM | LPIPS | VFID | TC | Mask PSNR |
|---|---|---:|---:|---:|---:|---:|---:|
| Exp12 adaptive outer b0.75 | DPO-S1 + SFT-S2 | 32.847530 | 0.971693 | 0.015612 | 0.184848 | 0.971164 | 24.001063 |
| Exp12 adaptive outer b0.75 | DPO-S1 + DPO-S2 | 32.856975 | 0.971585 | 0.015605 | 0.193578 | 0.971475 | 24.010508 |
| Exp11 boundary outer b0.75 | DPO-S1 + DPO-S2 | 33.013954 | 0.972295 | 0.015363 | 0.175423 | 0.971122 | 24.167487 |

Conclusion: Exp12 adaptive + outer b0.75 is an ablation; it does not beat Exp11 boundary outer b0.75 S2.
