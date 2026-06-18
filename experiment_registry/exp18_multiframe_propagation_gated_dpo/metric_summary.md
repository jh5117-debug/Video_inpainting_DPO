# Metric Summary

DAVIS10 hybrid metric summary:

| Method | PSNR | SSIM | strict mask PSNR | boundary PSNR |
|---|---:|---:|---:|---:|
| Exp11 boundary outer b0.75 S2 | 30.2413 | 0.9650 | 18.7114 | 24.8326 |
| Exp18a prop-only S1-500 | 30.1024 | 0.9650 | 18.5725 | 24.7090 |
| Exp18b prop+gen S1-500 | 29.6892 | 0.9609 | 18.1593 | 24.7152 |
| Exp18c oracle S1-500 | 29.7626 | 0.9632 | 18.2326 | 24.7991 |
| SFT-48000 baseline | 30.0126 | 0.9635 | 18.4827 | 24.4772 |

Decision:

```text
No Exp18 variant beats Exp11 on DAVIS10 primary metrics.
```

Source:

```text
reports/exp18_davis10_metric_summary.md
```
