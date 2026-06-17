# Metric Summary

DAVIS10 fixed protocol:

```text
raw6, no PCM, no mask dilation, no Gaussian blur, hard comp, frame-wise metrics, no VBench
```

| Method | PSNR | SSIM | strict mask PSNR | boundary PSNR | bbox PSNR | bbox SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exp11 boundary outer b0.75 S2 | 30.2950 | 0.9664 | 18.7651 | 24.7722 | 21.7774 | 0.7705 |
| SFT-48000 baseline | 29.6227 | 0.9616 | 18.0928 | 24.1247 | 21.1050 | 0.7390 |
| Exp17a positive S1-1000 | 29.7313 | 0.9632 | 18.2014 | 24.4509 | 21.2137 | 0.7466 |
| Exp17b saturation S1-1000 | 29.8542 | 0.9623 | 18.3243 | 24.4384 | 21.3366 | 0.7502 |
| Exp17c combined S1-1000 | 29.5117 | 0.9609 | 17.9818 | 24.4214 | 20.9941 | 0.7316 |

Conclusion:

```text
Exp17b is the best Exp17 variant, but no Exp17 variant beats Exp11.
```

Evidence:

```text
reports/exp17_davis10_gate_metric_summary.md
reports/exp17_davis10_gate_metric_summary.csv
reports/exp17_davis10_gate_decision.json
```
