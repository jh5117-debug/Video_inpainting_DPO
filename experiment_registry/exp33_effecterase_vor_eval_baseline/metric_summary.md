# Exp33 Metric Summary

Status: `EXP33_EFFECTERASE_BASELINE_WEAK`

The held-out VOR-Eval official81 EffectErase baseline has been evaluated on all
43 completed raw outputs.

| metric | value |
| --- | ---: |
| rows | 43 |
| technical-valid outputs | 43 |
| full PSNR mean | 21.9229 |
| full SSIM mean | 0.7349 |
| mask PSNR mean | 19.3942 |
| mask SSIM mean | 0.5889 |
| boundary PSNR mean | 20.0981 |
| outside L1 mean | 16.4051 |
| TC absdiff over winner mean | -0.5791 |
| Ewarp proxy mean | 6.4370 |

Classification counts:

- `BASELINE_USABLE`: 9
- `BASELINE_MIXED`: 17
- `BASELINE_WEAK`: 17

LPIPS is not included in the final aggregate because first-run AlexNet weight
download was too slow. No LPIPS proxy is reported as real LPIPS.

Reports:

- `reports/exp33_effecterase_vor_eval_official81_metrics_summary.md`
- `reports/exp33_effecterase_vor_eval_official81_metrics.csv`
- `reports/exp33_effecterase_vor_eval_official81_final_report.md`
