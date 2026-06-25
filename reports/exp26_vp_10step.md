# Exp26 VideoPainter 10-Step Gate

Status: `VIDEOPAINTER_10STEP_GATE_PASSED`

## Gate Reasons

- all pre-registered 10-step tolerances passed

## Comp Metrics

| step | PSNR | SSIM | LPIPS | Ewarp | mask PSNR | boundary PSNR |
|---|---:|---:|---:|---:|---:|---:|
| step0 | 24.301897 | 0.871558 | 0.070801 | 8.042740 | 16.012427 | 16.011952 |
| step1 | 27.186413 | 0.921252 | 0.078710 | 1.728870 | 18.946673 | 22.795688 |
| step10 | 25.279150 | 0.904199 | 0.066301 | 6.741283 | 16.987619 | 21.094159 |

## Step10 - Step0

- PSNR: `+0.977252`
- SSIM: `+0.032641`
- LPIPS: `-0.004499`
- Ewarp: `-1.301457`
- strict mask PSNR: `+0.975192`
- boundary pixel PSNR: `+5.082206`

## Diagnostics

- rows: `10`
- max grad norm: `170.04361478093455`
- final loser-dominant ratio: `0.0`
- NaN/Inf count: `0`

This report is a 10-step gate only. It does not start 50-step unless the status is `VIDEOPAINTER_10STEP_GATE_PASSED` and visual review is complete.
