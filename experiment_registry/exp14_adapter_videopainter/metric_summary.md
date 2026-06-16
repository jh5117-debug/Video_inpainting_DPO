# Metric Summary

Status: completed_davis50_eval.

DAVIS50 eval path:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/logs/target_eval/exp14_videopainter_adapter_gate2000_davis
```

Local metric snapshots:

```text
reports/videopainter_adapter_gate2000_davis_summary.csv
reports/videopainter_adapter_gate2000_davis_per_video.csv
exp14_adapter_videopainter/metrics/davis_summary.csv
exp14_adapter_videopainter/metrics/davis_per_video.csv
```

Protocol:

- DAVIS50, 50 videos, 2366 frames;
- VideoPainter inference steps: 50;
- hard comp with GT outside mask;
- no mask dilation;
- no Gaussian blur;
- no VBench;
- metric backend: `inference/metrics.py`.

| method | PSNR | SSIM | strict mask PSNR | LPIPS | videos | frames |
|---|---:|---:|---:|---:|---:|---:|
| VideoPainter baseline | 31.6124 | 0.9608 | 19.9691 | n/a | 50 | 2366 |
| VideoPainter + DPO adapter | 29.8028 | 0.9580 | 18.1595 | n/a | 50 | 2366 |

Conclusion: the adapter underperforms baseline by 1.8096 PSNR and 0.0028 SSIM.
It improved 16 / 50 videos but dropped on 34 / 50, so the full-set result is
negative.
