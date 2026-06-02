# Target Inpainting Metric Interface Report

status: **prepared**

## Backend Resolution

No repository file named `metric.py` is present in this checkout. The existing inpainting metric backend is:

`inference/metrics.py`

It already provides:

- `compute_psnr`
- `compute_ssim`
- `calc_psnr_and_ssim`
- `LPIPSMetric`
- `EwarpMetric`
- `MetricsCalculator`
- video/mask readers for OpenCV-backed video files

## Wrapper

New wrapper:

`tools/run_inpainting_metric_eval.py`

The wrapper does not reimplement PSNR/SSIM/LPIPS/Ewarp. It only:

- reads CSV or JSONL pair manifests
- resolves video and mask paths
- aligns frame counts
- crops whole, mask-region, and boundary regions
- calls `inference.metrics` for metric calculation
- writes per-sample and grouped summaries

## Manifest Contract

Default columns:

| column | meaning |
| --- | --- |
| `sample_id` | stable sample id |
| `model_label` | model/checkpoint name |
| `gt_video_path` | winner / ground-truth video |
| `prediction_video_path` | model output video |
| `mask_path` | partial-mask video or frame directory |

Column names can be overridden with CLI flags.

## Outputs

For `--output_dir <OUT>`:

- `<OUT>/metrics/per_sample_metrics.csv`
- `<OUT>/metrics/summary.csv`
- `<OUT>/metrics/summary.json`
- `<OUT>/metrics/summary.md`
- `<OUT>/metric_adapter_manifest.json`

## Policy

Use VBench only for video generation / full-mask prompt generation. Use this wrapper and `inference/metrics.py` for partial-mask inpainting on YouTube-VOS, DAVIS, D2, or D3.
