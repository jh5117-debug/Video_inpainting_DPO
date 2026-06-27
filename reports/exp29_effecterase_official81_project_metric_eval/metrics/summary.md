# Inpainting Metric Summary

pair_manifest: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_official81_inference_20260627/reports/exp29_effecterase_official81_metric_manifest.csv`
metric_backend: `inference/metrics.py`
rows_ok: 8
rows_skipped: 0

PSNR/SSIM/LPIPS/Ewarp are delegated to the existing project metric backend.
The wrapper only handles manifest pairing, mask/boundary crops, and aggregation.

## Summary

| model_label | rows | mask_region_psnr_mean | mask_region_ssim_mean | boundary_psnr_mean | boundary_ssim_mean | whole_video_psnr_mean | whole_video_ssim_mean | outside_region_diff_mean_mean | outside_region_diff_max_mean | temporal_diff_delta_vs_gt_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EffectErase_official81_raw | 8 | 25.7786 | 0.760667 | 25.696 | 0.768534 | 27.4169 | 0.84058 | 8.21069 | 141.559 | 1.7665 |
