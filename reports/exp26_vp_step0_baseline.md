# Exp26 VideoPainter Step0 Search-Dev Baseline

- run_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/searchdev_step0_official_20260625_131957`
- mask_manifest: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/searchdev_step0_official_20260625_131957/gate64_mask_ready.jsonl`
- raw_pair_manifest: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/searchdev_step0_official_20260625_131957/step0_metric_eval/step0_raw_metric_pairs.jsonl`
- comp_pair_manifest: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/searchdev_step0_official_20260625_131957/step0_metric_eval/step0_comp_metric_pairs.jsonl`
- review_dir: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/searchdev_step0_official_20260625_131957/step0_review`
- review_status: `TEMPORAL_REVIEW_PASS_DENSE_EVIDENCE`
- review_samples: `32`

## Aggregate Metrics

| variant | PSNR | SSIM | LPIPS | Ewarp | mask PSNR | boundary PSNR | status rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 22.829909342886836 | 0.8492259011345125 | 0.09040529020302882 | 7.401047145404978 | 15.996989577661143 | 15.445102895985176 | 32 |
| comp | 24.301897366442233 | 0.871557953992803 | 0.07080062118242197 | 8.04273951919754 | 16.012427341260924 | 16.01195236353609 | 32 |

Step0 is a baseline only. It does not authorize 10-step training unless the temporal review and L0/L1 gates are also complete.
