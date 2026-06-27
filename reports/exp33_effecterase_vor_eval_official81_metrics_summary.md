# Exp33 EffectErase VOR-Eval Official81 Metrics Summary

Status: `EXP33_EFFECTERASE_BASELINE_WEAK`

- Manifest: `exp33_effecterase_vor_eval_baseline/manifests/effecterase_vor_eval_official81_ready.jsonl`
- Rows: 43
- OK rows: 43
- Metrics CSV: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp33_effecterase_vor_eval_baseline/vor_eval_official81_compat_20260627_134945/reports/exp33_effecterase_vor_eval_official81_metrics.csv`
- Visual review CSV: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp33_effecterase_vor_eval_baseline/vor_eval_official81_compat_20260627_134945/reports/exp33_effecterase_vor_eval_official81_visual_review.csv`
- Review sheets: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp33_effecterase_vor_eval_baseline/vor_eval_official81_compat_20260627_134945/reports/exp33_effecterase_vor_eval_official81_visual_review_assets/review_sheets`
- Crop sheets: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp33_effecterase_vor_eval_baseline/vor_eval_official81_compat_20260627_134945/reports/exp33_effecterase_vor_eval_official81_visual_review_assets/crop_sheets`

## Classification Counts

- `BASELINE_MIXED`: 17
- `BASELINE_USABLE`: 9
- `BASELINE_WEAK`: 17

## Aggregate Metrics

- full_psnr_mean: 21.9229
- mask_psnr_mean: 19.3942
- boundary_psnr_mean: 20.0981
- outside_l1_mean: 16.4051
- lpips_mean: nan
- tc_absdiff_over_winner_mean: -0.5791
- ewarp_proxy_mean: 6.4370

LPIPS is computed on an evenly sampled frame subset and resized inputs; it is reported as a baseline signal, not as a training target.
This baseline remains held-out evaluation evidence only and is not used for adapter training or loser mining.
