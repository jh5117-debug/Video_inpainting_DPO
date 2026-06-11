# Exp11 Existing Eval Audit

Date: 2026-06-11

Interpretation: Exp11 rows are `Exp11-proxy` rows, not real optical-flow /
ProPainter-prior consistency method results.

## Verdict

```text
eval_complete_for_whole_frame_allmetrics = true
strict_mask_pixel_metrics_present = false
```

Existing DAVIS eval is complete for the current whole-frame / bbox all-metric
table. Do not retrain. If strict mask-pixel numbers are needed, run eval-only /
metric-only with the patched strict wrapper.

## Existing Eval Logs

| eval | log | exists | notes |
|---|---|---:|---|
| Exp11_1 v2 PSNR/SSIM | `logs/pipelines/exp09_10_11_20260610_framewise_raw6_davis50_v2/Exp11_1_DPO-S1_SFT-S2_.log` | true | raw6, no PCM, frame-wise, no fatal error |
| Exp11_2 v2 PSNR/SSIM | `logs/pipelines/exp09_10_11_20260610_framewise_raw6_davis50_v2/Exp11_2_DPO-S1_DPO-S2_.log` | true | raw6, no PCM, frame-wise, no fatal error |
| Exp11_1 LPIPS/VFID | `logs/pipelines/exp09_10_11_20260611_041757_framewise_raw6_davis50_lpips_vfid/Exp11_1_DPO-S1_SFT-S2_.log` | true | raw6, no PCM, frame-wise, no fatal error |
| Exp11_2 LPIPS/VFID | `logs/pipelines/exp09_10_11_20260611_041757_framewise_raw6_davis50_lpips_vfid/Exp11_2_DPO-S1_DPO-S2_.log` | true | raw6, no PCM, frame-wise, no fatal error |
| Exp11_1 allmetrics | `logs/pipelines/exp09_10_11_20260611_050013_framewise_raw6_davis50_allmetrics/Exp11_1_DPO-S1_SFT-S2_.log` | true | frame-wise allmetrics summary exists; no fatal error |
| Exp11_2 allmetrics | `logs/pipelines/exp09_10_11_20260611_050013_framewise_raw6_davis50_allmetrics/Exp11_2_DPO-S1_DPO-S2_.log` | true | frame-wise allmetrics summary exists; no fatal error |

## Output Roots

| eval | output root | rows | frames | PSNR/SSIM | LPIPS | VFID | TC | strict mask |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Exp11_1 allmetrics | `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp09_10_11_20260611_050013_framewise_raw6_davis50_allmetrics/Exp11_1_DPO-S1_SFT-S2_` | 50 | 24 | true | true | true | true | false |
| Exp11_2 allmetrics | `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp09_10_11_20260611_050013_framewise_raw6_davis50_allmetrics/Exp11_2_DPO-S1_DPO-S2_` | 50 | 24 | true | true | true | true | false |
| Exp11_1 LPIPS/VFID | `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp09_10_11_20260611_041757_framewise_raw6_davis50_lpips_vfid/Exp11_1_DPO-S1_SFT-S2_` | 50 | 24 | true | true | true | false | false |
| Exp11_2 LPIPS/VFID | `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp09_10_11_20260611_041757_framewise_raw6_davis50_lpips_vfid/Exp11_2_DPO-S1_DPO-S2_` | 50 | 24 | true | true | true | false | false |
| Exp11_1 v2 | `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp09_10_11_20260610_framewise_raw6_davis50_v2/Exp11_1_DPO-S1_SFT-S2_` | 50 | 24 | true | false | false | false | false |
| Exp11_2 v2 | `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp09_10_11_20260610_framewise_raw6_davis50_v2/Exp11_2_DPO-S1_DPO-S2_` | 50 | 24 | true | false | false | false | false |
| Exp11 pipeline stage1 val | `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp11_flow_prior_consistency_dpo_s1s2_2000_davis_pai_stage1_val_davis_20260609_2331_exp11_n16_gpus4_7_scratch` | 50 videos x 2 models | 24 | true | false | false | false | false |
| Exp11 pipeline stage2 val | `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp11_flow_prior_consistency_dpo_s1s2_2000_davis_pai_stage2_val_davis_20260609_2331_exp11_n16_gpus4_7_scratch` | 50 videos x 2 models | 24 | true | false | false | false | false |

## Metric Values From Current Allmetrics Summary

| eval | PSNR | SSIM | LPIPS | VFID | TC |
|---|---:|---:|---:|---:|---:|
| Exp11_1 DPO-S1 + SFT-S2 | 32.8913746855 | 0.9716430560 | 0.0157243171 | 0.1929596758 | 0.9712286016 |
| Exp11_2 DPO-S1 + DPO-S2 | 32.8838718041 | 0.9718927075 | 0.0157177797 | 0.1829321512 | 0.9711184379 |

## Missing Piece

Existing summaries do not contain:

- `strict_mask_pixel_psnr`
- `boundary_pixel_psnr`

Therefore only strict-mask eval is missing. If needed, rerun metric-only /
eval-only with the patched wrapper. Do not rerun Stage1 or Stage2 training.
