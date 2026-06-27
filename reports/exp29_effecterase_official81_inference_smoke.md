# Exp29 EffectErase Official 81F Inference Smoke

Status: `EFFECTERASE_OR_BASELINE_READY`
Rows: 8
Technical valid: 8 / 8
Manifest SHA256: `706cb09286fd8528d7efbbb91eb89673a9ec7ce61b0047e6b3b2e8ea4c9b1fb3`
Output root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_official81_20260626`

Raw output remains the primary OR diagnostic. Diagnostic hard-comp videos are generated only for inspection.

## Visual Review Decision

Codex opened all 8 temporal review pages and all 8 crop pages under
`reports/exp29_effecterase_official81_inference_previews/`.

- Technical-valid outputs: 8 / 8.
- Object/effect removal: 8 / 8.
- Black/purple/global collapse: 0 / 8.
- Baseline-ready diagnostic rows: 8 / 8.
- Medium-hard loser rows: 0 / 8.
- Main caveat: the model is strong and VOR-trained; raw outputs sometimes
  regenerate outside or large contextual regions, so these outputs are best
  treated as an OR strong baseline / diagnostic, not as primary on-policy
  DPO losers or scientific adapter-positive evidence.

Final EffectErase role for this milestone:

`EFFECTERASE_OR_BASELINE_READY`

Forbidden conclusions still hold:

- no `EFFECTERASE_TRUE_ADAPTER_FEASIBILITY_CONFIRMED`;
- no `SCIENTIFIC_POSITIVE`;
- no `UNIVERSAL_ADAPTER`.

## Aggregate Pixel Diagnostics

| metric | value |
| --- | ---: |
| boundary_psnr_mean | 26.143048 |
| boundary_ssim_mean | 0.768598 |
| condition_frames_mean | 81.000000 |
| condition_to_winner_mask_abs_mean | 48.822678 |
| fps_mean | 27.750000 |
| height_mean | 480.000000 |
| mask_bbox_ssim_mean | 0.760780 |
| mask_frames_mean | 81.000000 |
| object_effect_residual_ratio_mean | 0.321885 |
| outside_abs_diff_mean_mean | 8.210540 |
| raw_output_frames_mean | 81.000000 |
| raw_to_condition_mask_abs_mean | 48.558864 |
| raw_to_winner_mask_abs_mean | 12.661244 |
| strict_mask_psnr_mean | 25.485254 |
| temporal_abs_diff_delta_mean | -0.412184 |
| temporal_abs_diff_output_mean | 4.408185 |
| temporal_abs_diff_winner_mean | 4.820370 |
| whole_psnr_mean | 27.416948 |
| whole_ssim_mean | 0.840580 |
| width_mean | 832.000000 |
| winner_frames_mean | 81.000000 |

## Project Metric Wrapper

The existing `tools/run_inpainting_metric_eval.py` wrapper was run on the same
8 raw outputs with `inference/metrics.py`, `max_frames=81`, LPIPS enabled, and
Ewarp using the OpenCV DIS fallback.

| Metric | Mean |
| --- | ---: |
| whole_video_psnr | 27.416948 |
| whole_video_ssim | 0.840580 |
| whole_video_lpips | 0.085822 |
| mask_region_psnr | 25.778614 |
| mask_region_ssim | 0.760667 |
| boundary_psnr | 25.696018 |
| boundary_ssim | 0.768534 |
| ewarp_mask_region | 1.766501 |
| outside_region_diff_mean | 8.210687 |

The highest-risk rows are `REAL_ENV005_00003_003_05` and
`BLENDER_BEACH030_00003`, where visual review and metrics both show more
raw-output outside/context regeneration. The cleanest baseline-style rows are
`BLENDER_CON001_00218`, `BLENDER_BEACH036_00001`,
`REAL_ENV097_00001_002_02`, and `REAL_ENV102_00001_002_02`.
