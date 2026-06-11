# Strict Mask-Pixel Metric Audit

Date: 2026-06-11

Scope:

- `tools/run_davis50_framewise_protocol_eval.py`
- `tools/run_inpainting_metric_eval.py`
- `inference/metrics.py`

## Verdict

The previous DAVIS frame-wise table was valid for the fixed hard-comp
whole-frame protocol, but the previous `mask_region_psnr` / `boundary_psnr`
fields were bbox-style crop metrics in the main wrappers, not strict
mask-pixel metrics.

## Definitions

Hard comp / D+G off is the video-generation protocol:

```text
pred_comp = pred_raw * mask + GT * (1 - mask)
mask dilation = 0
Gaussian blur during comp = off
PCM = off
raw6
```

Strict mask-pixel metric is the scoring region:

```text
compute metric only where mask == 1
```

These are not the same. Hard comp intentionally copies GT outside the mask, so
whole-frame PSNR/SSIM can be high even if the inpainted region is weak.

## Current Code Findings

| Item | Before Fix | After Fix |
|---|---|---|
| whole-frame PSNR/SSIM | hard-comp whole frame | unchanged |
| `mask_region_psnr` | mask bbox crop | retained as legacy bbox-compatible field |
| `mask_region_ssim` | mask bbox crop | retained as legacy bbox-compatible field |
| strict mask PSNR | missing | added as `strict_mask_pixel_psnr` |
| boundary pixel PSNR | missing in main wrappers | added as `boundary_pixel_psnr` |
| bbox PSNR/SSIM names | ambiguous | added as `mask_bbox_psnr`, `mask_bbox_ssim` |
| outside GT leakage check | partial mean only | added `outside_diff_mean`, `outside_diff_max` |

SSIM note: `inference/metrics.py` computes image/crop SSIM. The wrappers do
not pretend to implement strict arbitrary-pixel SSIM. Bbox SSIM remains named
`mask_bbox_ssim`.

## GT Leakage

The fixed DAVIS protocol uses hard comp, so outside-mask pixels are copied from
GT by construction. This is correct for rendering the final inpainted video,
but final claims must not rely only on whole-frame PSNR/SSIM.

Required reporting fields after this audit:

- `whole_video_psnr`
- `whole_video_ssim`
- `strict_mask_pixel_psnr`
- `boundary_pixel_psnr`
- `outside_diff_mean`
- `outside_diff_max`
- `mask_bbox_psnr`
- `mask_bbox_ssim`

## Required Next Step

Do not reinterpret the existing DAVIS50 table as strict mask-pixel evidence.
Use it as whole-frame hard-comp evidence. To compare strict mask-pixel quality,
rerun the fixed eval wrapper after this patch and read
`strict_mask_pixel_psnr`.
