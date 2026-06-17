# Exp15 OR Metric Protocol: MiniMax-Compatible Draft

## Current Compatible Pieces

- No hard comp before scoring.
- Raw method output is compared against the original frame.
- Foreground mask nonzero means object/remove region.
- Background-region PSNR is computed on `mask == 0` pixels.

## Current Non-Compatible / Pending Pieces

- MiniMax Table 2 uses all 90 DAVIS videos; Exp15 currently uses DAVIS50.
- MiniMax TC follows COCOCO/AVID with CLIP-ViT-H/B-14 features. Exp15 `TC_bg` is a pixel temporal-difference proxy and should not be called paper TC.
- MiniMax VQ/Succ are GPT-O3 evaluations and are not implemented in Exp15.
- MiniMax paper does not fully specify SSIM implementation details. Exp15 currently reports `SSIM_bg_ignore_mask`, which zeros the foreground in GT and prediction before SSIM. This is a background-preservation proxy, not guaranteed identical to MiniMax.

## Safe Naming

Use these column names until full paper compatibility is implemented:

- `PSNR_bg`
- `SSIM_bg_ignore_mask`
- `TC_bg_pixel_proxy`
- `num_videos`
- `num_frames`
- `failed_cases`

Do not report Exp15 `TC_bg_pixel_proxy` as MiniMax paper `TC`.
