# Exp59 Metric Summary

- Rows: 8
- Decode pass: 8/8
- Frame count: 24 for all rows
- Resolution: 128x128 for all rows
- FPS: 8 for all rows
- Quadmask values: `0|63|127|255` for all rows
- `target_hit=true`: 0/8

Official inference metrics were computed in native diagnostic space: first 24 frames, official output downscaled from `384x672` to `128x128`, compared against `rgb_removed`.

Mean metrics:

- full PSNR: `30.152555`
- SSIM: `0.919492`
- object PSNR: `28.337691`
- overlap PSNR: `16.673219`
- affected PSNR: `17.527094`
- boundary PSNR: `22.267098`
- outside PSNR: `34.210532`
- outside L1: `0.014922`
- temporal flicker: `0.012497`
- object residual: `0.025721`
- effect residual: `0.078867`
- tone drift: `0.009077`

LPIPS, Ewarp, and TC are `NA` in this no-training diagnostic path.
