# Exp26 VideoPainter Shadow-Dev Seed Robustness

- status: `SEED_ROBUSTNESS_PASS`
- manifest_sha256: `4b3b18bc275eabcdc591ddf18173e34f544811fb2d7a206b014136befb243db2`
- seeds: `[20260619, 20260620, 20260621]`
- rows_per_seed: `[16, 16, 16]`
- primary_direction_pass_count: `3/3`
- mean strict-mask PSNR delta: `+8.005364`
- mean boundary PSNR delta: `+13.627635`
- mean LPIPS delta: `-0.056193`
- mean Ewarp delta: `-9.325711`

| seed | rows | strict_mask_delta | boundary_delta | lpips_delta | ewarp_delta | pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 20260619 | 16 | +7.019990 | +13.411947 | -0.054398 | -10.220416 | True |
| 20260620 | 16 | +7.871717 | +13.562566 | -0.054063 | -8.354823 | True |
| 20260621 | 16 | +9.124383 | +13.908393 | -0.060119 | -9.401894 | True |
