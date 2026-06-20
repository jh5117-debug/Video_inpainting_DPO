# Exp20 First-Wave Full Metrics Backfill

- source: existing hard-comp frame outputs; no DiffuEraser re-inference
- video_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/dev_boundary_search_v1_e385cc27_roots/JPEGImages_432_240`
- mask_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/dev_boundary_search_v1_e385cc27_roots/test_masks`
- I3D: `weights/i3d_rgb_imagenet.pt`
- TC model: `weights/open_clip_vit_h14`

| Method | PSNR | SSIM | LPIPS | VFID/FVD | TC | Ewarp | mask PSNR | boundary PSNR | rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT48000_baseline | 29.173336 | 0.967718 | 0.018903 | 0.238237 | 0.975772 | 11.997202 | 16.868025 | 22.530203 | 16 |
| Exp11_outer_b075_S1_plus_SFT_S2 | 29.333541 | 0.968432 | 0.018103 | 0.215391 | 0.976147 | 11.957009 | 17.028230 | 22.935509 | 16 |
| Exp11_outer_b075_S2 | 29.355372 | 0.968554 | 0.018211 | 0.215971 | 0.976050 | 11.939468 | 17.050061 | 22.953233 | 16 |
| Exp20_legacy_exact_untrained_control | 29.173324 | 0.967717 | 0.018902 | 0.238050 | 0.975807 | 11.997322 | 16.868013 | 22.530166 | 16 |
| P0 | 29.373852 | 0.968990 | 0.017842 | 0.223621 | 0.976131 | 12.016763 | 17.068540 | 22.947118 | 16 |
| P1 | 29.358342 | 0.968984 | 0.017876 | 0.226701 | 0.976171 | 12.030603 | 17.053031 | 22.893529 | 16 |
| P2 | 29.366833 | 0.969021 | 0.017936 | 0.227789 | 0.976126 | 12.025352 | 17.061521 | 22.911305 | 16 |
| P3 | 29.374159 | 0.969043 | 0.018020 | 0.229008 | 0.976174 | 12.016267 | 17.068848 | 22.923562 | 16 |
| P4 | 29.390553 | 0.969074 | 0.018198 | 0.232074 | 0.976040 | 11.994790 | 17.085242 | 22.946999 | 16 |
| P5 | 29.385103 | 0.968993 | 0.018433 | 0.234728 | 0.975934 | 11.976952 | 17.079791 | 22.930627 | 16 |
