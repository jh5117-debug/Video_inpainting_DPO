# Exp20 First-Wave Full Metrics Backfill

- source: existing hard-comp frame outputs; no DiffuEraser re-inference
- video_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/dev_boundary_search_v1_e385cc27_roots/JPEGImages_432_240`
- mask_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/dev_boundary_search_v1_e385cc27_roots/test_masks`
- I3D: `weights/i3d_rgb_imagenet.pt`
- TC model: `weights/open_clip_vit_h14`

| Method | PSNR | SSIM | LPIPS | VFID/FVD | TC | Ewarp | mask PSNR | boundary PSNR | rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MS_P0_s20260620 | 29.349671 | 0.968786 | 0.017994 | 0.228470 | 0.976084 | 12.031089 | 17.044359 | 22.910399 | 16 |
| MS_P0_s20260621 | 29.363087 | 0.968728 | 0.018019 | 0.223098 | 0.976116 | 12.002054 | 17.057776 | 22.950321 | 16 |
| MS_P4_s20260620 | 29.368310 | 0.968839 | 0.018242 | 0.231691 | 0.975967 | 12.004094 | 17.062998 | 22.932378 | 16 |
| MS_P4_s20260621 | 29.389904 | 0.968849 | 0.018327 | 0.225527 | 0.975967 | 11.976737 | 17.084592 | 22.968249 | 16 |
