# Exp20 First-Wave Full Metrics Backfill

- source: existing hard-comp frame outputs; no DiffuEraser re-inference
- video_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/dev_boundary_shadow_v1_roots/JPEGImages_432_240`
- mask_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/dev_boundary_shadow_v1_roots/test_masks`
- I3D: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp20_autoresearch/weights/i3d_rgb_imagenet.pt`
- TC model: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp20_autoresearch/weights/open_clip_vit_h14`

| Method | PSNR | SSIM | LPIPS | VFID/FVD | TC | Ewarp | mask PSNR | boundary PSNR | rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0_s20260619 | 29.502417 | 0.969243 | 0.017898 | 0.213700 | 0.975303 | 8.951974 | 18.209546 | 25.440119 | 20 |
| P4_s20260619 | 29.496018 | 0.969100 | 0.018127 | 0.216592 | 0.975500 | 8.928929 | 18.203147 | 25.439937 | 20 |
| BF07_s20260619 | 29.439405 | 0.968614 | 0.018533 | 0.223785 | 0.975361 | 8.905799 | 18.146534 | 25.400457 | 20 |
| AD04_s20260619 | 29.500996 | 0.969164 | 0.018039 | 0.215503 | 0.975393 | 8.936126 | 18.208125 | 25.446934 | 20 |
| BF07_s20260620 | 29.326340 | 0.967797 | 0.018716 | 0.225246 | 0.974955 | 8.903028 | 18.033470 | 25.385629 | 20 |
| BF07_s20260621 | 29.349741 | 0.967811 | 0.018822 | 0.228941 | 0.975098 | 8.896795 | 18.056870 | 25.385569 | 20 |
| P0_s20260620 | 29.368241 | 0.968481 | 0.017936 | 0.213478 | 0.975271 | 8.958385 | 18.075371 | 25.413374 | 20 |
| P0_s20260621 | 29.419253 | 0.968665 | 0.018070 | 0.219977 | 0.975372 | 8.949226 | 18.126382 | 25.423418 | 20 |
| P4_s20260620 | 29.365911 | 0.968287 | 0.018259 | 0.217352 | 0.975177 | 8.926212 | 18.073041 | 25.420541 | 20 |
| P4_s20260621 | 29.386545 | 0.968292 | 0.018421 | 0.221533 | 0.975225 | 8.921426 | 18.093675 | 25.422462 | 20 |
