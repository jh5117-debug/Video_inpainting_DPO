# Exp20 First-Wave Full Metrics Backfill

- source: existing hard-comp frame outputs; no DiffuEraser re-inference
- video_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/dev_boundary_search_v1_e385cc27_roots/JPEGImages_432_240`
- mask_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/dev_boundary_search_v1_e385cc27_roots/test_masks`
- I3D: `weights/i3d_rgb_imagenet.pt`
- TC model: `weights/open_clip_vit_h14`

| Method | PSNR | SSIM | LPIPS | VFID/FVD | TC | Ewarp | mask PSNR | boundary PSNR | rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EQ_AD04 | 29.387359 | 0.969029 | 0.017982 | 0.226749 | 0.976091 | 12.002448 | 17.082048 | 22.969489 | 16 |
| EQ_BF07 | 29.393079 | 0.968993 | 0.018441 | 0.232887 | 0.975930 | 11.967787 | 17.087768 | 22.947160 | 16 |
| EQ_P0 | 29.374471 | 0.968993 | 0.017839 | 0.223601 | 0.976136 | 12.016644 | 17.069160 | 22.947473 | 16 |
| EQ_P4 | 29.389916 | 0.969032 | 0.018072 | 0.227931 | 0.976041 | 11.993817 | 17.084604 | 22.966567 | 16 |
| EQ_RB08 | 29.361332 | 0.968911 | 0.017950 | 0.226904 | 0.976039 | 11.998562 | 17.056020 | 22.925615 | 16 |
