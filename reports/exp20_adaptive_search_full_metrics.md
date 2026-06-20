# Exp20 First-Wave Full Metrics Backfill

- source: existing hard-comp frame outputs; no DiffuEraser re-inference
- video_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/dev_boundary_search_v1_e385cc27_roots/JPEGImages_432_240`
- mask_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/dev_boundary_search_v1_e385cc27_roots/test_masks`
- I3D: `weights/i3d_rgb_imagenet.pt`
- TC model: `weights/open_clip_vit_h14`

| Method | PSNR | SSIM | LPIPS | VFID/FVD | TC | Ewarp | mask PSNR | boundary PSNR | rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AD01 | 29.357017 | 0.968920 | 0.017931 | 0.227119 | 0.976119 | 12.020403 | 17.051705 | 22.916456 | 16 |
| AD02 | 29.358743 | 0.968916 | 0.018003 | 0.228086 | 0.976130 | 12.013275 | 17.053431 | 22.916690 | 16 |
| AD03 | 29.359069 | 0.968895 | 0.018111 | 0.228540 | 0.976099 | 12.004064 | 17.053758 | 22.910063 | 16 |
| AD04 | 29.359881 | 0.968924 | 0.017984 | 0.227926 | 0.976143 | 12.014500 | 17.054570 | 22.917904 | 16 |
| AD05 | 29.357826 | 0.968900 | 0.018046 | 0.228550 | 0.976085 | 12.009945 | 17.052515 | 22.912990 | 16 |
| AD06 | 29.359513 | 0.968887 | 0.018168 | 0.228898 | 0.976085 | 11.998575 | 17.054202 | 22.906608 | 16 |
