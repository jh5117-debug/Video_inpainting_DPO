# Exp20 First-Wave Full Metrics Backfill

- source: existing hard-comp frame outputs; no DiffuEraser re-inference
- video_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/dev_boundary_search_v1_e385cc27_roots/JPEGImages_432_240`
- mask_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/dev_boundary_search_v1_e385cc27_roots/test_masks`
- I3D: `weights/i3d_rgb_imagenet.pt`
- TC model: `weights/open_clip_vit_h14`

| Method | PSNR | SSIM | LPIPS | VFID/FVD | TC | Ewarp | mask PSNR | boundary PSNR | rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RB01 | 29.327651 | 0.968747 | 0.017626 | 0.221882 | 0.976071 | 12.031673 | 17.022339 | 22.893952 | 16 |
| RB02 | 29.330825 | 0.968752 | 0.017712 | 0.222688 | 0.976094 | 12.023062 | 17.025514 | 22.898580 | 16 |
| RB03 | 29.336762 | 0.968762 | 0.017864 | 0.224356 | 0.976087 | 12.009635 | 17.031451 | 22.904697 | 16 |
| RB04 | 29.339192 | 0.968752 | 0.017976 | 0.225315 | 0.976095 | 11.999816 | 17.033881 | 22.908671 | 16 |
| RB05 | 29.341867 | 0.968748 | 0.018063 | 0.225225 | 0.976104 | 11.992294 | 17.036556 | 22.911410 | 16 |
| RB06 | 29.322438 | 0.968698 | 0.017714 | 0.222671 | 0.976100 | 12.022642 | 17.017127 | 22.887099 | 16 |
| RB07 | 29.322029 | 0.968642 | 0.017963 | 0.223824 | 0.976091 | 12.000329 | 17.016718 | 22.889239 | 16 |
| RB08 | 29.359323 | 0.968904 | 0.017981 | 0.227438 | 0.976056 | 11.999266 | 17.054011 | 22.919800 | 16 |
| RB09 | 29.353812 | 0.968889 | 0.017886 | 0.225863 | 0.976064 | 12.008453 | 17.048501 | 22.912110 | 16 |
| RB10 | 29.356687 | 0.968867 | 0.018082 | 0.228790 | 0.975989 | 11.991496 | 17.051376 | 22.914620 | 16 |
