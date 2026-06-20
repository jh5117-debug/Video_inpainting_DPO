# Exp20 First-Wave Full Metrics Backfill

- source: existing hard-comp frame outputs; no DiffuEraser re-inference
- video_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/dev_boundary_search_v1_e385cc27_roots/JPEGImages_432_240`
- mask_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/dev_boundary_search_v1_e385cc27_roots/test_masks`
- I3D: `weights/i3d_rgb_imagenet.pt`
- TC model: `weights/open_clip_vit_h14`

| Method | PSNR | SSIM | LPIPS | VFID/FVD | TC | Ewarp | mask PSNR | boundary PSNR | rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF01 | 29.344260 | 0.968809 | 0.017976 | 0.227433 | 0.976203 | 12.015804 | 17.038948 | 22.904445 | 16 |
| BF02 | 29.359382 | 0.968829 | 0.018278 | 0.229598 | 0.976047 | 11.986451 | 17.054071 | 22.908672 | 16 |
| BF03 | 29.363525 | 0.968835 | 0.018364 | 0.230150 | 0.975968 | 11.979235 | 17.058214 | 22.908234 | 16 |
| BF04 | 29.354699 | 0.968787 | 0.018369 | 0.229785 | 0.975960 | 11.982223 | 17.049388 | 22.894874 | 16 |
| BF05 | 29.361559 | 0.968802 | 0.018469 | 0.230818 | 0.975975 | 11.972037 | 17.056248 | 22.898742 | 16 |
| BF06 | 29.354710 | 0.968798 | 0.018317 | 0.229208 | 0.976010 | 11.985880 | 17.049399 | 22.898726 | 16 |
| BF07 | 29.369880 | 0.968801 | 0.018465 | 0.230950 | 0.975952 | 11.957151 | 17.064569 | 22.951337 | 16 |
| BF08 | 29.366428 | 0.968772 | 0.018372 | 0.230838 | 0.975907 | 11.963386 | 17.061117 | 22.954126 | 16 |
