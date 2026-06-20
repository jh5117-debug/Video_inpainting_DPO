# Exp20 First-Wave Full Metrics Backfill

- source: existing hard-comp frame outputs; no DiffuEraser re-inference
- video_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/dev_boundary_search_v1_e385cc27_roots/JPEGImages_432_240`
- mask_root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp20_autoresearch_scale_adaptive_region_dpo/dev_boundary_search_v1_e385cc27_roots/test_masks`
- I3D: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp20_autoresearch/weights/i3d_rgb_imagenet.pt`
- TC model: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp20_autoresearch/weights/open_clip_vit_h14`

| Method | PSNR | SSIM | LPIPS | VFID/FVD | TC | Ewarp | mask PSNR | boundary PSNR | rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0_s20260619 | 29.374471 | 0.968993 | 0.017839 | 0.223601 | 0.976136 | 12.016644 | 17.069160 | 22.947473 | 16 |
| P4_s20260619 | 29.389916 | 0.969032 | 0.018072 | 0.227931 | 0.976041 | 11.993817 | 17.084604 | 22.966567 | 16 |
| BF07_s20260619 | 29.393079 | 0.968993 | 0.018441 | 0.232887 | 0.975930 | 11.967787 | 17.087768 | 22.947160 | 16 |
| BF07_s20260620 | 29.367111 | 0.968731 | 0.018523 | 0.231996 | 0.975948 | 11.974237 | 17.061800 | 22.936356 | 16 |
| BF07_s20260621 | 29.380085 | 0.968694 | 0.018655 | 0.230802 | 0.975767 | 11.949698 | 17.074773 | 22.937501 | 16 |
| P0_s20260620 | 29.331430 | 0.968619 | 0.017871 | 0.224025 | 0.976162 | 12.034854 | 17.026119 | 22.926944 | 16 |
| P0_s20260621 | 29.364145 | 0.968710 | 0.018035 | 0.225900 | 0.976067 | 12.006843 | 17.058834 | 22.937845 | 16 |
| P4_s20260620 | 29.359948 | 0.968728 | 0.018164 | 0.228588 | 0.976031 | 12.003636 | 17.054637 | 22.952693 | 16 |
| P4_s20260621 | 29.383869 | 0.968768 | 0.018312 | 0.228234 | 0.975967 | 11.978300 | 17.078557 | 22.958114 | 16 |
