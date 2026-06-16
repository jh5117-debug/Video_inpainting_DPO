# Exp15 DAVIS50 OR Quantitative Summary

| Method | Status | Success | PSNR_bg | SSIM_bg_ignore_mask | TC_bg | Notes |
|---|---|---:|---:|---:|---:|---|
| propainter | ok | 50/50 | 35.5274 | 0.9927 | 35.7664 |  |
| videocomposer | failed_or_blocked | 0/50 | nan | nan | nan | no successful predictions |
| cococo | failed_or_blocked | 0/50 | nan | nan | nan | no successful predictions |
| floed | failed_or_blocked | 0/50 | nan | nan | nan | no successful predictions |
| diffueraser_sft48000 | ok | 50/50 | 28.6773 | 0.9686 | 28.8505 |  |
| videopainter | failed_or_blocked | 0/50 | nan | nan | nan | no successful predictions |
| vace | failed_or_blocked | 0/50 | nan | nan | nan | no successful predictions |
| ours_exp11_outer_b075_s2 | ok | 50/50 | 28.6795 | 0.9685 | 28.8682 |  |
| minimax_remover | failed_or_blocked | 0/50 | nan | nan | nan | no successful predictions |

Protocol: no comp; metrics are computed on raw method outputs. PSNR_bg is strict mask-outside pixels. SSIM_bg_ignore_mask ignores foreground by zeroing it in both images.
