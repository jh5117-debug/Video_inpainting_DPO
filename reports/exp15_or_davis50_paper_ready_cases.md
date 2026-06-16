# Exp15 DAVIS50 OR Paper-Ready Candidate Cases

These 20 cases are metric-preselected from DAVIS50 OR visual grids. Final paper figures still need human visual review because OR removal quality inside the mask has no GT target.

Visual grid root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp15_or_benchmark_davis50/visual_grids`

| # | Video | Selection note | Ours PSNR_bg | SFT PSNR_bg | ProPainter PSNR_bg | Ours-SFT PSNR | Ours SSIM_bg |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | kite-surf | metric-preselected: Ours improves over SFT | 29.8888 | 29.4031 | 35.1199 | 0.4857 | 0.9653 |
| 2 | mallard-water | metric-preselected: Ours improves over SFT | 26.9083 | 26.5860 | 33.1340 | 0.3222 | 0.9682 |
| 3 | bus | metric-preselected: Ours improves over SFT | 24.3364 | 24.0755 | 27.2330 | 0.2609 | 0.9210 |
| 4 | bear | metric-preselected: Ours improves over SFT | 26.4853 | 26.2285 | 32.4586 | 0.2568 | 0.9524 |
| 5 | drift-straight | metric-preselected: Ours improves over SFT | 27.9580 | 27.7326 | 36.1362 | 0.2254 | 0.9814 |
| 6 | goat | metric-preselected: Ours improves over SFT | 28.0573 | 27.8576 | 34.3981 | 0.1997 | 0.9804 |
| 7 | cows | metric-preselected: Ours improves over SFT | 25.4352 | 25.3234 | 30.2788 | 0.1118 | 0.9503 |
| 8 | hike | metric-preselected: Ours improves over SFT | 27.2533 | 27.1545 | 34.7416 | 0.0988 | 0.9668 |
| 9 | paragliding-launch | metric-preselected: Ours improves over SFT | 30.5231 | 30.4276 | 37.8140 | 0.0956 | 0.9660 |
| 10 | hockey | metric-preselected: Ours improves over SFT | 30.8040 | 30.7386 | 37.7171 | 0.0654 | 0.9818 |
| 11 | paragliding | metric-preselected: high Ours background preservation | 32.2343 | 32.2347 | 43.5516 | -0.0004 | 0.9649 |
| 12 | drift-chicane | metric-preselected: high Ours background preservation | 31.3264 | 31.2837 | 42.9929 | 0.0427 | 0.9710 |
| 13 | surf | metric-preselected: high Ours background preservation | 31.2744 | 31.3513 | 37.6141 | -0.0769 | 0.9760 |
| 14 | libby | metric-preselected: high Ours background preservation | 31.1240 | 31.0958 | 41.4208 | 0.0282 | 0.9693 |
| 15 | bmx-bumps | metric-preselected: high Ours background preservation | 30.6565 | 30.7168 | 38.6930 | -0.0604 | 0.9676 |
| 16 | motocross-bumps | metric-preselected: high Ours background preservation | 30.6064 | 30.6682 | 38.4455 | -0.0618 | 0.9716 |
| 17 | boat | representative harder/failure case | 28.5906 | 28.9143 | 34.3261 | -0.3237 | 0.9661 |
| 18 | motorbike | representative harder/failure case | 28.5660 | 28.8791 | 34.1223 | -0.3131 | 0.9662 |
| 19 | motocross-jump | representative harder/failure case | 29.3998 | 29.6731 | 36.5461 | -0.2733 | 0.9719 |
| 20 | car-turn | representative harder/failure case | 29.2314 | 29.4917 | 38.0130 | -0.2602 | 0.9543 |

Caution: ProPainter is much stronger on background preservation in this OR protocol. Ours should not be claimed as best OR baseline based on these metrics.
