# Exp20 First-Wave Fixed-Boundary Search

Status: FIRST_WAVE_COMPLETED

Branch/commit used on PAI: `research/exp20-adaptive-region-autoresearch-20260619` @ `64febe8122b0f67d9f5d982c7b0eba49e628ced3`.

Protocol:
- Stage1 only, true DiffuEraser trainer.
- Locked dev split: `exp20_autoresearch_scale_adaptive_region_dpo/manifests/dev_boundary_search_v1.jsonl`.
- 16 videos, 24 frames each.
- raw6, hard comp, D+G off, no PCM, no mask dilation, no Gaussian blur.
- Fixed inference seed: `20260619`.
- LPIPS and Ewarp available. VFID and TC are not available in this worktree because their dependency weights are missing.

Baselines:
- SFT_DEV_PSNR = 29.173336
- EXP11_S1_DEV_PSNR = 29.333541
- EXP11_S2_DEV_PSNR = 29.355372
- TARGET_DEV_PSNR = 29.523336

| Trial | Radius px | Boundary weight | Train steps | PSNR | SSIM | LPIPS | VFID | TC | Ewarp | Mask PSNR | Boundary PSNR | Mean loser degrade | Max grad | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| P0 | 0 | 0.75 | 112 | 29.3739 | 0.9690 | 0.0178 | n/a | n/a | 12.0168 | 17.0685 | 22.9471 | 1.000 | 12.909 | KEEP_CONTROL |
| P1 | 4 | 0.5 | 103 | 29.3583 | 0.9690 | 0.0179 | n/a | n/a | 12.0306 | 17.0530 | 22.8935 | 1.000 | 15.072 | DISCARD |
| P2 | 8 | 0.75 | 103 | 29.3668 | 0.9690 | 0.0179 | n/a | n/a | 12.0254 | 17.0615 | 22.9113 | 1.000 | 13.485 | DISCARD |
| P3 | 12 | 1.0 | 103 | 29.3742 | 0.9690 | 0.0180 | n/a | n/a | 12.0163 | 17.0688 | 22.9236 | 1.000 | 12.256 | KEEP_WITH_WARNING |
| P4 | 16 | 2.0 | 104 | 29.3906 | 0.9691 | 0.0182 | n/a | n/a | 11.9948 | 17.0852 | 22.9470 | 0.909 | 10.109 | KEEP |
| P5 | 24 | 4.0 | 103 | 29.3851 | 0.9690 | 0.0184 | n/a | n/a | 11.9770 | 17.0798 | 22.9306 | 0.727 | 9.340 | KEEP_WITH_WARNING |

Best fixed config by dev PSNR:
- `P4`: fixed_image_px radius 16, boundary weight 2.0.
- PSNR 29.390553; delta vs SFT +0.217217; delta vs Exp11-S1 +0.057012; delta vs Exp11-S2 +0.035181.
- It does not reach TARGET_DEV_PSNR (29.523336).
- LPIPS is +0.000095 vs Exp11-S1, within the first-wave tolerance but not strictly better.
- Ewarp is +0.037781 vs Exp11-S1, so temporal quality did not improve in this pilot.

Gate interpretation:
- P0 legacy control is valid and not collapsed.
- P4 is the current fixed-boundary candidate, but only KEEP as a dev pilot candidate, not a final success.
- P5 lowers Ewarp relative to P0/P4 but pays a larger LPIPS cost.
- No adaptive or region-balanced search has been started.
- No DAVIS50/YouTubeVOS100 final eval has been run.

PAI output root:
`/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp20_autoresearch_scale_adaptive_region_dpo/trials/`
