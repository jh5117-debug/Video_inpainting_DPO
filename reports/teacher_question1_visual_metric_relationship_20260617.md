# Teacher Question 1: Visual Results vs Metrics

Date: 2026-06-17

Question:

> 看可视化结果和指标之间的关系：是否比 DiffuEraser / 其他方法好；指标提升 0.3 是否正常；好在哪里；是否需要新指标；是否对于 OR 实验更合适。

## Evidence Paths On HAL

Main BR visual evidence:

- `/home/hj/dpo-2-1-exp/ours_exp11_outer_b075_s2_vs_diffueraser_visuals`
- `/home/hj/dpo-2-1-exp/this_week_exp11_exp12/visual_evidence/exp11_outer_b075_s2_selected_visuals`
- `/home/hj/dpo-2-1-exp/final_20_visual_cases_for_paper`
- `/home/hj/dpo-2-1-exp/this_week_exp11_exp12/visual_evidence/exp11_outer_b075_s2_davis50_light`

OR fixed visual evidence:

- `/home/hj/dpo-2-1-exp/exp15_or_benchmark_davis50_visuals_fixed`

## Main BR Result

Under the fixed BR protocol:

- raw6
- hard comp
- D+G off
- no PCM
- no mask dilation
- no Gaussian blur
- frame-wise in-memory metric

Exp11 outer b0.75 S2 consistently improves over SFT-48000 DiffuEraser.

| Dataset | Method | PSNR | SSIM | LPIPS | VFID | TC | Mask PSNR | Mask SSIM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DAVIS50 | SFT-48000 | 32.7314 | 0.9705 | 0.0167 | 0.2018 | 0.9712 | 23.8849 | 0.7976 |
| DAVIS50 | Exp11 outer b0.75 S2 | 33.0140 | 0.9723 | 0.0154 | 0.1754 | 0.9711 | 24.1675 | 0.8099 |
| YouTubeVOS100 | SFT-48000 | 33.3968 | 0.9701 | 0.0176 | 0.2007 | 0.9819 | 24.4262 | 0.7935 |
| YouTubeVOS100 | Exp11 outer b0.75 S2 | 33.7238 | 0.9711 | 0.0168 | 0.1925 | 0.9821 | 24.7532 | 0.7990 |

Deltas:

- DAVIS50: `+0.2826 PSNR`, `+0.0018 SSIM`, `-0.0013 LPIPS`, `-0.0264 VFID`, `+0.2826 mask PSNR`.
- YouTubeVOS100: `+0.3270 PSNR`, `+0.0009 SSIM`, `-0.0008 LPIPS`, `-0.0083 VFID`, `+0.3270 mask PSNR`.

This means the roughly `+0.3 PSNR` improvement is normal and meaningful for this setup, because it appears on two datasets and also appears in the mask-region metric, not only in whole-frame PSNR.

## Per-Video Relationship

DAVIS50 per-video deltas:

- PSNR improved on `34 / 50` videos.
- SSIM improved on `36 / 50` videos.
- LPIPS improved on `40 / 50` videos.
- TC is mixed: `25 / 50` improved and `25 / 50` worsened.

This shows that Exp11 outer b0.75 S2 mainly improves spatial fidelity and perceptual quality, while temporal consistency is not clearly improved by the current setting.

## Where It Looks Better

The strongest visual improvements are not global color/style changes. They are local:

1. Mask interior becomes cleaner.
   - `boat`: SFT-48000 creates obvious white fog / pasted blur around the boat wake; Exp11 keeps water texture and hull/wake structure cleaner.
   - `rhino`: SFT smears the head/body region; Exp11 preserves a more plausible foreground/background transition.

2. Boundary is more consistent with surrounding context.
   - This matches the method choice: `boundary_mode = outer`, `boundary_weight = 0.75`.
   - The benefit is visible when the mask touches object/background boundaries.

3. Large mask artifacts are reduced.
   - The old purple/white fog and patch-like artifacts are reduced in good cases.
   - The improvement is not always a perfect reconstruction; it is usually a cleaner and less disruptive inpainting result.

4. Thin structure and fast motion remain hard.
   - `dog-agility` has strong PSNR gain, but thin poles and moving dog parts still make SSIM/visual interpretation mixed.
   - `dance-jump` and `soccerball` should be kept as caution/failure cases, not positive examples.

## Do Metrics Match Visuals?

Mostly yes, but not perfectly.

They match well for:

- `boat`
- `rhino`
- `lucia`
- `blackswan`
- many YouTubeVOS100 positive cases

They are less aligned for cases with:

- thin structures
- fast motion
- strong blur
- object boundaries crossing narrow masks

Therefore the main conclusion should not be based only on whole-frame PSNR. It should be reported as:

> Exp11 outer b0.75 S2 improves the fixed BR metric table and the improvement is visually consistent mainly in mask interiors and outer boundaries. However, some motion/thin-structure cases still need qualitative inspection and boundary-aware metrics.

## Do We Need New Metrics?

Yes, but not to replace the current table. We should add diagnostic metrics.

Current main metrics are still useful:

- PSNR / SSIM for overall frame-wise quality.
- LPIPS / VFID for perceptual and distribution-level signal.
- TC for temporal signal.
- Mask PSNR / mask SSIM for region-local quality.

Needed additions:

1. Boundary-ring metric.
   - Because the method improvement is most visible at mask boundaries.
   - Report `boundary_pixel_psnr` or `boundary_psnr`.

2. Strict mask-pixel metric.
   - Avoid only using whole-frame hard-comp PSNR.
   - Report `strict_mask_pixel_psnr`.

3. Artifact / fog diagnostic.
   - Current PSNR does not directly measure purple fog, white fog, grid, or paste-like artifacts.
   - A simple artifact score could be a color-shift / local variance / edge-continuity diagnostic inside mask and boundary ring.

4. Case stratification.
   - Split by mask size, boundary length, motion, and thin-structure difficulty.
   - This would explain why some videos improve strongly and some remain mixed.

## Is This More Suitable For OR?

No, current evidence says it is not more suitable for OR.

For OR DAVIS50 fixed no-comp background-region protocol:

| Method | PSNR_bg | SSIM_bg | TC_bg_pixel_proxy |
|---|---:|---:|---:|
| ProPainter | 35.5274 | 0.9927 | 35.7664 |
| DiffuEraser SFT-48000 | 28.6773 | 0.9686 | 28.8505 |
| Ours Exp11 outer b0.75 S2 | 28.6795 | 0.9685 | 28.8682 |

OR conclusion:

- ProPainter is much better for object-removal background preservation.
- Ours is almost tied with DiffuEraser SFT-48000.
- The Exp11 gain was designed for BR / video inpainting with GT reconstruction style masks, not true object removal.
- OR does not currently support the main story; it should be treated as a failed/negative branch or future work, not the core claim.

## Answer To The Teacher Question

1. Is Ours better than DiffuEraser?
   - For BR: yes. Exp11 outer b0.75 S2 beats SFT-48000 DiffuEraser on DAVIS50 and YouTubeVOS100 under the fixed protocol.
   - For OR: no meaningful advantage. Ours is almost tied with DiffuEraser and far behind ProPainter.

2. Is `+0.3 PSNR` normal?
   - Yes. In this strong-SFT-baseline setting, `+0.28` on DAVIS50 and `+0.33` on YouTubeVOS100 is a meaningful and realistic improvement, especially because mask PSNR improves by a similar amount.

3. Where does it improve?
   - Mask interior cleanliness.
   - Outer-boundary continuity.
   - Less fog / patch-like artifact.
   - Better local structure in cases like `boat` and `rhino`.

4. Do we need new metrics?
   - Yes, as auxiliary diagnostics: strict mask-pixel PSNR, boundary-ring PSNR/SSIM, and artifact diagnostics.
   - Current PSNR/SSIM/LPIPS/VFID/TC table is still necessary, but not sufficient for explaining all visual differences.

5. Should we continue OR now?
   - No. OR/adapters have been tested enough for this week and did not produce a stronger direction. The main story should stay on BR with Exp11 outer b0.75 S2.

## Recommended Slide Sentence

Exp11 outer b0.75 S2 gives a consistent `+0.28 / +0.33 PSNR` improvement over the SFT-48000 DiffuEraser baseline on DAVIS50 / YouTubeVOS100, and the visual gains mainly appear as cleaner mask interiors and more natural outer-boundary transitions. The improvement is real but local, so the final paper should report whole-frame metrics together with mask-region, boundary-region, and qualitative evidence. OR does not currently support the method as a main result.
