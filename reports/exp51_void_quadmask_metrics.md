# Exp51 VOID Quadmask-Aware Metrics

Status: `VOID_QUADMASK_METRICS_READY`
Frame sampling: all 24 heldout frames per video for the isolated quadmask audit.

## Required Answers

1. Did 10-step actually improve affected region? Mixed: affected_only delta PSNR = -0.113883, overlap delta PSNR = -0.329240, affected_union delta PSNR = -0.266781.
2. Did it hurt object core? Yes: object_core delta PSNR = -0.816208; object_value0 delta PSNR = 0.813795.
3. Did it hurt boundary? Yes/mixed: object_boundary delta PSNR = 0.045300; affected_boundary delta PSNR = -0.190144.
4. Did outside improvement mask local damage in full PSNR? Yes. outside_background_255 delta PSNR = -0.011285, while object_core/object_value0 are negative and affected_union is not robustly positive.
5. Future loss priority: protect object_core/object_value0 and object_boundary first; use affected_union as local preference region but clip loser gradients and preserve outside/background.

## Mean Delta PSNR by Region

| run | region | delta_psnr | delta_l1 | step0_stepN_l1 | area |
|---|---|---:|---:|---:|---:|
| one_step | affected_boundary | -0.097311 | 0.089401 | 3.209782 | 0.276201 |
| one_step | affected_only_value127 | -0.120765 | 0.097225 | 3.625597 | 0.084732 |
| one_step | affected_union_63_127 | -0.103770 | 0.048899 | 3.408067 | 0.112348 |
| one_step | far_outside | 0.063817 | -0.053168 | 1.309577 | 0.667742 |
| one_step | full_frame | 0.020476 | -0.028684 | 2.310698 | 1.000000 |
| one_step | local_union_0_63_127 | -0.049313 | 0.028222 | 3.438904 | 0.115574 |
| one_step | object_boundary | 0.198857 | -0.283544 | 4.524670 | 0.015292 |
| one_step | object_core | -0.218031 | -0.034937 | 3.713866 | 0.000403 |
| one_step | object_value0 | 0.782992 | -0.672343 | 4.372029 | 0.003226 |
| one_step | outside_background_255 | 0.046695 | -0.034887 | 2.180207 | 0.884426 |
| one_step | outside_near_boundary | -0.102109 | 0.064855 | 2.911260 | 0.216685 |
| one_step | overlap_value63 | -0.146733 | 0.171929 | 4.648338 | 0.027616 |
| ten_step | affected_boundary | -0.190144 | 0.155064 | 3.280926 | 0.276201 |
| ten_step | affected_only_value127 | -0.113883 | 0.124189 | 3.718585 | 0.084732 |
| ten_step | affected_union_63_127 | -0.266781 | 0.160121 | 3.521574 | 0.112348 |
| ten_step | far_outside | 0.151556 | -0.064995 | 1.389533 | 0.667742 |
| ten_step | full_frame | -0.057789 | 0.007406 | 2.395351 | 1.000000 |
| ten_step | local_union_0_63_127 | -0.203887 | 0.141581 | 3.554912 | 0.115574 |
| ten_step | object_boundary | 0.045300 | -0.178015 | 4.580240 | 0.015292 |
| ten_step | object_core | -0.816208 | 0.457846 | 3.836609 | 0.000403 |
| ten_step | object_value0 | 0.813795 | -0.682981 | 4.398295 | 0.003226 |
| ten_step | outside_background_255 | -0.011285 | -0.004670 | 2.261547 | 0.884426 |
| ten_step | outside_near_boundary | -0.146235 | 0.096640 | 2.971231 | 0.216685 |
| ten_step | overlap_value63 | -0.329240 | 0.270912 | 4.739433 | 0.027616 |

## Notes

LPIPS/Ewarp/TC are not computed here; this isolated audit intentionally avoids `inference/metrics.py` and focuses on quadmask semantics, PSNR/L1/SSIM/flicker/tone/pixel-diff.
