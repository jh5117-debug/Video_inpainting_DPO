# Exp47 Pseudo-Success Teacher Quality Audit

Status: `EXP47_TEACHER_GLOBAL_DRIFT_CONFIRMED`

Rows audited: `48` search/shadow rows (`24/24`). Metrics use start/quarter/mid/three-quarter/end frames with spatial stride 4 for forensic speed; this audit is read-only and frame-space only; it does not train, run DPO, run GT-only SFT, or perform an optimizer step.

## Label Counts

- `PSEUDO_TARGET_GLOBAL_DRIFT`: `26`
- `PSEUDO_TARGET_USABLE_LOCAL_ONLY`: `22`

## Mean Metrics

- pseudo target vs V_bg full PSNR: `32.370202`
- pseudo target vs V_bg mask PSNR: `28.258355`
- pseudo target vs V_bg boundary PSNR: `25.815867`
- pseudo target vs V_bg outside PSNR: `32.921862`
- pseudo target vs V_bg outside L1: `0.017477`
- absolute brightness delta: `0.014741`
- absolute contrast delta: `0.000966`
- color histogram distance: `0.078622`
- low-frequency L1 drift proxy: `0.014869`
- temporal flicker delta proxy: `0.006926`
- mask removal PSNR gain over condition: `18.313552`
- Step0 vs V_bg full PSNR mean: `34.678017`

LPIPS was not available in the H20 Python environment. Ewarp is represented here by a temporal flicker delta proxy; Exp46 official Ewarp deltas remain recorded in Exp46 reports.

## Interpretation

Rows labelled `PSEUDO_TARGET_GLOBAL_DRIFT` or `PSEUDO_TARGET_OUTSIDE_BAD` are unsafe for full-video global SFT because they can teach tone/outside drift even when local removal succeeds. Rows labelled `PSEUDO_TARGET_USABLE_LOCAL_ONLY` may still be useful for localized pseudo-success targets or same-source preference, but not as global SFT targets without localization.

Outputs:

- `reports/exp47_pseudosuccess_teacher_quality_audit.csv`
- `reports/exp47_pseudosuccess_teacher_visual_review.csv`
- `reports/exp47_pseudosuccess_teacher_quality_summary.json`
- `reports/exp47_teacher_review_pages/`

## Codex Visual Inspection

Codex inspected all four contact review pages: `search_page_00`, `search_page_01`, `shadow_page_00`, and `shadow_page_01`, covering all search24 and shadow24 pseudo-success teacher rows. The pages show that pseudo-success often removes the local object/effect, but the target frequently changes global tone, color, contrast, or style relative to `V_bg`. This supports using these rows only as local pseudo-success or same-source preference evidence unless a stricter global-clean relabel passes.
