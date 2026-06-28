# Exp31 Qualitative Summary

Visual review status:
`VIDEOPAINTER_2000_VISUAL_REVIEW_COMPLETE_FORMAL_POSITIVE_BLOCKED`

Reviewed material:

- search-dev Step0/50/2000 all-32 evidence pages and crop pages.
- shadow-dev Step0/50/2000 all-32 evidence pages and crop pages.
- montage assets:
  `reports/exp31_vp_2000_visual_review_montages/`.
- per-sample visual review CSV:
  `reports/exp31_vp_2000_visual_review.csv`.

Observed pattern:

- Step0 is weak/noisy with poor local fill and boundary artifacts.
- Step50 improves the edited region but repeatedly introduces outside
  brightness/color pollution and purple/green local artifacts.
- Step2000 is visibly cleaner than Step50 and not collapsed.
- A minority of Step2000 rows still show residual local texture or mild
  darkening, so the result is not artifact-free.

Conclusion: Step2000 is better than Step0 and Step50 on the reviewed video
evidence, but formal positive is blocked by missing LPIPS/Ewarp metrics.
