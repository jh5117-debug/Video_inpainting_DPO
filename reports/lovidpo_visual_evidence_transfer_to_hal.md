# LoVI-DPO Visual Evidence Transfer To HAL

Generated: 2026-07-02T06:51:14

## Target

`/home/hj/LoVI_DPO_visual_evidence_for_advisor_20260702`

## Summary

- File count through symlinks: 767
- Symlink count: 18
- MP4 count: 122
- Apparent target size: 40M
- Transfer mode: symlink local HAL assets where possible; rsync selected VideoPainter mp4s from PAI/NAS.
- Training run: no
- Inference run: no
- Evaluation regenerated: no
- Original outputs deleted/overwritten: no

## Included DiffuEraser Assets

- `diffueraser/selected_8/exp11_outer_b075_s2_selected_visuals`
- `diffueraser/final_20/final_20_visual_cases_for_paper`
- `diffueraser/davis50_side_by_side/exp15_or_benchmark_davis50_visuals_fixed`
- `diffueraser/reports/teacher_question1_visual_metric_relationship_20260617.md`

## Included VideoPainter Assets

- 12 selected `side_by_side.mp4` files copied from PAI/NAS:
  - search-dev: two cases across Step0 / Step50 / Step2000
  - shadow-dev: two cases across Step0 / Step50 / Step2000
- Local report symlinks for Exp26 Gate64 / Step50 and Exp31 Step2000.

## Missing Or Intentionally Skipped

- Full raw VideoPainter roots were not copied because they are large.
- Full raw frame folders were not copied.
- No missing critical paper-facing reports were identified for this lightweight pack.

## Index Files

- `/home/hj/LoVI_DPO_visual_evidence_for_advisor_20260702/README_OPEN_FIRST.md`
- `/home/hj/LoVI_DPO_visual_evidence_for_advisor_20260702/index.md`
- `/home/hj/LoVI_DPO_visual_evidence_for_advisor_20260702/index.csv`
