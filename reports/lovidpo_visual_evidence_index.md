# LoVI-DPO Visual Evidence Index

Generated: 2026-07-02T06:51:14

## Quick Open List

1. Open `README_OPEN_FIRST.md`.
2. DiffuEraser: start with `diffueraser/selected_8/exp11_outer_b075_s2_selected_visuals`.
3. VideoPainter: open `videopainter/step2000_shadowdev/*step2000_side_by_side.mp4`, then compare the matching Step0/Step50 files.
4. Use `index.csv` for exact file paths and source provenance.

## Recommended Files

| model | case_id | split | file | reason |
|---|---|---|---|---|
| DiffuEraser | selected_8 | DAVIS50/YTVOS visual evidence | `/home/hj/LoVI_DPO_visual_evidence_for_advisor_20260702/diffueraser/selected_8/exp11_outer_b075_s2_selected_visuals` | Best quick starting point; includes strong and cautionary cases |
| DiffuEraser | final_20 | paper visual cases | `/home/hj/LoVI_DPO_visual_evidence_for_advisor_20260702/diffueraser/final_20/final_20_visual_cases_for_paper` | Broader paper case bank |
| DiffuEraser | davis50_side_by_side | DAVIS50 | `/home/hj/LoVI_DPO_visual_evidence_for_advisor_20260702/diffueraser/davis50_side_by_side/exp15_or_benchmark_davis50_visuals_fixed` | Good for eval-set discussion |
| DiffuEraser | teacher_question1_report | paper explanation | `/home/hj/LoVI_DPO_visual_evidence_for_advisor_20260702/diffueraser/reports/teacher_question1_visual_metric_relationship_20260617.md` | Open when explaining why metrics and visuals both matter |
| VideoPainter | vp2_vor_bg_49f_REAL_ENV242_00112_002_03_step2000 | search-dev | `/home/hj/LoVI_DPO_visual_evidence_for_advisor_20260702/videopainter/step50_searchdev/vp2_vor_bg_49f_REAL_ENV242_00112_002_03_step2000_side_by_side.mp4` | Search-dev representative side-by-side |
| VideoPainter | vp2_vor_bg_49f_REAL_ENV242_00112_002_03_step50 | search-dev | `/home/hj/LoVI_DPO_visual_evidence_for_advisor_20260702/videopainter/step50_searchdev/vp2_vor_bg_49f_REAL_ENV242_00112_002_03_step50_side_by_side.mp4` | Search-dev representative side-by-side |
| VideoPainter | vp2_vor_bg_49f_REAL_ENV800_00003_001_02_step2000 | search-dev | `/home/hj/LoVI_DPO_visual_evidence_for_advisor_20260702/videopainter/step50_searchdev/vp2_vor_bg_49f_REAL_ENV800_00003_001_02_step2000_side_by_side.mp4` | Search-dev representative side-by-side |
| VideoPainter | vp2_vor_bg_49f_REAL_ENV800_00003_001_02_step50 | search-dev | `/home/hj/LoVI_DPO_visual_evidence_for_advisor_20260702/videopainter/step50_searchdev/vp2_vor_bg_49f_REAL_ENV800_00003_001_02_step50_side_by_side.mp4` | Search-dev representative side-by-side |
| VideoPainter | vp2_vor_bg_49f_BLENDER_FOREST015_00002_step2000 | shadow-dev | `/home/hj/LoVI_DPO_visual_evidence_for_advisor_20260702/videopainter/step2000_shadowdev/vp2_vor_bg_49f_BLENDER_FOREST015_00002_step2000_side_by_side.mp4` | Shadow-dev representative side-by-side |
| VideoPainter | vp2_vor_bg_49f_REAL_ENV109_00002_005_02_step2000 | shadow-dev | `/home/hj/LoVI_DPO_visual_evidence_for_advisor_20260702/videopainter/step2000_shadowdev/vp2_vor_bg_49f_REAL_ENV109_00002_005_02_step2000_side_by_side.mp4` | Shadow-dev representative side-by-side |

## Reports

- `diffueraser/reports/teacher_question1_visual_metric_relationship_20260617.md`
- `videopainter/reports/exp31_vp_2000_paper_evidence.md`
- `videopainter/reports/exp31_vp_2000_final_decision.md`
- `videopainter/reports/exp26_gate64_primary32_final.md`

## Missing Or Intentionally Not Copied

Raw full VideoPainter generation roots were not copied because they are large. The HAL folder contains representative mp4s and report symlinks only. No original outputs were deleted or overwritten.

## Rsync Source Pattern

Representative VideoPainter mp4s were copied from PAI/NAS under `exp31_videopainter_2000step_longrun/exp31_vp2000_base_identity_replay_20260628_091019/{search,shadow}/{step0,step50,step2000}/official_generation/side_by_side/`.
