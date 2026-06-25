# Exp26 Gate64 Source Repair and Primary-32 Draft
## Status
- Duplicate-source deep audit: all 8 rejected rows are static-pixel duplicates with 49 unique decoded indices/timestamps, not true short sources.
- Repair materialization: 8/8 OK, 49 frames each.
- Repair official VideoPainter generation: 8/8 OK, 49 output frames each.
- Dense evidence/crop review: 8/8 repaired rows reviewed locally; no technical invalid, no global collapse.
- Strict mp4 playback review: pending. This report does not claim `VIDEO_REVIEW_PASS` or `DATA_READY`.
- Combined Gate64 formal-valid sources: 64/64.
- Combined eligible pool: 55 rows.
- DPO training: NOT STARTED.

## Combined Visual Classification
- hard-plausible: 18
- medium-hard: 37
- too-close: 1
- trivial-bad: 8

## Selection Decisions
- ELIGIBLE_AFTER_VISUAL_REVIEW: 55
- REJECT_TOO_CLOSE: 1
- REJECT_TRIVIAL_OR_TECHNICAL: 8

## Primary-32 Draft Manifest
- Candidate pool: `/home/hj/H20_Video_inpainting_DPO_exp26_videopainter/exp26_videopainter_dpo_v2/manifests/vp2_gate64_candidate_pool55_visual_reviewed_comp.jsonl`
- Candidate pool SHA256: `4bec06d78249b809ddf90a80eca900f8467ba3a2798c8ae9d03552d0f25ee5b8`
- Primary-32 draft: `/home/hj/H20_Video_inpainting_DPO_exp26_videopainter/exp26_videopainter_dpo_v2/manifests/vp2_gate64_primary32_visual_reviewed_comp.jsonl`
- Primary-32 SHA256: `21c0c617698bbf69859f3850979d777c3af3e1908926bca43b69ede447df3cc8`
- Main loser semantics: `final_loser_video_path = comp_loser_video_path`, matching current BR `selected_primary_comp` training source; raw loser is retained for diagnostics/ablation.

## Primary-32 Balance
- classification: `{'hard-plausible': 16, 'medium-hard': 16}`
- area_bucket: `{'small': 8, 'large': 8, 'medium': 16}`
- motion_bucket: `{'low': 8, 'high': 8, 'medium': 16}`
- mask_profile: `{'irregular_freeform': 8, 'object_like_polygon': 8, 'soft_blob': 5, 'ellipse_circle_subset': 3, 'thin_structure_freeform': 5, 'edge_touch_freeform': 3}`

## Remaining Blocker
- PAI user `hj` still cannot write `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp26_videopainter_dpo_v2`; DPO micro-training remains blocked until the root ACL fix in `reports/runtime/pai_postmaintenance_root_permission_fix.sh` is applied.
- DiffuEraser converted weights remain unreadable to `hj`, so Exp25/Exp27 true DiffuEraser tracks remain blocked.
