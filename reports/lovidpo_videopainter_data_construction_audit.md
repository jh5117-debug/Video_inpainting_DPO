# LoVI-DPO VideoPainter Data Construction Audit

Generated: 2026-07-02T06:51:14

## Paper-Safe Answer

VideoPainter is a controlled low-data cross-pipeline validation using a balanced primary32 set selected from a Gate64 self-loser pool. It is not a large-scale VideoPainter training dataset.

## Key Counts

- VOR-BG train source pool: 128 clean V_bg videos.
- Search-dev: 32, disjoint from train source and shadow-dev.
- Shadow-dev: 32, disjoint from train source and search-dev.
- Gate64: 64 official VideoPainter baseline/self-loser outputs from the train-source pool.
- Formal valid: 64/64; technical invalid: 0.
- Eligible: 55 = 37 medium-hard + 18 hard-plausible.
- Rejected: 9 = 8 trivial-bad + 1 too-close.
- Primary32: 32 = 16 medium-hard + 16 hard-plausible; reserve: 23.

| field | answer_for_paper | evidence_file | notes |
|---|---|---|---|
| VOR-BG train source pool | 128 VOR-Train-BG clean background source videos, each with a distinct scene_group/source_sample_id; all are 49-frame sources. | `exp26_videopainter_dpo_v2/manifests/vp2_vor_bg_train_source_128.jsonl; reports/exp26_gate64_readback.md` | This is the upstream source pool, not the final DPO train pair count. |
| VOR-BG search-dev and shadow-dev | Search-dev has 32 VOR-Train-BG videos and shadow-dev has 32 VOR-Train-BG videos; both are disjoint from train source and from each other by scene_group/source_id. | `exp26_videopainter_dpo_v2/manifests/vp2_vor_bg_search_dev_32.jsonl; exp26_videopainter_dpo_v2/manifests/vp2_vor_bg_shadow_dev_32.jsonl` | They are evaluation/dev splits, not used to pick primary32. |
| Primary32 selection source | Primary32 was not directly hand-picked from 128 raw videos. The path was 128 train source -> balanced Gate64 source subset -> official VideoPainter outputs -> visual/technical review -> 55 eligible -> balanced primary32 plus 23 reserve. | `exp26_videopainter_dpo_v2/manifests/vp2_gate64_source_manifest.jsonl; reports/exp26_gate64_primary32_final.md` | Gate64 and primary32 are fully inside VOR-Train-BG; no VOR-Eval use. |
| Gate64 official outputs | Gate64 means 64 official VideoPainter baseline/self-loser generations from the locked train-source subset. Each row has raw official output plus a BR-consistent comp loser candidate. | `reports/exp26_gate64_generation_status.md if present; reports/exp26_gate64_final_temporal_review.md; reports/exp26_gate64_manifest_identity.json` | Formal valid count was 64/64, with 49 real frames per output recorded in the Gate64 review. |
| Formal valid / invalid | Gate64 formal-valid count was 64/64 and technical-invalid count was 0 in the final temporal review. | `reports/exp26_gate64_final_temporal_review.md; reports/exp26_gate64_primary32_final.md` | Rejected rows were quality/eligibility decisions, not generation failures. |
| Eligible 55 | 55 of the 64 formal-valid outputs were eligible as medium-hard or hard-plausible same-pipeline loser candidates: 37 medium-hard and 18 hard-plausible. | `exp26_videopainter_dpo_v2/manifests/vp2_gate64_candidate_pool55_visual_reviewed_comp.jsonl` | The remaining 9 were rejected: 8 trivial-bad and 1 too-close. |
| Why only 32 selected | The final primary32 was a balanced low-data validation set: 16 medium-hard and 16 hard-plausible, balanced across REAL/BLENDER, mask profile, area bucket, and motion bucket. The other 23 eligible rows were kept as reserve. | `reports/exp26_gate64_primary32_final.md; exp26_videopainter_dpo_v2/manifests/vp2_gate64_primary32_final.jsonl; exp26_videopainter_dpo_v2/manifests/vp2_gate64_reserve_final.jsonl` | Selection happened before VideoPainter DPO training/step results; search-dev/shadow-dev were not used. |
| Cherry-picking risk | The correct claim is controlled low-data cross-pipeline validation, not large-scale VideoPainter generalization. Cherry-picking risk is mitigated by pre-training selection, explicit Gate64/eligible/reserve/rejected manifests, disjoint dev splits, and balance constraints, but the paper should not overclaim primary32. | `reports/exp26_gate64_primary32_final.md; reports/exp31_vp_2000_final_decision.md` | Recommended wording: VideoPainter is a controlled low-data cross-pipeline validation using a balanced primary32 set selected from a Gate64 self-loser pool. It is not a large-scale VideoPainter training dataset. |
| VOR meaning | VOR is the paired Video Object Removal dataset. V_obj/FG_BG is the object-present video, V_bg/BG is the clean removed/background video, and M/MASK is the object mask. | `reports/exp26_br_mask_source_audit.md; exp26_videopainter_dpo_v2/manifests/vp2_vor_bg_train_source_128.jsonl` | VideoPainter BR uses V_bg as clean source; object-removal backbones such as MiniMax/VOID/EffectErase use V_obj, V_bg, and M. |
| VideoPainter VOR usage | For this VideoPainter background-removal/inpainting pipeline, only V_bg is used as clean winner/source. V_obj and VOR object mask M are not training inputs for the BR pipeline; a generated moving BR mask defines the hole/condition. | `reports/exp26_br_mask_source_audit.md; exp26_videopainter_dpo_v2/configs/vp2_mixed_br_mask_v1.json` | Condition is winner outside the generated BR mask, not V_obj. |
| VideoPainter mask generation | VideoPainter uses vp2_mixed_br_mask_v1: 49-frame mixed moving BR masks with irregular/free-form, object-like polygon, soft blob, ellipse/circle subset, edge-touch, and thin-structure profiles across small/medium/large area and low/medium/high motion buckets. | `exp26_videopainter_dpo_v2/configs/vp2_mixed_br_mask_v1.json; reports/exp26_br_mask_source_audit.md` | Not ellipse-only. first_frame_gt semantics are used where the VideoPainter protocol expects first-frame guidance. |
| VideoPainter loser generation | The loser generator is the official VideoPainter baseline only. The raw loser is the official output; the comp loser is winner outside the generated mask plus raw output inside the mask. The comp loser is primary for DPO, while raw is retained for diagnostics. | `exp26_videopainter_dpo_v2/manifests/vp2_gate64_primary32_final.jsonl; reports/exp26_gate64_primary32_final.md` | All primary32 rows use final_loser_type=comp. |

## Detailed Audit Facts

- Gate64 source REAL/BLENDER balance: 56 REAL / 8 BLENDER.
- Eligible55 REAL/BLENDER balance: 51 REAL / 4 BLENDER.
- Primary32 REAL/BLENDER balance: 28 REAL / 4 BLENDER.
- Gate64 mask profile balance: irregular_freeform 16, object_like_polygon 16, soft_blob 8, edge_touch_freeform 8, ellipse_circle_subset 8, thin_structure_freeform 8.
- Primary32 mask profile balance: irregular_freeform 8, object_like_polygon 8, soft_blob 5, thin_structure_freeform 5, ellipse_circle_subset 3, edge_touch_freeform 3.
- Primary32 area buckets: medium 16, small 8, large 8.
- Primary32 motion buckets: medium 16, low 8, high 8.
- Primary32 frame count: all 32 rows record 49 frames.
- Eligible55 frame count: all 55 rows record 49 frames.
- Raw loser path fields exist for all eligible55 and all primary32 rows.
- Comp loser path fields exist for all eligible55 and all primary32 rows.
- Primary32 final_loser_video_path points to comp loser for all 32 rows.
- Search-dev and shadow-dev have zero scene/source overlap with train source, Gate64, and primary32.
- Selection happened before DPO training and before Step50/Step2000 outcomes; Step results were not used for primary32 selection.

## Claim Boundary

Allowed: controlled VideoPainter low-data validation, balanced primary32 selected before training from a Gate64 self-loser pool, disjoint search/shadow dev splits.

Not allowed: large-scale VideoPainter dataset, universal generalization, final SOTA, or primary32 as proof of broad VideoPainter coverage.
