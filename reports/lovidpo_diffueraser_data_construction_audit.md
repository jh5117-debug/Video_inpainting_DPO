# LoVI-DPO DiffuEraser Data Construction Audit

Generated: 2026-07-02T06:51:14

## Paper-Safe Answer

DiffuEraser main data is a YouTube-VOS / D3 partial-mask, DiffuEraser-only self-rollout dataset with K=4 masks per source. It records raw and comp losers, uses comp loser as the primary DPO loser, and has 3,327 confirmed primary-comp training pairs.

| field | answer_for_paper | evidence_file | notes |
|---|---|---|---|
| Final train pair count | 3,327 primary-comp DPO pairs, confirmed from selected_primary_comp.gtwin.pai_paths.jsonl on PAI/NAS. | `/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/exp09_10_11_youtubevos_gtwin_d3comp_pai/manifests/selected_primary_comp.gtwin.pai_paths.jsonl` | Also mirrored under /mnt/workspace/hj/nas_hj; wc -l returned 3327. |
| Source dataset | YouTube-VOS / D3 partial-mask source pool with GT-aligned winner videos. | `selected_primary_comp.gtwin.pai_paths.jsonl; PRD/25_paper_materials_and_writing_plan.md` | Manifest rows have source_dataset=youtubevos and data_asset=D3_youtubevos_partialmask_k4_diffueraser_only. |
| Mask K and generation method | K=4 partial masks per source under videodpo_partialmask_policy_v1_medium_hard_k4; png convention is 255 inpaint region and 0 keep region. | `selected_primary_comp.gtwin.pai_paths.jsonl` | Mask mode partial; canonical clips are 16 frames at 512x320 in this manifest. |
| Loser generator model count | DiffuEraser-only self-rollout for the main DiffuEraser DPO data, not an all-model loser source unless another manifest proves otherwise. | `selected_primary_comp.gtwin.pai_paths.jsonl` | Rows show generation_model=diffueraser and generation_source=diffueraser_only. |
| Raw and comp loser | Both raw_loser_video_path and comp_loser_video_path are retained; final_loser_type=comp and final_loser_variant=comp_loser are used for primary DPO training. | `selected_primary_comp.gtwin.pai_paths.jsonl` | Raw loser remains useful for diagnostics and visual explanations. |
| Winner semantics | Winner is the GT-aligned clean target video from YouTube-VOS/D3 canonical frame indices. | `selected_primary_comp.gtwin.pai_paths.jsonl` | Manifest rows show win_source=youtubevos_gt_aligned_by_canonical_frame_indices. |
| Reference checkpoint | DiffuEraser SFT-48000 is the reference/baseline checkpoint for the reported Exp11 outer b0.75 S2 comparison. | `PRD/25_paper_materials_and_writing_plan.md; reports/exp11_outer_b075_s2_youtubevos100_davis50_eval.md` | PAI path recorded in older PRDs as converted_weights_step48000. |
| Region loss mask semantics | Exp11 uses local/region-aware DPO with winner anchor, clipped loser gap, mask/boundary emphasis, outer-boundary b=0.75, and small outside weight. | `PRD/25_paper_materials_and_writing_plan.md; reports/exp11_outer_b075_s2_youtubevos100_davis50_eval.md` | This is the paper-positive DiffuEraser recipe, separate from VideoPainter BR masks. |
| Evaluation splits | Reported DiffuEraser validation uses DAVIS50 and YouTubeVOS100 fixed evaluation splits, with DAVIS50 visual packs prepared for teacher review. | `reports/exp11_outer_b075_s2_youtubevos100_davis50_eval.md; reports/teacher_question1_visual_metric_relationship_20260617.md` | These are evaluation splits, not training source selection pools. |

## Detailed Manifest Facts

- Selected primary-comp rows: 3,327.
- Final selected manifest source_video_id unique count: 3,327.
- `num_masks_per_video` field: 4 for all 3,327 rows.
- `mask_id` distribution: mask_002 847, mask_001 837, mask_003 831, mask_000 812.
- `generation_model`: diffueraser for all rows.
- `generation_source`: diffueraser_only for all rows.
- `final_loser_type`: comp for all rows.
- `source_dataset`: youtubevos for all rows.
- `canonical_num_frames`: 16 for all rows.
- `defect_bucket`: texture_or_structure_shift 2242, balanced_hard 788, too_good 296, too_bad 1.

## Claim Boundary

Allowed: DiffuEraser-only self-rollout DPO data with raw+comp losers and confirmed 3,327 primary-comp pairs.

Not allowed: claiming the DiffuEraser main loser set is an all-model loser pool unless a separate manifest proves that claim.
