# Exp30 Continuation V2 Readback

Status: `EXP30_CONTINUATION_V2_READBACK_COMPLETED`

Date: 2026-06-27

## Git

- Branch: `research/exp30-vor-or-multimodel-minimax-adapter-20260627`
- HEAD at readback: `f0f98d4066b45d0d62f47b652203c7d160211347`
- Worktree: `/home/hj/H20_Video_inpainting_DPO_exp30_vor_or_minimax`
- Worktree status before this report: clean

## Files Read

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/50_exp30_vor_or_multimodel_minimax.md`
- `experiment_registry/exp30_vor_or_multimodel_minimax/status.md`
- `experiment_registry/exp30_vor_or_multimodel_minimax/metric_summary.md`
- `experiment_registry/exp30_vor_or_multimodel_minimax/qualitative_summary.md`
- `experiment_registry/exp30_vor_or_multimodel_minimax/results.tsv`
- `reports/exp30_readback.md`
- `reports/exp30_three_backbone_paper_positioning.md`
- `reports/exp30_vor_or_source_pool_audit.md`
- `reports/exp30_vor_or_source_pool_audit.csv`
- `reports/exp30_vor_or_source_pool_summary.json`
- `reports/exp30_vor_or_source_pool_visual_review.csv`
- `reports/exp29_minimax_expanded_data_quality_v2.md`
- `reports/exp29_minimax_expanded_data_quality_summary_v2.json`
- `reports/exp29_effecterase_official81_inference_smoke.md`

## Left CLI Protection

- PAI hostname: `dsw-753014-85f54df947-bkp7h`
- Compute GPU processes at readback: none
- Left CLI monitor observed read-only: PID `258013`
- Left CLI runtime locks observed under
  `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/cli4`
- Reserved GPUs remain GPU1/GPU2/GPU3/GPU4 due to left-side locks.
- Signals sent: no
- Left-side files modified: no
- GPU tasks launched: no

## Why The Previous Source Pool Had Only 80 Rows

The previous Exp30 source-pool audit used only existing exact extraction caches:

- `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/extracted/vor_gate128_exact_20260623`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/extracted/vor_train_audit64_exact_20260623`

Those caches contain only 192 exact triplets. After excluding previous
MiniMax/EffectErase diagnostic scenes, only 80 usable scene groups remained.
This is a cache-subset limitation, not evidence that VOR-Train itself only has
80 usable VOR-OR triplets.

## Full VOR Index Preliminary Location

The full metadata index is present and readable on PAI:

- Path:
  `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp25_vor/exp25_vor_or_preference_data/manifests/vor_train_metadata_index.jsonl`
- Rows: 57,751
- SHA256:
  `33d57a3ea23c5799b583d476a311089f95cbce1b0d11280822a63b8c9edcddc4`
- Schema includes:
  `sample_id`, `scene_group`, `condition_member_path`, `winner_member_path`,
  `mask_member_path`, `condition_role`, `winner_role`, `mask_role`, `task`,
  `comp_mode`, `hard_comp`
- Pairing rule:
  `VOR-Train/FG_BG/<sample_id>.mp4`,
  `VOR-Train/BG/<sample_id>.mp4`, and `MASK/<sample_id>.mp4`

The mask/member index is also present:

- Path:
  `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp25_vor/reports/vor_train_mask_member_index.csv`
- Rows: 179,189
- SHA256:
  `3b3415c989e72b4df821f85903d01a754fb2c07412e40907749bf9844626d1f8`

## Readback Answers

1. Previous source pool only had 80 rows because it was built from two exact
   extraction caches totaling 192 triplets, then exclusions reduced the usable
   scene groups.
2. This was not a VOR data limit; the full metadata index exposes 57,751
   paired VOR-Train rows.
3. The full VOR metadata index path is listed above and will be promoted to a
   source-of-truth reference by the next milestone if identity checks pass.
4. The 57,751 metadata rows are readable from PAI.
5. The 192 exact triplets came from `vor_gate128_exact_20260623` and
   `vor_train_audit64_exact_20260623` extraction caches.
6. Exclusions included previous MiniMax/EffectErase scenes and removed 112 of
   the 192 cached triplets.
7. The next source-pool v2 must sample from the full metadata index, not from
   extracted-cache membership only.
8. The full-index recovery gate, source-pool v2 gate, smoke16 gate, and Gate64
   pool gate cannot be skipped.
9. Current results still do not support universal adapter, all-model support,
   MiniMax third-backbone quality-positive evidence, EffectErase adapter-ready
   language, final SOTA, or top-conference novelty confirmed.

## Next Milestone

Recover and verify the full VOR valid triplet index, then write
`exp30_vor_or_multimodel_minimax/manifests/vor_or_full_valid_triplet_index_ref.json`
only if the index identity and pairing checks pass.
