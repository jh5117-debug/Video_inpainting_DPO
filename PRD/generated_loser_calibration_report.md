# Generated Loser Calibration Report

Status: not run in this checked-in template.

This report must be overwritten on PAI after running the calibration subset.
Do not launch full offline generation before this report contains real
candidate metrics and selection distributions.

Required calibration command shape on PAI:

```bash
python tools/videodpo_generated_loser_calibration.py \
  --output_root data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4 \
  --models all \
  --limit 20 \
  --mask_policy_config configs/generation/videodpo_partialmask_policy_v1_medium_hard_k4.yaml \
  --selection_config configs/generation/medium_hard_balanced_selection_v1.yaml \
  --calibration_report PRD/generated_loser_calibration_report.md
```

Current 2026-05-25 production pass uses DiffuEraser-only generation for
throughput:

```bash
python tools/videodpo_generated_loser_calibration.py \
  --output_root data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4 \
  --models diffueraser \
  --limit 20 \
  --mask_policy_config configs/generation/videodpo_partialmask_policy_v1_medium_hard_k4.yaml \
  --selection_config configs/generation/medium_hard_balanced_selection_v1.yaml \
  --calibration_report PRD/generated_loser_calibration_report.md
```

## Pending Fields

- calibration_winners: `PENDING`
- masks_per_winner: `4`
- models_per_mask_default: `diffueraser, propainter, cococo, minimax_remover`
- models_per_mask_current: `diffueraser`
- candidate_count: `PENDING`
- successful_candidate_count: `PENDING`
- failed_candidate_count: `PENDING`
- fail_count_by_model: `PENDING`
- selected_model_distribution: `PENDING`
- quality_score_histogram: `PENDING`
- too_bad_ratio: `PENDING`
- eligible_ratio: `PENDING`
- too_good_ratio: `PENDING`
- selected_primary_distribution: `PENDING`
- selected_secondary_distribution: `PENDING`
- mask_area_distribution: `PENDING`
- mask_motion_distribution: `PENDING`
- outside_diff_check: `PENDING`
- sample_visual_preview_paths: `PENDING`

## Decision

Full generation approval: `PENDING_CALIBRATION`
