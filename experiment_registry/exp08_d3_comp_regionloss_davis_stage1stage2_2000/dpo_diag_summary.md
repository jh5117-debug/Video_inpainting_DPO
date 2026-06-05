# DPO Diagnostics

Status: pending.

Required files after launch:

- Stage1: `experiments/dpo/stage1/<exp08_stage1_run>/dpo_diagnostics.csv`
- Stage2: `experiments/dpo/stage2/<exp08_stage2_run>/dpo_diagnostics.csv`

Required region-loss diagnostic fields:

- `loss_region_mode`
- `region_mask_weight`
- `region_boundary_weight`
- `region_outside_weight`
- `mask_area_ratio`
- `boundary_area_ratio`
- `outside_area_ratio`
- `region_weighted_mse_w`
- `region_weighted_mse_l`
- `region_weighted_ref_mse_w`
- `region_weighted_ref_mse_l`

Summaries will be written to:

- `reports/exp08_stage1_dpo_diag_summary.md`
- `reports/exp08_stage2_dpo_diag_summary.md`
