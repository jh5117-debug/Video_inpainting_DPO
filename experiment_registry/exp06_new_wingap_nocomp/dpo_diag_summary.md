# DPO Diagnostic Summary

flags: COLLAPSE_RISK;LOSER_DOMINANT;WIN_GAP_EXPLODED
rows: 802

| metric | mean | p50 | p90 | max |
| --- | --- | --- | --- | --- |
| dpo_loss | 0.314314 | 0.027647 | 0.690621 | 17.319733 |
| implicit_acc | 0.960308 | 1 | 1 | 1 |
| win_gap | 0.036413 | -0.000664 | 0.007379 | 4.660285 |
| lose_gap | 2.780187 | 3.123741 | 5.338371 | 6.196559 |
| raw_win_gap |  |  |  |  |
| raw_lose_gap |  |  |  |  |
| norm_win_gap |  |  |  |  |
| norm_lose_gap |  |  |  |  |
| norm_lose_gap_clipped |  |  |  |  |
| winner_abs_reg | 0.145762 | 0.070861 | 0.30848 | 4.670134 |
| winner_gap_reg | 0.037992 | 7.11e-05 | 0.008091 | 4.660285 |
| mse_w_over_ref_mse_w | 6.034929 | 0.988949 | 1.154925 | 1397 |
| mse_l_over_ref_mse_l | 103.48356 | 30.725414 | 274.505646 | 1688 |
| sigma_term | 0.834513 | 0.974745 | 0.998742 | 0.999572 |
| kl_divergence | 0.70415 | 0.787237 | 1.344716 | 2.36141 |
| loser_dominant_ratio | 0.899917 | 1 | 1 | 1 |
| grad_norm | 51.222993 | 2.844719 | 109.429809 | 2291 |
| prior_loss |  |  |  |  |
| boundary_loss |  |  |  |  |
| flow_loss |  |  |  |  |

## Sources

- `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp09_10_11_pai_sync/reports/dpo_diag_snapshots/exp06_new_stage1_dpo_diagnostics.csv`
- `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp09_10_11_pai_sync/reports/dpo_diag_snapshots/exp06_new_stage2_dpo_diagnostics.csv`
