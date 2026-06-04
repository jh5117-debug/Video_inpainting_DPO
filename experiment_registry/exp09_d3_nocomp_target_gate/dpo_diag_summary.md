# DPO Diagnostics Summary: exp09_nocomp

- short_name: `d3_nocomp_target_gate`
- diag_file_status: `FOUND`
- diag_status: `DPO_SATURATED;LOSER_DOMINANT;COLLAPSE_RISK`
- dpo_diag_csv: `reports/dpo_diag_snapshots/exp09_nocomp_stage1_dpo_diagnostics.csv`
- row_count: `151`
- first_step: `1`
- last_step: `1500`

## Key Risk Fields

- dpo_loss_median: `0.565768`
- dpo_loss_frac_lt_1e_3: `0`
- implicit_acc_mean: `0.898455`
- win_gap_p90: `0.00440348`
- win_gap_frac_gt_0_5: `0`
- mse_w_over_ref_mse_w_p90: `1.1794`
- mse_w_over_ref_mse_w_frac_gt_5: `0.0264901`
- mse_l_over_ref_mse_l_p90: `22.5152`
- sigma_term_frac_gt_0_99: `0`
- loser_dominant_ratio_mean: `0.97351`

If `diag_file_status` is `MISSING_DIAG` or `REMOTE_DIAG_PATH_FOUND`, this experiment still needs local diagnostic CSV backfill before numeric DPO claims are final.
