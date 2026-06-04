# DPO Diagnostics Summary: exp06_new

- short_name: `new_d2_nocomp_wingap_lose025`
- diag_file_status: `FOUND`
- diag_status: `DPO_SATURATED;LOSER_DOMINANT;COLLAPSE_RISK`
- dpo_diag_csv: `reports/dpo_diag_snapshots/exp06_new_stage1_dpo_diagnostics.csv;reports/dpo_diag_snapshots/exp06_new_stage2_dpo_diagnostics.csv`
- row_count: `802`
- first_step: `1`
- last_step: `4000`

## Key Risk Fields

- dpo_loss_median: `0.0276825`
- dpo_loss_frac_lt_1e_3: `0.0723192`
- implicit_acc_mean: `0.960308`
- win_gap_p90: `0.00737511`
- win_gap_frac_gt_0_5: `0.0162095`
- mse_w_over_ref_mse_w_p90: `1.15361`
- mse_w_over_ref_mse_w_frac_gt_5: `0.0311721`
- mse_l_over_ref_mse_l_p90: `274.393`
- sigma_term_frac_gt_0_99: `0.410224`
- loser_dominant_ratio_mean: `0.899917`

If `diag_file_status` is `MISSING_DIAG` or `REMOTE_DIAG_PATH_FOUND`, this experiment still needs local diagnostic CSV backfill before numeric DPO claims are final.
