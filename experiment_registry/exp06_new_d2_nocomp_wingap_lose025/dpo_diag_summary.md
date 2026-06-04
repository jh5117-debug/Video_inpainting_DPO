# DPO Diagnostics Summary: exp06_new

- short_name: `new_d2_nocomp_wingap_lose025`
- diag_file_status: `FOUND`
- diag_status: `DPO_SATURATED;LOSER_DOMINANT;COLLAPSE_RISK`
- dpo_diag_csv: `reports/dpo_diag_snapshots/exp06_new_stage1_dpo_diagnostics.csv`
- row_count: `401`
- first_step: `1`
- last_step: `4000`

## Key Risk Fields

- dpo_loss_median: `0.422428`
- dpo_loss_frac_lt_1e_3: `0`
- implicit_acc_mean: `0.957606`
- win_gap_p90: `0.0281446`
- win_gap_frac_gt_0_5: `0.00498753`
- mse_w_over_ref_mse_w_p90: `1.35425`
- mse_w_over_ref_mse_w_frac_gt_5: `0.0299252`
- mse_l_over_ref_mse_l_p90: `38.5354`
- sigma_term_frac_gt_0_99: `0.0922693`
- loser_dominant_ratio_mean: `0.8133`

If `diag_file_status` is `MISSING_DIAG`, this experiment is incomplete as DPO evidence even if videos or metrics exist.
