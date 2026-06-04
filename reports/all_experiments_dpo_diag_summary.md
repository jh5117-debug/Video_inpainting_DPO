# All Experiments DPO Diagnostic Summary

This table is generated from `experiment_registry/experiment_matrix.csv`.
Experiments without a local diagnostic CSV are explicitly marked `MISSING_DIAG`.

|experiment_id|short_name|diag_file_status|row_count|first_step|last_step|dpo_loss_median|dpo_loss_frac_lt_1e_3|implicit_acc_mean|win_gap_p90|win_gap_frac_gt_0_5|lose_gap_p90|mse_w_over_ref_mse_w_p90|mse_w_over_ref_mse_w_frac_gt_5|mse_l_over_ref_mse_l_p90|sigma_term_frac_gt_0_99|loser_dominant_ratio_mean|diag_status|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|exp01|diffueraser_reproduction_sft|NOT_DPO|0||||||||||||||NOT_DPO|
|exp02|official_videodpo_vc2|MISSING_DIAG|0||||||||||||||MISSING_DIAG|
|exp03|official_videodpo_diffueraser|MISSING_DIAG|0||||||||||||||MISSING_DIAG|
|exp04|fullmask_loser_failed_gate|MISSING_DIAG|0||||||||||||||MISSING_DIAG|
|exp05_old|old_d2_comp_plain_failed|MISSING_DIAG|0||||||||||||||MISSING_DIAG|
|exp05_new|new_d2_comp_wingap_lose025|MISSING_DIAG|0||||||||||||||MISSING_DIAG|
|exp06_new|new_d2_nocomp_wingap_lose025|FOUND|401|1|4000|0.422428|0|0.957606|0.0281446|0.00498753|3.6606|1.35425|0.0299252|38.5354|0.0922693|0.8133|DPO_SATURATED;LOSER_DOMINANT;COLLAPSE_RISK|
|exp07_current|partialmask_task_current_failed|MISSING_DIAG|0||||||||||||||MISSING_DIAG|
|exp07_fix|fix_smallmask_prior|MISSING_DIAG|0||||||||||||||MISSING_DIAG|
|exp07_hybrid|dpoS1_sftS2_hybrid|MISSING_DIAG|0||||||||||||||MISSING_DIAG|
|exp08|regionloss_diagnostic|MISSING_DIAG|0||||||||||||||MISSING_DIAG|
|exp09_comp|d3_comp_target_gate|MISSING_DIAG|0||||||||||||||MISSING_DIAG|
|exp09_nocomp|d3_nocomp_target_gate|FOUND|151|1|1500|0.565768|0|0.898455|0.00440348|0|2.39721|1.1794|0.0264901|22.5152|0|0.97351|DPO_SATURATED;LOSER_DOMINANT;COLLAPSE_RISK|
|exp09_nolose|d3_nolose_gate|FOUND|101|1|1000|0.69124|0|0.991749|-4.90795e-05|0|-5.3132e-07|0.985498|0|0.999991|0|0|DPO_SATURATED;COLLAPSE_RISK|
