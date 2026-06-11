# DPO Diagnostic Summary

flags: COLLAPSE_RISK;LOSER_DOMINANT
rows: 402

| metric | mean | p50 | p90 | max |
| --- | --- | --- | --- | --- |
| dpo_loss | 0.421345 | 0.321039 | 0.633955 | 3.679688 |
| implicit_acc | 0.982587 | 1 | 1 | 1 |
| win_gap | 0.001933 | -0.00578 | 0.011027 | 0.926732 |
| lose_gap | 0.633771 | 0.857873 | 1 | 1 |
| raw_win_gap | 7.75e-06 | -0.000305 | 0.000262 | 0.179252 |
| raw_lose_gap | 0.467886 | 0.191233 | 1.381408 | 2.130708 |
| norm_win_gap | 0.001933 | -0.00578 | 0.011027 | 0.926732 |
| norm_lose_gap | 0.921948 | 1.008039 | 1.745806 | 2.37171 |
| norm_lose_gap_clipped | 0.633771 | 0.857873 | 1 | 1 |
| winner_abs_reg | 0.118429 | 0.071122 | 0.299345 | 0.742996 |
| winner_gap_reg | 0.013182 | 0.001718 | 0.016214 | 0.926732 |
| mse_w_over_ref_mse_w | 1.008837 | 0.994419 | 1.012117 | 2.852679 |
| mse_l_over_ref_mse_l | 3.204454 | 2.939457 | 5.7226 | 10.435728 |
| sigma_term | 0.677466 | 0.737518 | 0.787977 | 0.812856 |
| kl_divergence | 0.116973 | 0.047622 | 0.345185 | 0.531449 |
| loser_dominant_ratio | 0.998134 | 1 | 1 | 1 |
| grad_norm | 28.288247 | 3.674168 | 21.980754 | 1472 |
| prior_loss | 0.003495 | 0.001909 | 0.006034 | 0.211904 |
| boundary_loss | 0.003352 | 0.00196 | 0.008355 | 0.030064 |
| flow_loss | 0.004521 | 0.00281 | 0.008804 | 0.130897 |

## Sources

- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260609_2331_exp11_n16_gpus4_7_scratch_exp11_flow_prior_consistency_dpo_s1_2000_davis_pai/dpo_diagnostics.csv`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260609_2331_exp11_n16_gpus4_7_scratch_exp11_flow_prior_consistency_dpo_s2_2000_davis_pai/dpo_diagnostics.csv`
