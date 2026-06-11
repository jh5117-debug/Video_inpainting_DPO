# DPO Diagnostic Summary

flags: COLLAPSE_RISK;LOSER_DOMINANT
rows: 201

| metric | mean | p50 | p90 | max |
| --- | --- | --- | --- | --- |
| dpo_loss | 0.481473 | 0.481977 | 0.654175 | 3.679688 |
| implicit_acc | 0.972637 | 1 | 1 | 1 |
| win_gap | 0.007266 | -0.009219 | 0.014063 | 0.926732 |
| lose_gap | 0.544546 | 0.456565 | 1 | 1 |
| raw_win_gap | 0.000192 | -0.000666 | 0.000356 | 0.179252 |
| raw_lose_gap | 0.384245 | 0.111847 | 1.219874 | 2.130708 |
| norm_win_gap | 0.007266 | -0.009219 | 0.014063 | 0.926732 |
| norm_lose_gap | 0.74242 | 0.471969 | 1.645991 | 2.03959 |
| norm_lose_gap_clipped | 0.544546 | 0.456565 | 1 | 1 |
| winner_abs_reg | 0.146166 | 0.097476 | 0.365289 | 0.742996 |
| winner_gap_reg | 0.021923 | 0.001068 | 0.025214 | 0.926732 |
| mse_w_over_ref_mse_w | 1.020948 | 0.99063 | 1.015345 | 2.852679 |
| mse_l_over_ref_mse_l | 2.641681 | 1.770575 | 5.17882 | 7.456759 |
| sigma_term | 0.648005 | 0.625731 | 0.793304 | 0.812856 |
| kl_divergence | 0.096109 | 0.028603 | 0.303918 | 0.531449 |
| loser_dominant_ratio | 0.997512 | 1 | 1 | 1 |
| grad_norm | 52.380193 | 6.541419 | 53.075961 | 1472 |
| prior_loss | 0.005192 | 0.003106 | 0.007853 | 0.211904 |
| boundary_loss | 0.004798 | 0.003121 | 0.011154 | 0.030064 |
| flow_loss | 0.006263 | 0.004091 | 0.010572 | 0.130897 |

## Sources

- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260609_2331_exp11_n16_gpus4_7_scratch_exp11_flow_prior_consistency_dpo_s1_2000_davis_pai/dpo_diagnostics.csv`
