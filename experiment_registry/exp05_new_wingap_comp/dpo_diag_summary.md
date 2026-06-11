# DPO Diagnostic Summary

flags: COLLAPSE_RISK;LOSER_DOMINANT;WIN_GAP_EXPLODED
rows: 802

| metric | mean | p50 | p90 | max |
| --- | --- | --- | --- | --- |
| dpo_loss | 0.521324 | 0.663029 | 0.693026 | 5.736185 |
| implicit_acc | 0.922382 | 1 | 1 | 1 |
| win_gap | 0.010292 | -0.000127 | 0.006985 | 1.608719 |
| lose_gap | 0.734733 | 0.070938 | 2.730976 | 3.699698 |
| raw_win_gap |  |  |  |  |
| raw_lose_gap |  |  |  |  |
| norm_win_gap |  |  |  |  |
| norm_lose_gap |  |  |  |  |
| norm_lose_gap_clipped |  |  |  |  |
| winner_abs_reg | 0.122202 | 0.066639 | 0.302618 | 1.648957 |
| winner_gap_reg | 0.012655 | 0.000119 | 0.007946 | 1.608719 |
| mse_w_over_ref_mse_w | 1.227844 | 0.993688 | 1.068199 | 40.979805 |
| mse_l_over_ref_mse_l | 10.809275 | 2.500751 | 29.90877 | 204.150742 |
| sigma_term | 0.636738 | 0.516793 | 0.967509 | 0.99066 |
| kl_divergence | 0.186256 | 0.017753 | 0.693942 | 0.923994 |
| loser_dominant_ratio | 0.688797 | 0.857143 | 1 | 1 |
| grad_norm | 7.605838 | 2.977777 | 15.135008 | 211.704636 |
| prior_loss |  |  |  |  |
| boundary_loss |  |  |  |  |
| flow_loss |  |  |  |  |

## Sources

- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260531_134817_exp5_d2_comp_k4_wingap_lose025_beta10_s1s2_4000_stage1/dpo_diagnostics.csv`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260531_134817_exp5_d2_comp_k4_wingap_lose025_beta10_s1s2_4000_stage2/dpo_diagnostics.csv`
