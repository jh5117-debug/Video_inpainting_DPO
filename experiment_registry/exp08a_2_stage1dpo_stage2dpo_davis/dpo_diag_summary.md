# DPO Diagnostic Summary

flags: COLLAPSE_RISK;LOSER_DOMINANT;WIN_GAP_EXPLODED
rows: 402

| metric | mean | p50 | p90 | max |
| --- | --- | --- | --- | --- |
| dpo_loss | 0.384083 | 0.359237 | 0.693595 | 1.158791 |
| implicit_acc | 0.869447 | 1 | 1 | 1 |
| win_gap | 0.010664 | 0.003567 | 0.019219 | 0.222248 |
| lose_gap | 1.064249 | 0.885461 | 2.494966 | 3.089099 |
| raw_win_gap |  |  |  |  |
| raw_lose_gap |  |  |  |  |
| norm_win_gap |  |  |  |  |
| norm_lose_gap |  |  |  |  |
| norm_lose_gap_clipped |  |  |  |  |
| winner_abs_reg | 0.057541 | 0.036261 | 0.143367 | 0.31722 |
| winner_gap_reg | 0.010695 | 0.003567 | 0.019219 | 0.222257 |
| mse_w_over_ref_mse_w | 1.817705 | 1.180213 | 1.712155 | 72.704865 |
| mse_l_over_ref_mse_l | 41.876034 | 23.03742 | 108.842194 | 567.334473 |
| sigma_term | 0.721313 | 0.744192 | 0.954778 | 0.978703 |
| kl_divergence | 0.268728 | 0.22314 | 0.627844 | 0.773959 |
| loser_dominant_ratio | 0.946488 | 1 | 1 | 1 |
| grad_norm | 11.475752 | 3.933429 | 29.10891 | 405.044144 |
| prior_loss |  |  |  |  |
| boundary_loss |  |  |  |  |
| flow_loss |  |  |  |  |

## Sources

- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260605_142442_exp08_d3_comp_fullloss_wingap_lose025_s1_2000_davis_pai/dpo_diagnostics.csv`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260606_070556_exp08_d3_comp_fullloss_wingap_lose025_s2_2000_davis_pai/dpo_diagnostics.csv`
