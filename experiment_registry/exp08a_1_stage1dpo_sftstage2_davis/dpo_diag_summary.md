# DPO Diagnostic Summary

flags: COLLAPSE_RISK;LOSER_DOMINANT;WIN_GAP_EXPLODED
rows: 201

| metric | mean | p50 | p90 | max |
| --- | --- | --- | --- | --- |
| dpo_loss | 0.441029 | 0.546279 | 0.693204 | 1.158791 |
| implicit_acc | 0.906183 | 1 | 1 | 1 |
| win_gap | 0.01324 | 0.001608 | 0.028909 | 0.222248 |
| lose_gap | 0.894675 | 0.502587 | 2.408505 | 3.089099 |
| raw_win_gap |  |  |  |  |
| raw_lose_gap |  |  |  |  |
| norm_win_gap |  |  |  |  |
| norm_lose_gap |  |  |  |  |
| norm_lose_gap_clipped |  |  |  |  |
| winner_abs_reg | 0.066965 | 0.045101 | 0.173641 | 0.31722 |
| winner_gap_reg | 0.013303 | 0.001683 | 0.028909 | 0.222257 |
| mse_w_over_ref_mse_w | 2.274324 | 1.040103 | 2.41145 | 72.704865 |
| mse_l_over_ref_mse_l | 30.565873 | 16.106596 | 78.609283 | 300.585236 |
| sigma_term | 0.683815 | 0.61707 | 0.94945 | 0.978703 |
| kl_divergence | 0.226979 | 0.135253 | 0.624304 | 0.773959 |
| loser_dominant_ratio | 0.957652 | 1 | 1 | 1 |
| grad_norm | 18.441307 | 7.640489 | 44.781596 | 405.044144 |
| prior_loss |  |  |  |  |
| boundary_loss |  |  |  |  |
| flow_loss |  |  |  |  |

## Sources

- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260605_142442_exp08_d3_comp_fullloss_wingap_lose025_s1_2000_davis_pai/dpo_diagnostics.csv`
