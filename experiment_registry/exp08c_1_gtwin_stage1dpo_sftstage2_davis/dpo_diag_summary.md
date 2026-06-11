# DPO Diagnostic Summary

flags: COLLAPSE_RISK;LOSER_DOMINANT;WIN_GAP_EXPLODED
rows: 219

| metric | mean | p50 | p90 | max |
| --- | --- | --- | --- | --- |
| dpo_loss | 0.347569 | 0.228153 | 0.692382 | 4.206883 |
| implicit_acc | 0.941129 | 1 | 1 | 1 |
| win_gap | 0.014812 | 0.000617 | 0.005379 | 1.206849 |
| lose_gap | 1.323839 | 1.206956 | 2.978889 | 3.860331 |
| raw_win_gap |  |  |  |  |
| raw_lose_gap |  |  |  |  |
| norm_win_gap |  |  |  |  |
| norm_lose_gap |  |  |  |  |
| norm_lose_gap_clipped |  |  |  |  |
| winner_abs_reg | 0.07013 | 0.039284 | 0.155185 | 1.207721 |
| winner_gap_reg | 0.014918 | 0.000683 | 0.005379 | 1.206849 |
| mse_w_over_ref_mse_w | 10.570405 | 1.025425 | 1.464079 | 1385 |
| mse_l_over_ref_mse_l | 15.430525 | 4.55691 | 29.528479 | 297.101501 |
| sigma_term | 0.754868 | 0.80719 | 0.976459 | 0.988891 |
| kl_divergence | 0.334663 | 0.301868 | 0.744643 | 0.981912 |
| loser_dominant_ratio | 0.999429 | 1 | 1 | 1 |
| grad_norm | 58.789331 | 7.896222 | 214.900149 | 598.075086 |
| prior_loss |  |  |  |  |
| boundary_loss |  |  |  |  |
| flow_loss |  |  |  |  |

## Sources

- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260606_144527_exp08c_youtubevos_gtwin_d3comp_fullloss_wingap_lose025_s1_2000_davis_pai/dpo_diagnostics.csv`
