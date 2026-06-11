# DPO Diagnostic Summary

flags: COLLAPSE_RISK;LOSER_DOMINANT;WIN_GAP_EXPLODED
rows: 450

| metric | mean | p50 | p90 | max |
| --- | --- | --- | --- | --- |
| dpo_loss | 0.262079 | 0.092837 | 0.691041 | 4.206883 |
| implicit_acc | 0.94754 | 1 | 1 | 1 |
| win_gap | 0.013662 | 0.001452 | 0.006243 | 1.206849 |
| lose_gap | 1.993824 | 2.180007 | 3.573192 | 3.860331 |
| raw_win_gap |  |  |  |  |
| raw_lose_gap |  |  |  |  |
| norm_win_gap |  |  |  |  |
| norm_lose_gap |  |  |  |  |
| norm_lose_gap_clipped |  |  |  |  |
| winner_abs_reg | 0.059916 | 0.03535 | 0.139054 | 1.207721 |
| winner_gap_reg | 0.013714 | 0.001452 | 0.006243 | 1.206849 |
| mse_w_over_ref_mse_w | 6.620361 | 1.063169 | 1.854456 | 1385 |
| mse_l_over_ref_mse_l | 22.617287 | 8.36065 | 61.903423 | 297.101501 |
| sigma_term | 0.826348 | 0.934268 | 0.988479 | 0.991184 |
| kl_divergence | 0.501872 | 0.553784 | 0.894242 | 1.116555 |
| loser_dominant_ratio | 0.999722 | 1 | 1 | 1 |
| grad_norm | 28.720304 | 0.283812 | 68.630971 | 598.075086 |
| prior_loss |  |  |  |  |
| boundary_loss |  |  |  |  |
| flow_loss |  |  |  |  |

## Sources

- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260606_144527_exp08c_youtubevos_gtwin_d3comp_fullloss_wingap_lose025_s1_2000_davis_pai/dpo_diagnostics.csv`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260606_144527_exp08c_youtubevos_gtwin_d3comp_fullloss_wingap_lose025_s2_2000_davis_pai/dpo_diagnostics.csv`
