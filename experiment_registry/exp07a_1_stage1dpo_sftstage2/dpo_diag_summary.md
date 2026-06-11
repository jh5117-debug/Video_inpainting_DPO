# DPO Diagnostic Summary

flags: LOSER_DOMINANT
rows: 151

| metric | mean | p50 | p90 | max |
| --- | --- | --- | --- | --- |
| dpo_loss | 0.503999 | 0.667897 | 0.693125 | 1.301917 |
| implicit_acc | 0.898179 | 1 | 1 | 1 |
| win_gap | 0.006409 | 0.000323 | 0.015563 | 0.233819 |
| lose_gap | 0.598617 | 0.050862 | 2.035343 | 2.684299 |
| raw_win_gap |  |  |  |  |
| raw_lose_gap |  |  |  |  |
| norm_win_gap |  |  |  |  |
| norm_lose_gap |  |  |  |  |
| norm_lose_gap_clipped |  |  |  |  |
| winner_abs_reg | 0.053973 | 0.039775 | 0.125375 | 0.309829 |
| winner_gap_reg | 0.006851 | 0.000453 | 0.015676 | 0.233819 |
| mse_w_over_ref_mse_w | 1.203876 | 1.02165 | 1.543019 | 4.076163 |
| mse_l_over_ref_mse_l | 13.750707 | 3.588212 | 33.059013 | 156.025803 |
| sigma_term | 0.630634 | 0.513097 | 0.924159 | 0.965491 |
| kl_divergence | 0.151257 | 0.012032 | 0.511699 | 0.672273 |
| loser_dominant_ratio | 0.931488 | 1 | 1 | 1 |
| grad_norm | 13.270631 | 5.512669 | 29.588251 | 173.418631 |
| prior_loss |  |  |  |  |
| boundary_loss |  |  |  |  |
| flow_loss |  |  |  |  |

## Sources

- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260601_065618_exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500_stage1/dpo_diagnostics.csv`
