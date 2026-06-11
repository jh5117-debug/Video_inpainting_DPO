# DPO Diagnostic Summary

flags: COLLAPSE_RISK;LOSER_DOMINANT
rows: 201

| metric | mean | p50 | p90 | max |
| --- | --- | --- | --- | --- |
| dpo_loss | 0.455818 | 0.471578 | 0.646788 | 3.442898 |
| implicit_acc | 0.976368 | 1 | 1 | 1 |
| win_gap | -0.000553 | -0.009112 | 0.015889 | 0.839057 |
| lose_gap | 0.544708 | 0.468214 | 1 | 1 |
| raw_win_gap | -0.000787 | -0.000726 | 0.000534 | 0.058387 |
| raw_lose_gap | 0.387997 | 0.105564 | 1.218417 | 2.128723 |
| norm_win_gap | -0.000553 | -0.009112 | 0.015889 | 0.839057 |
| norm_lose_gap | 0.739509 | 0.470834 | 1.650418 | 2.029765 |
| norm_lose_gap_clipped | 0.544708 | 0.468214 | 1 | 1 |
| winner_abs_reg | 0.145188 | 0.090933 | 0.36544 | 0.742955 |
| winner_gap_reg | 0.014154 | 0.001247 | 0.022911 | 0.839391 |
| mse_w_over_ref_mse_w | 1.008382 | 0.990758 | 1.015832 | 3.05367 |
| mse_l_over_ref_mse_l | 2.628646 | 1.804211 | 5.238521 | 7.388337 |
| sigma_term | 0.656076 | 0.628542 | 0.793335 | 0.814276 |
| kl_divergence | 0.096803 | 0.026617 | 0.304915 | 0.530931 |
| loser_dominant_ratio | 0.992537 | 1 | 1 | 1 |
| grad_norm | 42.792656 | 6.756753 | 67.257269 | 1460 |
| prior_loss |  |  |  |  |
| boundary_loss |  |  |  |  |
| flow_loss |  |  |  |  |

## Sources

- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260609_1608_exp10_n16_gpus4_7_scratch_exp10_region_local_dpo_s1_2000_davis_pai/dpo_diagnostics.csv`
