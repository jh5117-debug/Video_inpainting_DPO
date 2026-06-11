# DPO Diagnostic Summary

flags: LOSER_DOMINANT
rows: 402

| metric | mean | p50 | p90 | max |
| --- | --- | --- | --- | --- |
| dpo_loss | 0.39716 | 0.341995 | 0.630648 | 0.963293 |
| implicit_acc | 0.979744 | 1 | 1 | 1 |
| win_gap | -0.011181 | -0.009926 | 0.006054 | 0.120385 |
| lose_gap | 0.60163 | 0.681381 | 1 | 1 |
| raw_win_gap | -0.000404 | -0.000132 | 0.000103 | 0.002303 |
| raw_lose_gap | 0.432811 | 0.253504 | 1.161061 | 2.004674 |
| norm_win_gap | -0.011181 | -0.009926 | 0.006054 | 0.120385 |
| norm_lose_gap | 0.697 | 0.712532 | 1.274978 | 2.108499 |
| norm_lose_gap_clipped | 0.60163 | 0.681381 | 1 | 1 |
| winner_abs_reg | 0.045513 | 0.026483 | 0.111366 | 0.316689 |
| winner_gap_reg | 0.004377 | 0.001376 | 0.010402 | 0.134045 |
| mse_w_over_ref_mse_w | 0.990262 | 0.990246 | 1.006415 | 1.344791 |
| mse_l_over_ref_mse_l | 2.310323 | 2.263231 | 3.567787 | 8.227013 |
| sigma_term | 0.683578 | 0.715301 | 0.784548 | 0.849295 |
| kl_divergence | 0.108102 | 0.063407 | 0.290068 | 0.501088 |
| loser_dominant_ratio | 0.98543 | 1 | 1 | 1 |
| grad_norm | 8.825244 | 3.207073 | 16.887006 | 249.463843 |
| prior_loss |  |  |  |  |
| boundary_loss |  |  |  |  |
| flow_loss |  |  |  |  |

## Sources

- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260609_025331_d3n16_val24_exp9_logratio_gap_dpo_s1_2000_davis_pai/dpo_diagnostics.csv`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260609_025331_d3n16_val24_exp9_logratio_gap_dpo_s2_2000_davis_pai/dpo_diagnostics.csv`
