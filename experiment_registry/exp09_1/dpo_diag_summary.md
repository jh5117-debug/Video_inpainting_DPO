# DPO Diagnostic Summary

flags: LOSER_DOMINANT
rows: 201

| metric | mean | p50 | p90 | max |
| --- | --- | --- | --- | --- |
| dpo_loss | 0.407822 | 0.376534 | 0.623161 | 0.963293 |
| implicit_acc | 0.988628 | 1 | 1 | 1 |
| win_gap | -0.018877 | -0.016028 | -0.002874 | 0.120385 |
| lose_gap | 0.544481 | 0.590892 | 1 | 1 |
| raw_win_gap | -0.00073 | -0.000469 | -1.564e-05 | 0.002303 |
| raw_lose_gap | 0.354853 | 0.189971 | 1.041332 | 1.734396 |
| norm_win_gap | -0.018877 | -0.016028 | -0.002874 | 0.120385 |
| norm_lose_gap | 0.618533 | 0.590918 | 1.253799 | 1.816505 |
| norm_lose_gap_clipped | 0.544481 | 0.590892 | 1 | 1 |
| winner_abs_reg | 0.053875 | 0.035972 | 0.134901 | 0.316689 |
| winner_gap_reg | 0.003908 | 0.000223 | 0.005978 | 0.134045 |
| mse_w_over_ref_mse_w | 0.983935 | 0.984186 | 0.997061 | 1.344791 |
| mse_l_over_ref_mse_l | 2.10416 | 1.896678 | 3.501266 | 6.147457 |
| sigma_term | 0.676772 | 0.688147 | 0.792471 | 0.849295 |
| kl_divergence | 0.088531 | 0.047548 | 0.259572 | 0.433246 |
| loser_dominant_ratio | 0.975124 | 1 | 1 | 1 |
| grad_norm | 12.737051 | 3.456027 | 27.198867 | 249.463843 |
| prior_loss |  |  |  |  |
| boundary_loss |  |  |  |  |
| flow_loss |  |  |  |  |

## Sources

- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260609_025331_d3n16_val24_exp9_logratio_gap_dpo_s1_2000_davis_pai/dpo_diagnostics.csv`
