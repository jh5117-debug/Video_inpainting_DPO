# DPO Diagnostic Summary

flags: COLLAPSE_RISK;LOSER_DOMINANT;WIN_GAP_EXPLODED
rows: 302

| metric | mean | p50 | p90 | max |
| --- | --- | --- | --- | --- |
| dpo_loss | 0.422761 | 0.485081 | 0.693424 | 1.301917 |
| implicit_acc | 0.879553 | 1 | 1 | 1 |
| win_gap | 0.010214 | 0.002821 | 0.021251 | 0.233819 |
| lose_gap | 0.876965 | 0.448459 | 2.375156 | 2.867071 |
| raw_win_gap |  |  |  |  |
| raw_lose_gap |  |  |  |  |
| norm_win_gap |  |  |  |  |
| norm_lose_gap |  |  |  |  |
| norm_lose_gap_clipped |  |  |  |  |
| winner_abs_reg | 0.046853 | 0.028449 | 0.109957 | 0.309829 |
| winner_gap_reg | 0.010435 | 0.002836 | 0.021265 | 0.233819 |
| mse_w_over_ref_mse_w | 1.6538 | 1.181805 | 2.460632 | 38.848175 |
| mse_l_over_ref_mse_l | 40.80874 | 15.954101 | 107.698891 | 457.9263 |
| sigma_term | 0.686272 | 0.622235 | 0.946639 | 0.971508 |
| kl_divergence | 0.221795 | 0.114193 | 0.595408 | 0.719499 |
| loser_dominant_ratio | 0.965744 | 1 | 1 | 1 |
| grad_norm | 9.445331 | 3.49059 | 18.361355 | 173.418631 |
| prior_loss |  |  |  |  |
| boundary_loss |  |  |  |  |
| flow_loss |  |  |  |  |

## Sources

- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260601_065618_exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500_stage1/dpo_diagnostics.csv`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260601_065618_exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500_stage2/dpo_diagnostics.csv`
