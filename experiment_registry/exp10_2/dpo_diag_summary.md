# DPO Diagnostic Summary

flags: COLLAPSE_RISK;LOSER_DOMINANT
rows: 402

| metric | mean | p50 | p90 | max |
| --- | --- | --- | --- | --- |
| dpo_loss | 0.42307 | 0.367565 | 0.643548 | 3.442898 |
| implicit_acc | 0.981965 | 1 | 1 | 1 |
| win_gap | 0.006685 | -0.00386 | 0.026959 | 0.839057 |
| lose_gap | 0.655429 | 0.926975 | 1 | 1 |
| raw_win_gap | 0.001997 | -0.00014 | 0.001621 | 0.331279 |
| raw_lose_gap | 0.487326 | 0.206193 | 1.390401 | 2.128723 |
| norm_win_gap | 0.006685 | -0.00386 | 0.026959 | 0.839057 |
| norm_lose_gap | 0.970025 | 1.132162 | 1.797368 | 2.460637 |
| norm_lose_gap_clipped | 0.655429 | 0.926975 | 1 | 1 |
| winner_abs_reg | 0.120419 | 0.072253 | 0.299275 | 0.980958 |
| winner_gap_reg | 0.017345 | 0.002715 | 0.029918 | 0.839391 |
| mse_w_over_ref_mse_w | 1.014728 | 0.996237 | 1.030044 | 3.05367 |
| mse_l_over_ref_mse_l | 3.36575 | 3.241859 | 5.972056 | 11.690132 |
| sigma_term | 0.678128 | 0.722665 | 0.787765 | 0.814276 |
| kl_divergence | 0.122331 | 0.051373 | 0.347594 | 0.531676 |
| loser_dominant_ratio | 0.995647 | 1 | 1 | 1 |
| grad_norm | 23.781061 | 4.277386 | 22.777389 | 1460 |
| prior_loss |  |  |  |  |
| boundary_loss |  |  |  |  |
| flow_loss |  |  |  |  |

## Sources

- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260609_1608_exp10_n16_gpus4_7_scratch_exp10_region_local_dpo_s1_2000_davis_pai/dpo_diagnostics.csv`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260609_1608_exp10_n16_gpus4_7_scratch_exp10_region_local_dpo_s2_2000_davis_pai/dpo_diagnostics.csv`
