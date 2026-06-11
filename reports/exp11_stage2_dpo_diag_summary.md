# Exp11 Stage2 DPO Diagnostic Summary

diag_csv: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260609_2331_exp11_n16_gpus4_7_scratch_exp11_flow_prior_consistency_dpo_s2_2000_davis_pai/dpo_diagnostics.csv`
rows: 201

classification: `LOSER_DOMINANT`

| field | present | last | mean_last20 | min | max |
|---|---:|---:|---:|---:|---:|
| dpo_loss | True | 0.58194 | 0.353873 | 0.230433 | 0.718753 |
| implicit_acc | True | 1 | 1 | 0.5 | 1 |
| raw_win_gap | True | 7.46512e-05 | 0.000214788 | -0.0034957 | 0.0183955 |
| raw_lose_gap | True | 0.00809556 | 0.463136 | 8.11274e-05 | 2.0509 |
| norm_win_gap | True | 0.00841569 | -0.00460133 | -0.0398447 | 0.0541024 |
| norm_lose_gap | True | 0.250008 | 1.08225 | 0.0468652 | 2.37171 |
| norm_lose_gap_clipped | True | 0.250008 | 0.735848 | 0.0468652 | 1 |
| winner_abs_reg | True | 0.00892319 | 0.0923244 | 0.00131796 | 0.660134 |
| winner_gap_reg | True | 0.0125137 | 0.00405367 | 0 | 0.0589912 |
| prior_loss | True | 0.000378961 | 0.00215155 | 5.99011e-05 | 0.0285898 |
| boundary_loss | True | 0.000325829 | 0.00189999 | 5.67521e-05 | 0.0122461 |
| flow_loss | True | 0.000467358 | 0.00354244 | 4.11012e-05 | 0.0565324 |
| mse_w_over_ref_mse_w | True | 1.00844 | 0.995325 | 0.960883 | 1.05524 |
| mse_l_over_ref_mse_l | True | 1.64774 | 3.66076 | 1.05017 | 10.4357 |
| sigma_term | True | 0.567199 | 0.711743 | 0.491166 | 0.79453 |
| kl_divergence | True | 0.00204255 | 0.115838 | 1.24088e-05 | 0.512188 |
| loser_dominant_ratio | True | 1 | 1 | 0.75 | 1 |
| grad_norm | True | 6.91057 | 3.92018 | 0.690442 | 18.1866 |

Interpretation: this Exp11 run is `Exp11-proxy`, not real optical-flow / ProPainter-prior consistency.
Proxy losses, if present, must be reported as frozen-ref prior, boundary, and adjacent-frame residual proxy terms.
