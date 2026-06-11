# Exp11 Stage1 DPO Diagnostic Summary

diag_csv: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260609_2331_exp11_n16_gpus4_7_scratch_exp11_flow_prior_consistency_dpo_s1_2000_davis_pai/dpo_diagnostics.csv`
rows: 201

classification: `LOSER_DOMINANT`

| field | present | last | mean_last20 | min | max |
|---|---:|---:|---:|---:|---:|
| dpo_loss | True | 0.483551 | 0.313629 | 0.20761 | 3.67969 |
| implicit_acc | True | 1 | 0.9875 | 0.25 | 1 |
| raw_win_gap | True | -0.000622909 | -0.00127012 | -0.0149577 | 0.179252 |
| raw_lose_gap | True | 0.0107282 | 0.551003 | 2.20034e-05 | 2.13071 |
| norm_win_gap | True | -0.0499748 | -0.0186187 | -0.0637402 | 0.926732 |
| norm_lose_gap | True | 0.243032 | 1.20903 | 4.29292e-05 | 2.03959 |
| norm_lose_gap_clipped | True | 0.221833 | 0.808554 | 4.29292e-05 | 1 |
| winner_abs_reg | True | 0.0121646 | 0.129514 | 0.00138197 | 0.742996 |
| winner_gap_reg | True | 0 | 0.00666574 | 0 | 0.926732 |
| prior_loss | True | 0.00060351 | 0.00310608 | 1.47458e-05 | 0.211904 |
| boundary_loss | True | 0.000444438 | 0.00298227 | 1.85197e-05 | 0.0300642 |
| flow_loss | True | 0.000836993 | 0.0048248 | 2.94659e-05 | 0.130897 |
| mse_w_over_ref_mse_w | True | 0.951288 | 0.983437 | 0.939611 | 2.85268 |
| mse_l_over_ref_mse_l | True | 1.67952 | 3.98099 | 1.00006 | 7.45676 |
| sigma_term | True | 0.628822 | 0.744267 | 0.0328101 | 0.812856 |
| kl_divergence | True | 0.00252632 | 0.137433 | 1.23051e-07 | 0.531449 |
| loser_dominant_ratio | True | 1 | 1 | 0.5 | 1 |
| grad_norm | True | 27.0997 | 63.5615 | 0.425082 | 1471.88 |

Interpretation: this Exp11 run is `Exp11-proxy`, not real optical-flow / ProPainter-prior consistency.
Proxy losses, if present, must be reported as frozen-ref prior, boundary, and adjacent-frame residual proxy terms.
