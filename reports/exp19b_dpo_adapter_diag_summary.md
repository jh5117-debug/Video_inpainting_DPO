# Exp19b DPO / Adapter Diagnostic Summary

- diag_csv: `exp19_boundary_gated_flow_adapter_dpo/dpo_diag/exp19b_stage2_500_dpo_diagnostics.csv`
- rows: `51`
- first_step: `1`
- last_step: `500`

## Key Diagnostics

| field | mean | min | max | last |
|---|---:|---:|---:|---:|
| total_loss | 0.698331 | 0.690689 | 0.711688 | 0.69541 |
| dpo_loss | 0.693076 | 0.690085 | 0.695524 | 0.694272 |
| m_w | 0.0962873 | 0.00113602 | 0.349743 | 0.00709884 |
| m_l | 0.108912 | 0.000877686 | 0.561446 | 0.00605107 |
| m_w_ref | 0.0962887 | 0.00113621 | 0.349661 | 0.00709574 |
| m_l_ref | 0.108916 | 0.00087832 | 0.561604 | 0.00605313 |
| norm_win_gap | -6.83313e-06 | -0.00100749 | 0.00109512 | 0.000363523 |
| norm_lose_gap | 9.67318e-05 | -0.000709998 | 0.00146279 | -0.000331697 |
| winner_abs_reg | 0.0962873 | 0.00113602 | 0.349743 | 0.00709884 |
| winner_gap_reg | 0.000440752 | 0 | 0.00142026 | 0.00078383 |
| loser_dominant_ratio | 0.176471 | 0 | 1 | 0 |
| adapter_grad_norm | 0.000468085 | 5.81127e-05 | 0.00190417 | 0.000792602 |
| base_grad_norm | 0 | 0 | 0 | 0 |
| adapter_residual_norm | 0.0644607 | 0 | 0.21669 | 0.0979491 |
| gate_mean | 0.00666145 | 0.00111179 | 0.0183542 | 0.00910551 |
| nonzero_gate_ratio | 0.0229342 | 0.00446429 | 0.0591518 | 0.0234375 |
| flow_conf_mean | 0.469913 | 0.0849687 | 0.681804 | 0.457159 |
| valid_flow_ratio | 0.602173 | 0.183823 | 0.737442 | 0.650823 |
| mean_flow_magnitude | 8.37127 | 0.078441 | 46.5943 | 11.8908 |

## Labels

- BASE_FROZEN_OK
- ADAPTER_GRAD_NONZERO
- ADAPTER_RESIDUAL_STABLE

## Notes

Exp19b 500-step adapter-only training completed. This validates training mechanics and adapter stability; DAVIS10 metric/visual evaluation is blocked until an Exp19 inference wrapper safely injects flow context during pipeline denoising.
