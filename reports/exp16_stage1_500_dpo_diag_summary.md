# Exp16 Stage1-500 DPO Diagnostic Summary

Date: 2026-06-17

Source:

```text
exp16_prior_confidence_gated_dpo/dpo_diag/stage1_500_dpo_diagnostics.csv
```

Rows: `51`

## Loss Summary

| Field | First | Last | Mean | Min | Max |
|---|---:|---:|---:|---:|---:|
| loss | 0.795262 | 0.612881 | 0.665445 | 0.487197 | 0.795262 |
| L_base | 0.694449 | 0.537362 | 0.587136 | 0.410535 | 0.694449 |
| dpo_loss | 0.693051 | 0.528544 | 0.578113 | 0.407069 | 0.693051 |
| L_prior | 0.400244 | 0.376475 | 0.413692 | 0.313802 | 0.537265 |
| L_gen | 0.714177 | 0.376985 | 0.372089 | 0.093265 | 0.728087 |
| L_boundary_extra | 0.250795 | 0.190216 | 0.183348 | 0.072910 | 0.291013 |

Loss weights:

| Field | Value |
|---|---:|
| lambda_prior | 0.1 |
| lambda_gen | 0.05 |
| lambda_boundary_extra | 0.1 |

## Gap / DPO Signal

| Field | First | Last | Mean | Min | Max |
|---|---:|---:|---:|---:|---:|
| raw_win_gap | 0.000001 | -0.003254 | -0.002011 | -0.012359 | 0.000706 |
| raw_lose_gap | 0.000011 | 0.078920 | 0.047908 | 0.000011 | 0.203958 |
| norm_win_gap | 0.000378 | -0.018266 | -0.018835 | -0.101897 | 0.001242 |
| norm_lose_gap | 0.001695 | 0.216844 | 0.128319 | 0.001695 | 0.319801 |
| norm_lose_gap_clipped | 0.001695 | 0.216844 | 0.128319 | 0.001695 | 0.319801 |
| mse_w_over_ref_mse_w | 1.000326 | 0.981827 | 0.981490 | 0.904262 | 1.001217 |
| mse_l_over_ref_mse_l | 1.001522 | 1.242328 | 1.141839 | 1.001522 | 1.377483 |

`implicit_acc` is `1.0` for both mean and final row. `loser_dominant_ratio`
mean is `0.980392`, final row is `1.0`.

## Regularization / Stability

| Field | First | Last | Mean | Min | Max |
|---|---:|---:|---:|---:|---:|
| winner_abs_reg | 0.003871 | 0.175780 | 0.167578 | 0.002139 | 0.623738 |
| winner_gap_reg | 0.001204 | 0.000029 | 0.000644 | 0.000000 | 0.005598 |
| grad_norm | 34.777272 | 2.563846 | 6.026017 | 1.554639 | 34.777272 |

The first logged gradient norm is high, but training later settles to a smaller
range. No NaN/OOM was observed in the Stage1-500 gate.

## Confidence Fields In This Historical CSV

The Stage1-500 run predates the corrected mass fields, so the CSV contains the
old area-style fields:

| Field | First | Last | Mean |
|---|---:|---:|---:|
| prior_conf_mean | 0.857152 | 0.915170 | 0.882837 |
| reliable_area_ratio | 0.232837 | 0.256641 | 0.250894 |
| generate_area_ratio | 0.232715 | 0.254932 | 0.249257 |
| mask_area_ratio | 0.234654 | 0.254382 | 0.250563 |
| boundary_area_ratio | 0.048584 | 0.054395 | 0.052145 |

For corrected confidence statistics, use:

```text
reports/exp16_confidence_limit100_offline_summary.md
reports/exp16_confidence_limit100_offline_summary.csv
```

## Label

```text
IMPLEMENTATION_ONLY
DPO_SATURATED
LOSER_DOMINANT
GRAD_SPIKE
```

## Interpretation

This run validates implementation wiring: `L_prior`, `L_gen`, and
`L_boundary_extra` are nonzero and entered `loss`. However, the DPO signal is
already saturated (`implicit_acc = 1.0`) and strongly loser-dominant. Combined
with DAVIS10 visual sanity, this should be treated as an implementation gate,
not evidence that Exp16 beats Exp11.
