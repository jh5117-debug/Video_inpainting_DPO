# Exp11 Outer B0.75 S2 DPO Diagnostics Summary

Date: 2026-06-15

## Paths

PAI Stage1:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s1_2000_davis_pai/dpo_diagnostics.csv
```

PAI Stage2:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s2_2000_davis_pai/dpo_diagnostics.csv
```

HAL snapshots:

```text
/home/hj/dpo-2-1-exp/this_week_exp11_exp12/dpo_diag/exp11_boundary_outer_b075_s1_dpo_diagnostics.csv
/home/hj/dpo-2-1-exp/this_week_exp11_exp12/dpo_diag/exp11_boundary_outer_b075_s2_dpo_diagnostics.csv
```

## Setting

- boundary mode: `outer`
- mask weight: `1.0`
- boundary weight: `0.75`
- outside weight: `0.05`
- gap normalization: `log_ratio`
- loss region mode: `region`
- lose gap clip tau: `1.0`

## Stage1 Summary

Rows: 201. Steps: 1 to 2000.

| field | mean all | mean last20 | last | max abs |
|---|---:|---:|---:|---:|
| dpo_loss | 0.439787 | 0.310162 | 0.464496 | 0.824353 |
| implicit_acc | 0.978856 | 0.987500 | 1.000000 | 1.000000 |
| raw_win_gap | -0.000920 | -0.000332 | -0.000565 | 0.026263 |
| raw_lose_gap | 0.383648 | 0.550082 | 0.020849 | 2.132240 |
| norm_win_gap | -0.005040 | -0.015532 | -0.046134 | 0.190633 |
| norm_lose_gap | 0.724962 | 1.213880 | 0.368998 | 2.096830 |
| norm_lose_gap_clipped | 0.541821 | 0.829626 | 0.288182 | 1.000000 |
| winner_abs_reg | 0.142423 | 0.128041 | 0.011913 | 0.737299 |
| winner_gap_reg | 0.009944 | 0.008638 | 0.000000 | 0.193743 |
| mse_w_over_ref_mse_w | 0.998135 | 0.987136 | 0.954751 | 1.291080 |
| mse_l_over_ref_mse_l | 2.576420 | 3.958190 | 2.321660 | 7.871790 |
| sigma_term | 0.659728 | 0.747539 | 0.643572 | 0.809711 |
| kl_divergence | 0.095682 | 0.137437 | 0.005071 | 0.531673 |
| loser_dominant_ratio | 0.997512 | 1.000000 | 1.000000 | 1.000000 |
| grad_norm | 35.874300 | 58.189300 | 15.166700 | 809.145000 |
| mask_area_ratio | 0.253660 | 0.258682 | 0.244141 | 0.576172 |
| boundary_area_ratio | 0.051663 | 0.053279 | 0.050391 | 0.070752 |
| outside_area_ratio | 0.694678 | 0.688040 | 0.705469 | 0.768750 |
| region_weight_sum | 837.480000 | 852.589000 | 812.050000 | 1647.550000 |

Label: `LOSER_DOMINANT`.

Stage1 does not show old-style win-gap explosion. The loser branch remains dominant, which is expected for this DPO construction and should be reported as a residual risk rather than ignored.

## Stage2 Summary

Rows: 201. Steps: 1 to 2000.

| field | mean all | mean last20 | last | max abs |
|---|---:|---:|---:|---:|
| dpo_loss | 0.383098 | 0.338639 | 0.551337 | 2.497290 |
| implicit_acc | 0.987562 | 1.000000 | 1.000000 | 1.000000 |
| raw_win_gap | 0.001937 | 0.000083 | 0.000154 | 0.099624 |
| raw_lose_gap | 0.595174 | 0.513343 | 0.015378 | 2.140280 |
| norm_win_gap | 0.013580 | -0.000614 | 0.017749 | 0.633911 |
| norm_lose_gap | 1.225410 | 1.208410 | 0.391761 | 2.454030 |
| norm_lose_gap_clipped | 0.787115 | 0.787290 | 0.345636 | 1.000000 |
| winner_abs_reg | 0.091341 | 0.090691 | 0.008809 | 0.648329 |
| winner_gap_reg | 0.019875 | 0.006708 | 0.020878 | 0.633911 |
| mse_w_over_ref_mse_w | 1.022700 | 0.999107 | 1.017820 | 2.268220 |
| mse_l_over_ref_mse_l | 4.140930 | 4.114820 | 2.220380 | 11.618500 |
| sigma_term | 0.706309 | 0.721619 | 0.584992 | 0.800738 |
| kl_divergence | 0.149278 | 0.128356 | 0.003883 | 0.534562 |
| loser_dominant_ratio | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| grad_norm | 4.806910 | 4.303000 | 7.694890 | 24.848300 |
| mask_area_ratio | 0.253660 | 0.258682 | 0.244141 | 0.576172 |
| boundary_area_ratio | 0.051663 | 0.053279 | 0.050391 | 0.070752 |
| outside_area_ratio | 0.694678 | 0.688040 | 0.705469 | 0.768750 |
| region_weight_sum | 837.480000 | 852.589000 | 812.050000 | 1647.550000 |

Label: `LOSER_DOMINANT`.

Stage2 remains much more stable than old raw DPO runs: no large winner gap explosion and no sustained collapse signal in the last diagnostics. It still has a fully loser-dominant diagnostic pattern, so the safest claim is: metrics and visual evidence improve under the fixed DAVIS50 protocol, while dpo-diag still shows a residual loser-dominant tendency.

