# Exp16 DPO Diagnostics Summary: Stage1 500 Limit100

Date: 2026-06-17

Scope:

```text
Exp16 Stage1 500, limit=100 real ProPainter prior cache
```

Paths:

```text
preflight_diag = exp16_prior_confidence_gated_dpo/dpo_diag/preflight_dpo_diagnostics.csv
stage1_diag = exp16_prior_confidence_gated_dpo/dpo_diag/stage1_500_dpo_diagnostics.csv
stage1_run = /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260617_exp16_limit100_exp16_prior_confidence_s1_500_limit100_pai
```

## Summary

| field | first | last | mean | min | max |
|---|---:|---:|---:|---:|---:|
| loss | 0.795262 | 0.612881 | 0.665445 | 0.487197 | 0.795262 |
| L_base | 0.694449 | 0.537362 | 0.587136 | 0.410535 | 0.694449 |
| dpo_loss | 0.693051 | 0.528544 | 0.578113 | 0.407069 | 0.693051 |
| L_prior | 0.400244 | 0.376475 | 0.413692 | 0.313802 | 0.537265 |
| L_gen | 0.714177 | 0.376985 | 0.372089 | 0.093265 | 0.728087 |
| L_boundary_extra | 0.250795 | 0.190216 | 0.183348 | 0.072910 | 0.291013 |
| implicit_acc | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| loser_dominant_ratio | 1.000000 | 1.000000 | 0.980392 | 0.000000 | 1.000000 |
| grad_norm | 34.777272 | 2.563846 | 6.026017 | 1.554639 | 34.777272 |
| prior_conf_mean | 0.857152 | 0.915170 | 0.882837 | 0.800452 | 0.945813 |
| reliable_area_ratio | 0.232837 | 0.256641 | 0.250894 | 0.206738 | 0.298828 |
| generate_area_ratio | 0.232715 | 0.254932 | 0.249257 | 0.205688 | 0.298535 |
| mask_area_ratio | 0.234654 | 0.254382 | 0.250563 | 0.206902 | 0.298871 |
| boundary_area_ratio | 0.048584 | 0.054395 | 0.052145 | 0.043506 | 0.061328 |

## Interpretation

The engineering gate passed: `L_prior`, `L_gen`, and `L_boundary_extra` are
nonzero and enter `total_loss`; the run completed 500 steps and saved both
checkpoint-500 and `last_weights`.

The optimization signal is still risky: `implicit_acc` stays at 1.0 and
`loser_dominant_ratio` averages about 0.98. This looks like an easy-pair /
strong-separation regime rather than a clean final training setting. Treat this
as implementation validation, not a final Exp16 result.

Recommended next step is not Stage2/full training yet. First inspect a small
validation or qualitative decode from the Stage1 500 checkpoint, then decide
whether to build the full prior cache and tune pair hardness / loss weights.
