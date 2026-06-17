# Exp16 Confidence Diagnostic Fix Report

Date: 2026-06-17

## Problem

The original Exp16 dpo_diag fields:

```text
reliable_area_ratio
generate_area_ratio
```

were area-style nonzero statistics. Because both `M * C_prior` and
`M * (1 - C_prior)` are usually nonzero inside the same mask, both values can
look close to `mask_area_ratio`. That is technically true but not useful for
interpreting how much of the mask is being treated as prior-reliable versus
generation-needed.

## Fix

The old fields were kept for backward compatibility. New fields were added for
future dpo_diag rows:

```text
prior_conf_mean_inside_mask
prior_conf_std_inside_mask
prior_conf_p10_inside_mask
prior_conf_p50_inside_mask
prior_conf_p90_inside_mask
reliable_weight_mass
generate_weight_mass
reliable_generate_mass_sum
confidence_alpha
```

Definitions:

```text
reliable_weight_mass = sum(M * C_prior) / (sum(M) + eps)
generate_weight_mass = sum(M * (1 - C_prior)) / (sum(M) + eps)
reliable_generate_mass_sum = reliable_weight_mass + generate_weight_mass
```

These are easier to read: they describe the confidence-weighted split inside
the mask, and the mass sum should be approximately 1.0.

## Code Touched

Only Exp16-local code was changed:

```text
exp16_prior_confidence_gated_dpo/code/exp16_loss.py
exp16_prior_confidence_gated_dpo/code/exp16_dpo_diag.py
exp16_prior_confidence_gated_dpo/code/train_exp16_stage1.py
exp16_prior_confidence_gated_dpo/code/audit_prior_confidence_cache.py
```

No Exp9/10/11/12 code or shared `training/dpo` code was modified.

## Offline Limit100 Summary

Offline recomputation over the limit=100 ProPainter prior cache:

```text
reports/exp16_confidence_limit100_offline_summary.md
reports/exp16_confidence_limit100_offline_summary.csv
```

Key aggregate values:

| Field | Mean |
|---|---:|
| prior_conf_mean_inside_mask | 0.656014 |
| prior_conf_std_inside_mask | 0.264408 |
| prior_conf_p10_inside_mask | 0.239536 |
| prior_conf_p50_inside_mask | 0.725268 |
| prior_conf_p90_inside_mask | 0.940553 |
| reliable_weight_mass | 0.656014 |
| generate_weight_mass | 0.343986 |
| reliable_generate_mass_sum | 1.000000 |
| mask_area_ratio | 0.256022 |
| boundary_area_ratio | 0.008180 |

## Interpretation

The corrected fields look healthy: the mass split is interpretable and sums to
1.0. This supports the implementation of the confidence map, but it does not by
itself prove that Exp16 improves generation quality.
