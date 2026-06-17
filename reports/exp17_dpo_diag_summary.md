# Exp17 DPO Diagnostic Summary

Date: 2026-06-17

Run:

```text
20260617_171347_exp17_saturation_positive
```

Exp17 tested three Stage1-1000 gates on top of Exp11 outer b0.75 S2:

- `exp17a_positive_s1_1000`: stronger positive-side winner protection.
- `exp17b_saturation_s1_1000`: saturation-aware DPO gate.
- `exp17c_combined_s1_1000`: positive protection plus saturation gate.

## Diagnostic Table

| Variant | implicit_acc mean | loser_dominant mean | grad_norm mean / max | mse_l/ref mean | sat_weight mean | saturated_pair_ratio | Final label |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Exp17a positive | 0.9752 | 0.9224 | 11.77 / 178.18 | 1.5859 | 0.9887 | 0.0000 | `LOSER_DOMINANT`, `GRAD_SPIKE`, `NEGATIVE_ABLATION` |
| Exp17b saturation | 0.9703 | 0.9455 | 11.19 / 164.62 | 1.7092 | 0.9883 | 0.0000 | `DPO_SATURATED`, `LOSER_DOMINANT`, `NEGATIVE_ABLATION` |
| Exp17c combined | 0.9752 | 0.9224 | 11.51 / 172.43 | 1.5810 | 0.9887 | 0.0000 | `LOSER_DOMINANT`, `GRAD_SPIKE`, `NEGATIVE_ABLATION` |

## Interpretation

The positive-side variants did not remove the loser-dominant failure mode. The
mean `loser_dominant_ratio` stays above 0.92 for all variants, and the final
step is again `1.0`.

The saturation gate technically reduces `dpo_loss_raw` slightly, but it does not
meaningfully activate under the current margin definition. `sat_weight_mean`
stays around 0.988 and `saturated_pair_ratio` remains 0.0, so Exp17b is not a
true saturation fix yet.

All three variants still show high implicit accuracy and occasional large
gradient spikes. This is exactly the diagnostic pattern that motivated Exp17,
so the current implementation is a useful negative ablation rather than a new
best method.

## Paths

```text
exp17_saturation_positive_dpo/dpo_diag/exp17a_positive_s1_1000_stage1_1000_dpo_diagnostics.csv
exp17_saturation_positive_dpo/dpo_diag/exp17b_saturation_s1_1000_stage1_1000_dpo_diagnostics.csv
exp17_saturation_positive_dpo/dpo_diag/exp17c_combined_s1_1000_stage1_1000_dpo_diagnostics.csv
```
