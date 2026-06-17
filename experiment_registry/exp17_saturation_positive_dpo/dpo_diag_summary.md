# DPO Diagnostics

All three Exp17 Stage1-1000 gates wrote `dpo_diagnostics.csv`.

| Variant | implicit_acc mean | loser_dominant mean | grad_norm mean / max | mse_l/ref mean | sat_weight mean | saturated_pair_ratio | Label |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Exp17a positive | 0.9752 | 0.9224 | 11.77 / 178.18 | 1.5859 | 0.9887 | 0.0000 | `LOSER_DOMINANT`, `GRAD_SPIKE`, `NEGATIVE_ABLATION` |
| Exp17b saturation | 0.9703 | 0.9455 | 11.19 / 164.62 | 1.7092 | 0.9883 | 0.0000 | `DPO_SATURATED`, `LOSER_DOMINANT`, `NEGATIVE_ABLATION` |
| Exp17c combined | 0.9752 | 0.9224 | 11.51 / 172.43 | 1.5810 | 0.9887 | 0.0000 | `LOSER_DOMINANT`, `GRAD_SPIKE`, `NEGATIVE_ABLATION` |

The saturation gate did not truly activate under this margin definition:
`sat_weight_mean` stayed close to 1 and `saturated_pair_ratio` stayed 0.

Evidence:

```text
reports/exp17_dpo_diag_summary.md
exp17_saturation_positive_dpo/dpo_diag/
```
