# DPO Diagnostics Summary

Diagnostics are present:

- `exp18_multiframe_propagation_gated_dpo/dpo_diag/exp18a_prop_only_stage1_500_dpo_diagnostics.csv`
- `exp18_multiframe_propagation_gated_dpo/dpo_diag/exp18b_prop_gen_stage1_500_dpo_diagnostics.csv`
- `exp18_multiframe_propagation_gated_dpo/dpo_diag/exp18c_oracle_stage1_500_dpo_diagnostics.csv`

Summary:

| Variant | mean loser_dominant | mean prop coverage | mean prop conf | label |
|---|---:|---:|---:|---|
| Exp18a | 0.9412 | 0.0277 | 0.0316 | `NON_ORACLE_SPARSE_CONFIDENCE`, `LOSER_DOMINANT` |
| Exp18b | 0.9216 | 0.0228 | 0.0273 | `NON_ORACLE_SPARSE_CONFIDENCE`, `NEGATIVE_ABLATION` |
| Exp18c | 0.9608 | 0.9463 | 0.9400 | `ORACLE_UPPER_BOUND_NEGATIVE`, `DIAGNOSTIC_ONLY` |

Source:

```text
reports/exp18_dpo_diag_summary.md
```
