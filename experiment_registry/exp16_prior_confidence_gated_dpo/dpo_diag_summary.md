# DPO Diagnostics

Latest summary:

```text
reports/exp16_dpo_diag_summary_limit100.md
```

Local CSV:

```text
exp16_prior_confidence_gated_dpo/dpo_diag/stage1_500_dpo_diagnostics.csv
```

Interpretation:

- Stage1 500 completed.
- `L_prior`, `L_gen`, and `L_boundary_extra` are nonzero and enter total loss.
- `implicit_acc` is 1.0 and loser-dominant remains high, so this is an
  implementation gate, not a final result.
