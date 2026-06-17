# DPO Diagnostics

Latest summary:

```text
reports/exp16_dpo_diag_summary_limit100.md
reports/exp16_stage1_500_dpo_diag_summary.md
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
- Corrected confidence mass fields were added for future diagnostics:
  `reliable_weight_mass`, `generate_weight_mass`,
  `reliable_generate_mass_sum`, and inside-mask confidence percentiles.
- Offline limit100 confidence summary is in
  `reports/exp16_confidence_limit100_offline_summary.md`.
