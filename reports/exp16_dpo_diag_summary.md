# Exp16 DPO Diagnostic Summary

Latest status: Stage1 500 limit=100 completed on PAI with real ProPainter prior
frames and latent-x0 prior/generation/boundary losses.

Primary summary:

```text
reports/exp16_dpo_diag_summary_limit100.md
```

Local diagnostic CSVs:

```text
exp16_prior_confidence_gated_dpo/dpo_diag/preflight_dpo_diagnostics.csv
exp16_prior_confidence_gated_dpo/dpo_diag/stage1_500_dpo_diagnostics.csv
```

Short interpretation:

- `L_prior`, `L_gen`, and `L_boundary_extra` are nonzero and enter total loss.
- Stage1 500 completed and saved `last_weights`.
- `implicit_acc` remains 1.0 and `loser_dominant_ratio` is high, so this is an
  implementation small gate, not a final method result.
