# Exp19b Exploratory 2000 DPO / Adapter Diagnostic Summary

Status: completed.

Source:

```text
exp19b_exploratory_2000/dpo_diag/exp19b_exploratory_s2_2000_dpo_diagnostics.csv
```

This continuation started from the Exp19b Stage2-500 adapter and ran 1500
additional adapter-only optimization steps, for 2000 total adapter steps.

Key diagnostics:

| Field | Value |
| --- | ---: |
| diagnostic rows | 151 |
| final total loss | 0.699858 |
| mean total loss | 0.697558 |
| final dpo loss | 0.693484 |
| mean dpo loss | 0.693246 |
| final norm win gap | 0.000093 |
| final norm lose gap | -0.000161 |
| mean loser dominant ratio | 0.211921 |
| final loser dominant ratio | 0.000000 |
| mean adapter grad norm | 0.000183 |
| final adapter grad norm | 0.000096 |
| mean base grad norm | 0.000000 |
| final base grad norm | 0.000000 |
| mean adapter residual norm | 0.071805 |
| final adapter residual norm | 0.073974 |
| mean gate | 0.003739 |
| mean flow confidence | 0.451832 |
| mean valid flow ratio | 0.594241 |

Interpretation:

- The base model remained frozen (`base_grad_norm = 0`).
- The adapter did continue to receive finite, non-zero gradients.
- The DPO objective stayed close to saturation (`dpo_loss ~= 0.693`), and the
  adapter residual remained small.
- No NaN/OOM/collapse was observed.
- DAVIS50 did not improve over Exp11, so the run is best treated as a safe
  no-op / negative ablation rather than a promising continuation.
