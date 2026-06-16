# VideoPainter Adapter Gate2000 DPO Diagnostic Summary

Status: completed_training.

Rows: 201
First step: 1
Last step: 2000

Labels: DPO_SATURATED, LOSER_DOMINANT, GRAD_SPIKE_OBSERVED

Key values:

```text
mean_loss = 0.0790636
mean_dpo_loss = 0.0719959
mean_implicit_acc = 0.995025
mean_loser_dominant_ratio = 0.840796
max_grad_norm = 80.3213
last_loss = 0.0946371
last_dpo_loss = 0.0854610
last_loser_dominant_ratio = 1.0
```

Interpretation:

The 2000-step gate completed and diagnostics remained finite. However, the run
is DPO-saturated and loser-dominant, so it should not be treated as a clean
adapter improvement without a proper DAVIS metric / visual evaluation.
