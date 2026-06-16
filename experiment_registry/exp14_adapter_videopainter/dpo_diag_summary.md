# DPO Diagnostic Summary

Status: completed_training.

DPO diagnostics path on PAI:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/exp14_adapter_videopainter/dpo_diag/dpo_diagnostics.csv
```

Current observation:

- diagnostics were written every 10 steps;
- training reached step 2000;
- losses are finite;
- no `NaN`, `OOM`, `OutOfMemory`, `Traceback`, or `BLOCKED` observed;
- `checkpoint-500`, `checkpoint-1000`, `checkpoint-1500`, `checkpoint-2000`, and `last_weights` exist.

Labels:

```text
DPO_SATURATED
LOSER_DOMINANT
GRAD_SPIKE_OBSERVED
```

Key values:

```text
rows = 201
last_step = 2000
mean_loss = 0.0790636
mean_dpo_loss = 0.0719959
mean_implicit_acc = 0.995025
mean_loser_dominant_ratio = 0.840796
max_grad_norm = 80.3213
```

Interpretation:

The adapter trainer completed, but diagnostics are saturated and loser-dominant.
Do not claim VideoPainter adapter improvement until DAVIS eval / visualization
is implemented and reviewed.
