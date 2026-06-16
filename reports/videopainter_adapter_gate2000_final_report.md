# VideoPainter Adapter Gate2000 Final Report

Date: 2026-06-16

## Result

```text
training_status = completed_2000_steps
eval_status = blocked_pending_exp14_thin_eval_adapter
```

Exp14 VideoPainter adapter gate2000 successfully ran to 2000 optimization steps
on PAI with the isolated direct Diff-DPO trainer.

It did not run upstream VideoPainter official training as a substitute.

## Main Paths

PAI clean worktree:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate
```

Diagnostics:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/exp14_adapter_videopainter/dpo_diag/dpo_diagnostics.csv
```

Final weights:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/exp14_adapter_videopainter/runs/gate2000/last_weights
```

Checkpoints:

```text
checkpoint-500
checkpoint-1000
checkpoint-1500
checkpoint-2000
```

## DPO Diagnostic Summary

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
last_loss = 0.0946371
last_dpo_loss = 0.0854610
last_loser_dominant_ratio = 1.0
```

The run completed, but diagnostics show saturation and loser-dominant behavior.
That means this is a valid training gate result, not yet evidence of a useful
VideoPainter adapter.

## Eval Status

No DAVIS metric table was produced.

The upstream VideoPainter DAVIS eval path is not acceptable for this project
result because:

- it is not the fixed raw6 hard-comp protocol;
- it does not call the project `inference/metrics.py` metric backend;
- it uses VideoPainter's own metric path and mask dilation behavior;
- it currently fails import without the Exp14 compatibility shim;
- it unconditionally expects a `FluxFillPipeline` first-frame model in the
  DAVIS path, while `ckpt/flux_inp` is not present.

Next action is an Exp14-only thin eval adapter that loads VideoPainter baseline
and `last_weights`, generates raw6 hard-comp outputs, and then calls the
existing project metric backend.
