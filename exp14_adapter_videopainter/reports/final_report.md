# VideoPainter Adapter Gate2000 Final Report

Date: 2026-06-16

## Status

```text
training_status = completed_2000_steps
eval_status = blocked_pending_exp14_thin_eval_adapter
```

The Exp14 VideoPainter adapter gate2000 completed 2000 optimization steps on
PAI using the isolated Exp14 DPO trainer:

```text
exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py
```

It did not use upstream VideoPainter official training as a substitute.

## Paths

Clean worktree:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate
```

DPO diagnostics:

```text
exp14_adapter_videopainter/dpo_diag/dpo_diagnostics.csv
```

Checkpoints:

```text
exp14_adapter_videopainter/runs/gate2000/checkpoint-500
exp14_adapter_videopainter/runs/gate2000/checkpoint-1000
exp14_adapter_videopainter/runs/gate2000/checkpoint-1500
exp14_adapter_videopainter/runs/gate2000/checkpoint-2000
exp14_adapter_videopainter/runs/gate2000/last_weights
```

Logs:

```text
logs/pipelines/exp14_adapter_videopainter_gate2000.log
exp14_adapter_videopainter/runs/gate2000/train.log
```

## Training Result

The run completed all 2000 steps. The process exited and GPU0 was released.

No `Traceback`, `OOM`, `OutOfMemory`, `NaN`, or `BLOCKED` lines were found in
the monitored logs.

`last_weights` exists and contains:

```text
last_weights/branch/config.json
last_weights/branch/diffusion_pytorch_model.safetensors
last_weights/run_manifest.json
```

## DPO Diagnostics

Summary report:

```text
reports/videopainter_adapter_gate2000_dpo_diag_summary.md
exp14_adapter_videopainter/reports/dpo_diag_summary.md
```

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

Interpretation:

The trainer is functional and completed, but the DPO signal is highly saturated
and frequently loser-dominant. This is not enough evidence to claim the
VideoPainter adapter improves inpainting quality.

## DAVIS Eval Status

DAVIS eval did not complete in this run.

Reason:

- Upstream `evaluate/eval_inpainting.py` is not the project fixed raw6
  hard-comp protocol.
- It imports VideoPainter's vendored Diffusers without the Exp14 compatibility
  shim and fails on `FLAX_WEIGHTS_NAME` in the current PAI environment.
- It unconditionally calls `FluxFillPipeline.from_pretrained(args.img_inpainting_model)`
  in the DAVIS path; the required `ckpt/flux_inp` model is not present.
- It uses VideoPainter's own metrics and mask dilation path, not the project
  `inference/metrics.py` / raw6 / no dilation / hard-comp protocol.

Therefore no VideoPainter baseline-vs-adapter DAVIS metric table is available
yet. A separate isolated Exp14 thin eval adapter is required before producing
paper-quality VideoPainter adapter metrics or four-column videos.

## Conclusion

- Training: completed.
- Reference model: constructed and frozen in preflight / training path.
- Adapter type: direct Diff-DPO branch adapter.
- dpo_diag: present and finite, but saturated / loser-dominant.
- DAVIS eval: blocked pending a proper Exp14 eval adapter.
- Recommendation: do not claim VideoPainter adapter improvement yet. Next step
  is to implement a small Exp14-only DAVIS inference/eval wrapper that loads
  baseline branch and `last_weights`, generates raw6 hard-comp outputs, and
  calls the project metric backend.
