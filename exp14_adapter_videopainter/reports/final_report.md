# VideoPainter Adapter Gate2000 Final Report

Date: 2026-06-16

## Status

```text
training_status = completed_2000_steps
eval_status = completed_davis50
adapter_type = direct_diff_dpo_branch_adapter
conclusion = adapter_underperforms_baseline
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

DAVIS eval:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/logs/target_eval/exp14_videopainter_adapter_gate2000_davis
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
and frequently loser-dominant.

## DAVIS50 Eval

DAVIS50 eval completed with the Exp14 thin eval adapter.

| method | PSNR | SSIM | strict mask PSNR | LPIPS | videos | frames |
|---|---:|---:|---:|---:|---:|---:|
| VideoPainter baseline | 31.6124 | 0.9608 | 19.9691 | n/a | 50 | 2366 |
| VideoPainter + DPO adapter | 29.8028 | 0.9580 | 18.1595 | n/a | 50 | 2366 |

The adapter is lower by 1.8096 PSNR and 0.0028 SSIM. It improves some individual
videos but fails overall.

## Conclusion

- Training: completed.
- Reference model: constructed and frozen in preflight / training path.
- Adapter type: direct Diff-DPO branch adapter.
- dpo_diag: present and finite, but saturated / loser-dominant.
- DAVIS eval: completed on full DAVIS50.
- Recommendation: do not continue this VideoPainter adapter as a longer run
  without redesigning the adapter objective / pair construction.
