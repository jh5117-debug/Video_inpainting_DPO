# Experiment Registry: Exp14 VideoPainter Adapter

Status: completed_negative_gate.

Exp14 tests whether the current best DiffuEraser-side DPO recipe can transfer
to VideoPainter as a branch adapter:

```text
source method = Exp11 outer b0.75 S2
adapter type = direct Diff-DPO branch adapter
```

The isolated trainer was implemented under:

```text
exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py
```

The DAVIS eval adapter was implemented under:

```text
exp14_adapter_videopainter/code/eval_videopainter_adapter_davis.py
```

Outcome:

- trainer preflight passed on PAI;
- gate2000 completed 2000 steps;
- adapter checkpoint was real and different from the baseline checkpoint;
- full DAVIS50 eval completed;
- adapter underperformed the VideoPainter baseline.

Main metric:

| method | PSNR | SSIM | strict mask PSNR |
|---|---:|---:|---:|
| VideoPainter baseline | 31.6124 | 0.9608 | 19.9691 |
| VideoPainter + DPO adapter | 29.8028 | 0.9580 | 18.1595 |

Decision: do not continue this exact VideoPainter adapter with longer training
without redesigning the objective / data pairing.
