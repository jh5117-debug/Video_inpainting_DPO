# PRD 30: VideoPainter Adapter DAVIS Eval

Date: 2026-06-16

## Summary

Exp14 VideoPainter adapter gate2000 has now been evaluated on full DAVIS50 with
an Exp14-only thin eval adapter. The result is negative: the DPO adapter
checkpoint is real and loadable, but it underperforms the VideoPainter baseline.

## What Was Evaluated

Baseline:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch
```

Adapter:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/exp14_adapter_videopainter/runs/gate2000/last_weights
```

Checkpoint audit:

```text
fallback_used = false
weights_different = true
```

So the adapter eval did not silently use the baseline checkpoint.

## Eval Protocol

This eval is not upstream VideoPainter's official metric path. It uses the
project Exp14 thin adapter and the existing project metric backend.

```text
dataset = DAVIS50
videos = 50
frames = 2366
VideoPainter inference steps = 50
hard comp = prediction inside mask + GT outside mask
mask dilation = off
Gaussian blur = off
VBench = off
metric backend = inference/metrics.py
```

VideoPainter does not have DiffuEraser's `raw6` denoising-step convention, so
the generation uses VideoPainter's inference-step setting. The metric protocol
still follows the project hard-comp / no-dilation / no-blur / frame-wise rule.

## Main Table

| method | PSNR | SSIM | strict mask PSNR | LPIPS | videos | frames |
|---|---:|---:|---:|---:|---:|---:|
| VideoPainter baseline | 31.6124 | 0.9608 | 19.9691 | n/a | 50 | 2366 |
| VideoPainter + DPO adapter | 29.8028 | 0.9580 | 18.1595 | n/a | 50 | 2366 |

Delta adapter minus baseline:

```text
PSNR = -1.8096
SSIM = -0.0028
strict_mask_pixel_psnr = -1.8096
```

Per-video:

```text
16 / 50 videos improved in PSNR
34 / 50 videos dropped in PSNR
median PSNR delta = -1.4387
```

## Visual Cases

Success candidates:

```text
rollerblade
scooter-black
dog-agility
bus
motorbike
libby
bear
flamingo
```

Failure candidates:

```text
hockey
paragliding-launch
hike
car-turn
dog
dance-jump
bmx-bumps
swing
```

Four-column videos and contact sheets are under:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/logs/target_eval/exp14_videopainter_adapter_gate2000_davis
```

Columns:

1. GT
2. mask overlay
3. VideoPainter baseline
4. VideoPainter + DPO adapter

## DPO Diagnostics

The training diagnostics explain why the eval result should be treated as a
negative adapter gate:

```text
DPO_SATURATED
LOSER_DOMINANT
GRAD_SPIKE_OBSERVED
mean_dpo_loss = 0.0719959
mean_implicit_acc = 0.995025
mean_loser_dominant_ratio = 0.840796
max_grad_norm = 80.3213
```

Interpretation:

The trainer ran and saved a real adapter checkpoint, but the objective saturated
and became loser-dominant. The full DAVIS50 result confirms that this
Exp11-style DPO objective does not transfer cleanly to VideoPainter as a branch
adapter.

## Decision

Do not continue this exact VideoPainter adapter with longer training. It is a
valid negative feasibility result and should be reported as such if discussed:
the infrastructure works, but the current DPO adapter objective is not useful
for VideoPainter without redesign.
