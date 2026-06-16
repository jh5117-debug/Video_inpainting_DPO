# VideoPainter Adapter Gate2000 Final Report

Date: 2026-06-16

## Result

```text
training_status = completed_2000_steps
eval_status = completed_davis50
adapter_type = direct_diff_dpo_branch_adapter
conclusion = adapter_underperforms_videopainter_baseline
```

Exp14 VideoPainter adapter gate2000 completed 2000 optimization steps on PAI
with the isolated direct Diff-DPO trainer. It did not run upstream VideoPainter
official training as a substitute.

DAVIS eval is now complete with the Exp14 thin eval adapter. The adapter is not
better than the VideoPainter baseline under the project hard-comp frame-wise
metric protocol.

## Main Paths

PAI clean worktree:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate
```

Baseline checkpoint:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch
```

Adapter checkpoint:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/exp14_adapter_videopainter/runs/gate2000/last_weights
```

DPO diagnostics:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/exp14_adapter_videopainter/dpo_diag/dpo_diagnostics.csv
```

DAVIS eval output:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/logs/target_eval/exp14_videopainter_adapter_gate2000_davis
```

## Checkpoint Loading

The adapter eval did not fall back to baseline weights.

```text
baseline_weight_exists = true
adapter_weight_exists = true
fallback_used = false
weights_different = true
```

Audit:

```text
reports/videopainter_adapter_checkpoint_loading_audit.md
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

The run completed, but diagnostics show saturation, loser-dominant behavior, and
a large gradient spike. That is consistent with the negative DAVIS result.

## DAVIS50 Eval

Protocol:

- dataset: DAVIS50, 50 videos, 2366 frames;
- VideoPainter inference steps: 50;
- frames capped at 49 and trimmed to 4k+1;
- hard comp: prediction inside mask + GT outside mask;
- no mask dilation;
- no Gaussian blur;
- no VBench;
- metric backend: `inference/metrics.py`;
- mp4 is only for visualization, not metric computation.

Metrics:

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

Per-video split:

```text
PSNR improved on 16 / 50 videos
PSNR dropped on 34 / 50 videos
median PSNR delta = -1.4387
```

Positive candidates include `rollerblade`, `scooter-black`, `dog-agility`,
`bus`, `motorbike`, `libby`, `bear`, and `flamingo`.

Failure candidates include `hockey`, `paragliding-launch`, `hike`, `car-turn`,
`dog`, `dance-jump`, `bmx-bumps`, and `swing`.

## Conclusion

The VideoPainter adapter trainer is functional, and the checkpoint can be
loaded and evaluated. However, the current Exp11-style region-local normalized
DPO adapter does not improve VideoPainter on full DAVIS50. It should not be
continued as a longer training run or claimed as a paper result without a new
adapter design.
