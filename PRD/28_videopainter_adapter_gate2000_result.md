# PRD 28: VideoPainter Adapter Gate2000 Result

Date: 2026-06-16

## Current Status

```text
status = completed_training_and_davis50_eval
adapter_type = direct_diff_dpo_isolated_trainer
result = adapter_underperforms_videopainter_baseline
```

The VideoPainter adapter gate2000 completed 2000 optimization steps on PAI
after resolving the previous missing-weight blocker and passing the trainer
preflight. The follow-up Exp14 thin eval adapter has now completed full DAVIS50
evaluation.

This run uses the isolated Exp14 trainer:

```text
exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py
```

It does **not** use upstream VideoPainter official training as a substitute.

PAI clean worktree:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate
```

Training PID:

```text
659269 (finished)
```

GPU:

```text
CUDA_VISIBLE_DEVICES=0
```

## Why This Gate Was Previously Blocked

The first gate attempt was blocked because there was no isolated trainer that
could compute the requested DPO adapter objective:

- policy winner loss `m_w`
- policy loser loss `m_l`
- frozen-reference winner loss `m_w_ref`
- frozen-reference loser loss `m_l_ref`
- region-local MSE with outer-boundary weighting
- log-ratio normalized gaps
- clipped loser gap
- winner-anchor terms
- `dpo_diag` / `adapter_diag`

Launching upstream VideoPainter training would have optimized the official
VideoPainter objective, not the Exp11 outer b0.75 S2 style DPO adapter.

That implementation blocker has been resolved by the isolated Exp14 trainer.

## Weight Resolution

The second blocker was missing PAI weights:

```text
third_party/VideoPainter/ckpt/CogVideoX-5b-I2V
third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch
```

PAI could not download from Hugging Face, so the weights were downloaded on HAL
from:

- `TencentARC/VideoPainter`
- `THUDM/CogVideoX-5b-I2V`

and transferred to PAI via:

```text
rsync --partial --append-verify
```

PAI weight target:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate/third_party/VideoPainter/ckpt
```

The required files were validated on PAI and were not LFS pointer stubs.

## PAI Preflight

Preflight report:

```text
exp14_adapter_videopainter/runs/preflight/preflight_report.json
```

Preflight result:

```text
status = passed
loss = 0.7026171684
dpo_loss = 0.6931471825
m_w = 0.1893996745
m_l = 0.2312944233
m_w_ref = 0.1893996745
m_l_ref = 0.2312944233
grad_norm = 81.0422735139
reference_has_grad = false
```

Interpretation:

- policy branch loaded;
- frozen reference branch loaded;
- winner / loser forwards ran;
- `m_w`, `m_l`, `m_w_ref`, `m_l_ref` were computed;
- region-local normalized DPO loss was finite;
- backward succeeded;
- reference parameters stayed frozen.

## Gate2000 Training

Training setup:

```text
max_steps = 2000
checkpointing_steps = 500
boundary_mode = outer
mask_weight = 1.0
boundary_weight = 0.75
outside_weight = 0.05
beta_dpo = 10
lose_gap_weight = 0.25
lose_gap_clip_tau = 1.0
winner_abs_reg_weight = 0.05
winner_gap_reg_weight = 1.0
winner_gap_reg_margin = 0.0
```

Training paths:

```text
log = logs/pipelines/exp14_adapter_videopainter_gate2000.log
train_log = exp14_adapter_videopainter/runs/gate2000/train.log
dpo_diag = exp14_adapter_videopainter/dpo_diag/dpo_diagnostics.csv
run_dir = exp14_adapter_videopainter/runs/gate2000
```

Final observed state:

```text
checkpoint-500 = present
checkpoint-1000 = present
checkpoint-1500 = present
checkpoint-2000 = present
last_weights = present
dpo_diagnostics.csv = completed_through_step_2000
Traceback/OOM/NaN/BLOCKED = not observed
```

Final dpo diagnostics:

```text
labels = DPO_SATURATED, LOSER_DOMINANT, GRAD_SPIKE_OBSERVED
mean_dpo_loss = 0.0719959
mean_implicit_acc = 0.995025
mean_loser_dominant_ratio = 0.840796
max_grad_norm = 80.3213
```

DAVIS evaluation is no longer blocked. The Exp14 thin eval adapter loaded both
the official VideoPainter baseline branch and the gate2000 `last_weights`
adapter checkpoint, then evaluated full DAVIS50 with hard comp and the project
metric backend.

## DAVIS50 Evaluation

Checkpoint loading audit:

```text
baseline_checkpoint = third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch
adapter_checkpoint = exp14_adapter_videopainter/runs/gate2000/last_weights
fallback_used = false
weights_different = true
```

Eval protocol:

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

VideoPainter does not use DiffuEraser's `raw6` convention, so generation uses
VideoPainter's inference-step setting. The metric protocol is still the project
hard-comp / no-dilation / no-blur / frame-wise rule.

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
16 / 50 videos improved in PSNR
34 / 50 videos dropped in PSNR
median PSNR delta = -1.4387
```

The adapter has visible positive cases, but the full-set metric result is
negative. This matches the saturated / loser-dominant DPO diagnostics.

## Final Decision

This is a completed negative adapter gate. The infrastructure works, the
adapter checkpoint is real, and the eval is complete. However, this exact
Exp11-style VideoPainter branch adapter should not be continued as a longer run
or claimed as a useful result without redesigning the adapter objective.
