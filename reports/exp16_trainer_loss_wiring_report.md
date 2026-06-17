# Exp16 Trainer Loss Wiring Report

Date: 2026-06-17

## Trainer Scope

Only the isolated Exp16 Stage1 trainer is wired in this pass:

```text
exp16_prior_confidence_gated_dpo/code/train_exp16_stage1.py
```

Shared `training/dpo` files and old Exp9/10/11/12 code are untouched.

## Real Prior Target

The trainer requires an Exp16 manifest with real ProPainter prior frames. The
dataset loader refuses rows without one of the explicit prior fields:

```text
prior_frame_dir
propainter_prior_frame_dir
propainter_frame_dir
prior_video_path
propainter_prior_video_path
propainter_video_path
propainter_mp4
propainter_path
```

No frozen-reference epsilon proxy is used as a prior target.

## Latent x0 Path

Stage1 computes:

```text
z_prior = VAE(ProPainter prior)
z_gt = VAE(GT winner)
z_hat_x0 = scheduler-derived clean latent from policy winner prediction
```

`z_hat_x0` reconstruction supports scheduler `prediction_type` values:

```text
epsilon
v_prediction
sample
```

If the scheduler cannot provide `alphas_cumprod` for epsilon/v-prediction,
Exp16 raises instead of silently falling back to a proxy.

## Total Loss Wiring

Base loss inherits Exp11 outer b0.75 S2:

```text
L_base =
    L_DPO(log-ratio normalized gap, clipped loser gap)
  + 0.05 * m_w
  + ReLU(g_w)
```

Exp16 adds:

```text
L_prior = |z_hat_x0 - z_prior| over M_reliable
L_gen = |z_hat_x0 - z_gt| over M_generate
L_boundary_extra = |z_hat_x0 - z_gt| over boundary_outer

L_total =
    L_base
  + 0.1 * L_prior
  + 0.05 * L_gen
  + 0.1 * L_boundary_extra
```

Required answers:

```text
L_prior enters total_loss = yes
L_gen enters total_loss = yes
L_boundary_extra enters total_loss = yes
prior_target_mode = latent_x0
confidence_mode = gt_error
```

## Remaining Limitation

Stage2 is not wired for Exp16 prior-confidence loss in this pass. It must not
be launched until the same real-prior latent-x0 wiring is implemented there.
