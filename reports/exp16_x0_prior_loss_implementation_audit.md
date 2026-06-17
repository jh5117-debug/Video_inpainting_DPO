# Exp16 X0 / Prior Loss Implementation Audit

Date: 2026-06-17

## Requirement

Exp16 must use real ProPainter prior targets. It must not reuse the old
frozen-reference epsilon proxy.

Required tensors:

```text
z_hat_x0 = predicted clean latent reconstructed from model output
z_prior = VAE(ProPainter prior)
z_gt = VAE(GT clean video)
```

Losses:

```text
L_prior = |z_hat_x0 - z_prior| over M_reliable
L_gen = |z_hat_x0 - z_gt| over M_generate
L_boundary_extra = |z_hat_x0 - z_gt| over boundary_outer
```

## Implemented

Implemented helper:

```text
exp16_prior_confidence_gated_dpo/code/exp16_loss.py
  predict_x0_from_model_output()
  compute_prior_gated_losses()
```

`predict_x0_from_model_output()` supports:

- `prediction_type = epsilon`
- `prediction_type = v_prediction`
- `prediction_type = sample`

It raises if the scheduler does not expose `alphas_cumprod` or uses an
unsupported prediction type.

## Not Yet Passed

The full Stage1 / Stage2 training loops have not yet been updated to:

1. load `prior_pixel_values` from a prior manifest;
2. encode `z_prior` through the DiffuEraser VAE;
3. reconstruct `z_hat_x0` from real model outputs;
4. add `L_prior`, `L_gen`, and `L_boundary_extra` to `total_loss`;
5. write the new Exp16 diagnostics.

Therefore:

```text
status = BLOCKED_PENDING_FULL_TRAINER_INTEGRATION
```

The copied Stage1 / Stage2 scripts are guarded by
`EXP16_ENABLE_REAL_PRIOR_X0_TRAINING`. Without that environment variable they
exit with `BLOCKED` instead of running the old Exp11 loss.

## Not A Proxy

No code path in Exp16 is allowed to use:

```text
frozen-ref epsilon prior
reference-only prior proxy
generated loser as prior target
```

The current implementation blocks rather than falling back to any of these.

