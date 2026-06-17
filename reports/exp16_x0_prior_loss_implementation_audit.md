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

## Passed For Stage1 Small Gate

The isolated Stage1 trainer has been updated and verified to:

1. load `prior_pixel_values` from a prior manifest;
2. encode `z_prior` through the DiffuEraser VAE;
3. reconstruct `z_hat_x0` from real model outputs;
4. add `L_prior`, `L_gen`, and `L_boundary_extra` to `total_loss`;
5. write the new Exp16 diagnostics.

Therefore:

```text
status = STAGE1_PASSED_LIMIT100_PREFLIGHT_AND_SMALL_GATE
```

Evidence:

```text
preflight_diag = exp16_prior_confidence_gated_dpo/dpo_diag/preflight_dpo_diagnostics.csv
stage1_diag = exp16_prior_confidence_gated_dpo/dpo_diag/stage1_500_dpo_diagnostics.csv
stage1_run = /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260617_exp16_limit100_exp16_prior_confidence_s1_500_limit100_pai
```

Stage2 has not been updated with this wiring and remains blocked.

## Not A Proxy

No code path in Exp16 is allowed to use:

```text
frozen-ref epsilon prior
reference-only prior proxy
generated loser as prior target
```

The current implementation blocks rather than falling back to any of these.
