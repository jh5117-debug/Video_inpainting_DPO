# Exp16 Prior Confidence Map Audit

Date: 2026-06-17

## Definition

First version:

```text
P = ProPainter prior
x_GT = GT clean video
M = mask, 1 = hole

err_prior = mean_abs(P - x_GT) over RGB
C_prior = exp(-alpha * normalize(err_prior))
alpha = 5.0
```

Interpretation:

- high `C_prior`: ProPainter prior is close to GT and should be preserved;
- low `C_prior`: prior is unreliable and diffusion should be allowed to
  generate using GT / context / preference.

Regions:

```text
M_reliable = M * C_prior
M_generate = M * (1 - C_prior)
boundary_outer = dilate(M) - M
```

The implementation lives in:

```text
exp16_prior_confidence_gated_dpo/code/exp16_loss.py
```

## Current Numeric Status

Tensor-level sanity is implemented in:

```text
exp16_prior_confidence_gated_dpo/code/preflight_exp16.py
```

Real dataset statistics are pending because the training manifest does not yet
have a verified real ProPainter prior cache.

Required future dataset stats:

```text
prior_conf_mean
prior_conf_p10
prior_conf_p50
prior_conf_p90
reliable_area_ratio
generate_area_ratio
mask_area_ratio
boundary_area_ratio
```

## Mask Convention

Current DiffuEraser DPO dataset uses:

```text
brushnet mask: 0 = hole, 1 = known
hole mask for Exp16: M = 1 - brushnet_mask
```

Nearest-neighbor resize is required for masks. Confidence can be resized with
bilinear interpolation because it is continuous.

## Status

`PENDING_REAL_PRIOR_CACHE`

No training should start until this audit is updated with real prior-cache
statistics from PAI.

