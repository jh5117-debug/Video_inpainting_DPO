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
exp16_prior_confidence_gated_dpo/code/exp16_loss.py
```

Real dataset statistics were computed from the PAI limit=100 ProPainter prior
cache:

```text
manifest = /mnt/nas/hj/H20_Video_inpainting_DPO/data/cache/exp16_propainter_prior_cache_limit100/manifests/exp16_train_with_prior_limit100.jsonl
rows_ok = 100
rows_failed = 0
confidence_mode = gt_error
confidence_alpha = 5.0
```

| metric | mean |
|---|---:|
| prior_conf_mean | 0.656014 |
| prior_conf_p10 | 0.239536 |
| prior_conf_p50 | 0.725268 |
| prior_conf_p90 | 0.940553 |
| reliable_area_ratio | 0.256022 |
| generate_area_ratio | 0.254534 |
| mask_area_ratio | 0.256022 |
| boundary_area_ratio | 0.008180 |

## Mask Convention

Current DiffuEraser DPO dataset uses:

```text
brushnet mask: 0 = hole, 1 = known
hole mask for Exp16: M = 1 - brushnet_mask
```

Nearest-neighbor resize is required for masks. Confidence can be resized with
bilinear interpolation because it is continuous.

## Status

`PASSED_LIMIT100_CACHE_AUDIT`

Stage1 500 small gate has run. Do not start Stage2/full training until the
Stage1 500 diagnostics and a small visual/validation check are reviewed.
