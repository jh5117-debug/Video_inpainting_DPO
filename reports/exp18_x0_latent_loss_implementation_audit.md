# Exp18 X0 Latent Loss Implementation Audit

Status:

```text
IMPLEMENTED_NOT_YET_PREFLIGHTED_ON_PAI
```

Implemented files:

```text
exp18_multiframe_propagation_gated_dpo/code/exp18_loss.py
exp18_multiframe_propagation_gated_dpo/code/train_exp18_stage1.py
```

Required tensors:

```text
z_hat_x0 = predicted clean latent from real policy model output
z_prop = VAE(P_prop)
z_gt = VAE(GT)
```

Implemented losses:

```text
L_prop = sum(M * C_prop * |z_hat_x0 - z_prop|) / (sum(M * C_prop) + eps)
L_gen = sum(M_gen * |z_hat_x0 - z_gt|) / (sum(M_gen) + eps)
L_boundary = sum(B_outer * |z_hat_x0 - z_gt|) / (sum(B_outer) + eps)
```

Total:

```text
L_total = L_base + lambda_prop * L_prop + lambda_gen * L_gen + lambda_boundary_extra * L_boundary
```

Defaults:

```text
lambda_prop = 0.1
lambda_gen = 0.05
lambda_boundary_extra = 0.1
```

Guardrails:

- no frozen-reference epsilon proxy;
- no generated loser as propagation prior;
- no ProPainter prior directly substituted for `P_prop`;
- no GT-error confidence for non-oracle variants.

The implementation still needs PAI preflight before training can be trusted.

