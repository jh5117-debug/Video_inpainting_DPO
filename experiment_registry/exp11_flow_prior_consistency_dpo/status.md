# Status

status: invalid_mislabeled_blocked
updated_at: 2026-06-11

Truth-audit result:

- The isolated proxy code under `exp11_flow_prior_consistency_dpo/code/` does
  not implement real train-time ProPainter-prior consistency.
- `L_prior` uses frozen-ref epsilon prediction, not prior frames/tensors.
- `L_flow` is an adjacent-frame residual proxy, not optical-flow warp
  consistency.
- Old Exp11 outputs are invalid / mislabeled and should not be used as method
  results.

Launcher behavior:

- `exp11_flow_prior_consistency_dpo/scripts/launch_exp11_pai.sh` writes the
  blocked audit and exits nonzero before training.
