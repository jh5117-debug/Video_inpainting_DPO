# Exp11 Status

status: invalid_mislabeled_blocked
updated_at: 2026-06-11

Truth-audit result:

- The isolated proxy code under `exp11_flow_prior_consistency_dpo/code/` is not
  a valid flow-prior consistency DPO implementation.
- `L_prior` uses frozen SFT/ref epsilon prediction as target, not ProPainter
  prior frames/tensors.
- `L_flow` is an adjacent-frame residual proxy, not a flow-warp consistency
  loss.
- Old Exp11 outputs must be treated as invalid / mislabeled and not as method
  results.

Current launcher behavior:

- `exp11_flow_prior_consistency_dpo/scripts/launch_exp11_pai.sh` writes a
  blocked audit and exits before training.
- Do not set `EXP11_ENABLE_TRAINING=1` until real train-time ProPainter-prior
  and/or flow targets are implemented and re-audited.
