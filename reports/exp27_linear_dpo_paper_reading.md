# Exp27 Linear-DPO Paper Reading

Paper: Linear-DPO: Linear Direct Preference Optimization for Diffusion and Flow-Matching Generative Models  
arXiv: 2605.21123  
Official code: `https://github.com/Whynot0101/Linear-DPO`

## Core Reading

Linear-DPO argues that sigmoid/logistic DPO weighting can pseudo-converge too early in regression-style diffusion or flow-matching training. It replaces the sigmoid utility with a clipped linear utility and optionally uses an EMA reference.

Official implementation details:

- Utility ratio: `0.2 * beta_dpo * (model_diff - ref_diff) + 0.5`.
- Official code clamps to `[eta, 1-eta]`, while paper text describes the upper clip as `1`.
- Utility is computed under `torch.no_grad()` and multiplied by `(model_losses_w - model_losses_l)`.
- EMA reference is updated after optimizer step when `--use_ema_ref` is enabled.
- SD and SD3 paths both implement linear utility; run scripts are not perfectly uniform.

## Relevance To Exp27

Linear-DPO threatens any claim that Exp27 invented sustained/non-saturating DPO utility, EMA reference DPO, or unified diffusion/flow DPO. Exp27 can still use Linear-DPO as an exact baseline or ablation, especially for VideoPainter/flow-matching style backbones.

We must split:

- `Linear-Frozen`: linear utility with frozen reference.
- `Linear-EMA`: linear utility with EMA reference.

Combining them into one experiment would confound utility shape and reference drift.

## Parity Status

Exp27 CPU parity passed:

- loss: `0.09172474592924118`
- EMA update max_abs_diff: `0.0`
- gradients finite: `true`

This verifies the toy utility and EMA update. Full DiffuEraser/VideoPainter DDP save-resume parity remains pending.
