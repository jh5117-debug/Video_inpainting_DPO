# Exp27 Diffusion-SDPO Paper Reading

Paper: Diffusion-SDPO: Safeguarded Direct Preference Optimization for Diffusion Models  
arXiv: 2511.03317  
Official code: `https://github.com/AIDC-AI/Diffusion-SDPO`

## Core Reading

SDPO addresses a failure mode we have repeatedly seen in DPO diagnostics: preference margin can improve by degrading the rejected sample while the preferred sample does not improve or may even get worse. SDPO derives a first-order safety condition for winner loss under a gradient update and scales the loser branch gradient by an adaptive safe coefficient.

Official implementation details:

- Key function: `get_adaptive_lose_l_scale` in `train.py`.
- It computes output-space winner and loser MSE gradients.
- If the winner/loser gradient dot product is non-positive, loser scaling is safe and set to 1.
- If the dot product is positive, safe lambda is `(1-mu) * ||g_w||^2 / dot(g_w,g_l)`, then clamped.
- The loser loss value remains in the objective, but its gradient is scaled through a detach trick.

## Mathematical Boundary

The guarantee is first-order and approximate. It applies to the simplified preference objective whose winner and loser gradients define the safe condition. It does not automatically apply to our LoVI loss because LoVI adds:

- region-weighted residuals;
- reference-normalized log-ratio gaps;
- clipped loser gaps;
- reduced loser weight;
- winner absolute regularization;
- winner gap ReLU.

Therefore, an Exp27 variant that adds SDPO-like scaling to LoVI must be named `heuristic` unless Reviewer B derives a new region/log-ratio-safe theorem.

## Parity Status

Exp27 CPU parity passed for the scalar safe-lambda helper:

- official scalar: `1.0`
- Exp27 scalar: `1.0`
- max_abs_diff: `0.0`

This only verifies the toy helper, not full DiffuEraser batch parity.
