# Exp12 Adaptive Outer Boundary

Purpose: test whether batch-level adaptive normalization still helps when the
region-local loss uses the current best boundary setting from Exp11:
`boundary_mode=outer`, `boundary_weight=0.75`.

Exp9 log-ratio is reference-level normalization:

`g = log((policy_mse + eps) / (reference_mse + eps))`

Exp12 adds batch z-score normalization after log-ratio:

`g_adapt = (g - mean(g_batch)) / (std(g_batch) + eps)`

This branch is isolated from the original Exp12 implementation. The original
Exp12 used `boundary_mode=exp10_default` and `boundary_weight=0.5`; this branch
uses the Exp11 outer-boundary setting only for comparison against the current
best `Exp11 outer b0.75 S2` result.

Decision rule:

- If `Exp12 adaptive + outer b0.75` beats `Exp11 outer b0.75 S2` under the
  fixed DAVIS50 raw6 hard-comp protocol, keep this branch as the next candidate.
- If it does not beat `Exp11 outer b0.75 S2`, keep the original Exp12
  normalization result as an ablation and do not replace the current best.
