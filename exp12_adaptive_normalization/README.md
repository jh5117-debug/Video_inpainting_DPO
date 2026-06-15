# Exp12 Adaptive Normalization

Purpose: test whether batch-level adaptive normalization improves Exp10-style region-local log-ratio DPO.

Exp9 log-ratio is reference-level normalization:

`g = log((policy_mse + eps) / (reference_mse + eps))`

Exp12 adds batch z-score normalization after log-ratio:

`g_adapt = (g - mean(g_batch)) / (std(g_batch) + eps)`

Only the batch z-score variant is launched by default. A timestep-level running-normalization config is prepared as a TODO and is not launched.
