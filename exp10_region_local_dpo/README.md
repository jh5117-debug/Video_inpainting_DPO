# Exp10 Region-Local DPO

Experiment name: `exp10_region_local_dpo_s1s2_2000_davis_pai`

Purpose: build on Exp9 normalized-gap DPO, but compute winner/loser/reference MSE as region-weighted inpainting MSE. This targets the mask and boundary regions instead of letting full-frame background dominate the objective.

Region weights:

- mask: `1.0`
- boundary: `0.5`
- outside: `0.05`

The MSE is normalized by `sum(region_weight_map)`, not `mean(weight * mse)`.
