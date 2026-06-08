# Exp10: Region-Local DPO

Experiment name:

```text
exp10_region_local_dpo_s1s2_2000_davis_pai
```

Goal: inherit Exp9 normalized-gap DPO and move the MSE computation from
full-video average to region-local weighted MSE.

Regions:

```text
mask_region_weight = 1.0
boundary_region_weight = 0.5
outside_region_weight = 0.05
```

Weighted MSE:

```text
m = sum(region_weight_map * mse_map) / (sum(region_weight_map) + eps)
```

Do not use `mean(region_weight_map * mse_map)`.

Mask handling:

- Use nearest interpolation if mask resolution differs from latent/noise tensor.
- Boundary ring is derived from maxpool-style dilation/erosion of the binary
  mask.
- Record raw and normalized win/lose gap statistics at every diag interval.

Required extra outputs:

```text
dpo_gap_trace.csv
dpo_gap_samples.jsonl.gz
mask_region_mse
boundary_region_mse
outside_region_mse
mask_area_ratio
boundary_area_ratio
outside_area_ratio
region_weight_sum
```

Run policy:

- Prepared but not launched by default.
- New PAI runs use `NFRAMES=24`.
- DAVIS validation uses `DAVIS_VIDEO_LENGTH=24`; 16-frame validation is invalid
  because DiffuEraser/ProPainter requires effective duration greater than 22.
- Launch only with explicit `RUN_EXPERIMENTS=exp10` or a sequential list.
