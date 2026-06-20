# Exp23: Two-Stage Pool Morphology Sweep

Status: `BLOCKED_GPU4_7_NOT_AVAILABLE`

Branch: `research/exp23-two-stage-pool-morphology-sweep`

HAL worktree: `/home/hj/H20_Video_inpainting_DPO_exp23_pool_sweep`

PAI worktree target: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp23_pool_sweep`

## Motivation

Exp20 found that broad image-space boundary tuning produced small search-dev signals but did not survive multi-seed shadow-dev confirmation. Exp23 restarts this line with a more faithful DiffuEraser setting:

- full Stage1 2000 + Stage2 2000 training;
- fresh Exp11 twin for every candidate/seed;
- DAVIS50 explicitly used as `DAVIS50_MODEL_SELECTION_BENCHMARK`;
- independent Stage1/Stage2 inner and outer max-pool morphology.

## Morphology Definition

Mask convention: `M=1` means hole/unknown.

The public geometry parameters are:

- `pool_grid_scale`
- `inner_pool_steps`
- `outer_pool_steps`
- `inner_weight`
- `outer_weight`

No candidate is named by image-space pixel radius.

One pool step is:

```python
F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
```

Regions:

- `mask_core = erode_steps(M, inner_pool_steps)`
- `inner_ring = M - mask_core`
- `outer_ring = dilate_steps(M, outer_pool_steps) - M`
- `far_outside = 1 - mask_core - inner_ring - outer_ring`

`pool_grid_scale=1` is legacy latent/loss-grid pooling. `pool_grid_scale=2/4` upsamples the binary latent mask using nearest, builds the partition on the finer pool grid, then area-pools back to loss grid.

Exp11 exact setting:

- `pool_grid_scale=1`
- Stage1: `inner=0`, `outer=1`, `inner_weight=0`, `outer_weight=0.75`
- Stage2: `inner=0`, `outer=1`, `inner_weight=0`, `outer_weight=0.75`
- `mask_core_weight=1.0`
- `outside_weight=0.05`

## Aggregation

Two aggregation modes are implemented:

- `legacy_weighted`
- `region_balanced`

The first parity target is `legacy_weighted` with Exp11 exact morphology.

## Data

Training manifest target:

`/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/exp09_10_11_youtubevos_gtwin_d3comp_pai/manifests/selected_primary_comp.gtwin.pai_paths.jsonl`

This path must be revalidated on PAI before training.

DAVIS50 target:

`/mnt/workspace/hj/nas_hj/data/external/davis_432_240`

## Current Implementation

Implemented:

- `pool_morphology.py`
- `region_aggregation.py`
- `process_title.py`
- initial morphology and process-title tests
- GPU4-7 release audit

Not yet launched:

- Stage1/Stage2 trainer copy;
- Exp11 full loss parity;
- PAI DDP training;
- DAVIS50 paired evaluation.

## GPU Status

GPU4-7 audit result: `BLOCKED_GPU4_7_NOT_AVAILABLE`.

GPUs 4-6 are occupied by active `/mnt/workspace/xiaoqi/multigen/...` training processes. GPU7 has 58GB NVML memory with no live `/proc` PID. No process was killed.

See:

- `reports/exp23_gpu4_7_release_audit.md`
- `reports/exp23_gpu4_7_release_audit.csv`

## Next Gate

Do not start Exp23 training until GPU4-7 are safely available or the user/admin confirms ownership and termination permission for the listed processes.
