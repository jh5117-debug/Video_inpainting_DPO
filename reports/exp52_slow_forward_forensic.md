# Exp52 Slow-Forward Forensic

Status: `VOID_CACHE_BLOCKED`

## Profile

- Device: `cuda:0`
- Runtime: 657.10 sec
- Frames after official patch-size adjustment: 13
- Peak allocated VRAM: 26.375 GiB
- Peak reserved VRAM: 27.578 GiB
- Row0 loss: 0.6961396336555481
- Grad finite: True
- Optimizer step: no

Slow stages are recorded in `reports/exp52_slow_forward_profile.csv`.

## Interpretation

The Exp51 blocker came from doing VAE/text/quadmask encoding and base-policy reference forwards inside each recipe process. Exp52 fixes that by materializing CPU tensor caches for train4/heldout4 under Q0/Q1/Q2/Q3, including fixed noise, timestep, text embeddings, VAE latents, inpaint latents, region weights, scheduler target, and reference predictions/losses.


## Parity Explanation

Status updated to `VOID_CACHE_PARITY_EXPLAINED`.

The cache itself is usable: cached and uncached multi-dimensional policy loss inputs matched exactly on row0 after deterministic VAE seeding. The observed max diff came from the parity helper downcasting cached 0-dim reference-loss scalars to bf16 (`0.0705375 -> 0.0703125`). The Exp52 helper now preserves scalar loss precision during cache load.
