# Exp52 Cache Plan

Status: `VOID_CACHE_BLOCKED`

Cache root:

`/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp52_void_winner_preserving_allgpu/cache/tensor_cache`

Cached artifacts:

- rgb_full-derived condition latents
- winner VAE latents and v-prediction targets
- loser VAE latents and v-prediction targets
- quadmask/inpaint latents
- prompt tokens/text embeddings
- fixed timestep `500`
- fixed noise seeds beginning at `1234`
- object/affected/boundary/outside recipe weights
- reference predictions/losses for R1 row-compatible parity
- source path SHA256 metadata

No VOR-Eval, hard comp, optimizer step, or official VOID source edit was used.


## Parity Explanation

Status updated to `VOID_CACHE_PARITY_EXPLAINED`.

The cache itself is usable: cached and uncached multi-dimensional policy loss inputs matched exactly on row0 after deterministic VAE seeding. The observed max diff came from the parity helper downcasting cached 0-dim reference-loss scalars to bf16 (`0.0705375 -> 0.0703125`). The Exp52 helper now preserves scalar loss precision during cache load.
