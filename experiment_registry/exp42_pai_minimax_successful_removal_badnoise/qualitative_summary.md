# Exp42 Qualitative Summary

No Exp42 video review has been run yet.

Readback qualitative state:

- Previous MiniMax checkpoints can move outputs, but existing direct SFT/DPO
  recipes either tie visually or introduce fogging, over-erasure,
  boundary damage, or outside/global drift.
- Exp42 will not promote any milestone without actual video evidence,
  temporal strips, per-video review rows, and aggregate metrics.

## 2026-06-29 Successful-Removal Mining Review

Codex opened all compact 16-frame temporal review pages generated for the
selected candidates:

- success pages: `26`
- failure pages: `40`
- selected success rows represented: `52`
- selected failure rows represented: `80`

Qualitative findings:

- Successful MiniMax raw removals do exist under the official executable
  protocol. Clean signals appeared in forest/grass/water/indoor/staircase
  scenes.
- Success rows are heavily clustered by source and seed. Row count therefore
  overstates diversity.
- Failure rows are technical-valid and mostly coherent; no systematic
  black/purple collapse or wrapper failure was observed.
- The selected failure pool is not label-pure. Many auto-failure rows are
  borderline clean or metric/boundary driven at compact-strip scale.
- Success/failure same-source overlap is only `7` scene groups, too low for
  the preregistered `>=24` bad-noise v3 pair gate.

Conclusion: `MINIMAX_SUCCESSFUL_REMOVAL_POOL_WEAK`. Use this as evidence that
MiniMax has seed-dependent successful-removal signal, not as a ready
preference/training pool.
