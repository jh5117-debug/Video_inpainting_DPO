# Exp26 Qualitative Summary

## 2026-06-25 Gate64 Final Temporal Review

All 64 Gate64 samples were checked with 16-frame temporal strips and existing
dense evidence/crop sheets. The review found no new systematic frame-order
failure, first-frame failure, or global collapse. Rejected rows remain rejected
for being too-close or trivial-bad; the 55 eligible rows have finite local
defects suitable for preference-data construction.

Primary-32 uses hard-composited VideoPainter outputs as the training loser.
Raw outputs remain diagnostics for outside-damage and ablation.

## 2026-06-25 Primary-32 50-Step Gate

Manual review opened dense temporal evidence pages and crop pages covering all
`32/32` step50 search-dev rows.

Reviewer conclusion:

- no global black/purple collapse;
- no frame-order mismatch;
- no first-frame failure;
- no systematic outside damage in the comp protocol;
- no gate-blocking flicker/ghosting pattern;
- local failures remain in mask/affected regions, especially green/purple
  residual patches, local blur, texture discontinuity, and boundary tinting in
  water/foliage/grass cases.

The 50-step result is visually acceptable for the pre-registered micro gate,
but it is not a final held-out benchmark result and should not be labelled
`SCIENTIFIC_POSITIVE`.
