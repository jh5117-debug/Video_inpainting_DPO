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

## 2026-06-26 Shadow-Dev Visual Review

Review covered all `32/32` independent shadow-dev rows with anonymous pages
followed by informed pages.

Final classification:

- Step50 clearly better: `12`
- Step50 slightly better: `13`
- tie: `3`
- Step0 better or Step50 new artifact: `4`
- Step50 new artifact count: `3`

Positive pattern: Step50 often removes Step0 translucent oval/ring residuals
and produces cleaner local object removal. Negative pattern: water, grass, and
foliage cases can still develop green/purple local patches or color smears.

No systematic frame-order failure, first-frame failure, unexpected winner
leakage, comp mask overwrite, or global collapse was observed. Visual evidence
supports `VIDEOPAINTER_SHADOWDEV_CONFIRMED` for the current VOR-BG
distribution, while leaving external cross-dataset validation as the next
scientific requirement.

## 2026-06-26 Post-Confirmation Sanity Audit

The post-confirmation readback did not change the qualitative conclusion.
Shadow-dev review evidence remains `32/32` complete: Step50 better in `25`
rows, tied in `3`, and worse or newly artifacted in `4`. The observed benefit
is local object-removal cleanup rather than a universal failure-free generator:
water, grass, and foliage cases still account for the main visible artifacts.

No new VideoPainter training was run, and no Step10/30/50 reselection was made.
External clean-source validation is the next required qualitative check.

## 2026-06-26 External 49F Inventory

The external source pool is DAVIS-derived clean `gt_frames`, not visualization
`comparison.mp4` files. The selected 32 rows cover animals, people, vehicles,
water, foliage/grass, urban scenes, and high-motion cases. No external videos
have been generated or reviewed yet; visual review remains pending until the
pre-registered Step0/Step50 outputs exist.

## 2026-06-26 External Validation Preregistration

The external validation visual protocol is locked but no external model videos
exist yet. Each row now has an exact 49-frame materialized source directory and
a deterministic mixed moving BR mask sequence. Video review remains pending and
must cover Step0 and Step50 raw/comp outputs after inference, including
frame1-48 local artifacts, raw outside preservation, water/grass/foliage
failure modes, and any new Step50 artifacts.

## 2026-06-26 External Validation Visual Review

External visual review covered `32/32` preregistered DAVIS-derived rows using
anonymous A/B pages followed by informed Step0/Step50 pages and local crop
sheets. Final classification: Step50 slightly better in `3`, tie in `5`, Step0
better in `24`, and Step50 new local artifact in `29` rows.

The dominant Step50 failure pattern is a dark/green blob or color smear in the
masked/affected region, especially for water, grass, foliage, fast motion,
animals, people, and vehicles. This agrees with the external strict-mask PSNR
and LPIPS failures. No new evidence of frame-order failure, first-frame error,
or unexpected winner leakage was found.
