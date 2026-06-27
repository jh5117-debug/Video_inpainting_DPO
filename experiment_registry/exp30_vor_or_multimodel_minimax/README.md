# Exp30 Registry

Exp30 tracks VOR-OR multi-model medium-hard pool construction, MiniMax
quality-positive adapter gates, DiffuEraser VOR-OR micro preparation, and
paper-ready three-backbone evidence boundaries.

Current gate sequence:

1. Recover full VOR valid triplet index.
2. Build source-pool v2 from the full index.
3. Run multi-model OR smoke16 only if source-pool v2 passes.
4. Run limited Gate64 only if smoke16 passes.
5. Run MiniMax 10-step adapter gate only if Gate64 provides scene-disjoint
   train/heldout data.

EffectErase remains an OR strong baseline / diagnostic, not an adapter target.

## 2026-06-27 Continuation V3

Current gate remains `MULTIMODEL_OR_SMOKE16_V2_BLOCKED`. Continuation v3 starts
with failure analysis and controlled-corruption calibration planning before any
new candidate generation. Direct Gate64, MiniMax adapter gates, long training,
and universal-adapter claims remain forbidden until the preregistered smoke
gates pass.

Smoke16 v2 failure analysis is complete. It confirms a quality-yield failure:
controlled corruption needs calibrated temporal/local profiles, and MiniMax
must remain one family in a multi-model pool rather than the sole generator.

Controlled-corruption v3 planning is locked. It caps smoke16 v3 controlled
candidates at 24 and keeps controlled corruption as a fallback/data-source
only.

Smoke16 v3 has now passed as a multi-model smoke gate, driven primarily by
controlled corruption v3. Smoke32 v3 is preregistered with 16 new disjoint
source groups and BLENDER/REAL 8/8 balance. Gate64 and all adapter/training
gates remain stopped until Smoke32 materialization, generation, metrics, and
visual review pass.


## 2026-06-27 Smoke32 V3

Smoke32 v3 passed the preregistered confirmation gate with 64/64 technical-valid candidates, 14 usable candidates, 10/16 usable source groups, and usable contributions from controlled corruption v3, MiniMax official v3, and ProPainter. DiffuEraser no-PCM was technical-valid but contributed 0 usable candidates. This unlocks limited Gate64 pool preparation only; adapter gates remain locked until Gate64 produces scene-disjoint train/heldout data.

## 2026-06-27 Gate64 V3 Preregistration

Gate64 v3 source rows are locked: 64 scene-disjoint rows, BLENDER/REAL 32/32. This is metadata-only and unlocks selective extraction/materialization, not adapter training.

## 2026-06-27 Gate64 V3 Repair

Gate64 v3 source manifest was repaired before inference after 9 empty-mask materialization failures. Final manifest remains 64 scene groups and BLENDER/REAL 32/32; model generation is still pending final materialization.
