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
