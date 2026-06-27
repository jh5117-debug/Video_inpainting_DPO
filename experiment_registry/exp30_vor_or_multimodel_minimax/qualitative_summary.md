# Exp30 Qualitative Summary

No Exp30 videos have been generated or reviewed yet.

Readback imported the following qualitative constraints:

- EffectErase official 81F outputs are strong OR baseline / diagnostic outputs,
  but too strong/VOR-confounded for primary on-policy loser claims.
- MiniMax is plumbing-positive but data-yield-limited with MiniMax-only loser
  mining.
- VideoPainter is a second adapter backbone for VOR-BG/BR-style evidence, not a
  standard VOR-OR adapter.
- DiffuEraser needs VOR-OR micro evidence if the paper frames VOR-OR and
  MiniMax together.

## 2026-06-27 Three-Backbone Positioning

No new videos were generated. The qualitative role split is:

- DiffuEraser: original backbone and VOR-OR micro target.
- VideoPainter: second backbone for VOR-BG BR/inpainting evidence.
- MiniMax: flow-style third-backbone candidate pending quality-positive micro.
- EffectErase: OR strong baseline / diagnostic only.

## 2026-06-27 VOR-OR Source Pool Audit

Codex opened all 10 batch preview pages covering 80 source rows. The available
rows show aligned condition/winner/mask strips with non-empty masks and visible
affected regions. However, they are dominated by REAL scenes and do not provide
the requested pool size or reserve. No source-pool-ready, data-ready, smoke, or
training claim is supported.

## 2026-06-27 Continuation V2

No new video generation or qualitative promotion occurred. The next work is
full-index recovery and source-pool v2 sampling before any smoke or training.

## 2026-06-27 Full VOR Index Recovery

No videos were extracted or reviewed in this milestone. The milestone verifies
metadata-only VOR-OR triplet identity and unlocks source-pool v2 sampling.

## 2026-06-27 Source-Pool V2

No videos were extracted or reviewed. Source-pool v2 is metadata-ready only;
visual preview status is `metadata_only_visual_preview_pending`. It unlocks
smoke16 but not data-ready, pool-ready after visual generation, or adapter
claims.

## 2026-06-27 Smoke16 V2 Preregistration

No video review has occurred for smoke16 v2 yet. The row lock is metadata-only
and preserves the rule that `MULTIMODEL_OR_SMOKE16_V2_PASS` requires generated
videos, metrics, per-sample review sheets, and visual inspection.

## 2026-06-27 Smoke16 V2 Pre-Inference Repair

No model candidate videos have been generated or reviewed yet. The repair only
removes technical-invalid source rows discovered during materialization: one
short decoded row and two empty-mask rows. The final manifest remains balanced
at BLENDER/REAL 8/8, but qualitative smoke status is still pending candidate
generation, per-video review sheets, and Codex visual inspection.

## 2026-06-27 Smoke16 V2 Final Materialization

Source materialization is now technically valid for 16/16 rows. The
materializer wrote source evidence strips for condition/winner/mask and rejected
empty-mask rows via the new guard. No OR candidate videos have been generated
yet, so qualitative smoke status remains pending model/controlled-corruption
outputs and per-video visual inspection.

## 2026-06-27 Multi-Model OR Smoke16 V2

Codex opened 32/32 temporal evidence strips: all 16 controlled-corruption
candidates and all 16 MiniMax official candidates. Controlled corruption
preserved outside pixels but often produced hard local residuals or temporal
artifacts. MiniMax produced a few usable local-defect examples, but most
candidates were too close, retained the object/effect, or had black/smudged
local artifacts. The qualitative result matches the metric/classification
failure, so Gate64 and training remain stopped.
