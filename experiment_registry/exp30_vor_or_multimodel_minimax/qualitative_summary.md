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

## 2026-06-27 Continuation V3 Readback

No new videos were generated or reviewed in this milestone. The readback
classifies v2 as a quality-yield failure, not a source/materialization failure:
controlled corruption needs softer and more temporally coherent medium-hard
profiles, while MiniMax needs supplemental generator families rather than being
treated as a sole candidate source. DiffuEraser and ProPainter require verified
stack smoke before they can be enabled in smoke16 v3.

## 2026-06-27 Smoke16 V2 Failure Analysis

Continuation v3 opened 4 controlled-corruption overview pages and 4 MiniMax
review pages, covering all 32 candidates. Controlled corruption often shows
hard mask-local residuals or frame-to-frame inconsistent local texture while
leaving the outside clean. MiniMax shows a few usable medium-hard examples, but
many rows have outside damage, black/smudged local artifacts, residual object or
effect content, or temporal instability. The qualitative conclusion supports a
preregistered v3 repair rather than direct Gate64.

## 2026-06-27 Controlled Corruption V3 Plan

The qualitative target for v3 is bounded local defect: visible object/effect or
boundary/texture imperfection in the task region while preserving far outside
content and avoiding frame-wise flicker. CC-v3-D is deliberately held out of the
smoke16 v3 run to avoid profile over-search.

## 2026-06-27 DiffuEraser / ProPainter Candidate Audit

No new videos were generated or reviewed. Qualitative readiness is split:

- DiffuEraser has strong Exp25 visual/metric evidence for no-PCM
  ProPainter-prior OR losers, but Exp30 has not yet ported the exact wrapper
  identity. It cannot be enabled from the legacy Exp30 wrapper.
- ProPainter has a valid PAI weight directory and wrapper, so it can enter a
  two-sample smoke after the Exp30 PAI runtime snapshot is created.

Smoke16 v3 remains pending until smoke2 verifies the generator outputs in the
current Exp30 runtime.

## 2026-06-27 Verified Generator Smoke2

Codex opened 4/4 review sheets:

- ProPainter and DiffuEraser on `BLENDER_FOREST006_00001` were too close to
  the clean winner; they are not useful medium-hard losers.
- ProPainter on `BLENDER_FOREST007_00001` removed the car but introduced a
  dark smear/grid-like blur and local seam, making it hard-but-plausible.
- DiffuEraser no-PCM on `BLENDER_FOREST007_00001` removed the car with smoky
  local residual and boundary/texture defects, making it medium-hard.

The generator wiring is now visually smoke-tested, but the sample count is too
small for pool readiness. Smoke16 v3 still needs to run and pass.

## 2026-06-27 Smoke32 V3 Preregistration

No videos were generated or reviewed. The preregistration only locks the
confirmation sources for the next Smoke32 validation run. It therefore carries
no qualitative pass, no data-ready claim, and no Gate64 unlock by itself.

## 2026-06-27 Smoke32 V3 Materialization

Source evidence was generated for all 16 Smoke32 confirmation rows:
condition, winner, mask overlay, side-by-side mp4s, and 16-frame temporal
strips under the PAI Smoke32 materialization output root. This confirms the
locked sources can be decoded and prepared for candidate generation.

No OR candidate videos were generated in this milestone, so there is still no
Smoke32 qualitative pass, no pool-ready claim, and no Gate64 unlock.


## 2026-06-27 Smoke32 V3 Visual Review

Codex opened the Smoke32 v3 controlled, MiniMax, ProPainter, and DiffuEraser review pages before final classification. Controlled corruption v3 provided the strongest usable source coverage but still produced 8 trivial-bad cases from over-strong or temporally unstable local defects. MiniMax contributed three usable candidates but was mostly trivial-bad due to outside damage, flicker, or over-strong local changes. ProPainter contributed three usable candidates but most rows were either too clean/strong or artifacted. DiffuEraser no-PCM showed systematic white/blur artifacts or outside damage in this split and contributed no usable candidates.

The qualitative conclusion matches the low-margin pass: proceed only to limited Gate64 pool preparation, with no adapter-training claim yet.

## 2026-06-27 Gate64 V3 Preregistration

No video review was performed because no videos were extracted or generated. Qualitative review remains pending materialization and candidate generation.
