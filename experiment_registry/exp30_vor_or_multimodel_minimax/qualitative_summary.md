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
