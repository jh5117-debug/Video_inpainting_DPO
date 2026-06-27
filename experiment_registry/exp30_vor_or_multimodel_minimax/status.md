# Exp30 Status

Current status: `EXP30_READBACK_COMPLETED`

Exp30 starts from Exp29 and is scoped to:

- VOR-OR multi-model medium-hard preference pool construction;
- MiniMax true-adapter quality-positive micro gates;
- DiffuEraser VOR-OR Stage1/Stage2 micro validation;
- paper-ready three-backbone evidence planning.

## 2026-06-27 Readback

- Branch: `research/exp30-vor-or-multimodel-minimax-adapter-20260627`.
- Start HEAD: `6bc6c67c60b5cf2fe8d937ffd1e1d88a4684991c`.
- Base: `origin/research/exp29-minimax-effecterase-adapter-feasibility-20260626`.
- EffectErase status read from Exp29:
  `EFFECTERASE_OR_BASELINE_READY` and
  `EFFECTERASE_BASELINE_ONLY_FOR_NOW`.
- MiniMax status read from Exp29:
  `MINIMAX_EXPANDED_DATA_YIELD_INSUFFICIENT`.
- VideoPainter status read from Exp26:
  `VIDEOPAINTER_SHADOWDEV_CONFIRMED`, with external DAVIS-derived validation
  not confirmed.
- Missing files in this branch were recorded rather than fabricated:
  `PRD/47_exp25_vor_or_preference_data.md`,
  `experiment_registry/exp25_vor_or_preference_data/status.md`,
  `reports/exp25_gate32_yield_review_20260624.md`, and
  `reports/exp25_diffueraser_or_root_cause_matrix_v2.md`.
- PAI GPU readback found no compute processes. Left CLI runtime locks still
  reserve GPU1/GPU2/GPU3/GPU4.
- No GPU task, inference, training, RC-FPO, or left-side action was launched by
  readback.

