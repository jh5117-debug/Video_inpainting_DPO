# Exp30 VOR-OR Multi-Model MiniMax

Status: `EXP30_READBACK_COMPLETED`

Exp30 is an isolated track for VOR-OR multi-model medium-hard preference pool
construction, MiniMax true-adapter micro validation, DiffuEraser VOR-OR
Stage1/Stage2 micro validation, and paper-ready three-backbone evidence
planning.

## Ground Rules

- Do not continue EffectErase as an adapter attempt.
- Keep EffectErase as OR strong baseline / diagnostic only.
- Do not use VOR-Eval for training, loser mining, threshold selection, or
  checkpoint selection.
- Do not start 500/1000/2000-step long training.
- Do not start RC-FPO.
- Do not modify Exp1-Exp28, shared trainer code, or `inference/metrics.py`.
- Do not write universal-adapter, all-models-supported, final-SOTA, or
  top-conference-novelty-confirmed claims.

## 2026-06-27 Readback

- Branch:
  `research/exp30-vor-or-multimodel-minimax-adapter-20260627`.
- Start HEAD:
  `6bc6c67c60b5cf2fe8d937ffd1e1d88a4684991c`.
- Base branch:
  `origin/research/exp29-minimax-effecterase-adapter-feasibility-20260626`.
- EffectErase:
  `EFFECTERASE_OR_BASELINE_READY` and
  `EFFECTERASE_BASELINE_ONLY_FOR_NOW`.
- MiniMax:
  `MINIMAX_EXPANDED_DATA_YIELD_INSUFFICIENT`.
- VideoPainter:
  `VIDEOPAINTER_SHADOWDEV_CONFIRMED`; external DAVIS-derived validation is not
  confirmed.
- DiffuEraser:
  existing BR success is part of project lineage; Exp30 still needs VOR-OR
  micro evidence before using it as VOR-OR evidence in the paper.
- Left CLI:
  checked read-only; no signals or file mutations. Runtime locks reserve
  GPU1/GPU2/GPU3/GPU4.
- GPU:
  PAI reported no compute processes during readback.

Readback report:

- `reports/exp30_readback.md`

No GPU task, inference, training, RC-FPO, or left-side action was launched by
this readback milestone.

## 2026-06-27 Three-Backbone Paper Positioning

- Status: `EXP30_THREE_BACKBONE_POSITIONING_LOCKED`
- DiffuEraser: primary original backbone plus VOR-OR adapter baseline to
  validate with Stage1/Stage2 micro.
- VideoPainter: second backbone with VOR-BG BR/inpainting adapter evidence;
  external DAVIS-derived validation remains not confirmed.
- MiniMax: Wan2.1 / DiT / flow-matching third-backbone candidate with
  plumbing-positive gates but no quality-positive heldout micro yet.
- EffectErase: OR strong baseline / diagnostic / upper reference only.

Allowed claim:

- Model-specific backend adapters with cross-backbone evidence currently from
  DiffuEraser plus VideoPainter.

Forbidden claims:

- Universal adapter.
- All models supported.
- EffectErase adapter-ready.
- MiniMax quality-positive before Exp30 heldout micro.
- Final SOTA or top-conference novelty confirmed.

Reports:

- `reports/exp30_three_backbone_paper_positioning.md`
- `reports/exp30_three_backbone_paper_positioning.csv`
