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

## 2026-06-27 VOR-OR Source Pool Audit

- Status: `VOR_OR_SOURCE_POOL_BLOCKED`
- Method: existing exact extraction caches only; no VOR tar archive rescan.
- Discovered extracted triplets: 192.
- Candidate scene groups after previous MiniMax/EffectErase exclusions: 80.
- Source manifest rows: 80.
- Reserve rows: 0.
- Source type counts: REAL 71, BLENDER 9.
- Mask bucket counts: small 22, medium 40, large 18.
- Source manifest SHA256:
  `58696bc504e79eec1342f00cbbb93d244b96d8311f128cf14156c3c6283cb595`
- Reserve manifest SHA256:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Codex opened 10/10 batch preview pages covering 80/80 source rows.
- Visual source sanity passed for the rows that exist: condition/winner/mask
  strips are aligned, masks are non-empty, and affected regions are visible.
- Gate failure: the pool does not meet the requested 128 source + 128 reserve
  design and is severely source-type imbalanced.

Decision:

`VOR_OR_SOURCE_POOL_BLOCKED`

No multi-model OR smoke, Gate128 generation, MiniMax adapter gate, DiffuEraser
VOR-OR micro, GPU task, or training was launched after this blocked source-pool
audit.

Reports/manifests:

- `exp30_vor_or_multimodel_minimax/manifests/vor_or_pool128_sources.jsonl`
- `exp30_vor_or_multimodel_minimax/manifests/vor_or_pool128_reserve.jsonl`
- `exp30_vor_or_multimodel_minimax/manifests/vor_or_source_pool_rejected.jsonl`
- `reports/exp30_vor_or_source_pool_audit.md`
- `reports/exp30_vor_or_source_pool_audit.csv`
- `reports/exp30_vor_or_source_pool_visual_review.csv`
- `reports/exp30_vor_or_source_pool_summary.json`
- `reports/exp30_vor_or_source_pool_previews/`

## 2026-06-27 Continuation V2 Readback

- Status: `EXP30_CONTINUATION_V2_READBACK_COMPLETED`
- Previous blocked source pool was traced to using exact extraction caches only,
  not to a full VOR data shortage.
- Previous cache roots contained 192 extracted triplets; after prior diagnostic
  exclusions only 80 usable scene groups remained.
- Full metadata index located on PAI:
  `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp25_vor/exp25_vor_or_preference_data/manifests/vor_train_metadata_index.jsonl`
- Full metadata rows: 57,751.
- Full metadata SHA256:
  `33d57a3ea23c5799b583d476a311089f95cbce1b0d11280822a63b8c9edcddc4`
- Pairing rule uses exact basename across:
  `VOR-Train/FG_BG/<sample_id>.mp4`,
  `VOR-Train/BG/<sample_id>.mp4`, and `MASK/<sample_id>.mp4`.
- Left CLI was checked read-only. No signals were sent and no left-side files
  were modified.
- No GPU task, inference, generation, training, or RC-FPO was launched.

Next gate:

`FULL_VOR_VALID_TRIPLET_INDEX_READY` or `FULL_VOR_INDEX_BLOCKED`.

Report:

- `reports/exp30_continuation_v2_readback.md`
