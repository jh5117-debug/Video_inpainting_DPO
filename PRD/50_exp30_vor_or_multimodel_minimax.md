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

## 2026-06-27 Full VOR Valid Triplet Index Recovery

- Status: `FULL_VOR_VALID_TRIPLET_INDEX_READY`
- Metadata index:
  `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp25_vor/exp25_vor_or_preference_data/manifests/vor_train_metadata_index.jsonl`
- Metadata rows: 57,751.
- Valid triplets after quarantining known bad sample
  `BLENDER_RIVER007_00001`: 57,750.
- Scene groups: 1,449.
- Source type counts: BLENDER 21,495; REAL 36,256.
- Metadata SHA256:
  `33d57a3ea23c5799b583d476a311089f95cbce1b0d11280822a63b8c9edcddc4`
- Member index:
  `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp25_vor/reports/vor_train_mask_member_index.csv`
- Member index SHA256:
  `3b3415c989e72b4df821f85903d01a754fb2c07412e40907749bf9844626d1f8`
- Pairing rule verified:
  `VOR-Train/FG_BG/<sample_id>.mp4`,
  `VOR-Train/BG/<sample_id>.mp4`, `MASK/<sample_id>.mp4`.
- VOR tar archives scanned: no.
- Videos extracted: no.
- VOR-Eval used: no.
- Exp25 worktree modified: no.

Reports/manifests:

- `reports/exp30_full_vor_index_recovery.md`
- `reports/exp30_full_vor_index_recovery.csv`
- `reports/exp30_full_vor_index_summary.json`
- `exp30_vor_or_multimodel_minimax/manifests/vor_or_full_valid_triplet_index_ref.json`

## 2026-06-27 Source-Pool V2 Sampling From Full Index

- Status: `VOR_OR_SOURCE_POOL_V2_READY`
- Metadata rows after known-bad quarantine: 57,750.
- Scene groups after known-bad quarantine: 1,449.
- Strict excluded scene groups: 209.
- Strict candidates after exclusion: 1,240.
- Relaxed diagnostic exclusion used: false.
- Primary rows: 128.
- Reserve rows: 128.
- Reserve2 rows: 128.
- Primary source type counts: BLENDER 64; REAL 64.
- Reserve source type counts: BLENDER 20; REAL 108.
- Reserve2 source type counts: REAL 128.
- Mask bucket metadata: unavailable, recorded as `unknown`.
- Effect type metadata: unavailable, recorded as `unknown`.
- Visual preview status: `metadata_only_visual_preview_pending`.
- Videos extracted: no.
- Archives scanned: no.

Manifest SHA256:

- Primary:
  `cf8104cbf8a859b66a5c3e7358b8c72c3c77177279ed78297063f7538903975b`
- Reserve:
  `cfb661b2ce86b936b1c85961173f2c83e16b6cc094505ff3b6dfa0c940c55cb0`
- Reserve2:
  `0f457bc5439580bb13ee2390ae1f19fb53438a2a6b28a758ce0201a6270b4823`

This unlocks multi-model OR smoke16 only. It does not unlock Gate64, MiniMax
adapter training, DiffuEraser micro training, or any data-ready/scientific
claim without video generation, metrics, and visual review.

Reports/manifests:

- `exp30_vor_or_multimodel_minimax/manifests/vor_or_source_pool_v2_primary128.jsonl`
- `exp30_vor_or_multimodel_minimax/manifests/vor_or_source_pool_v2_reserve128.jsonl`
- `exp30_vor_or_multimodel_minimax/manifests/vor_or_source_pool_v2_reserve2_128.jsonl`
- `exp30_vor_or_multimodel_minimax/manifests/vor_or_source_pool_v2_rejected.jsonl`
- `reports/exp30_vor_or_source_pool_v2_sampling.md`
- `reports/exp30_vor_or_source_pool_v2_sampling.csv`
- `reports/exp30_vor_or_source_pool_v2_summary.json`
- `reports/exp30_vor_or_source_pool_v2_preview_review.csv`
