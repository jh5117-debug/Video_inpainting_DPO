# Exp30 Status

Current status: `MULTIMODEL_OR_SMOKE16_V2_BLOCKED`

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

## 2026-06-27 Three-Backbone Paper Positioning

- Status: `EXP30_THREE_BACKBONE_POSITIONING_LOCKED`.
- DiffuEraser role: primary original backbone and VOR-OR adapter baseline to
  validate with Stage1/Stage2 micro.
- VideoPainter role: second backbone for VOR-BG BR/inpainting adapter evidence;
  not a standard VOR-OR result.
- MiniMax role: flow-style Wan/DiT third-backbone adapter candidate, blocked
  until multi-model medium-hard data and heldout micro gates pass.
- EffectErase role: OR strong baseline / diagnostic / upper reference only.
- Allowed language: model-specific backend adapters and cross-backbone evidence
  from DiffuEraser plus VideoPainter.
- Forbidden language: universal adapter, all models supported, EffectErase
  adapter-ready, MiniMax quality-positive before heldout micro, final SOTA.

Reports:

- `reports/exp30_three_backbone_paper_positioning.md`
- `reports/exp30_three_backbone_paper_positioning.csv`

## 2026-06-27 VOR-OR Source Pool Audit

- Status: `VOR_OR_SOURCE_POOL_BLOCKED`.
- Method: used existing exact extraction caches only; no VOR tar archive rescan.
- Discovered extracted triplets: 192.
- Candidate scene groups after previous MiniMax/EffectErase exclusions: 80.
- Source manifest rows: 80.
- Reserve rows: 0.
- Source type counts: REAL 71, BLENDER 9.
- Mask buckets: small 22, medium 40, large 18.
- Source manifest SHA256:
  `58696bc504e79eec1342f00cbbb93d244b96d8311f128cf14156c3c6283cb595`.
- Reserve manifest SHA256:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Codex opened 10/10 batch preview pages covering 80/80 source rows.
- Visual source sanity passed for the available rows, but the pool cannot meet
  the requested 128 source + 128 reserve target or source-type balance.
- No multi-model OR smoke, Gate128, MiniMax adapter gate, DiffuEraser micro, or
  GPU task was launched.

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

- Status: `EXP30_CONTINUATION_V2_READBACK_COMPLETED`.
- Previous source pool failed because it used exact extraction caches only.
- Full metadata index located:
  `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp25_vor/exp25_vor_or_preference_data/manifests/vor_train_metadata_index.jsonl`
- Rows: 57,751.
- SHA256:
  `33d57a3ea23c5799b583d476a311089f95cbce1b0d11280822a63b8c9edcddc4`.
- Pairing rule: exact basename across FG_BG, BG, and MASK.
- No GPU task or left-side action was launched.

Report:

- `reports/exp30_continuation_v2_readback.md`

## 2026-06-27 Full VOR Valid Triplet Index Recovery

- Status: `FULL_VOR_VALID_TRIPLET_INDEX_READY`.
- Metadata rows: 57,751.
- Valid triplets: 57,750 after quarantining known bad
  `BLENDER_RIVER007_00001`.
- Scene groups: 1,449.
- Source type counts: BLENDER 21,495; REAL 36,256.
- Pairing rule verified by exact basename across FG_BG, BG, and MASK.
- Archives scanned: no.
- Videos extracted: no.
- VOR-Eval used: no.

Reports/manifests:

- `reports/exp30_full_vor_index_recovery.md`
- `reports/exp30_full_vor_index_recovery.csv`
- `reports/exp30_full_vor_index_summary.json`
- `exp30_vor_or_multimodel_minimax/manifests/vor_or_full_valid_triplet_index_ref.json`

## 2026-06-27 Source-Pool V2 Sampling

- Status: `VOR_OR_SOURCE_POOL_V2_READY`.
- Primary rows: 128.
- Reserve rows: 128.
- Reserve2 rows: 128.
- Strict candidates after exclusion: 1,240.
- Relaxed diagnostic exclusion used: false.
- Primary source type counts: BLENDER 64; REAL 64.
- Reserve source type counts: BLENDER 20; REAL 108.
- Mask/effect metadata unavailable; values recorded as `unknown`.
- Visual preview status: `metadata_only_visual_preview_pending`.
- No videos were extracted and no smoke/training was launched.

Reports/manifests:

- `exp30_vor_or_multimodel_minimax/manifests/vor_or_source_pool_v2_primary128.jsonl`
- `exp30_vor_or_multimodel_minimax/manifests/vor_or_source_pool_v2_reserve128.jsonl`
- `exp30_vor_or_multimodel_minimax/manifests/vor_or_source_pool_v2_reserve2_128.jsonl`
- `exp30_vor_or_multimodel_minimax/manifests/vor_or_source_pool_v2_rejected.jsonl`
- `reports/exp30_vor_or_source_pool_v2_sampling.md`
- `reports/exp30_vor_or_source_pool_v2_sampling.csv`
- `reports/exp30_vor_or_source_pool_v2_summary.json`
- `reports/exp30_vor_or_source_pool_v2_preview_review.csv`

## 2026-06-27 Smoke16 V2 Preregistration

- Status: `EXP30_SMOKE16_V2_PREREGISTERED`.
- Rows: 16.
- Scene groups: 16.
- Source type counts: BLENDER 8; REAL 8.
- Mask/effect metadata unavailable; values recorded as `unknown`.
- Manifest SHA256:
  `1871f8e1aa23579425a87661040f91a992e934492aaa98c196f924ff21990ca3`.
- VOR-Eval used: no.
- Model outputs generated: no.
- GPU used: no.
- Next gate: `MULTIMODEL_OR_SMOKE16_V2_PASS`.

## 2026-06-27 Smoke16 V2 Pre-Inference Repair

- Status: `EXP30_SMOKE16_V2_MANIFEST_REPAIRED_PRE_INFERENCE`.
- Invalid rows: `BLENDER_CARTOON006_00001`,
  `REAL_ENV044_00004_001_01`, `REAL_ENV046_00001_001_01`.
- Replacement rows: `BLENDER_FOREST019_00001`,
  `REAL_ENV046_00004_001_01`, `REAL_ENV046_00005_001_01`.
- Replacement rule: same source type, scene-disjoint, deterministic source-pool
  v2 replacement before any model output review.
- Final rows: 16.
- Final scene groups: 16.
- Final source type counts: BLENDER 8; REAL 8.
- Final manifest SHA256:
  `7e8cfd1b672b17b131476c9dd82804841d22d7450adf26301cf9ae8ff83f7f76`.
- Still not unlocked: smoke pass, Gate64, MiniMax adapter training,
  DiffuEraser VOR-OR micro, data-ready, or scientific-positive claims.

## 2026-06-27 Smoke16 V2 Final Materialization

- Status: `EXP30_SMOKE16_V2_MATERIALIZED`.
- Requested rows: 16.
- Materialized rows: 16.
- Failed rows: 0.
- Frames per row: 17.
- Resolution: 512 x 512.
- Source type counts: BLENDER 8; REAL 8.
- Materialized manifest SHA256:
  `72be9884335fef61926c307c66878fdc05dec85e9be4da28ab1547db98f8c26d`.
- This unlocks smoke16 candidate generation only. It does not unlock
  multi-model smoke pass, Gate64, MiniMax recipe/training, DiffuEraser VOR-OR
  micro, or any data-ready/scientific claim.

## 2026-06-27 Multi-Model OR Smoke16 V2

- Status: `MULTIMODEL_OR_SMOKE16_V2_BLOCKED`.
- Candidate generators completed: controlled corruption, MiniMax official.
- Non-EffectErase candidates: 32.
- Technical valid: 32.
- Usable: 9.
- Classification totals: MEDIUM_HARD_ELIGIBLE 6; HARD_BUT_PLAUSIBLE 3;
  TRIVIAL_BAD 23.
- Controlled-corruption usable fallback: 5/16, required >=6/16: fail.
- MiniMax usable: 4/16: low-yield documented.
- Codex visual review: opened 32/32 temporal strips locally.
- Not launched: Gate64, MiniMax recipe/training, DiffuEraser VOR-OR micro,
  long training, RC-FPO.

## 2026-06-27 Continuation V3 Readback

- Status: `EXP30_CONTINUATION_V3_READBACK_COMPLETED`.
- Branch/HEAD: `research/exp30-vor-or-multimodel-minimax-adapter-20260627`
  at `bd8777274dfe898dc9278cadcc1dd971536a5e2c`.
- Reread PRD, registry, smoke16 v2 reports, source-pool/full-index reports,
  and Exp30 scripts/code.
- Confirmed v2 blocker: controlled corruption only reached 5/16 usable
  fallback candidates, below the preregistered >=6/16 threshold. The blocker is
  quality yield, not technical validity.
- Protected PAI lanes remain active: Exp31 on GPU1, Exp33 on GPU3, and cli4
  locks reserving GPU1-GPU4. No protected process or file was modified.
- No GPU generation, Gate64, MiniMax adapter gate, training, RC-FPO, or
  protected-lane action was launched.

Report:

- `reports/exp30_continuation_v3_readback.md`

## 2026-06-27 Smoke16 V2 Failure Analysis

- Status: `SMOKE16_V2_FAILURE_ANALYZED`.
- Analyzed 32/32 v2 non-EffectErase candidates; no new candidate generation,
  GPU task, Gate64, adapter gate, or training ran.
- Controlled corruption: 16/16 technical-valid, 5/16 usable; dominant blocker
  is temporal discontinuity from a single aggressive frame-wise corruption
  profile.
- MiniMax official: 16/16 technical-valid, 4/16 usable; dominant blockers are
  outside damage and temporal flicker/instability, with a few usable
  medium-hard/hard-plausible rows.
- Continuation v3 visual readback opened 4 controlled overview pages and 4
  MiniMax review pages covering all 32 v2 candidates.

Reports:

- `reports/exp30_smoke16_v2_failure_analysis.md`
- `reports/exp30_smoke16_v2_failure_analysis.csv`
- `reports/exp30_smoke16_v2_failure_summary.json`
