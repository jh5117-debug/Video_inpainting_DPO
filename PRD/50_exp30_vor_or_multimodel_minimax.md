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

## 2026-06-27 Smoke16 V2 Preregistration

- Status: `EXP30_SMOKE16_V2_PREREGISTERED`
- Locked 16 source groups from source-pool v2 before any extraction or model
  output review.
- Source types: BLENDER 8; REAL 8.
- Scene groups: 16.
- Mask/effect labels remain metadata-unavailable and are recorded as
  `unknown`, not inferred.
- VOR-Eval used: no.
- Model outputs generated: no.
- GPU used: no.
- Manifest:
  `exp30_vor_or_multimodel_minimax/manifests/vor_or_smoke16_v2_sources.jsonl`
- Manifest SHA256:
  `1871f8e1aa23579425a87661040f91a992e934492aaa98c196f924ff21990ca3`

This preregistration only locks the smoke16 rows. It does not by itself satisfy
`MULTIMODEL_OR_SMOKE16_V2_PASS`, unlock Gate64, unlock MiniMax adapter
training, or support any data-ready/scientific claim.

Reports:

- `reports/exp30_multimodel_or_smoke16_v2_preregistration.md`
- `reports/exp30_multimodel_or_smoke16_v2_preregistration.csv`
- `reports/exp30_multimodel_or_smoke16_v2_preregistration.json`

## 2026-06-27 Smoke16 V2 Pre-Inference Technical Repair

- Status: `EXP30_SMOKE16_V2_MANIFEST_REPAIRED_PRE_INFERENCE`.
- Reason: first materialization found one preregistered row with fewer than 17
  decodable frames and two rows with empty masks. These are technical source
  invalidity failures, not model-result failures.
- Invalid rows:
  `BLENDER_CARTOON006_00001`,
  `REAL_ENV044_00004_001_01`,
  `REAL_ENV046_00001_001_01`.
- Replacement rows:
  `BLENDER_FOREST019_00001`,
  `REAL_ENV046_00004_001_01`,
  `REAL_ENV046_00005_001_01`.
- Replacement rule: deterministic same-source-type source-pool-v2 replacement,
  scene-disjoint, before any candidate/model output review.
- Final rows: 16.
- Final scene groups: 16.
- Final source type counts: BLENDER 8; REAL 8.
- Final manifest:
  `exp30_vor_or_multimodel_minimax/manifests/vor_or_smoke16_v2_sources_final.jsonl`
- Final manifest SHA256:
  `7e8cfd1b672b17b131476c9dd82804841d22d7450adf26301cf9ae8ff83f7f76`.

This repair does not by itself satisfy `MULTIMODEL_OR_SMOKE16_V2_PASS`, unlock
Gate64, unlock MiniMax adapter training, or support any data-ready/scientific
claim. The next required step is selective extraction/materialization of the
final smoke16 manifest followed by actual candidate generation, metrics, and
per-video review.

Reports:

- `reports/exp30_multimodel_or_smoke16_v2_manifest_repair.md`
- `reports/exp30_multimodel_or_smoke16_v2_manifest_repair.csv`
- `reports/exp30_multimodel_or_smoke16_v2_manifest_repair.json`

## 2026-06-27 Smoke16 V2 Final Materialization

- Status: `EXP30_SMOKE16_V2_MATERIALIZED`.
- Requested rows: 16.
- Materialized rows: 16.
- Failed rows: 0.
- Frames per row: 17.
- Resolution: 512 x 512.
- Source type counts: BLENDER 8; REAL 8.
- Empty-mask guard: active; no empty-mask row passed.
- Materialized manifest:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp30_vor_or_multimodel_minimax/smoke16_v2_20260627/materialized_17f_512_final/smoke16_v2_materialized_final.jsonl`
- Materialized manifest SHA256:
  `72be9884335fef61926c307c66878fdc05dec85e9be4da28ab1547db98f8c26d`.
- Local reference manifest:
  `exp30_vor_or_multimodel_minimax/manifests/vor_or_smoke16_v2_materialized_final_ref.jsonl`

This unlocks candidate generation for smoke16 only. It still does not satisfy
`MULTIMODEL_OR_SMOKE16_V2_PASS`, unlock Gate64, unlock MiniMax adapter
training, or support data-ready/scientific claims without generated OR
candidates, metrics, and per-video visual review.

Reports:

- `reports/exp30_smoke16_v2_materialization_final.md`
- `reports/exp30_smoke16_v2_materialization_final.csv`
- `reports/exp30_smoke16_v2_materialization_final.json`

## 2026-06-27 Multi-Model OR Smoke16 V2

- Status: `MULTIMODEL_OR_SMOKE16_V2_BLOCKED`.
- Non-EffectErase candidates generated/reviewed: 32.
- Technical-valid non-EffectErase candidates: 32.
- Usable non-EffectErase candidates: 9.
- Classification totals: MEDIUM_HARD_ELIGIBLE 6; HARD_BUT_PLAUSIBLE 3;
  TRIVIAL_BAD 23.
- Controlled corruption:
  - Technical valid: 16/16.
  - Usable fallback: 5/16.
  - Required usable fallback: >= 6/16.
  - Gate result: fail.
- MiniMax official:
  - Technical valid: 16/16.
  - Usable: 4/16.
  - Gate result: low-yield documented; MiniMax-only criterion passes, but
    smoke remains blocked by controlled fallback.
- ProPainter: wrapper/weights observed in the worktree, but not launched after
  the preregistered controlled-fallback criterion had already failed. Additional
  model outputs cannot repair that specific gate without changing the rule.
- DiffuEraser OR stack: `DIFFUSERASER_OR_STACK_PENDING_NOT_BLOCKING_SMOKE16`.
- EffectErase: diagnostic-only, not used for training or smoke promotion.
- Codex visual review: opened 32/32 temporal strips locally, covering all
  controlled-corruption and MiniMax candidates.

Gate64, MiniMax adapter recipe/training, and DiffuEraser VOR-OR micro remain
stopped. The smoke did not support data-ready or scientific-positive language.

Reports:

- `reports/exp30_multimodel_or_smoke16_v2.md`
- `reports/exp30_multimodel_or_smoke16_v2.csv`
- `reports/exp30_multimodel_or_smoke16_metrics_v2.csv`
- `reports/exp30_multimodel_or_smoke16_visual_review_v2.csv`
- `reports/exp30_multimodel_or_smoke16_summary_v2.json`
- `reports/exp30_controlled_corruption_smoke16_v2.md`
- `reports/exp30_controlled_corruption_smoke16_v2_review.csv`
- `reports/exp30_controlled_corruption_smoke16_v2_metrics.csv`
- `reports/exp30_controlled_corruption_smoke16_v2_summary.json`
- `reports/exp30_minimax_smoke16_v2_metrics.csv`
- `reports/exp30_minimax_smoke16_v2_summary.json`

## 2026-06-27 Continuation V3 Readback

- Status: `EXP30_CONTINUATION_V3_READBACK_COMPLETED`.
- Branch/HEAD: `research/exp30-vor-or-multimodel-minimax-adapter-20260627`
  at `bd8777274dfe898dc9278cadcc1dd971536a5e2c`.
- Readback covered the current PRDs, Exp30 registry, smoke16 v2 reports,
  source-pool/full-index reports, Exp30 scripts/code, and read-only runtime
  state on PAI.
- Smoke16 v2 blocker is confirmed as quality yield only:
  controlled corruption reached 5/16 usable fallback candidates, MiniMax
  reached 4/16 usable, and there were no systemic decode/frame/mask failures.
- Repaired smoke16 source rows remain valid for planning smoke16 v3:
  materialized manifest SHA256
  `72be9884335fef61926c307c66878fdc05dec85e9be4da28ab1547db98f8c26d`.
- Protected-lane readback found Exp31 on GPU1 and Exp33 on GPU3; left CLI locks
  reserve GPU1-GPU4. Exp30 must not touch those tasks, locks, branches, or
  output roots.
- Next allowed work: v2 failure analysis, controlled-corruption v3 plan,
  DiffuEraser/ProPainter stack audit, and smoke16 v3 preregistration.
- Still not allowed: direct Gate64, Smoke32 before smoke16 v3 pass, MiniMax
  adapter gate before Gate64 pool ready, any long training, RC-FPO, universal
  adapter language, or modification of `inference/metrics.py` / shared trainer.

## 2026-06-27 Smoke16 V2 Failure Analysis

- Status: `SMOKE16_V2_FAILURE_ANALYZED`.
- Inputs: `reports/exp30_multimodel_or_smoke16_metrics_v2.csv`,
  `reports/exp30_multimodel_or_smoke16_visual_review_v2.csv`, existing v2
  temporal strips/review assets, and continuation v3 visual evidence readback.
- Candidate coverage: 32/32 non-EffectErase candidates.
- Technical validity: 32/32.
- Controlled corruption:
  - Usable: 5/16.
  - Failure counts: temporal discontinuity 11, local residual too sharp 2,
    bounded local texture mismatch 3.
  - Interpretation: outside preservation is not the dominant issue; the single
    frame-wise corruption profile is too aggressive and not temporally coherent.
- MiniMax official:
  - Usable: 4/16.
  - Failure counts: outside damage 7, temporal flicker/instability 4, too bad 1,
    bounded residual/medium-hard 3, strong but plausible local defect 1.
  - Interpretation: MiniMax can contribute a few useful candidates, but cannot
    be the only VOR-OR loser source.
- Required v3 fixes: preregistered controlled-corruption profiles, no infinite
  sampling, and DiffuEraser/ProPainter stack audit before enabling those
  candidate families.
- Still stopped: direct Gate64, Smoke32 before smoke16 v3 pass, MiniMax adapter
  gate, DiffuEraser micro training, RC-FPO, and long training.

Reports:

- `reports/exp30_smoke16_v2_failure_analysis.md`
- `reports/exp30_smoke16_v2_failure_analysis.csv`
- `reports/exp30_smoke16_v2_failure_summary.json`

## 2026-06-27 Controlled Corruption V3 Calibration Plan

- Status: `CONTROLLED_CORRUPTION_V3_PLAN_LOCKED`.
- Motivation: v2 controlled corruption failed by temporal discontinuity and
  hard local residuals, not by outside leakage.
- Locked profiles:
  - CC-v3-A mild-object: lower noise, lower condition mix, high temporal
    smoothing.
  - CC-v3-B medium-object: default all-source profile, still softer than v2.
  - CC-v3-C affected-soft: object plus soft affected map, far outside
    preserved.
  - CC-v3-D boundary-focused: defined but not enabled in smoke16 v3.
- Smoke16 v3 controlled schedule:
  - CC-v3-B for all 16 sources.
  - CC-v3-A for six temporal-discontinuity repair sources.
  - CC-v3-C for two affected-soft sources.
  - Maximum 24 controlled candidates total and at most 2 per source.
- Success target: technical-valid >=15/16, usable source coverage >=8/16,
  trivial-bad <=6/16 in the selected controlled view, and outside systematic
  damage = 0.
- Controlled corruption remains a fallback/data-source only; it is not a final
  model or ground truth.
- No video generation, smoke run, Gate64, adapter gate, or training was run.

Reports:

- `reports/exp30_controlled_corruption_v3_plan.md`
- `reports/exp30_controlled_corruption_v3_plan.json`
- `reports/exp30_controlled_corruption_v3_profile_table.csv`
