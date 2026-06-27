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

## 2026-06-27 Controlled Corruption V3 Plan

- Status: `CONTROLLED_CORRUPTION_V3_PLAN_LOCKED`.
- Locked at most four profiles, with smoke16 v3 using CC-v3-B for all sources,
  CC-v3-A for six repair rows, CC-v3-C for two affected-soft rows, and
  CC-v3-D reserved only.
- Smoke16 v3 controlled candidate cap: 24 total, at most 2 per source.
- Success target: technical-valid >=15/16, usable source coverage >=8/16,
  trivial-bad <=6/16 in selected controlled view, outside systematic damage 0.
- No generation, GPU task, Gate64, adapter gate, or training was launched.

Reports:

- `reports/exp30_controlled_corruption_v3_plan.md`
- `reports/exp30_controlled_corruption_v3_plan.json`
- `reports/exp30_controlled_corruption_v3_profile_table.csv`

## 2026-06-27 DiffuEraser / ProPainter Candidate Audit

- Status: `NEW_GENERATORS_SMOKE2_PENDING`.
- DiffuEraser verified stack found in Exp25:
  `DE-B_sft_raw6_d8_propainter`, 12/12 ok, 9 medium-hard, 3 hard-plausible,
  0 trivial-bad.
- Exp30 DiffuEraser blocker: current wrapper identity is not the explicit
  no-PCM overlay wrapper used by Exp25. Port and smoke2 are required.
- ProPainter status: candidate assets ready. Use
  `/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter`, not
  `/mnt/nas/hj/weights/propainter`.
- PAI Exp30 worktree was missing at audit time; no GPU smoke or training was
  launched.

Reports:

- `reports/exp30_diffueraser_propainter_candidate_audit.md`
- `reports/exp30_diffueraser_propainter_candidate_audit.csv`
- `reports/exp30_diffueraser_propainter_candidate_audit.json`

## 2026-06-27 Verified Generator Wrapper Port

- Status: `EXP30_VERIFIED_GENERATOR_WRAPPERS_PORTED_SMOKE2_PENDING`.
- Added:
  `exp30_vor_or_multimodel_minimax/scripts/infer_diffueraser_or_exp30.py`
  and
  `exp30_vor_or_multimodel_minimax/scripts/run_verified_or_generator_smoke.py`.
- No GPU smoke or model output was launched.
- Next gate: PAI runtime sync and two-sample DiffuEraser/ProPainter smoke.

Report:

- `reports/exp30_verified_generator_wrapper_port.md`

## 2026-06-27 Verified Generator Smoke2

- Status: `NEW_GENERATORS_SMOKE2_PARTIAL_PASS`.
- ProPainter: 2/2 generated, 17/17 frames each.
- DiffuEraser no-PCM: 2/2 generated, 17/17 frames each.
- Codex visual review: 4/4 review sheets opened.
- Final visual classes: 2 too-close, 1 hard-but-plausible, 1 medium-hard,
  0 final trivial-bad.
- Smoke16 v3 may include these generator families, but Smoke16 v3 itself has
  not run. Smoke32, Gate64, adapter gates, and training remain stopped.

Reports/assets:

- `reports/exp30_new_generators_smoke2.md`
- `reports/exp30_new_generators_smoke2_metrics.csv`
- `reports/exp30_new_generators_smoke2_visual_review.csv`
- `reports/exp30_new_generators_smoke2_summary.json`
- `reports/exp30_verified_generators_smoke2_assets/`

## 2026-06-27 Controlled-Corruption V3 Generator

- Status: `CONTROLLED_CORRUPTION_V3_GENERATOR_IMPLEMENTED`.
- Implemented `run_controlled_corruption_smoke16_v3.py` with the previously
  locked CC-v3-A/B/C schedule.
- Candidate cap: 24 controlled candidates across the 16 Smoke16 sources.
- The script emits all candidate metrics/review rows plus one deterministic
  primary controlled candidate per source.
- No Smoke16 v3 candidate generation, Gate64, MiniMax adapter training, long
  training, or RC-FPO was launched by this implementation milestone.
- Left CLI, Exp31, and Exp33 paths/processes remain untouched.

Report:

- `reports/exp30_controlled_corruption_v3_generator_impl.md`

## 2026-06-27 Controlled-Corruption Smoke16 V3

- Status: `CONTROLLED_CORRUPTION_V3_READY`.
- Ran the locked CC-v3-A/B/C schedule on the fixed 16-row Smoke16 manifest.
- Generated 24/24 controlled v3 candidates.
- Automatic metrics: 16 medium-hard, 8 trivial-bad across all candidates.
- Deterministic primary controlled view: 13 medium-hard, 3 trivial-bad,
  16/16 technical-valid, outside-fail count 0.
- Codex opened 6 all-candidate temporal pages and 4 primary pages, covering
  24/24 candidates and 16/16 primary choices.
- This repairs the controlled fallback subgate only. Full Smoke16 v3, Smoke32,
  Gate64, MiniMax adapter gates, DiffuEraser micro, RC-FPO, and long training
  remain stopped until the aggregate gate is run and passes.

Reports/assets:

- `reports/exp30_controlled_corruption_smoke16_v3.md`
- `reports/exp30_controlled_corruption_smoke16_v3_metrics.csv`
- `reports/exp30_controlled_corruption_smoke16_v3_review.csv`
- `reports/exp30_controlled_corruption_smoke16_v3_primary.csv`
- `reports/exp30_controlled_corruption_smoke16_v3_summary.json`
- `reports/exp30_controlled_corruption_smoke16_v3_visual_review.csv`
- `reports/exp30_controlled_corruption_smoke16_v3_visual_summary.json`
- `reports/exp30_controlled_corruption_v3_assets/`

## 2026-06-27 Multi-Model OR Smoke16 V3

- Status: `MULTIMODEL_OR_SMOKE16_V3_PASS`.
- Candidate rows: 64.
- Source rows: 16.
- Technical-valid candidates: 64/64.
- Best-per-source usable: 13/16.
- Best-per-source visual counts: 13 medium-hard, 3 trivial-bad.
- Usable by family:
  - controlled corruption v3 primary: 13.
  - MiniMax v2 reused without new seed selection: 4.
  - ProPainter verified full16: 2.
  - DiffuEraser no-PCM full16: 0.
- Codex opened controlled v3 pages and 8 verified-generator overview pages.
- Interpretation: the Smoke16 v3 aggregate gate passes, driven primarily by
  controlled corruption v3. ProPainter and DiffuEraser wrappers are technically
  valid but low-yield on this fixed Smoke16 set.
- Unlocked next: Smoke32 validation only.
- Still stopped: Gate64, MiniMax adapter gates, DiffuEraser micro, RC-FPO, and
  all long training.

Reports/assets:

- `reports/exp30_multimodel_or_smoke16_v3.md`
- `reports/exp30_multimodel_or_smoke16_v3_candidates.csv`
- `reports/exp30_multimodel_or_smoke16_v3_best_per_source.csv`
- `reports/exp30_multimodel_or_smoke16_v3_summary.json`
- `reports/exp30_verified_generators_smoke16_v3.md`
- `reports/exp30_verified_generators_smoke16_v3_metrics.csv`
- `reports/exp30_verified_generators_smoke16_v3_visual_review.csv`
- `reports/exp30_verified_generators_smoke16_v3_visual_review_final.csv`
- `reports/exp30_verified_generators_smoke16_v3_summary.json`
- `reports/exp30_verified_generators_smoke16_v3_assets/`

## 2026-06-27 Smoke32 V3 Preregistration

- Status: `EXP30_SMOKE32_V3_PREREGISTERED`.
- Locked 16 new confirmation source groups, disjoint from Smoke16, after
  Smoke16 v3 passed.
- Excluded known pre-inference invalid rows from the Smoke16 repair audit:
  `BLENDER_CARTOON006_00001`, `REAL_ENV044_00004_001_01`,
  `REAL_ENV046_00001_001_01`.
- Source balance: BLENDER / REAL = 8 / 8.
- Manifest SHA256:
  `ee8e056b05b9dcdd6d9d4a842637d32711eac4e397fbac0fdf728a33d65ddf45`.
- No extraction, model generation, GPU task, Gate64, adapter gate, or training
  was launched in this milestone.

Reports:

- `reports/exp30_smoke32_v3_preregistration.md`
- `reports/exp30_smoke32_v3_preregistration.csv`
- `reports/exp30_smoke32_v3_preregistration.json`

## 2026-06-27 Smoke32 V3 Selective Extraction and Materialization

- Status: `EXP30_SMOKE32_V3_MATERIALIZED`.
- Exact selective extraction completed from existing VOR split archives:
  `VOR-Train` 32/32 members and `VOR-Train-MASK` 16/16 members.
- Extraction missing members: 0.
- Extraction unsafe members: 0.
- Materialization completed 16/16 rows at 17 frames, 512 x 512.
- Failed materialized rows: 0.
- Materialized manifest SHA256:
  `320bb89ba16fb61a005e533ab319a2f4fb9ee6362cb8c269d4f2f0223a3e2ce9`.
- This is not a Smoke32 pass. It only unlocks the preregistered Smoke32 v3
  candidate-generation step.
- Gate64, MiniMax adapter gates, DiffuEraser micro, RC-FPO, and training remain
  stopped.

Reports:

- `reports/exp30_smoke32_v3_selective_extraction.csv`
- `reports/exp30_smoke32_v3_selective_extraction_state.json`
- `reports/exp30_smoke32_v3_materialization.md`
- `reports/exp30_smoke32_v3_materialization.csv`
- `reports/exp30_smoke32_v3_materialization.json`


## 2026-06-27 Smoke32 V3 Multi-Model Candidate Confirmation

- Status: `MULTIMODEL_OR_SMOKE32_V3_PASS`.
- Candidate rows: 64 non-EffectErase candidates across controlled corruption v3, MiniMax official v3, ProPainter verified full16, and DiffuEraser no-PCM full16.
- Source rows: 16 confirmation source groups, disjoint from Smoke16.
- Technical-valid candidates: 64/64.
- Total usable candidates: 14.
- Best-per-source usable coverage: 10/16.
- Controlled corruption usable source coverage: 8/16.
- Usable generator families: controlled corruption v3, MiniMax official v3, ProPainter.
- Model classification counts:
  - controlled corruption v3: 8 medium-hard, 8 trivial-bad.
  - MiniMax official v3: 2 medium-hard, 1 hard-plausible, 1 too-close, 12 trivial-bad.
  - ProPainter: 2 medium-hard, 1 hard-plausible, 13 trivial-bad.
  - DiffuEraser no-PCM: 16 trivial-bad, 0 usable.
- Codex opened 4 controlled review pages, 4 MiniMax review pages, and 8 verified-generator review pages before final classification.
- Interpretation: Smoke32 v3 passes the preregistered confirmation gate but with low margin. The pass is primarily controlled-corruption driven, with small supporting contributions from MiniMax and ProPainter; DiffuEraser remains technically runnable but unusable for this Smoke32 pool.
- This unlocks limited Gate64 pool preparation only. It does not unlock MiniMax adapter training by itself, does not make the pool data-ready, and does not support universal-adapter language.
- Left CLI, Exp31, and Exp33 were treated as protected lanes and not modified.

Reports/assets:

- `reports/exp30_multimodel_or_smoke32_v3.md`
- `reports/exp30_multimodel_or_smoke32_v3.csv`
- `reports/exp30_multimodel_or_smoke32_metrics_v3.csv`
- `reports/exp30_multimodel_or_smoke32_visual_review_v3.csv`
- `reports/exp30_multimodel_or_smoke32_summary_v3.json`
- `reports/exp30_multimodel_or_smoke32_best_per_source_v3.csv`
- `reports/exp30_verified_generators_smoke32_v3_visual_review_final.csv`
- `reports/exp30_smoke32_v3_assets/`


## 2026-06-27 Gate64 V3 Preregistration

- Status: `EXP30_GATE64_V3_PREREGISTERED`.
- Locked 64 source groups after Smoke16 v3 and Smoke32 v3 both passed.
- Excluded Smoke16 and Smoke32 scene groups plus known invalid rows.
- Source balance: BLENDER / REAL = 32 / 32.
- Manifest SHA256: `c4a0f5e07ef75aae57c9b40010f7fec85d10d5aa6c26a8056e1079d807bcf7f2`.
- No extraction, model output, visual selection, VOR-Eval access, adapter gate, RC-FPO, or training was launched by this milestone.
- This only unlocks Gate64 selective extraction/materialization and candidate generation.

Reports:

- `reports/exp30_vor_or_gate64_preregistration_v3.md`
- `reports/exp30_vor_or_gate64_preregistration_v3.csv`
- `reports/exp30_vor_or_gate64_preregistration_v3.json`
- `exp30_vor_or_multimodel_minimax/manifests/vor_or_gate64_sources_v3.jsonl`


## 2026-06-27 Gate64 V3 Pre-Inference Manifest Repair

- Status: `EXP30_GATE64_V3_MANIFEST_REPAIRED_PRE_INFERENCE`.
- Initial Gate64 materialization decoded 55/64 rows; 9 rows failed because masks were empty across the first 17 decoded frames.
- Failed rows were repaired before any Gate64 model output, visual selection, adapter gate, or training.
- Replacement rows: 9 total, matching source type distribution of failures (1 BLENDER, 8 REAL).
- Final manifest rows: 64.
- Final scene groups: 64.
- Final source balance: BLENDER / REAL = 32 / 32.
- Final manifest SHA256: `c2da063118934f0b03d13d88015cfc1cc57e881aca257307ca42de20cc944eb0`.
- Replacement manifest SHA256: `32a11315e55ce59d54864568c8225f6d329040282a2e34c26968581317b0062c`.
- This unlocks replacement-member extraction and final materialization only. Candidate generation and training remain stopped until final materialization passes.

Reports:

- `reports/exp30_vor_or_gate64_materialization_v3_partial.md`
- `reports/exp30_vor_or_gate64_materialization_v3_partial.csv`
- `reports/exp30_vor_or_gate64_materialization_v3_partial.json`
- `reports/exp30_vor_or_gate64_materialization_failed_v3.jsonl`
- `reports/exp30_vor_or_gate64_manifest_repair_v3.md`
- `reports/exp30_vor_or_gate64_manifest_repair_v3.csv`
- `reports/exp30_vor_or_gate64_manifest_repair_v3.json`
- `exp30_vor_or_multimodel_minimax/manifests/vor_or_gate64_sources_v3_final.jsonl`
- `exp30_vor_or_multimodel_minimax/manifests/vor_or_gate64_sources_v3_replacements.jsonl`


## 2026-06-27 Gate64 V3 Final Materialization

- Status: `EXP30_GATE64_V3_MATERIALIZED`.
- Final round3 manifest materialized successfully after pre-inference repairs.
- Requested rows: 64.
- Materialized rows: 64.
- Failed rows: 0.
- Frames: 17.
- Resolution: 512 x 512.
- Source balance: BLENDER / REAL = 32 / 32.
- Materialized manifest SHA256: `a32d42b9d5f9894e3e4c8f177b04e8d98271670b864f2388f72a5cb98dc02d13`.
- This unlocks Gate64 candidate generation only. It is not yet a pool-ready or adapter-training signal.
- No MiniMax adapter gate, RC-FPO, or long training has been launched.

Reports:

- `reports/exp30_vor_or_gate64_materialization_final_v3.md`
- `reports/exp30_vor_or_gate64_materialization_final_v3.csv`
- `reports/exp30_vor_or_gate64_materialization_final_v3.json`
- `exp30_vor_or_multimodel_minimax/manifests/vor_or_gate64_materialized_final_v3_ref.jsonl`


## 2026-06-27 Gate64 V3 Multi-Model OR Pool

- Status: `VOR_OR_GATE64_MULTIMODEL_POOL_READY`.
- Aggregated candidates: 256.
- Selected primary usable pairs: 50.
- Selected classes: 48 medium-hard, 2 hard-plausible.
- Train rows: 32.
- Heldout rows: 16.
- Train/heldout scene overlap: 0.
- Selected model counts: controlled corruption v3 26, MiniMax official v3 17,
  ProPainter 6, DiffuEraser no-PCM 1.
- EffectErase primary used: false.
- VOR-Eval used: false.
- Training started: false.
- Gate unlocked: preregistered MiniMax 10-step adapter gate only.
