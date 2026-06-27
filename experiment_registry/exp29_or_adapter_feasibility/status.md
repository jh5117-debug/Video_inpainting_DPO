# Exp29 Status

Current status: `READBACK_AND_SCAFFOLD_CREATED`

Exp29 is an isolated feasibility audit for MiniMax-Remover and EffectErase.
It inherits the Exp26 conclusion that DiffuEraser plus VideoPainter support
`CROSS_BACKBONE_ADAPTER_EVIDENCE_CONFIRMED`, but not a universal-adapter
claim.

No GPU inference, trainable-forward gate, DPO step, or long training has been
launched by the scaffold milestone.

## 2026-06-26 Repo And Weight Audit

- MiniMax: `MINIMAX_REPO_READY`, `MINIMAX_WEIGHTS_READY`.
- EffectErase: `EFFECTERASE_REPO_READY`, `EFFECTERASE_BLOCKED_NO_WEIGHTS`.
- No inference smoke or trainable-forward gate has run yet.

Reports:

- `reports/exp29_minimax_repo_weight_audit.md`
- `reports/exp29_minimax_repo_weight_audit.csv`
- `reports/exp29_minimax_asset_matrix.json`
- `reports/exp29_effecterase_repo_weight_audit.md`
- `reports/exp29_effecterase_repo_weight_audit.csv`
- `reports/exp29_effecterase_asset_matrix.json`

## 2026-06-26 Inference Smoke And Trainable Forward

- MiniMax inference: `MINIMAX_INFERENCE_SMOKE_PASSED_WITH_VISUAL_QUALITY_RISKS`.
- MiniMax visual quality: mixed; one medium-hard candidate and three
  trivial-bad outputs.
- MiniMax trainable forward: `MINIMAX_TRAINABLE_FORWARD_PASSED`.
- EffectErase inference: `EFFECTERASE_INFERENCE_SMOKE_BLOCKED_NO_WEIGHTS`.

## 2026-06-26 MiniMax Adapter Gates

- Zero-gap: `MINIMAX_ZERO_GAP_PASSED`.
- One-step strict reload: `MINIMAX_ONE_STEP_STRICT_RELOAD_PASSED`.
- 10-step: `MINIMAX_10STEP_PARETO_MIXED`.
- Heldout visual result: Step10 is nearly unchanged from Step0 on two heldout
  rows; no visible quality gain.
- MiniMax final: `MINIMAX_ADAPTER_POSSIBLE_NEEDS_MORE_WORK`.
- EffectErase final for this run: `EFFECTERASE_BLOCKED`.

## 2026-06-26 Continuation Readback

- Status: `EXP29_CONTINUATION_READBACK_COMPLETED`.
- Branch/HEAD confirmed:
  `research/exp29-minimax-effecterase-adapter-feasibility-20260626` at
  `4b8d68af3ebd0f6981e697baee952b5f0e1ca76f`.
- Previous PRD, registry, reports, MiniMax gate JSON, EffectErase asset matrix,
  and relevant code pointers were reread before any GPU task.
- Left CLI state was checked read-only. GPU1/GPU2/GPU3/GPU4 remain reserved by
  CLI runtime locks; Exp28 DAVIS50 evaluation was observed on GPU3.
- No left CLI signal was sent; no left runtime/worktree/output file was
  modified.
- Report: `reports/exp29_continuation_readback.md`.

## 2026-06-26 MiniMax 10-Step Failure Analysis

- Status: `MINIMAX_10STEP_FAILURE_ANALYZED`.
- Root cause: the previous stable recovery recipe was intentionally too
  conservative (`SGD(lr=1e-7)`), producing only
  `1.1061271569642785e-10` step10 parameter-probe delta.
- Backward path was not missing: mean preclip grad norm was `0.7237282794`, max
  `1.2341757971`, with 461 gradient tensors.
- Quality signal was weak: 3/4 previous smoke training losers were
  trivial-bad, and the heldout set had only 2 rows.
- Decision: do not run longer from the same recipe; require a medium-hard
  train16/heldout16 data-quality gate before further MiniMax optimizer tests.
- Reports:
  - `reports/exp29_minimax_10step_failure_analysis.md`
  - `reports/exp29_minimax_10step_failure_analysis.csv`
  - `reports/exp29_minimax_next_micro_plan.md`

## 2026-06-26 MiniMax Preference Data Quality Gate

- Status: `MINIMAX_DATA_YIELD_INSUFFICIENT`.
- Candidate protocol: 32 sources, 17 frames each, 3 fixed seeds each, raw
  MiniMax output only.
- Reviewed candidates: 96.
- Counts:
  - `MEDIUM_HARD_ELIGIBLE`: 23
  - `HARD_BUT_PLAUSIBLE`: 4
  - `TOO_CLOSE`: 3
  - `TRIVIAL_BAD`: 60
  - `TECHNICAL_INVALID`: 6
- Eligible candidates: 27, but only 9 unique scene groups.
- The required scene-disjoint train16/heldout16 split cannot be formed.
- Decision: recipe search and 30-step MiniMax micro are not allowed from this
  data gate.
- Reports:
  - `reports/exp29_minimax_preference_data_quality.md`
  - `reports/exp29_minimax_preference_data_quality.csv`
  - `reports/exp29_minimax_preference_video_review.csv`
  - `reports/exp29_minimax_preference_data_quality_summary.json`

## 2026-06-26 EffectErase Weight Recovery

- Status: `EFFECTERASE_WEIGHTS_READY`.
- Official EffectErase and Wan2.1-Fun InP assets recovered into the Exp29
  autoresearch cache.
- SHA256 verification: 19 manifest entries checked, all `OK`.
- No inference smoke, trainable forward, zero-gap, one-step, or adapter micro
  has run yet.
- EffectErase remains VOR baseline/diagnostic only until a non-confounded smoke
  and evaluation design are completed.

## 2026-06-26 Continuation V2 Readback

- Status: `EXP29_CONTINUATION_V2_READBACK_COMPLETED`.
- Confirmed branch HEAD `6c97d4b74f331ce4db089224f7dcf9ec6eb283ce`.
- MiniMax remains `MINIMAX_DATA_YIELD_INSUFFICIENT`.
- EffectErase remains `EFFECTERASE_WEIGHTS_READY`.
- Left CLI was read-only audited; no signal and no file mutation.
- Right-side GPU tasks remain blocked until architecture-family audit and
  EffectErase smoke pre-registration are committed.

## 2026-06-26 Architecture Family Audit

- Status: `EXP29_ARCHITECTURE_FAMILY_AUDIT_COMPLETED`.
- DiffuEraser: SD1.5 / UNet-style latent diffusion, confirmed adapter backbone.
- VideoPainter: CogVideoX-style video model, confirmed adapter backbone.
- MiniMax: Wan2.1 / DiT / flow matching with `epsilon - z0` target, candidate
  only; no target mismatch in current Exp29 code path.
- EffectErase: Wan / DiT / flow-style removal pipeline, baseline/diagnostic
  until smoke and training-forward gates are proven.

## 2026-06-26 EffectErase Smoke Pre-Registration

- Status: `EFFECTERASE_SMOKE_PREREGISTERED`.
- Locked 6 diagnostic VOR rows:
  `exp29_or_adapter_feasibility/manifests/effecterase_smoke_preregistered.jsonl`.
- Manifest SHA256:
  `54fd62a97fa69f2f17590488136d426cee77de0ed02548c46a83d8818be2b137`.
- Balanced source/mask coverage: REAL/BLENDER = 3/3 and small/medium/large
  masks = 2/2/2.
- The protocol fixes 17 frames, 832x480, seed 2025, CFG 1.0, 50 steps, raw
  output primary, diagnostic comp optional.
- The official EffectErase default is 81 frames. This 17-frame run is therefore
  diagnostic compatibility smoke only.
- VOR-Eval and non-VOR OR rows are not used. All rows are tagged
  `diagnostic_only_vor_confounded` and `eligible_for_training=false`.

## 2026-06-26 Continuation V3 Readback

- Status: `EXP29_CONTINUATION_V3_READBACK_COMPLETED`.
- Confirmed branch HEAD `972deab321a518638102a1ace6ed87a13456a261`.
- EffectErase remains weights-ready and smoke-preregistered; no inference has
  run yet.
- MiniMax remains data-yield insufficient; no recipe or 30-step task is
  allowed before expanded source-pool evidence.
- Left CLI was read-only audited; no signal and no file mutation.
- Report: `reports/exp29_continuation_v3_readback.md`.

## 2026-06-26 EffectErase Smoke Input Materialization

- Status: `EFFECTERASE_SMOKE_INPUTS_BLOCKED`.
- Manifest SHA256:
  `54fd62a97fa69f2f17590488136d426cee77de0ed02548c46a83d8818be2b137`.
- Ready rows: 5/6.
- Blocker: `REAL_ENV249_00103_004_04` has an empty mask in the locked
  materialization.
- No replacement row, mask edit, seed change, or frame-index change was made.
- EffectErase inference was not launched.
- Reports:
  - `reports/exp29_effecterase_smoke_input_materialization.md`
  - `reports/exp29_effecterase_smoke_input_materialization.csv`
  - `reports/exp29_effecterase_smoke_input_materialization.json`

## 2026-06-26 EffectErase Command Dry-Run

- Status: `EFFECTERASE_COMMAND_READY`.
- Official script import now succeeds in a dedicated Exp29 venv using
  `transformers==4.51.3`, `diffusers==0.31.0`, and `decord==0.6.0`.
- The official script accepts `--num_frames`, `--cfg`,
  `--num_inference_steps`, and `--seed`.
- A locked-command line was constructed for a ready row
  `REAL_ENV231_00010_003_03`.
- No full inference was run because the preregistered six-row input set remains
  blocked by one empty-mask row.
- Reports:
  - `reports/exp29_effecterase_command_dryrun.md`
  - `reports/exp29_effecterase_command_dryrun.json`

## 2026-06-26 MiniMax Expanded Source-Pool Plan

- Status: `MINIMAX_EXPANDED_GENERATION_BLOCKED`.
- Required expanded pool: 96 or 128 sources.
- Current audit CSV has 64 rows, 63 valid aligned rows, and only 31 remaining
  valid rows after excluding the previous source32 gate.
- Remaining inventory manifest:
  `exp29_or_adapter_feasibility/manifests/minimax_expanded_source_pool_v2.jsonl`.
- Manifest SHA256:
  `bb31cfa5abd320dc88a5471036a3b2bb54b91257d3f65380dc43ecdf29c60929`.
- No MiniMax generation, recipe search, 30-step micro, or training was
  launched.
- Reports:
  - `reports/exp29_minimax_expanded_source_pool_plan.md`
  - `reports/exp29_minimax_expanded_source_pool_plan.json`

## 2026-06-26 Continuation V4 Readback

- Status: `EXP29_CONTINUATION_V4_READBACK_COMPLETED`.
- Branch/HEAD confirmed:
  `research/exp29-minimax-effecterase-adapter-feasibility-20260626` at
  `5e20149363b16f4728016260ff3e6d79dace299d`.
- EffectErase remains `EFFECTERASE_COMMAND_READY` but the old six-row smoke is
  input-invalid because `REAL_ENV249_00103_004_04` has an empty mask across all
  17 frames.
- MiniMax remains `MINIMAX_EXPANDED_GENERATION_BLOCKED` until a larger
  full-VOR source audit finds at least 128 valid candidate groups.
- Left CLI was checked read-only. GPU1/GPU2/GPU3/GPU4 remain reserved by CLI
  runtime locks. No signal was sent and no left-side file was modified.
- Report: `reports/exp29_continuation_v4_readback.md`.

## 2026-06-26 EffectErase Smoke V2 Pre-Registration

- Status: `EFFECTERASE_SMOKE_V2_PREREGISTERED`.
- Old manifest preserved at SHA256
  `54fd62a97fa69f2f17590488136d426cee77de0ed02548c46a83d8818be2b137`.
- Rejected row: `REAL_ENV249_00103_004_04`, because the smoke materialized mask
  is empty across all 17 frames.
- Replacement row: `REAL_ENV248_00118_005_03`.
- New v2 manifest SHA256:
  `b16a0007a22f190bb7894a673092063efb5dd2eda26dbd53737cdc987d9d4f36`.
- Rejected manifest SHA256:
  `4d092e0a0e3c126d7ede674bcc76f166638bf6b00bd12c10e435b1da6bc260f8`.
- Preview review passed for 6/6 rows; VOR-Eval use is false and training
  eligibility is false.
- No EffectErase inference has run yet.

## 2026-06-26 EffectErase Smoke V2 Input Materialization

- Status: `EFFECTERASE_SMOKE_V2_INPUTS_READY`.
- Rows: 6.
- Ready rows: 6.
- Blocked rows: 0.
- Each condition/winner/mask stream decodes as 17 frames at 832x480.
- Each mask video is non-empty in all 17 decoded frames.
- VOR-Eval use remains false and training eligibility remains false.
- No EffectErase inference has run yet.

## 2026-06-26 EffectErase Official Inference Smoke V2

- Status: `EFFECTERASE_SMOKE_BLOCKED_FRAME_COUNT_INCOMPATIBLE`.
- Attempted row: `REAL_ENV231_00010_003_03`.
- GPU/PID/PGID: GPU0 / `594851` / `594851`.
- Attempt 1 failed because `diffsynth` was not on `PYTHONPATH`.
- Automatic fix attempted once by adding the official EffectErase repo root to
  `PYTHONPATH` in the Exp29 runner.
- Attempt 2 loaded official model assets and LoRA, then failed with latent time
  mismatch: 81-frame default noise time dimension `21` vs 17-frame input latent
  time dimension `5`.
- No output video was produced; no metrics or visual quality pass are available.
- EffectErase remains not OR baseline-ready and not adapter-ready.

## 2026-06-26 MiniMax Full-VOR Source Audit

- Status: `MINIMAX_FULL_VOR_SOURCE_AUDIT_READY`.
- Full index:
  `/mnt/nas/hj/H20_Video_inpainting_DPO_exp25_vor/exp25_vor_or_preference_data/manifests/vor_train_metadata_index.jsonl`.
- Full index SHA256:
  `33d57a3ea23c5799b583d476a311089f95cbce1b0d11280822a63b8c9edcddc4`.
- Raw rows: 57,751.
- Raw scene groups: 1,449.
- Exclusions: previous MiniMax source32 = 32 scene groups; EffectErase smoke
  rows = 7 scene groups.
- Valid candidate groups after exclusions: 1,417.
- Locked candidate manifest:
  `exp29_or_adapter_feasibility/manifests/minimax_full_vor_source_candidates_v2.jsonl`.
- Manifest SHA256:
  `16e128282da110eeefd6cb56a517c8b6de82e42a5241c9b845e01315d9800f10`.
- Selected groups: 192, REAL/BLENDER = 96/96.
- Mask/effect/motion labels are pending materialization or metadata and were
  not guessed.
- No MiniMax generation, recipe search, 30-step micro, or training launched.

## 2026-06-26 MiniMax Expanded Data-Yield Review V2

- Status: `MINIMAX_EXPANDED_DATA_YIELD_INSUFFICIENT`.
- Source96 first-pass generation completed for seed A.
- Conditional seed B ran only for 32 preregistered near-miss candidates.
- Combined classification counts across 128 attempts:
  - `MEDIUM_HARD_ELIGIBLE`: 24
  - `HARD_BUT_PLAUSIBLE`: 2
  - `TOO_CLOSE`: 14
  - `TRIVIAL_BAD`: 77
  - `TECHNICAL_INVALID`: 11
- Best-candidate merge yielded 26 eligible unique scene groups, below the
  required 32 scene-disjoint groups for train16+heldout16.
- Train trace manifest SHA256:
  `84d3b3ce06216a05ea005fb29a91fdd40a1e73b7b1cd2ab7a49bb3e311683c95`.
- Heldout manifest is empty, SHA256:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Codex opened all 32 review pages: seed A 24/24 and seed B 8/8.
- No recipe, 30-step, training, or third-backbone-positive claim is allowed.

## 2026-06-27 Continuation V5 Readback

- Status: `EXP29_CONTINUATION_V5_READBACK_COMPLETED`.
- Branch: `research/exp29-minimax-effecterase-adapter-feasibility-20260626`.
- HEAD at readback: `c06958c762996dfe327e4a3024ad58550eb20d46`.
- Worktree was clean before readback edits.
- Required PRD, registry, reports, and relevant code paths were read.
- Left-side CLI was checked read-only; GPU1-GPU4 remain reserved by runtime
  heartbeat files even though all GPUs currently report 0 MiB usage.
- No left-side signal or file modification occurred.
- EffectErase remains blocked before baseline readiness by the 17F/81F official
  frame-count incompatibility.
- MiniMax remains blocked before recipe/training by insufficient eligible
  scene groups: 26 available vs 32 required.
- Next allowed milestones are EffectErase official-81F source audit and MiniMax
  top-up source audit.

## 2026-06-27 EffectErase Official 81F Source Audit

- Status: `EFFECTERASE_OFFICIAL81_PREREGISTERED`.
- Candidate triplets audited from existing Exp25 exact extraction caches: 24.
- Accepted by 81F/frame/mask rules: 24.
- Locked manifest rows: 8.
- Rejected/reserve rows recorded: 16.
- Manifest SHA256:
  `706cb09286fd8528d7efbbb91eb89673a9ec7ce61b0047e6b3b2e8ea4c9b1fb3`.
- Rejected manifest SHA256:
  `7700a99c5585d4b8759527bd48029b6ad90a8ca8c3c304a877b0bc1fbcce0f6e`.
- Source type counts: REAL 5, BLENDER 3.
- Mask bucket counts: small 3, medium 3, large 2.
- Codex opened all 8 preview sheets and marked 8/8 source sanity pass.
- The rows remain diagnostic-only VOR-confounded smoke inputs. No EffectErase
  inference, OR baseline-ready claim, trainable-forward claim, adapter claim, or
  training was launched.

## 2026-06-27 EffectErase Official 81F Input Materialization

- Status: `EFFECTERASE_OFFICIAL81_INPUTS_READY`.
- Manifest SHA256:
  `706cb09286fd8528d7efbbb91eb89673a9ec7ce61b0047e6b3b2e8ea4c9b1fb3`.
- Rows materialized: 8.
- Ready rows: 8.
- Blocked rows: 0.
- Resolution: 832x480.
- Frames per stream: 81.
- VOR-Eval use: false.
- Training eligibility: false.
- Codex opened all 8 materialized preview sheets and marked 8/8 input sanity
  pass.
- No EffectErase inference, OR baseline-ready claim, trainable-forward claim,
  adapter claim, or training was launched.
