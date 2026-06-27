# PRD 49: Exp29 OR Adapter Feasibility

## Objective

Exp29 audits MiniMax-Remover and EffectErase as:

1. OR baselines;
2. loser generators;
3. possible future true DPO adapter backbones.

This is a feasibility and evidence-quality track. It does not start long
training, RC-FPO, VideoPainter 100-step continuation, or third-backbone DPO
training.

## Source State

Exp26 VideoPainter LoVI-DPO completed search-dev, independent shadow-dev, and
external DAVIS-derived validation. The accepted statement is:

`CROSS_BACKBONE_ADAPTER_EVIDENCE_CONFIRMED`

The forbidden statement remains:

`UNIVERSAL_ADAPTER`

External DAVIS-derived VideoPainter validation was not confirmed, so cross-data
robustness remains open.

## Left CLI Protection

The left CLI controller and its worktrees are treated as read-only external
state. Exp29 may inspect GPU/process status but must not modify files, locks,
heartbeats, branches, worktrees, or processes associated with:

- `/home/hj/cli4_controller`
- `/home/hj/H20_Video_inpainting_DPO_exp25_cli4`
- `/home/hj/H20_Video_inpainting_DPO_exp27_cli4`
- `/home/hj/H20_Video_inpainting_DPO_exp28_inner_boundary`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/cli4`

## Initial Readback

- Branch: `research/exp29-minimax-effecterase-adapter-feasibility-20260626`
- Base: `origin/research/exp26-videopainter-dpo-v2`
- Initial base HEAD: `568a7dfb48bcdfce893176a1dd48c653414a13a8`
- PRD selected: `PRD/49_exp29_or_adapter_feasibility.md`
- New isolated code root: `exp29_or_adapter_feasibility/`
- New registry: `experiment_registry/exp29_or_adapter_feasibility/`

## Initial Local Asset Hints

- MiniMax local candidate repo:
  `/home/hj/dpo-2-1-exp/third_party_baselines/MiniMax-Remover/repo`
- EffectErase local candidate repo:
  `/home/hj/video_inpainting_third_party/EffectErase`

These are hints only. Exp29 must still perform repo, license, weight, code, and
trainable-forward audits before running any smoke.

## 2026-06-26 Repo And Weight Audit

MiniMax:

- `MINIMAX_REPO_READY`
- `MINIMAX_WEIGHTS_READY`
- Local repo: `/home/hj/dpo-2-1-exp/third_party_baselines/MiniMax-Remover/repo`
- PAI/NAS weights:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/minimax_remover/current`
- Flow target from paper/code audit: velocity / `epsilon - z0`.
- No inference smoke, trainable-forward gate, zero-gap, one-step, or 10-step
  gate has run yet.

EffectErase:

- `EFFECTERASE_REPO_READY`
- `EFFECTERASE_BLOCKED_NO_WEIGHTS`
- Local repo: `/home/hj/video_inpainting_third_party/EffectErase`
- Official inference requires `EffectErase.ckpt` and `Wan2.1-Fun-1.3B-InP`
  assets, which were not found in the audited paths.
- Generic Wan training utilities are present, but removal-specific adapter
  feasibility is blocked until official weights are available.
- VOR use remains diagnostic/baseline only because of the VOR-training
  confound recorded in Exp26.

Reports:

- `reports/exp29_minimax_repo_weight_audit.md`
- `reports/exp29_effecterase_repo_weight_audit.md`

## Promotion Gates

MiniMax can only become `MINIMAX_TRUE_ADAPTER_FEASIBILITY_CONFIRMED` after:

- verified repo and weights;
- official inference smoke with video review;
- native trainable forward;
- policy/reference zero-gap;
- one-step strict reload;
- 10-step micro gate with held-out review.

EffectErase can only become `EFFECTERASE_TRUE_ADAPTER_FEASIBILITY_CONFIRMED`
after the same technical evidence plus a non-confounded data design. If its
available evidence is VOR-trained only, it must be labeled technical or
baseline/diagnostic rather than scientific positive.

## 2026-06-26 Inference Smoke And Trainable Forward

MiniMax:

- `MINIMAX_INFERENCE_SMOKE_PASSED_WITH_VISUAL_QUALITY_RISKS`
- 4/4 fixed smoke samples ran with official local weights and produced 9
  frames each.
- Visual review:
  - `davis_bear`: medium-hard candidate; mostly successful local removal.
  - `davis_bus`: trivial-bad; large bus remains.
  - `davis_mallard-water`: trivial-bad; duck remains with blue/black artifact.
  - `davis_elephant`: trivial-bad; elephant remains with haze/smoothing.
- `MINIMAX_TRAINABLE_FORWARD_PASSED`
- Flow target implemented as `epsilon - z0`; native transformer backward
  produced finite gradients.

EffectErase:

- `EFFECTERASE_INFERENCE_SMOKE_BLOCKED_NO_WEIGHTS`
- Official repo is available, but `EffectErase.ckpt` and required Wan assets
  were not found. No fallback checkpoint was used.

Reports:

- `reports/exp29_minimax_inference_smoke.md`
- `reports/exp29_minimax_inference_visual_review.csv`
- `reports/exp29_minimax_trainable_forward_audit.md`
- `reports/exp29_effecterase_inference_smoke.md`

## 2026-06-26 Adapter Gate Decision

MiniMax:

- `MINIMAX_ZERO_GAP_PASSED`
- `MINIMAX_ONE_STEP_STRICT_RELOAD_PASSED`
- `MINIMAX_10STEP_PARETO_MIXED`
- Final status: `MINIMAX_ADAPTER_POSSIBLE_NEEDS_MORE_WORK`

The first fp16 AdamW micro attempt produced NaNs after step 1. A conservative
SGD micro-update was used as an engineering fix to verify finite update
mechanics. It completed 10 steps without NaNs and strict-loaded step1/step10,
but heldout Step10 videos were visually almost unchanged from Step0. Therefore
MiniMax is a credible next-adapter candidate, but not a confirmed third
backbone yet.

EffectErase:

- Final status: `EFFECTERASE_BLOCKED`

No EffectErase smoke or adapter gate was run because official weights were not
available.

Reports:

- `reports/exp29_minimax_zero_gap_gate.md`
- `reports/exp29_minimax_one_step_gate.md`
- `reports/exp29_minimax_10step_micro.md`
- `reports/exp29_minimax_effecterase_adapter_summary.md`

## 2026-06-26 Continuation Readback

- Status: `EXP29_CONTINUATION_READBACK_COMPLETED`
- Branch: `research/exp29-minimax-effecterase-adapter-feasibility-20260626`
- HEAD: `4b8d68af3ebd0f6981e697baee952b5f0e1ca76f`
- Left CLI protection was re-audited read-only. The observed left runtime keeps
  GPU1/GPU2/GPU3/GPU4 reserved, with Exp28 DAVIS50 evaluation active on GPU3.
  No signal was sent and no left-side file was modified.
- Right-side eligible GPU candidates after two checks: GPU0/GPU5/GPU6/GPU7,
  max two concurrent tasks.
- MiniMax remains `MINIMAX_ADAPTER_POSSIBLE_NEEDS_MORE_WORK` pending a
  medium-hard data-quality gate and a better small optimizer/precision recipe.
- EffectErase remains `EFFECTERASE_BLOCKED` until official weights and Wan base
  assets are recovered or confirmed unavailable.

Report:

- `reports/exp29_continuation_readback.md`

## 2026-06-26 MiniMax 10-Step Failure Analysis

- Status: `MINIMAX_10STEP_FAILURE_ANALYZED`
- The previous MiniMax 10-step gate used `SGD(lr=1e-7)` after fp16 AdamW
  produced NaNs. This stabilized plumbing but produced a step10 parameter delta
  probe of only `1.1061271569642785e-10`.
- Gradients were finite, so the primary issue was not a missing backward path.
  The effective update was too small, the training losers were mostly
  trivial-bad, and the heldout set had only two rows.
- Decision: do not extend the same recipe. MiniMax must first pass a
  medium-hard train16/heldout16 data-quality gate, then a bounded optimizer /
  precision recipe gate, before any 30-step micro can run.

Reports:

- `reports/exp29_minimax_10step_failure_analysis.md`
- `reports/exp29_minimax_10step_failure_analysis.csv`
- `reports/exp29_minimax_next_micro_plan.md`

## 2026-06-26 MiniMax Preference Data Quality Gate

- Status: `MINIMAX_DATA_YIELD_INSUFFICIENT`
- Candidate generation used 32 VOR-train/search/shadow non-VOR-Eval sources,
  17 frames per source, and three pre-registered seeds per source.
- Total candidates: 96.
- Classification:
  - `MEDIUM_HARD_ELIGIBLE`: 23
  - `HARD_BUT_PLAUSIBLE`: 4
  - `TOO_CLOSE`: 3
  - `TRIVIAL_BAD`: 60
  - `TECHNICAL_INVALID`: 6
- Although 27 candidates were eligible, they came from only 9 unique scene
  groups. A scene-disjoint train16/heldout16 split is therefore impossible.
- Locked manifests:
  - train rows: 9, SHA256
    `8d9986537f04ef36a9907f093663593b5e9d87e131ed130935796d7df29cd33d`
  - heldout rows: 0, SHA256
    `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
  - rejected rows: 87, SHA256
    `38f190abc1005cbbae5a475dc3023149132aee5fa69eeaad2066d8f987ee932f`
- Decision: do not run optimizer recipe search or 30-step MiniMax micro gate
  from this data. MiniMax remains `ADAPTER_POSSIBLE_NEEDS_MORE_WORK`, not a
  third-backbone quality positive.

Reports:

- `reports/exp29_minimax_preference_data_quality.md`
- `reports/exp29_minimax_preference_data_quality.csv`
- `reports/exp29_minimax_preference_video_review.csv`
- `reports/exp29_minimax_preference_data_quality_summary.json`

## 2026-06-26 EffectErase Weight Recovery

- Status: `EFFECTERASE_WEIGHTS_READY`
- Official EffectErase and Wan2.1-Fun InP assets were downloaded on HAL and
  rsynced to the Exp29 PAI/NAS autoresearch cache.
- PAI SHA256 verification checked 19 manifest entries and all returned `OK`.
- Key recovered assets: `EffectErase.ckpt`, `Wan2.1_VAE.pth`,
  `diffusion_pytorch_model.safetensors`,
  `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`, and
  `models_t5_umt5-xxl-enc-bf16.pth`.
- Asset root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626`
- This milestone does not run inference smoke or adapter gates. EffectErase
  remains VOR baseline/diagnostic only until non-confounded evaluation is
  locked.

Reports:

- `reports/exp29_effecterase_weight_recovery.md`
- `reports/exp29_effecterase_weight_recovery.csv`
- `reports/exp29_effecterase_weight_recovery.json`

## 2026-06-26 Continuation V2 Readback

- Status: `EXP29_CONTINUATION_V2_READBACK_COMPLETED`
- Branch/HEAD confirmed:
  `research/exp29-minimax-effecterase-adapter-feasibility-20260626` at
  `6c97d4b74f331ce4db089224f7dcf9ec6eb283ce`.
- Re-read Exp29 PRD, registry, previous MiniMax reports, EffectErase weight
  recovery reports, and current code pointers before any GPU task.
- Left CLI was inspected read-only. Runtime locks reserve GPU1/GPU2/GPU3/GPU4
  even though PAI GPUs currently report 0 MiB and no compute process.
- No left-side signal was sent and no left-side file was modified.
- Next allowed milestones are architecture-family audit, EffectErase smoke
  pre-registration/smoke, and MiniMax expanded source-pool planning.

Report:

- `reports/exp29_continuation_v2_readback.md`

## 2026-06-26 Architecture Family Audit

- Status: `EXP29_ARCHITECTURE_FAMILY_AUDIT_COMPLETED`
- DiffuEraser and VideoPainter are confirmed LoVI-DPO adapter backbones but do
  not imply one shared SD1.5 trainer.
- MiniMax is audited as Wan2.1 / DiT / flow-matching with target `epsilon - z0`.
  The current Exp29 MiniMax code path does not trigger
  `MINIMAX_GATE_INVALID_TARGET_MISMATCH`.
- EffectErase is audited as Wan / DiT / flow-style remove pipeline with
  recovered weights, but remains OR baseline/diagnostic until inference smoke
  and any trainable-forward gates are proven.
- Required language: model-specific backend adapter. Forbidden language:
  universal adapter, all models supported, final SOTA.

Reports:

- `reports/exp29_architecture_family_audit.md`
- `reports/exp29_architecture_family_audit.csv`

## 2026-06-26 EffectErase Smoke Pre-Registration

- Status: `EFFECTERASE_SMOKE_PREREGISTERED`
- Locked manifest:
  `exp29_or_adapter_feasibility/manifests/effecterase_smoke_preregistered.jsonl`
- Manifest SHA256:
  `54fd62a97fa69f2f17590488136d426cee77de0ed02548c46a83d8818be2b137`
- Rows: 6 diagnostic rows, balanced as REAL/BLENDER = 3/3 and
  small/medium/large masks = 2/2/2.
- All rows use verified 17-frame Exp29 materialized condition/winner/mask frame
  directories. The official EffectErase default is 81 frames, so this is
  explicitly a diagnostic compatibility smoke, not an official full benchmark.
- VOR-Eval is not used. No non-VOR OR triplet was available at preregistration
  time.
- Every row is tagged `diagnostic_only_vor_confounded` and
  `eligible_for_training=false`.
- Fixed protocol: removal task, raw output primary, diagnostic comp optional,
  832x480, seed 2025, CFG 1.0, 50 steps, frame interval 1.
- This milestone does not run inference and cannot support a true adapter or
  scientific-positive claim.

Reports:

- `reports/exp29_effecterase_smoke_preregistration.md`
- `reports/exp29_effecterase_smoke_preregistration.json`

## 2026-06-26 Continuation V3 Readback

- Status: `EXP29_CONTINUATION_V3_READBACK_COMPLETED`
- Branch/HEAD confirmed:
  `research/exp29-minimax-effecterase-adapter-feasibility-20260626` at
  `972deab321a518638102a1ace6ed87a13456a261`.
- Re-read Exp29 PRD, registry, latest EffectErase recovery/preregistration
  reports, MiniMax data-yield reports, and relevant code pointers before any
  new GPU inference.
- EffectErase remains `EFFECTERASE_WEIGHTS_READY` and
  `EFFECTERASE_SMOKE_PREREGISTERED`; no EffectErase inference output has run.
- MiniMax remains `MINIMAX_DATA_YIELD_INSUFFICIENT`; recipe search and 30-step
  micro remain forbidden.
- Left CLI was audited read-only. Runtime locks reserve GPU1/GPU2/GPU3/GPU4;
  no left signal was sent and no left file was modified.

Report:

- `reports/exp29_continuation_v3_readback.md`

## 2026-06-26 EffectErase Smoke Input Materialization

- Status: `EFFECTERASE_SMOKE_INPUTS_BLOCKED`
- Locked smoke manifest remained unchanged:
  `54fd62a97fa69f2f17590488136d426cee77de0ed02548c46a83d8818be2b137`.
- Materialized input mp4s for the preregistered rows under:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_20260626/inputs/`
- Ready rows: 5/6.
- Blocked row: `REAL_ENV249_00103_004_04`; its locked mask is empty across the
  materialized 17 frames.
- No row was replaced, no seed/mask/frame index was changed, and no EffectErase
  inference was launched.
- Because the preregistered six-row smoke is not input-valid, Milestone C
  cannot run as a 6/6 official smoke without a new preregistration.

Reports:

- `reports/exp29_effecterase_smoke_input_materialization.md`
- `reports/exp29_effecterase_smoke_input_materialization.csv`
- `reports/exp29_effecterase_smoke_input_materialization.json`

## 2026-06-26 EffectErase Command Dry-Run

- Status: `EFFECTERASE_COMMAND_READY`
- Official runtime repo copy:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/third_party/EffectErase`
- Dedicated venv:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/venvs/effecterase_smoke/bin/python`
- Environment fix: `transformers==4.51.3`, `diffusers==0.31.0`, and
  `decord==0.6.0` allow official `examples/remove_wan/infer_remove_wan.py` to
  import successfully.
- Official script supports `--num_frames`, `--cfg`, `--num_inference_steps`,
  and `--seed`; 17-frame override is accepted by command construction.
- Core recovered asset SHA256 values were rechecked for LoRA, VAE, DiT, image
  encoder, and text encoder.
- No full inference was run because Milestone A is still
  `EFFECTERASE_SMOKE_INPUTS_BLOCKED` for the locked six-row manifest.

Reports:

- `reports/exp29_effecterase_command_dryrun.md`
- `reports/exp29_effecterase_command_dryrun.json`

## 2026-06-26 MiniMax Expanded Source-Pool Plan

- Status: `MINIMAX_EXPANDED_GENERATION_BLOCKED`
- Planned first-pass requirement: 96 or 128 non-VOR-Eval sources, excluding the
  previous 32 sources.
- Available source audit:
  `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp25_vor/reports/vor_triplet_audit64_semantic.csv`
- Audit rows: 64.
- Valid aligned rows: 63.
- Previous source32 rows excluded: 32.
- Remaining valid rows: 31.
- Remaining inventory manifest:
  `exp29_or_adapter_feasibility/manifests/minimax_expanded_source_pool_v2.jsonl`
- Manifest SHA256:
  `bb31cfa5abd320dc88a5471036a3b2bb54b91257d3f65380dc43ecdf29c60929`
- The manifest rows are marked `eligible_for_generation=false` because the
  preregistered source-count requirement cannot be met.
- No MiniMax inference, recipe search, 30-step micro, or training was launched.

Reports:

- `reports/exp29_minimax_expanded_source_pool_plan.md`
- `reports/exp29_minimax_expanded_source_pool_plan.json`

## 2026-06-26 Exp29 Continuation V4 Readback

- Status: `EXP29_CONTINUATION_V4_READBACK_COMPLETED`
- HEAD: `5e20149363b16f4728016260ff3e6d79dace299d`
- Re-read Exp29 PRD, registry, EffectErase smoke materialization/dry-run,
  EffectErase preregistration/weight recovery, MiniMax source-plan and
  data-quality reports, plus current Exp29 helper code.
- EffectErase remains command-ready but blocked by the old locked manifest's
  empty-mask row `REAL_ENV249_00103_004_04`; v2 may replace that row only after
  explicit non-empty-mask input audit and preview review.
- MiniMax remains expanded-generation blocked because the prior 64-row Exp25
  semantic audit left only 31 unused valid rows. This round must use a larger
  full-VOR source audit before any new generation.
- Left CLI was checked read-only on PAI. Runtime locks still reserve
  GPU1/GPU2/GPU3/GPU4. No signal was sent and no left-side file was modified.
- Right Exp29 may consider only GPU0/GPU5/GPU6/GPU7 after repeated availability
  checks, with at most two concurrent GPU tasks.
- No EffectErase inference, MiniMax generation, recipe, 30-step, adapter
  training, long training, or RC-FPO was launched by this readback.

Report:

- `reports/exp29_continuation_v4_readback.md`

## 2026-06-26 EffectErase Smoke V2 Pre-Registration

- Status: `EFFECTERASE_SMOKE_V2_PREREGISTERED`
- Old manifest preserved:
  `exp29_or_adapter_feasibility/manifests/effecterase_smoke_preregistered.jsonl`
- Old manifest SHA256:
  `54fd62a97fa69f2f17590488136d426cee77de0ed02548c46a83d8818be2b137`
- Rejected old row: `REAL_ENV249_00103_004_04`, because its smoke
  materialized mask is empty across all 17 frames.
- Replacement row: `REAL_ENV248_00118_005_03`
- New v2 manifest:
  `exp29_or_adapter_feasibility/manifests/effecterase_smoke_preregistered_v2.jsonl`
- New v2 manifest SHA256:
  `b16a0007a22f190bb7894a673092063efb5dd2eda26dbd53737cdc987d9d4f36`
- Rejected manifest:
  `exp29_or_adapter_feasibility/manifests/effecterase_smoke_rejected_v2.jsonl`
- Rejected manifest SHA256:
  `4d092e0a0e3c126d7ede674bcc76f166638bf6b00bd12c10e435b1da6bc260f8`
- Balance preserved: REAL/BLENDER = 3/3 and small/medium/large masks = 2/2/2.
- VOR-Eval use: false. Training eligibility: false for all rows.
- Preview review: 6/6 rows inspected via single-sample temporal sheets and
  classified `PREVIEW_INPUT_VALID`.
- No EffectErase inference, baseline-ready claim, adapter feasibility claim, or
  scientific-positive claim is made by this preregistration.

Reports:

- `reports/exp29_effecterase_smoke_v2_input_audit.md`
- `reports/exp29_effecterase_smoke_v2_input_audit.csv`
- `reports/exp29_effecterase_smoke_v2_rejected_rows.csv`
- `reports/exp29_effecterase_smoke_v2_preview_review.csv`
- `reports/exp29_effecterase_smoke_v2_preregistration.md`
- `reports/exp29_effecterase_smoke_v2_preregistration.json`

## 2026-06-26 EffectErase Smoke V2 Input Materialization

- Status: `EFFECTERASE_SMOKE_V2_INPUTS_READY`
- Manifest:
  `exp29_or_adapter_feasibility/manifests/effecterase_smoke_preregistered_v2.jsonl`
- Rows: 6
- Ready rows: 6
- Blocked rows: 0
- Resolution: 832x480
- Condition/winner/mask frame counts: 17/17/17 for every row.
- Mask non-empty frames: 17/17 for every row.
- VOR-Eval use: false. Training eligibility: false.
- Output root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_inference_smoke_v2_20260626`
- No EffectErase inference was launched by this materialization milestone.

Reports:

- `reports/exp29_effecterase_smoke_v2_input_materialization.md`
- `reports/exp29_effecterase_smoke_v2_input_materialization.csv`
- `reports/exp29_effecterase_smoke_v2_input_materialization.json`

## 2026-06-26 EffectErase Official Inference Smoke V2

- Status: `EFFECTERASE_SMOKE_BLOCKED_FRAME_COUNT_INCOMPATIBLE`
- Attempted sample: `REAL_ENV231_00010_003_03`
- GPU: PAI GPU0
- Runner PID/PGID: `594851` / `594851`
- Attempt 1 failed with `ModuleNotFoundError: No module named 'diffsynth'`.
- Automatic fix attempted once: add the EffectErase official repo root to
  `PYTHONPATH` in the Exp29 runner.
- Attempt 2 loaded the DiT, text encoder, VAE, image encoder, and LoRA, then
  failed at inference with latent time mismatch: pipeline noise time dimension
  `21` versus input condition/mask latent time dimension `5`.
- Code audit shows official `infer_remove_wan.py` reads inputs with
  `args.num_frames`, but does not pass `num_frames=args.num_frames` into
  `WanRemovePipeline.__call__`, whose default is 81 frames.
- Per preregistration, this milestone did not patch the official script, did
  not expand inputs to 81 frames, did not change the manifest, and did not
  claim baseline readiness.
- No output video, metrics, visual quality pass, adapter feasibility, or
  scientific-positive claim is supported.

Reports:

- `reports/exp29_effecterase_inference_smoke_v2.md`
- `reports/exp29_effecterase_inference_smoke_v2.csv`
- `reports/exp29_effecterase_inference_visual_review_v2.csv`
- `reports/exp29_effecterase_inference_metrics_v2.csv`
- `reports/exp29_effecterase_inference_summary_v2.json`

## 2026-06-26 MiniMax Full-VOR Source Audit

- Status: `MINIMAX_FULL_VOR_SOURCE_AUDIT_READY`
- Full VOR Train metadata index:
  `/mnt/nas/hj/H20_Video_inpainting_DPO_exp25_vor/exp25_vor_or_preference_data/manifests/vor_train_metadata_index.jsonl`
- Full metadata SHA256:
  `33d57a3ea23c5799b583d476a311089f95cbce1b0d11280822a63b8c9edcddc4`
- Raw rows read: 57,751.
- Raw scene groups: 1,449.
- Previous MiniMax source32 excluded: 32 rows / 32 scene groups.
- EffectErase smoke rows excluded: 12 rows / 7 scene groups.
- Valid candidate groups after exclusions: 1,417.
- Locked candidate manifest:
  `exp29_or_adapter_feasibility/manifests/minimax_full_vor_source_candidates_v2.jsonl`
- Candidate manifest SHA256:
  `16e128282da110eeefd6cb56a517c8b6de82e42a5241c9b845e01315d9800f10`
- Selected groups: 192, with REAL/BLENDER = 96/96.
- Mask bucket, effect type, and motion bucket are explicitly recorded as
  `unknown_pending_materialization` / `unknown_pending_metadata` because the
  full metadata index does not contain those labels.
- This milestone fixes the previous 31-row source-pool blocker only. It does
  not claim MiniMax micro-data quality, does not create train16/heldout16, and
  does not run generation, recipe search, 30-step, or training.

Reports:

- `reports/exp29_minimax_full_vor_source_audit.md`
- `reports/exp29_minimax_full_vor_source_audit.csv`
- `reports/exp29_minimax_full_vor_source_audit.json`

## 2026-06-26 MiniMax Expanded Source-Pool Candidate Review V2

- Status: `MINIMAX_EXPANDED_DATA_YIELD_INSUFFICIENT`
- Source96 first-pass manifest SHA256:
  `267d03a2991894a47e26a14b698f6fd28423a726e6968890448cd460e5de1928`
- Final materialized source96 SHA256:
  `a8998902daa8e771afd111e798df017e4b64f5f21f5a43bf5fa6ef82aa4ce428`
- Seed A generation completed: 96/96 candidates.
- Seed A classification: `MEDIUM_HARD_ELIGIBLE` 23,
  `HARD_BUT_PLAUSIBLE` 2, `TOO_CLOSE` 7, `TRIVIAL_BAD` 53,
  `TECHNICAL_INVALID` 11.
- Conditional seed B near-miss manifest SHA256:
  `1d45c60cdd54a28fe98373bd88d53b4cb277c649c292ad9a3e00c4aa718a6aad`
- Seed B generation completed: 32/32 near-miss candidates.
- Seed B classification: `MEDIUM_HARD_ELIGIBLE` 1, `TOO_CLOSE` 7,
  `TRIVIAL_BAD` 24.
- Combined attempts: 128.
- Best-candidate merge unique scene groups attempted: 96.
- Eligible unique scene groups after seed A/B merge: 26, below the 32
  scene-disjoint train16+heldout16 requirement.
- Train trace manifest SHA256:
  `84d3b3ce06216a05ea005fb29a91fdd40a1e73b7b1cd2ab7a49bb3e311683c95`
- Heldout manifest is empty, SHA256:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Codex opened and reviewed all per-sample evidence pages: seed A 24/24 and
  seed B 8/8.
- No MiniMax recipe search, 30-step micro, adapter training, long training, or
  RC-FPO was launched.

Reports:

- `reports/exp29_minimax_expanded_data_quality_v2.md`
- `reports/exp29_minimax_expanded_data_quality_v2.csv`
- `reports/exp29_minimax_expanded_video_review_v2.csv`
- `reports/exp29_minimax_expanded_data_quality_summary_v2.json`
- `reports/exp29_minimax_expanded_review_pages_v2/`

## 2026-06-27 Continuation V5 Readback

- Status: `EXP29_CONTINUATION_V5_READBACK_COMPLETED`
- Branch: `research/exp29-minimax-effecterase-adapter-feasibility-20260626`
- HEAD at readback: `c06958c762996dfe327e4a3024ad58550eb20d46`
- Worktree status before edits: clean.
- Read PRD, registry, prior EffectErase v2 reports, MiniMax full-VOR source
  audit, MiniMax expanded data-yield v2, architecture-family audit, and
  EffectErase weight recovery reports.
- Left CLI was checked read-only. Runtime heartbeats still reserve GPU1-GPU4
  for Exp25/Exp27/Exp28 lanes, even though PAI currently reports all GPUs at
  0 MiB and 0% utilization.
- No signal was sent to left CLI and no left-side file was modified.
- EffectErase state remains
  `EFFECTERASE_SMOKE_BLOCKED_FRAME_COUNT_INCOMPATIBLE`; the 17-frame v2 smoke
  is not an official result and must not be promoted.
- V5 EffectErase plan is official 81-frame diagnostic smoke only:
  source audit, preregistration, materialization, command validation, then
  inference/metrics/visual review if gates pass.
- MiniMax state remains `MINIMAX_EXPANDED_DATA_YIELD_INSUFFICIENT`, with 26
  eligible unique scene groups after 128 candidate attempts.
- V5 MiniMax plan is top-up data-yield only until a scene-disjoint
  train16+heldout16 split exists. MiniMax 30-step remains forbidden.

Report:

- `reports/exp29_continuation_v5_readback.md`

## 2026-06-27 EffectErase Official 81F Source Audit

- Status: `EFFECTERASE_OFFICIAL81_PREREGISTERED`
- Switched away from the blocked 17-frame diagnostic smoke. The official
  EffectErase smoke is now preregistered for 81 real frames without patching the
  official pipeline to fit 17-frame inputs.
- Used existing Exp25 full-VOR metadata and exact selective-extraction caches
  read-only; no VOR archive rescan, VOR-Eval use, or row replacement after
  viewing outputs occurred.
- Candidate triplets audited: 24.
- Accepted by 81F/frame/mask rules: 24.
- Locked rows: 8.
- Source type balance: REAL 5, BLENDER 3.
- Mask bucket balance: small 3, medium 3, large 2.
- Manifest SHA256:
  `706cb09286fd8528d7efbbb91eb89673a9ec7ce61b0047e6b3b2e8ea4c9b1fb3`.
- Codex opened all 8 source preview sheets and marked 8/8 as valid source
  sanity passes. No empty mask, loop, padding, or visible condition/winner frame
  mismatch was found.
- This is source preregistration only. No EffectErase inference, baseline-ready
  claim, trainable-forward claim, or adapter claim is made yet.

Reports:

- `reports/exp29_effecterase_official81_source_audit.md`
- `reports/exp29_effecterase_official81_source_audit.csv`
- `reports/exp29_effecterase_official81_preregistration.md`
- `reports/exp29_effecterase_official81_preregistration.json`
- `reports/exp29_effecterase_official81_preview_review.csv`
- `reports/exp29_effecterase_official81_previews/`

## 2026-06-27 EffectErase Official 81F Input Materialization

- Status: `EFFECTERASE_OFFICIAL81_INPUTS_READY`
- Materialized all 8 locked official-81F diagnostic rows into 832x480
  condition/winner/mask MP4s under the official81 output root.
- Frame audit: 8/8 rows have condition/winner/mask = 81/81/81 decoded frames.
- Mask audit: 8/8 rows have non-empty masks; no row is VOR-Eval or training
  eligible.
- Codex opened all 8 materialized preview sheets and marked 8/8 as input sanity
  pass after resize/encoding.
- No EffectErase inference, baseline-ready claim, trainable-forward claim, or
  adapter claim is made yet.

Reports:

- `reports/exp29_effecterase_official81_input_materialization.md`
- `reports/exp29_effecterase_official81_input_materialization.csv`
- `reports/exp29_effecterase_official81_input_materialization.json`
- `reports/exp29_effecterase_official81_materialized_preview_review.csv`
- `reports/exp29_effecterase_official81_materialized_previews/`

## 2026-06-27 EffectErase Official 81F Command Validation

- Status: `EFFECTERASE_OFFICIAL81_COMMAND_READY`
- Validated the official EffectErase command in dry-run mode only; no full model
  inference was launched.
- Official script import/help returned successfully in the pinned EffectErase
  venv.
- Assets ready: true.
- Inputs ready: true, 8/8 rows.
- Command uses `--num_frames 81`, `--height 480`, `--width 832`, seed 2025,
  CFG 1.0, and 50 inference steps.
- VOR-Eval use: false.
- Training eligibility: false.
- This unlocks 81F inference smoke, but still does not support baseline-ready,
  trainable-forward, adapter, or scientific-positive claims.

Reports:

- `reports/exp29_effecterase_official81_command_validation.md`
- `reports/exp29_effecterase_official81_command_validation.json`
- `reports/exp29_effecterase_official81_command_validation_inputs.csv`

## 2026-06-27 EffectErase Official 81F Inference Smoke

- Status: `EFFECTERASE_OR_BASELINE_READY`
- Full official EffectErase inference ran on the locked 8-row official-81F
  diagnostic manifest.
- Manifest SHA256:
  `706cb09286fd8528d7efbbb91eb89673a9ec7ce61b0047e6b3b2e8ea4c9b1fb3`.
- GPU used: right-side GPU0 only.
- Output rows: 8/8.
- Exit codes: 8/8 zero.
- Frame audit: 8/8 raw outputs decode as 81 frames at 832x480.
- Project metric wrapper:
  - whole PSNR: `27.416948`
  - whole SSIM: `0.840580`
  - whole LPIPS: `0.085822`
  - mask PSNR: `25.778614`
  - boundary PSNR: `25.696018`
  - Ewarp mask-region: `1.766501`
  - outside diff mean: `8.210687`
- Codex opened all 8 temporal review pages and all 8 crop pages.
- Visual result: 8/8 show strong object/effect removal and no black/purple
  collapse.
- Caveat: the model is strong and VOR-trained; raw outputs sometimes regenerate
  outside/context regions and are too clean/strong for primary medium-hard loser
  use.

Decision:

`EFFECTERASE_OR_BASELINE_READY`

This is an OR baseline / diagnostic readiness result only. It does not support
`EFFECTERASE_TRUE_ADAPTER_FEASIBILITY_CONFIRMED`, `SCIENTIFIC_POSITIVE`, or
`UNIVERSAL_ADAPTER`.

Reports:

- `reports/exp29_effecterase_official81_inference_smoke.md`
- `reports/exp29_effecterase_official81_inference_smoke.csv`
- `reports/exp29_effecterase_official81_inference_visual_review.csv`
- `reports/exp29_effecterase_official81_aggregate_metrics.csv`
- `reports/exp29_effecterase_official81_inference_summary.json`
- `reports/exp29_effecterase_official81_project_metric_eval/`
- `reports/exp29_effecterase_official81_inference_previews/`
