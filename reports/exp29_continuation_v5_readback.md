# Exp29 Continuation V5 Readback

Date: 2026-06-27

Status: `EXP29_CONTINUATION_V5_READBACK_COMPLETED`

## Scope

This readback starts Exp29 continuation V5 only. It does not run EffectErase
inference, MiniMax generation, MiniMax recipe search, MiniMax 30-step,
EffectErase adapter training, long training, RC-FPO, or any left-side CLI task.

## Git

- Branch: `research/exp29-minimax-effecterase-adapter-feasibility-20260626`
- HEAD at readback: `c06958c762996dfe327e4a3024ad58550eb20d46`
- Worktree status before edits: clean
- Latest source-of-truth commit: `c06958c Review MiniMax expanded source-pool candidates v2`

## Files Read

PRD and registry:

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/49_exp29_or_adapter_feasibility.md`
- `experiment_registry/exp29_or_adapter_feasibility/status.md`
- `experiment_registry/exp29_or_adapter_feasibility/paths.yaml`
- `experiment_registry/exp29_or_adapter_feasibility/config.yaml`
- `experiment_registry/exp29_or_adapter_feasibility/results.tsv`
- `experiment_registry/exp29_or_adapter_feasibility/metric_summary.md`
- `experiment_registry/exp29_or_adapter_feasibility/qualitative_summary.md`

Reports:

- `reports/exp29_continuation_v4_readback.md`
- `reports/exp29_effecterase_smoke_v2_input_audit.md`
- `reports/exp29_effecterase_smoke_v2_preregistration.md`
- `reports/exp29_effecterase_smoke_v2_input_materialization.md`
- `reports/exp29_effecterase_command_dryrun.md`
- `reports/exp29_effecterase_inference_smoke_v2.md`
- `reports/exp29_minimax_full_vor_source_audit.md`
- `reports/exp29_minimax_expanded_source_pool_plan.md`
- `reports/exp29_minimax_expanded_data_quality_v2.md`
- `reports/exp29_minimax_expanded_video_review_v2.csv`
- `reports/exp29_minimax_expanded_data_quality_summary_v2.json`
- `reports/exp29_architecture_family_audit.md`
- `reports/exp29_effecterase_weight_recovery.md`
- `reports/exp29_minimax_effecterase_adapter_summary.md`

Code and assets read or located:

- Official EffectErase repo: `/home/hj/video_inpainting_third_party/EffectErase`
- Exp29 scripts: `exp29_or_adapter_feasibility/scripts/`
- Exp29 code: `exp29_or_adapter_feasibility/code/`
- Exp25 full VOR metadata index path from prior audit:
  `/mnt/nas/hj/H20_Video_inpainting_DPO_exp25_vor/exp25_vor_or_preference_data/manifests/vor_train_metadata_index.jsonl`

## Left CLI Protection

Left-side runtime was checked read-only at:

`/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/cli4`

Observed heartbeat reservations:

- `exp25_gate16_deb`: GPU1, status `running`
- `exp27_localdpo_24f`: GPU2, status `running`
- `exp28_pairB_inner4_eval_resume`: GPU1-2, status `running`
- `exp28_pairC_inner8` / `exp28_pairC_inner8_eval_resume_after_iowait`: GPU3-4, status `running`

No signal was sent to any left-side process. No left-side worktree, branch,
PID file, heartbeat, lock, or output file was modified.

## GPU State

Two PAI GPU checks were run approximately 60 seconds apart on
`dsw-753014-85f54df947-bkp7h`.

Both checks showed GPUs 0-7 at `0 MiB` used and `0%` utilization with no compute
processes. Despite this, GPU1-GPU4 remain reserved for left-side CLI due to
runtime heartbeats. Right-side Exp29 may only use non-left-reserved free GPUs,
preferably GPU0/GPU5/GPU6/GPU7, and at most two concurrently.

## Current EffectErase State

- Repo and weights are ready from prior milestones.
- V2 non-empty-mask 17-frame inputs were materialized successfully.
- Official v2 inference attempted one row and loaded model assets after a
  `PYTHONPATH` fix.
- Final blocker: official `WanRemovePipeline` used default 81-frame noise latent
  time dimension 21 while the locked 17-frame inputs encoded to latent time 5.
- Current status: `EFFECTERASE_SMOKE_BLOCKED_FRAME_COUNT_INCOMPATIBLE`.
- EffectErase is not OR baseline-ready and not adapter-ready.

V5 plan: abandon the 17-frame smoke as an official result and preregister a new
official-compatible 81-frame diagnostic smoke manifest. Do not patch the
official pipeline to force 17-frame compatibility.

## Current MiniMax State

- Expanded source-pool v2 completed 128 candidate attempts:
  seed A = 96, conditional seed B near-miss = 32.
- Combined classification counts:
  - `MEDIUM_HARD_ELIGIBLE`: 24
  - `HARD_BUT_PLAUSIBLE`: 2
  - `TOO_CLOSE`: 14
  - `TRIVIAL_BAD`: 77
  - `TECHNICAL_INVALID`: 11
- Eligible unique scene groups after merge: 26.
- Required for scene-disjoint train16+heldout16: 32.
- Current status: `MINIMAX_EXPANDED_DATA_YIELD_INSUFFICIENT`.
- No recipe, 30-step, or training is allowed until top-up data-yield creates a
  valid scene-disjoint train16/heldout16 split.

V5 plan: build a top-up source audit from full VOR valid triplet metadata,
excluding all previous MiniMax and EffectErase rows, then run top-up generation
only if the source audit is ready. Do not run MiniMax 30-step.

## Forbidden Repeats And Claims

Do not repeat architecture-family audit, EffectErase weight recovery, old 17F
smoke as official output, MiniMax recipe/30-step before data readiness, long
training, RC-FPO, Exp1-Exp28 edits, `inference/metrics.py` edits,
shared-trainer edits, or left-side CLI operations.

Allowed scientific language remains:

- DiffuEraser + VideoPainter support cross-backbone adapter evidence.
- MiniMax remains plumbing-positive but data-yield-limited.
- EffectErase is command/weight-ready but not OR baseline-ready until official
  81-frame smoke succeeds and is visually/quantitatively reviewed.

Forbidden language:

- `UNIVERSAL_ADAPTER`
- `ALL_MODELS_SUPPORTED`
- `FINAL_SOTA`
- `TOP_CONFERENCE_NOVELTY_CONFIRMED`

## Next Milestones

1. EffectErase official-81F source audit and preregistration.
2. EffectErase official-81F input materialization.
3. EffectErase official-81F command validation.
4. EffectErase official-81F inference smoke and OR baseline diagnostic, only
   after the above gates pass.
5. MiniMax top-up source audit.
6. MiniMax top-up candidate generation, only if the source audit is ready.
7. Optional MiniMax 10-step recipe gate, only if top-up creates
   scene-disjoint train16+heldout16.
