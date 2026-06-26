# Exp29 Continuation V4 Readback

Date: 2026-06-26 / PAI 2026-06-27

Status: `EXP29_CONTINUATION_V4_READBACK_COMPLETED`

## Git State

- Branch: `research/exp29-minimax-effecterase-adapter-feasibility-20260626`
- HEAD at readback: `5e20149363b16f4728016260ff3e6d79dace299d`
- Worktree status before this report: clean
- Remote fetch: completed with `git fetch --all --prune`
- `git diff --check`: clean

Recent commits read:

- `5e20149 Plan MiniMax expanded source pool`
- `286ecf3 Validate EffectErase smoke command`
- `8509c37 Materialize EffectErase smoke inputs`
- `0033fc7 Record Exp29 continuation v3 readback`
- `972deab Preregister EffectErase inference smoke`

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

- `reports/exp29_continuation_v3_readback.md`
- `reports/exp29_effecterase_smoke_input_materialization.md`
- `reports/exp29_effecterase_command_dryrun.md`
- `reports/exp29_effecterase_smoke_preregistration.md`
- `reports/exp29_effecterase_weight_recovery.md`
- `reports/exp29_architecture_family_audit.md`
- `reports/exp29_minimax_expanded_source_pool_plan.md`
- `reports/exp29_minimax_preference_data_quality.md`
- `reports/exp29_minimax_preference_video_review.csv`
- `reports/exp29_minimax_effecterase_adapter_summary.md`

Code pointers reread:

- `exp29_or_adapter_feasibility/code/exp29_status.py`
- `exp29_or_adapter_feasibility/scripts/exp29_status_report.py`
- `exp29_or_adapter_feasibility/tests/test_exp29_scaffold.py`
- EffectErase official runtime repo path from prior dry-run:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/third_party/EffectErase`
- MiniMax prior candidate and source manifests under
  `exp29_or_adapter_feasibility/manifests/`

## Current EffectErase State

- Prior status: `EFFECTERASE_COMMAND_READY_BUT_SMOKE_BLOCKED_BY_INPUT_VALIDITY`
- Old preregistered manifest:
  `exp29_or_adapter_feasibility/manifests/effecterase_smoke_preregistered.jsonl`
- Old manifest SHA256:
  `54fd62a97fa69f2f17590488136d426cee77de0ed02548c46a83d8818be2b137`
- Rows: 6
- Ready rows from prior materialization: 5/6
- Blocked row: `REAL_ENV249_00103_004_04`
- Blocker: locked mask is empty across all 17 materialized frames.
- Official EffectErase command dry-run status: `EFFECTERASE_COMMAND_READY`
- Recovered weights status: `EFFECTERASE_WEIGHTS_READY`

This round may create a new v2 manifest, but must not overwrite the old
manifest or edit the empty mask row. V2 may proceed only after a replacement row
passes non-empty-mask, 17-frame, resolution, alignment, no-VOR-Eval, and
diagnostic-only checks.

## Current MiniMax State

- Prior status: `MINIMAX_EXPANDED_GENERATION_BLOCKED`
- Prior expanded source manifest:
  `exp29_or_adapter_feasibility/manifests/minimax_expanded_source_pool_v2.jsonl`
- Prior expanded source manifest SHA256:
  `bb31cfa5abd320dc88a5471036a3b2bb54b91257d3f65380dc43ecdf29c60929`
- Prior blocker: only 31 unused valid rows remained from the small Exp25
  64-row semantic audit.
- MiniMax previous 96-candidate gate remains data-yield insufficient:
  27 eligible candidates but only 9 unique eligible scene groups.

This round may do a read-only full-VOR source audit using existing indexes or
committed/cached metadata. It must not rescan the 363GB archive unless
necessary, must not write into Exp25 worktrees, and must not run MiniMax
generation unless the full-VOR audit finds at least 128 valid candidate groups.

## Left CLI Protection

PAI hostname:

- `dsw-753014-85f54df947-bkp7h`

Observed left monitor:

- PID `258013`
- Command: `bash /mnt/nas/hj/H20_Video_inpainting_DPO/runtime/cli4/cli4_remote_monitor_5min.sh`

Observed left runtime locks reserve:

- GPU1
- GPU2
- GPU3
- GPU4

Representative lock paths read-only:

- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/cli4/gpu1.lock`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/cli4/gpu1_2.lock`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/cli4/gpu2.lock`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/cli4/gpu3_4.lock`

No left-side process signal was sent. No left-side worktree, branch, lock,
heartbeat, output, or controller file was modified.

## GPU State

HAL check:

- Hostname: `hal-9000`
- GPU0 has unknown non-Exp29 GPU memory holders and is not selected from HAL.

PAI check:

- Hostname: `dsw-753014-85f54df947-bkp7h`
- GPUs 0-7 reported 0 MiB memory used and no compute-app rows.
- Because left runtime locks reserve GPU1-GPU4, right Exp29 may only consider
  GPU0/GPU5/GPU6/GPU7 after the required repeated availability checks.
- Maximum right-side parallelism remains two GPUs.

## This Round Milestones

1. `EffectErase smoke v2 input audit and re-preregistration`
2. `EffectErase smoke v2 input materialization`
3. `EffectErase official inference smoke v2`
4. `EffectErase trainable-forward audit v2`, only if smoke technical-valid
   count is at least 5/6
5. `MiniMax full-VOR source audit`
6. `MiniMax expanded candidate first pass`, only if the full-VOR audit finds
   at least 128 valid candidate groups

## Banned Repeats And Claims

This round must not:

- redo architecture-family audit;
- redo EffectErase weight recovery;
- redo EffectErase command dry-run unless code/environment changed;
- start MiniMax recipe search;
- start MiniMax 30-step;
- start EffectErase adapter training;
- start long training or RC-FPO;
- modify `inference/metrics.py`, shared trainer, or Exp1-Exp28;
- overwrite old manifests, recovered weights, or previous outputs;
- use VOR-Eval for training or tuning;
- claim universal adapter, all models supported, final SOTA, or top-conference
  novelty confirmed.

## Promotion Gates

EffectErase can become `EFFECTERASE_OR_BASELINE_READY` only after v2 inference
runs on six valid rows, outputs decode, metrics are computed, and per-video
review finds technical valid count at least 5/6 with no systemic artifact.

MiniMax can become `MINIMAX_EXPANDED_MICRO_DATA_READY` only after a committed
full-VOR source audit and generation/review of expanded candidates produce at
least 32 eligible unique scene groups and scene-disjoint train16/heldout16
manifests. No training is allowed in this prompt even if data becomes ready.
