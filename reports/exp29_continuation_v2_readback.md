# Exp29 Continuation V2 Readback

Date: 2026-06-26

Status: `EXP29_CONTINUATION_V2_READBACK_COMPLETED`

No GPU inference, candidate generation, trainable-forward gate, adapter step, or
training task was launched before this readback.

## Git State

- Branch: `research/exp29-minimax-effecterase-adapter-feasibility-20260626`
- HEAD: `6c97d4b74f331ce4db089224f7dcf9ec6eb283ce`
- Worktree: `/home/hj/H20_Video_inpainting_DPO_exp29_or_adapters`
- Fetch: completed with `git fetch --all --prune`
- Worktree status at readback: clean

Recent commits:

- `6c97d4b` Recover EffectErase weights
- `cc8d4d6` Build MiniMax medium-hard micro preferences
- `c05630c` Analyze MiniMax 10-step micro failure
- `bd44f67` Record Exp29 continuation readback
- `4b8d68a` Run MiniMax adapter gates

## Files Read

PRD:

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/49_exp29_or_adapter_feasibility.md`

Registry:

- `experiment_registry/exp29_or_adapter_feasibility/status.md`
- `experiment_registry/exp29_or_adapter_feasibility/paths.yaml`
- `experiment_registry/exp29_or_adapter_feasibility/config.yaml`
- `experiment_registry/exp29_or_adapter_feasibility/results.tsv`
- `experiment_registry/exp29_or_adapter_feasibility/metric_summary.md`
- `experiment_registry/exp29_or_adapter_feasibility/qualitative_summary.md`

Previous reports:

- `reports/exp29_continuation_readback.md`
- `reports/exp29_minimax_10step_failure_analysis.md`
- `reports/exp29_minimax_next_micro_plan.md`
- `reports/exp29_minimax_preference_data_quality.md`
- `reports/exp29_minimax_preference_data_quality.csv`
- `reports/exp29_minimax_preference_video_review.csv`
- `reports/exp29_minimax_preference_data_quality_summary.json`
- `reports/exp29_effecterase_weight_recovery.md`
- `reports/exp29_effecterase_weight_recovery.csv`
- `reports/exp29_effecterase_weight_recovery.json`
- `reports/exp29_minimax_effecterase_adapter_summary.md`

## Current Model Status

MiniMax:

- Repo ready.
- Weights ready.
- Inference smoke passed with visual quality risks.
- Trainable forward passed.
- Target audited as flow velocity `epsilon - z0`.
- Zero-gap and one-step strict reload passed.
- Previous 10-step was `MINIMAX_10STEP_PARETO_MIXED`.
- Expanded 32-source data gate produced 27 eligible seed-level candidates but
  only 9 eligible source groups.
- Current status: `MINIMAX_DATA_YIELD_INSUFFICIENT`.
- Recipe search and 30-step micro remain forbidden until a scene-disjoint
  train16/heldout16 medium-hard pool exists.

EffectErase:

- Repo ready at commit `bcee0a5da5ef387c2ba39390dc4d579503669fb8`.
- License: CC BY-NC 4.0.
- Official weights recovered and SHA-verified.
- Asset root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp29_or_adapter_feasibility/effecterase_weight_recovery_20260626`
- Current status: `EFFECTERASE_WEIGHTS_READY`.
- No inference smoke, trainable forward, zero-gap, one-step, or adapter micro
  has run yet.

## Left CLI Protection

Left CLI was inspected read-only only.

- Controller path: `/home/hj/cli4_controller`
- Runtime path: `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/cli4`
- Observed process: PID `258013`,
  `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/cli4/cli4_remote_monitor_5min.sh`
- Runtime locks still reserve GPU1/GPU2/GPU3/GPU4, including historical Exp28
  locks and `gpu3_4.failed_final_iowait_recurred_20260627_012316.json`.
- No signal was sent.
- No left worktree, branch, output, lock, heartbeat, or controller file was
  modified.

## GPU State

PAI hostname: `dsw-753014-85f54df947-bkp7h`

At readback, GPUs 0-7 reported `0 MiB` used and no compute process. Despite
that, GPU1/GPU2/GPU3/GPU4 remain reserved for the left CLI because of runtime
locks. Right-side Exp29 may only use non-left-reserved idle GPUs, with at most
two concurrent GPU tasks.

## Current Milestones

Allowed next milestones:

1. Architecture-family audit.
2. EffectErase smoke pre-registration.
3. EffectErase official inference smoke if pre-registration and assets pass.
4. MiniMax expanded source-pool plan.
5. MiniMax expanded candidate generation and review only after the plan is
   committed.

Banned repeats and claims:

- No long training.
- No RC-FPO.
- No VideoPainter continuation.
- No MiniMax recipe / 30-step until expanded data is ready.
- No EffectErase adapter claim from weight recovery or inference smoke alone.
- No `UNIVERSAL_ADAPTER`, `ALL_MODELS_SUPPORTED`, `FINAL_SOTA`, or
  `TOP_CONFERENCE_NOVELTY_CONFIRMED` language.
