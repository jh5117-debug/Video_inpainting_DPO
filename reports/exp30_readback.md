# Exp30 Readback

Date: 2026-06-27

Status: `EXP30_READBACK_COMPLETED`

## Git

- Branch: `research/exp30-vor-or-multimodel-minimax-adapter-20260627`
- Start HEAD: `6bc6c67c60b5cf2fe8d937ffd1e1d88a4684991c`
- Base branch: `origin/research/exp29-minimax-effecterase-adapter-feasibility-20260626`
- Worktree: `/home/hj/H20_Video_inpainting_DPO_exp30_vor_or_minimax`
- Worktree status before scaffold: clean

## Files Read

Core PRD / registry files read or checked:

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/48_exp26_videopainter_dpo_v2.md`
- `PRD/49_exp29_or_adapter_feasibility.md`
- `experiment_registry/exp26_videopainter_dpo_v2/status.md`
- `experiment_registry/exp29_or_adapter_feasibility/status.md`

Missing in this branch and recorded as missing rather than fabricated:

- `PRD/47_exp25_vor_or_preference_data.md`
- `experiment_registry/exp25_vor_or_preference_data/status.md`
- `reports/exp25_gate32_yield_review_20260624.md`
- `reports/exp25_diffueraser_or_root_cause_matrix_v2.md`

Reports read or checked:

- `reports/exp29_effecterase_official81_inference_smoke.md`
- `reports/exp29_effecterase_official81_inference_visual_review.csv`
- `reports/exp29_effecterase_official81_aggregate_metrics.csv`
- `reports/exp29_effecterase_trainable_forward_audit.md`
- `reports/exp29_effecterase_trainable_forward_audit.json`
- `reports/exp29_minimax_expanded_data_quality_v2.md`
- `reports/exp29_minimax_expanded_video_review_v2.csv`
- `reports/exp29_minimax_expanded_data_quality_summary_v2.json`
- `reports/exp29_minimax_trainable_forward_audit.md`
- `reports/exp29_minimax_zero_gap_gate.md`
- `reports/exp29_minimax_one_step_gate.md`
- `reports/exp26_vp_shadowdev_final_decision.md`
- `reports/exp26_vp_shadowdev_metrics_and_statistics.md`
- `reports/exp26_vp_shadowdev_visual_review.md`

## Current Source-Of-Truth

### EffectErase

- `EFFECTERASE_OR_BASELINE_READY` from official 81F smoke.
- `EFFECTERASE_BASELINE_ONLY_FOR_NOW` from trainable-forward audit.
- Official removal path has removal-specific adapters/task tokens but no
  removal-specific `training_loss`.
- Generic Wan training is not an acceptable substitute for EffectErase removal
  adapter training.
- EffectErase remains OR strong baseline / diagnostic / upper reference only.

### MiniMax

- Repo/weights/inference/trainable-forward/zero-gap/one-step are ready.
- Native target: flow velocity `epsilon - z0`.
- MiniMax-only data mining remains insufficient:
  128 attempts, 24 medium-hard, 2 hard-plausible, 26 eligible scene groups,
  below the 32 groups required for scene-disjoint train16+heldout16.
- Exp30 must solve data yield using a multi-model OR pool.

### VideoPainter

- `VIDEOPAINTER_SHADOWDEV_CONFIRMED`.
- Search/shadow-dev evidence supports cross-backbone adapter evidence together
  with DiffuEraser.
- External DAVIS-derived validation was not confirmed.
- VideoPainter is VOR-BG/BR-style evidence, not standard VOR-OR.

### DiffuEraser

- DiffuEraser is the primary original backbone in the project lineage.
- Exp30 needs VOR-OR Stage1/Stage2 micro evidence before treating it as VOR-OR
  paper evidence alongside MiniMax.

## Left CLI Protection

PAI readback:

- Hostname: `dsw-753014-85f54df947-bkp7h`
- Compute processes: none at readback.
- Left CLI runtime locks present under
  `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/cli4`.
- Reserved GPUs from locks: GPU1, GPU2, GPU3, GPU4.
- Signals sent: no.
- Left files modified: no.

## This Round Milestones

1. Paper three-backbone role positioning.
2. VOR-OR source pool and affected-region audit.
3. Multi-model OR candidate smoke.
4. Gate128 multi-model candidate generation if smoke passes.
5. MiniMax adapter gate if the pool is ready.
6. DiffuEraser VOR-OR Stage1/Stage2 micro if the pool is ready.
7. Paper-ready three-backbone evidence summary.

## Forbidden Actions

- EffectErase adapter training.
- Generic Wan training as EffectErase adapter.
- VOR-Eval training or selection.
- Long training: 500/1000/2000 steps.
- RC-FPO.
- MiniMax 30/50-step before a positive 10-step gate.
- Modifying Exp1-Exp28, shared trainer, or `inference/metrics.py`.
- Universal-adapter, all-models-supported, final-SOTA, or
  top-conference-novelty-confirmed claims.

