# Exp30 Continuation V3 Readback

Date: 2026-06-27

Status: `EXP30_CONTINUATION_V3_READBACK_COMPLETED`

## Git State

- Branch: `research/exp30-vor-or-multimodel-minimax-adapter-20260627`
- Start HEAD: `bd8777274dfe898dc9278cadcc1dd971536a5e2c`
- Worktree status at readback: clean.
- Latest commits read:
  - `bd87772 Run Exp30 multi-model OR smoke16 v2`
  - `ed5f19d Materialize Exp30 smoke16 repaired sources`
  - `a1eeb9f Repair Exp30 smoke16 technical source rows`

## Files Read

PRD and registry:

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/50_exp30_vor_or_multimodel_minimax.md`
- `experiment_registry/exp30_vor_or_multimodel_minimax/status.md`
- `experiment_registry/exp30_vor_or_multimodel_minimax/README.md`
- `experiment_registry/exp30_vor_or_multimodel_minimax/metric_summary.md`
- `experiment_registry/exp30_vor_or_multimodel_minimax/qualitative_summary.md`
- `experiment_registry/exp30_vor_or_multimodel_minimax/results.tsv`

Previous reports:

- `reports/exp30_multimodel_or_smoke16_v2.md`
- `reports/exp30_multimodel_or_smoke16_summary_v2.json`
- `reports/exp30_multimodel_or_smoke16_metrics_v2.csv`
- `reports/exp30_multimodel_or_smoke16_visual_review_v2.csv`
- `reports/exp30_controlled_corruption_smoke16_v2.md`
- `reports/exp30_minimax_smoke16_v2_metrics.csv`
- `reports/exp30_vor_or_source_pool_v2_sampling.md`
- `reports/exp30_full_vor_index_recovery.md`
- `reports/exp30_three_backbone_paper_positioning.md`

Code and wrappers:

- `exp30_vor_or_multimodel_minimax/code/*`
- `exp30_vor_or_multimodel_minimax/scripts/*`
- `DPO_finetune/generate_multimodel_dpo_dataset.py`
- `inference/run_OR.py`
- `inference/metrics.py` read-only

## Current Source Rows

- Smoke16 final rows: 16.
- Scene groups: 16.
- Source balance: BLENDER 8, REAL 8.
- Materialization: 16/16 success, 17 frames, 512 x 512.
- Materialized manifest SHA256: `72be9884335fef61926c307c66878fdc05dec85e9be4da28ab1547db98f8c26d`.
- Current source rows remain technically credible for smoke16 v3. They were repaired before model-output review and have no failed materialization rows.

## Why Smoke16 V2 Was Blocked

Smoke16 v2 was not blocked by decode, frame-count, mask alignment, or missing-output failures. It was blocked by quality yield:

- Non-EffectErase candidates: 32/32 technical-valid.
- Total usable non-EffectErase candidates: 9/32.
- Controlled corruption: 5/16 usable, below the preregistered fallback requirement of at least 6/16.
- MiniMax official: 4/16 usable, documented as low-yield.
- Gate64, MiniMax recipe/training, DiffuEraser VOR-OR micro, long training, and RC-FPO remain stopped.

## Controlled Corruption Failure Pattern

Controlled corruption was technically valid and preserved outside pixels by construction, but it was too often not medium-hard:

- Usable: 5/16.
- Medium-hard: 3.
- Hard-plausible: 2.
- Trivial-bad: 11.
- Dominant failure mode from the v2 CSV/review: temporal instability and hard local residuals.
- The likely generator issue is overly aggressive or poorly temporally-smoothed region corruption, especially on small/structured masks, rather than outside-region leakage.

This makes controlled corruption a candidate fallback/data source only after calibration. It is not a final method and cannot be treated as ground truth.

## MiniMax Failure Pattern

MiniMax official was also technically valid but low-yield:

- Usable: 4/16.
- Medium-hard: 3.
- Hard-plausible: 1.
- Trivial-bad: 12.
- Dominant failure modes recorded in v2 evidence: residual object/effect, black or smudged local artifacts, too-close outputs, and local texture hallucination.
- MiniMax remains a flow-style adapter candidate, not quality-positive evidence.

## Generator Stack Status

- DiffuEraser OR candidate stack: not yet verified in Exp30. Must audit Exp25/Exp30 stack identity before any smoke16 v3 use.
- ProPainter: wrapper and weights were observed in the repository/weights tree, but v2 did not launch it. It requires a 2-sample verified smoke before enabling in smoke16 v3.
- EffectErase: diagnostic/baseline only in this lane. It is trained on VOR and must not count for smoke pass or training-pair selection.

## Protected Lanes

Read-only PAI audit found:

- Left CLI runtime exists under `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/cli4`.
- Left CLI locks reserve GPU1/GPU2/GPU3/GPU4.
- Exp31 VideoPainter 2000-step is running on GPU1, PID `1215136`.
- Exp33 EffectErase VOR-Eval baseline is running on GPU3, launcher PGID `1349871`, active inference PID `1414858`.
- No signal was sent. No protected file was modified.

For Exp30, GPU1-GPU4 are reserved. Eligible GPUs, subject to fresh checks before any future GPU task, are GPU0/GPU5/GPU6/GPU7 only.

## Why Gate64 Cannot Start Now

Gate64 requires a passed smoke gate. Smoke16 v2 is explicitly blocked, and smoke16 v3 has not been preregistered or run. Starting Gate64 now would bypass the preregistered medium-hard quality gate and would overfit a known low-yield candidate generation setup.

## Overfitting Controls For V3

- Keep the repaired smoke16 source rows unless a source becomes technically invalid.
- Analyze v2 failure modes before modifying generation.
- Pre-register at most four controlled-corruption profiles.
- Limit controlled-corruption v3 to at most two candidates per source in smoke16 v3.
- Enable DiffuEraser/ProPainter only after verified stack audit and smoke2.
- Do not use VOR-Eval.
- Do not use EffectErase for promotion or primary training losers.
- Do not start Smoke32, Gate64, or MiniMax adapter gates unless the preceding gate passes.

## Next Milestones

1. Analyze smoke16 v2 per-candidate failure modes.
2. Lock controlled-corruption v3 calibration profiles.
3. Audit DiffuEraser and ProPainter candidate stacks.
4. Preregister smoke16 v3.
5. Only then run smoke16 v3.

No GPU generation or training was launched by this readback.
