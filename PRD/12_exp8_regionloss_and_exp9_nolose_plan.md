# Exp8 Region-Loss And Exp9 No-Lose Plan

Updated: 2026-06-04

## Current Pre Conclusion

The current weekly conclusion is:

- DPO has a short useful window.
- Long direct DPO training degrades.
- DPO Stage2 should not be trained under the current objective.
- Current best DPO-S1 candidate is still Exp9 D3-comp checkpoint-500, pending
  full PAI artifact registration.

## Artifact Registry Addendum

Do not launch or present additional results from this plan until the artifact
registry is clean:

- `PRD/12_experiment_artifact_registry.md`
- `PRD/13_dpo_diag_audit.md`
- `PRD/14_pai_manual_artifact_search_commands.md`

Current audit status:

- Exp8 region-loss: H20 scan found launcher evidence but no completed run
  folder, no dpo-diag, and no target eval output. Treat as **missing / pending
  PAI search**, not completed.
- Exp9 no-lose H20: H20 scan found a run folder and `dpo_diagnostics.csv` at
  `experiments/dpo/stage1/20260604_080411_exp9_youtubevos_d3_comp_wingap_nolose_stage1_gate1000_h20_stage1`.
- PAI clean Exp9 comp: needs PAI manual artifact search before final registry
  and PPT quantitative claim.

Artifact rule:

- No dpo-diag means no complete DPO result.
- No independent folder means artifact gap.
- No target eval report means no target-domain conclusion.

## Why Exp8 Exists

Exp8 is a target-domain D3 region-loss diagnostic, not a VideoDPO bridge
experiment. It tests whether mask/boundary-weighted loss can improve
partial-mask inpainting stability.

## Why No-Lose-Gap Exists

No-lose-gap removes the loser-degradation reward shortcut by setting:

```text
LOSE_GAP_WEIGHT=0.0
```

It tests whether winner-preserving DPO is more stable than the current
`lose_gap_weight=0.25` setting.

## Metric Policy

For Exp8/Exp9 video inpainting:

```text
tools/run_inpainting_metric_eval.py
inference/metrics.py
```

Do not use VBench for partial-mask inpainting.

