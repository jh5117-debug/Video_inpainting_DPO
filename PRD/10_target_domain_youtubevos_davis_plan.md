# Target-Domain YouTube-VOS / DAVIS Plan

Updated: 2026-06-04 with artifact registry repair.

## Metric Policy

For video inpainting / partial-mask inpainting:

```text
tools/run_inpainting_metric_eval.py
inference/metrics.py
```

Do not use VBench for Exp8/Exp9 partial-mask inpainting gates.

For video generation / full-mask bridge:

```text
VBench + DPO diagnostics + qualitative side-by-side
```

## Current Target-Domain Status

| Item | Status | Artifact note |
| --- | --- | --- |
| Exp9 D3 nocomp H20 | training complete; target eval folder found | H20 dpo-diag found; target eval at `logs/target_eval/exp9_d3_nocomp_gate_h20_20260604_023243`. |
| Exp9 D3 comp PAI clean | reportedly completed and ckpt500 was best by target metric | Needs PAI manual artifact search to register run folder, dpo-diag, eval outputs. |
| Exp9 D3 comp no-lose H20 | training folder and dpo-diag found | Eval status must be checked separately; do not claim final result without eval report. |
| Exp8 region loss | launcher found, completed artifact not found on H20 | Do not claim result until PAI returns artifact evidence. |

## Artifact Gate Before Any Next Experiment

Before launching any longer sweep:

1. Register independent folder for the experiment.
2. Register dpo-diag CSV.
3. Register target eval output folder.
4. Register qualitative side-by-side folder.
5. Compare against DiffuEraser-base and current best Exp9 comp ckpt500.

If the above artifacts are missing, the next action is artifact recovery, not
new training.

