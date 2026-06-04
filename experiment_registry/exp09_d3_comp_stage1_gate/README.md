# Exp9 D3 comp Stage1 gate

Short name: `d3_comp_stage1_gate`

Status: `pai_clean_gate_complete_eval_complete_diag_pending`

## What This Experiment Does

Move from D2 to target-domain YouTube-VOS D3 comp data and test a short Stage1 DPO window.

## How It Was Run / Intended

Partial-mask task, full loss, winner anchoring, Stage1-only gate1500 with checkpoints 500/1000/1500.

## Current Result

ckpt500 beat base on mask/whole metrics in pasted PAI summary; 1000/1500/last were worse than ckpt500, supporting short-window conclusion.

## Conclusion Boundary

Best metric checkpoint is ckpt500; longer steps degraded several metrics.

## Next Action

Recover PAI dpo_diag; compare against nocomp and no-lose using target-domain report.
