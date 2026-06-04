# Exp7 D2 partial-mask task with winner anchoring

Short name: `d2_partialmask_task_wingap`

Status: `partial_metrics_complete_pai_diag_pending`

## What This Experiment Does

Change the task so DiffuEraser sees the same partial mask used to create the loser.

## How It Was Run / Intended

Reuse D2 comp data, train with partial mask from manifest and full DPO loss.

## Current Result

Stage1_last beat base on true partial-mask metrics; Stage2_last regressed.

## Conclusion Boundary

Task alignment helps Stage1, but DPO Stage2 regresses.

## Next Action

Keep Stage2 DPO stopped; recover dpo_diag CSV from PAI/NAS if available.
