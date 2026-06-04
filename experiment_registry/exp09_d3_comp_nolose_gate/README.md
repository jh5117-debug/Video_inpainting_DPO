# Exp9 D3 comp no-lose-gap gate

Short name: `d3_comp_nolose_gate`

Status: `h20_gate_complete_eval_pending`

## What This Experiment Does

Ablate lose_gap_weight=0 to test winner-only/winner-preserving DPO.

## How It Was Run / Intended

D3 comp data, partial-mask task, full loss, Stage1-only gate1000, lose_gap_weight=0.

## Current Result

Gate artifacts and dpo_diag found on H20; target-domain eval still pending in registry.

## Conclusion Boundary

Training gate exists; diagnostic purpose is to test whether removing loser-gap reduces shortcut behavior.

## Next Action

Run target-domain eval only after registry review; compare to Exp9 comp ckpt500.
