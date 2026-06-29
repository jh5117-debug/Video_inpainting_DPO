# Exp44 Same-Source Pair Construction

- Status: MINIMAX_SAME_SOURCE_PAIR_GATE_PASSED
- Usable same-source pairs: 40
- Minimum gate: 24
- Target: 48
- Max pairs per group: 4
- Split counts: train=24, search=8, shadow=8
- Split group overlap ok: True

## Pairing Rule

Winner is the GT background path for DPO preference construction. The MiniMax successful-removal output is preserved as `pseudo_success_path` for possible Stage2-style distillation, but it is not treated as GT.

Loser is a same-source MiniMax raw output labeled `FAILURE_MEDIUM_HARD` after visual relabeling. Cross-source pairing is not used.

## Safety

No training, optimizer step, bad-noise scan, Stage2 handoff, hard comp, or VOR-Eval use occurred in this milestone.
