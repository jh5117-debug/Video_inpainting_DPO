# Exp41 H20 MiniMax Parallel BF16

H20-only MiniMax adapter/debug track based on Exp40. PAI is read-only. The
track may write Exp41 reports, configs, launchers, registry files, and runtime
outputs only.

Forbidden in this branch:

- modifying MiniMax official source files;
- modifying `inference/metrics.py`;
- modifying shared trainers;
- rewriting Exp1-Exp40 source/result history;
- claiming universal adapter, final SOTA, or top-conference novelty.

## 2026-06-29 Data Ready Gate

`H20_MINIMAX_DATA_READY` passed. H20 mirror and MiniMax weights are ready for
BF16/SIGFPE preflight; no training has been launched.

## 2026-06-29 BF16 Safe Gate

`H20_MINIMAX_BF16_SAFE_READY` passed for P0-P7, including DDP8 one-batch
MiniMax training. This does not authorize quality claims; official protocol
audit remains pending.
