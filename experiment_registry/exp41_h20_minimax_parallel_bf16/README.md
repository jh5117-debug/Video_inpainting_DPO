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

## 2026-06-29 Official Protocol Gate

`H20_MINIMAX_PROTOCOL_MATCHES_OFFICIAL` passed for the executable official
README/test protocol: `UniPCMultistepScheduler`, `float16`,
`num_inference_steps=12`, and `iterations=6`. The README prose claim of
"6 inference steps" is recorded as a diagnostic ambiguity only.

This gate does not claim MiniMax quality improvement. The next allowed lane is
the gated SFT-only bad-noise ladder after fresh readback and H20 GPU audit.

## 2026-06-29 SFT Ladder Blocker

`H20_MINIMAX_SFT_BLOCKED` was recorded before training. Existing MiniMax
training scripts cap SFT/DPO micro gates at 10 steps, but Lane A requires a
30-step SFT gate and longer gated continuations. Exp41 wrote a patch proposal
instead of changing training source.
