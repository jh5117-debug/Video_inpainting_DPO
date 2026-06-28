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
