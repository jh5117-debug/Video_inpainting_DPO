# Exp45 PAI MiniMax Pair Scale-Up Helpers

This directory is reserved for Exp45-only helper scripts and manifests.

Final handoff status: `MINIMAX_STAGE2_FORMAL_DATA_READY`.

The manifest set under `manifests/` contains the formal `64/24/24`
pseudo-success, GT distillation, and same-source preference splits for a later
H20 mirror and pseudo-success SFT 30-step preflight.

Rules:

- no H20 execution;
- no training;
- no optimizer step;
- no edits to shared trainer, `inference/metrics.py`, or MiniMax official repo.
