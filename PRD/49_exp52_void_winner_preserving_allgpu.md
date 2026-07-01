# Exp52 VOID Winner-Preserving All-GPU Rescue

Branch: `research/exp52-void-winner-preserving-allgpu-20260701`

Status: `EXP52_ALL_GPU_READY`

Goal: fix Exp51 slow-forward/no-checkpoint blocker, cache VOID-native training inputs, run bounded winner-preserving rescue objectives, and only unlock 10-step if one-step video/metric gates pass.

Forbidden: VOR-Eval for training/filtering/tuning, hard comp, long training, official VOID source edits, shared trainer edits, `inference/metrics.py` edits, universal adapter claims, final SOTA claims, and VOID third-adapter evidence without a true heldout-positive micro gate.

## Milestone A - Readback and GPU Audit

Status: `EXP52_ALL_GPU_READY`

H20 GPU0-7 are free and no stale Exp50/51/52 GPU processes were killed. Exp52 will not run long training before cache and row0 smoke gates pass.
